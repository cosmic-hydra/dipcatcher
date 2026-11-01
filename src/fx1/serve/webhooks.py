"""Signed job-completion webhooks — sign + verify for the harness's
``callback_url`` deliveries.

When a job submission carries ``callback_secret``, the POSTed job record
arrives with two headers:

- ``X-Fx1-Webhook-Timestamp`` — unix seconds at delivery time
- ``X-Fx1-Webhook-Signature`` — ``sha256=<hmac-sha256 hex>`` over
  ``<timestamp>.<raw body>``

Receivers authenticate a delivery with :func:`verify_webhook` over the
*raw* request body (no re-serialization — signature bytes are exact).
"""

from __future__ import annotations

import hashlib
import hmac
import http.client
import ipaddress
import math
import os
import socket
import ssl
import urllib.parse

__all__ = [
    "WEBHOOK_BACKOFF_S",
    "WEBHOOK_MAX_ATTEMPTS",
    "WEBHOOK_SIGNATURE_HEADER",
    "WEBHOOK_TIMESTAMP_HEADER",
    "check_callback_url",
    "deliver_signed",
    "sign_webhook",
    "verify_webhook",
]

WEBHOOK_TIMESTAMP_HEADER = "X-Fx1-Webhook-Timestamp"
WEBHOOK_SIGNATURE_HEADER = "X-Fx1-Webhook-Signature"
WEBHOOK_MAX_ATTEMPTS = 3
WEBHOOK_BACKOFF_S = 0.5

_DEFAULT_TOLERANCE_S = 300.0  # 5 min — rejects replayed stale deliveries
_ALLOW_PRIVATE_NETWORKS_ENV = "FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS"


def _private_networks_allowed() -> bool:
    """Return whether this process explicitly permits private callbacks."""
    return os.environ.get(_ALLOW_PRIVATE_NETWORKS_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _is_public_unicast(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Classify an address conservatively for an outbound callback."""
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_global and not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def _validate_literal_host(host: str) -> None:
    """Reject a special-use IP literal without performing DNS at submit time."""
    candidate = host.split("%", 1)[0]
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError:
        return
    if not _private_networks_allowed() and not _is_public_unicast(address):
        raise ValueError("callback_url must not target a private or special-use address")


def _resolved_addresses(host: str, port: int) -> tuple[str, ...]:
    """Resolve a host and return only validated, distinct numeric addresses."""
    addresses: list[str] = []
    for family, socktype, _proto, _canonname, sockaddr in socket.getaddrinfo(
        host,
        port,
        type=socket.SOCK_STREAM,
    ):
        if family not in (socket.AF_INET, socket.AF_INET6) or socktype != socket.SOCK_STREAM:
            continue
        raw = str(sockaddr[0])
        candidate = raw.split("%", 1)[0]
        address = ipaddress.ip_address(candidate)
        if not _private_networks_allowed() and not _is_public_unicast(address):
            raise ValueError(
                f"callback_url resolved to a private or special-use address: {address}"
            )
        normalized = str(address)
        if normalized not in addresses:
            addresses.append(normalized)
    if not addresses:
        raise OSError(f"callback_url host {host!r} did not resolve to an IP address")
    return tuple(addresses)


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """HTTP connection whose TCP destination is the validated numeric address."""

    source_address: tuple[str, int] | None

    def __init__(self, host: str, port: int, address: str, timeout: float) -> None:
        super().__init__(host, port=port, timeout=timeout)
        self._address = address

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self._address, self.port),
            self.timeout,
            self.source_address,
        )


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection pinned to an IP while authenticating the URL host."""

    source_address: tuple[str, int] | None
    _context: ssl.SSLContext

    def __init__(self, host: str, port: int, address: str, timeout: float) -> None:
        context = ssl.create_default_context()
        # Explicit contexts must advertise the protocol used by http.client.
        context.set_alpn_protocols(["http/1.1"])
        super().__init__(
            host,
            port=port,
            timeout=timeout,
            context=context,
        )
        self._address = address

    def connect(self) -> None:
        raw_sock = socket.create_connection(
            (self._address, self.port),
            self.timeout,
            self.source_address,
        )
        self.sock = self._context.wrap_socket(raw_sock, server_hostname=self.host)


def _post_once(
    parsed: urllib.parse.ParseResult,
    address: str,
    body: bytes,
    headers: dict[str, str],
    timeout_s: float,
) -> int:
    """POST once to a previously validated address; never follow redirects."""
    host = parsed.hostname
    if host is None:  # defensive; check_callback_url rejects this first
        raise ValueError("callback_url has no host")
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    connection_cls = _PinnedHTTPSConnection if parsed.scheme == "https" else _PinnedHTTPConnection
    connection = connection_cls(host, port, address, timeout_s)
    target = urllib.parse.urlunparse(("", "", parsed.path or "/", parsed.params, parsed.query, ""))
    try:
        connection.request("POST", target, body=body, headers=headers)
        # Delivery depends only on status; an untrusted response body may be
        # unbounded. This connection is never reused, so close without draining.
        with connection.getresponse() as response:
            return response.status
    finally:
        connection.close()


def _signed_payload(timestamp: str, body: bytes) -> bytes:
    return timestamp.encode("ascii") + b"." + body


def sign_webhook(secret: str, timestamp: str, body: bytes) -> str:
    """Return ``sha256=<hex>`` — the signature header value."""
    sig = hmac.new(
        secret.encode("utf-8"), _signed_payload(timestamp, body), hashlib.sha256
    ).hexdigest()
    return f"sha256={sig}"


def verify_webhook(
    secret: str,
    timestamp: str | None,
    signature: str | None,
    body: bytes,
    *,
    tolerance_s: float = _DEFAULT_TOLERANCE_S,
    now: float | None = None,
) -> bool:
    """Authenticate a webhook delivery using a constant-time signature compare.

    Timestamps and tolerances must be finite. Nonnegative ``tolerance_s``
    allows at most that many seconds of past or future clock skew (inclusive);
    a finite negative tolerance explicitly disables the freshness check.
    Any malformed input is a plain ``False`` — never an exception.
    """
    if not isinstance(secret, str) or not secret:
        return False
    if not isinstance(timestamp, str) or not timestamp or not timestamp.isascii():
        return False
    if not isinstance(signature, str) or not signature.isascii():
        return False
    if not signature.startswith("sha256="):
        return False
    try:
        ts = float(timestamp)
        if not math.isfinite(ts) or not math.isfinite(tolerance_s):
            return False
        if tolerance_s >= 0:
            import time  # noqa: PLC0415 — local import keeps the module leaf

            ref = time.time() if now is None else now
            if not math.isfinite(ref) or abs(ref - ts) > tolerance_s:
                return False
        expected = sign_webhook(secret, timestamp, body)
        return hmac.compare_digest(expected, signature)
    except (TypeError, ValueError, OverflowError):
        # Includes encoding failures (UnicodeError is a ValueError subclass).
        return False


def check_callback_url(url: str | None) -> str | None:
    """The shared ``callback_url`` field validator — a webhook target must
    be a real http(s) URL with a host. Returns ``url`` unchanged; raises
    ``ValueError`` on anything else."""
    if url is None:
        return url
    parsed = urllib.parse.urlparse(url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(
            f"callback_url must be an http(s) URL with a host and no "
            f"userinfo credentials, got {url!r}"
        )
    try:
        if parsed.port == 0:
            raise ValueError("port zero is not a callback destination")
    except ValueError as exc:
        raise ValueError(f"callback_url has an invalid port, got {url!r}") from exc
    _validate_literal_host(parsed.hostname)
    return url


def deliver_signed(
    url: str,
    secret: str | None,
    body: bytes,
    *,
    max_attempts: int = WEBHOOK_MAX_ATTEMPTS,
    backoff_s: float = WEBHOOK_BACKOFF_S,
    timeout_s: float = 10.0,
) -> tuple[bool, str | None, int]:
    """POST ``body`` to a webhook ``url`` — HMAC-signed when ``secret`` is
    set. Returns ``(delivered, error, attempts)``.

    Transient faults (network errors, 5xx) retry ``max_attempts`` times
    with capped exponential backoff; a 4xx is a definitive rejection and
    is never retried. Delivery faults return as the error string — this
    helper never raises into its caller's worker."""
    import time  # noqa: PLC0415 — local import keeps the module leaf

    error: str | None = None
    try:
        check_callback_url(url)
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname
        if host is None:
            raise ValueError("callback_url has no host")
        port = parsed.port
        if port is None:
            port = 443 if parsed.scheme == "https" else 80
    except Exception as exc:  # noqa: BLE001 — invalid targets are delivery data
        return False, f"{type(exc).__name__}: {exc}", 0

    for attempt in range(max_attempts):
        if attempt:
            time.sleep(backoff_s * (1 << (attempt - 1)))
        try:
            headers = {"Content-Type": "application/json"}
            if secret:
                ts = str(int(time.time()))
                headers[WEBHOOK_TIMESTAMP_HEADER] = ts
                headers[WEBHOOK_SIGNATURE_HEADER] = sign_webhook(secret, ts, body)
            addresses = _resolved_addresses(host, port)
            last_exception: Exception | None = None
            for address in addresses:
                try:
                    status = _post_once(parsed, address, body, headers, timeout_s)
                except Exception as exc:  # noqa: BLE001 — try another validated address
                    last_exception = exc
                    continue
                if status < 300:
                    return True, None, attempt + 1
                error = f"callback endpoint returned {status}"
                if 400 <= status < 500:
                    return False, error, attempt + 1
                break
            else:
                if last_exception is not None:
                    raise last_exception
        except Exception as exc:  # noqa: BLE001 — delivery faults are data, never raised
            error = f"{type(exc).__name__}: {exc}"
    return False, error, max_attempts
