"""Inference backends for fx-1.

Two backends, one interface:
- ``HostedK3Backend`` — Moonshot's hosted K3 API. Serves as fx-1's teacher
  (tool-use trace generation) and as the day-one serving path.
- ``LocalFx1Backend`` — a fine-tuned fx-1 checkpoint. Fail-closed until a
  checkpoint with a valid model card exists on disk.

Credentials come from environment variables only — never hardcoded.
"""

from __future__ import annotations

import contextlib
import http.client
import ipaddress
import json
import os
import shlex
import socket
import ssl
import string
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

from fx1.modelcard import ModelCard

MOONSHOT_API_URL = "https://api.moonshot.ai/v1/chat/completions"

BYOK_BASE_URL_ENV = "FX1_BYOK_BASE_URL"
BYOK_API_KEY_ENV = "FX1_BYOK_API_KEY"
BYOK_MODEL_ENV = "FX1_BYOK_MODEL"
BYOK_ALLOW_PRIVATE_NETWORKS_ENV = "FX1_BYOK_ALLOW_PRIVATE_NETWORKS"

LOCAL_SERVE_URL_ENV = "FX1_LOCAL_SERVE_URL"
LOCAL_SERVE_CMD_ENV = "FX1_LOCAL_SERVE_CMD"
LOCAL_MODEL_ENV = "FX1_LOCAL_MODEL"
LOCAL_API_KEY_ENV = "FX1_LOCAL_API_KEY"
LOCAL_TIMEOUT_S_ENV = "FX1_LOCAL_TIMEOUT_S"
LOCAL_START_TIMEOUT_S_ENV = "FX1_LOCAL_START_TIMEOUT_S"


@dataclass(frozen=True)
class SamplingParams:
    """Declared per-call parameters for one completion.

    Only declared fields reach the wire beyond ``temperature`` — a
    provider that doesn't know ``seed`` never sees it. ``temperature``
    defaults to 0.0: eval/teacher runs stay deterministic unless the
    caller explicitly opts out, and the *resolved* set is what lands in
    the completion record as evidence of what was sampled.

    Decode knobs (``temperature``/``top_p``/``max_tokens``/``seed``/
    ``stop``/the penalty pair/``logit_bias``) shape generation; provider
    hints (``reasoning_effort``/``service_tier``/``prompt_cache_key``/
    ``prompt_cache_retention``/``verbosity``/``user``) pass through
    verbatim — the upstream decides whether each
    is meaningful; the harness records that they were requested.
    ``stop`` is additionally enforced harness-side
    (:func:`truncate_at_stops`) so providers that ignore it still ship
    the cut text.
    """

    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    seed: int | None = None
    stop: tuple[str, ...] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, int] | None = None
    reasoning_effort: str | None = None
    service_tier: str | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None
    verbosity: str | None = None
    user: str | None = None

    def body_fields(self) -> dict[str, Any]:
        """The exact fields merged into the request body."""
        fields: dict[str, Any] = {
            "temperature": 0.0 if self.temperature is None else self.temperature
        }
        if self.top_p is not None:
            fields["top_p"] = self.top_p
        if self.max_tokens is not None:
            fields["max_tokens"] = self.max_tokens
        if self.seed is not None:
            fields["seed"] = self.seed
        if self.stop is not None:
            fields["stop"] = list(self.stop)
        if self.presence_penalty is not None:
            fields["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            fields["frequency_penalty"] = self.frequency_penalty
        if self.logit_bias is not None:
            fields["logit_bias"] = dict(self.logit_bias)
        if self.reasoning_effort is not None:
            fields["reasoning_effort"] = self.reasoning_effort
        if self.service_tier is not None:
            fields["service_tier"] = self.service_tier
        if self.prompt_cache_key is not None:
            fields["prompt_cache_key"] = self.prompt_cache_key
        if self.prompt_cache_retention is not None:
            fields["prompt_cache_retention"] = self.prompt_cache_retention
        if self.verbosity is not None:
            fields["verbosity"] = self.verbosity
        if self.user is not None:
            fields["user"] = self.user
        return fields


def _stop_cut(text: str, stops: tuple[str, ...] | list[str] | None) -> int:
    """Index of the earliest stop-sequence occurrence, or len(text)."""
    cut = len(text)
    for s in stops or ():
        i = text.find(s)
        if i >= 0:
            cut = min(cut, i)
    return cut


def truncate_at_stops(text: str, stops: tuple[str, ...] | list[str] | None) -> str:
    """Cut ``text`` before the earliest stop sequence — OpenAI ``stop``
    semantics, applied harness-side so every backend honors the contract
    even when the upstream doesn't (the matched sequence itself is
    excluded, per spec)."""
    return text[: _stop_cut(text, stops)]


def truncate_chunks(chunks: list[str], stops: tuple[str, ...] | list[str] | None) -> list[str]:
    """``truncate_at_stops`` over a delta stream — preserves the
    provider's chunk boundaries, emitting each chunk up to the cut and a
    slice of the straddling chunk (a stop that lands mid-delta truncates
    that delta rather than dropping whole chunks early)."""
    if not stops:
        return chunks
    cut = _stop_cut("".join(chunks), stops)
    out: list[str] = []
    pos = 0
    for chunk in chunks:
        if pos >= cut:
            break
        piece = chunk[: cut - pos]
        if piece:
            out.append(piece)
        pos += len(chunk)
    return out


class BackendNotConfiguredError(RuntimeError):
    """A backend whose required configuration is absent — a 503-class fault."""

    def __init__(self, message: str = "", *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


class TokenCountUnavailableError(RuntimeError):
    """The provider has no token-counting route — a 501-class capability gap.

    Token counts come only from the provider's own tokenizer endpoint;
    when it refuses the route outright (404/405/501 or a rejected chat
    shape) the honest verdict is "unavailable", never an estimate."""


def _chat_completions_url(base_url: str) -> str:
    """Normalize a BYOK base to the chat-completions route.

    Accepts ``https://host/v1``, ``https://host/v1/``, or an already-
    complete ``.../chat/completions`` URL.
    """
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _openai_sibling_url(chat_url: str, route: str) -> str:
    """Derive a sibling route from a chat-completions URL.

    ``https://host/v1/chat/completions`` + ``"embeddings"`` →
    ``https://host/v1/embeddings``; a URL already ending in the route
    passes through unchanged.
    """
    base = chat_url.rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    if base.endswith(f"/{route}"):
        return base
    return f"{base}/{route}"


def _env_float(name: str, override: float | None, default: float) -> float:
    """Resolve a seconds-valued knob: kwarg beats env beats default."""
    raw = override if override is not None else os.environ.get(name)
    if raw is None or raw == "":
        return default
    val = float(raw)
    if val <= 0:
        raise ValueError(f"{name} must be positive, got {val!r}")
    return val


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Fail closed instead of replaying a credentialed request elsewhere.

    ``urllib`` follows redirects by default and preserves caller-supplied
    headers such as ``Authorization`` even when the destination changes
    origin.  Every OpenAI-compatible request carries a provider credential,
    so no redirect response is safe to follow implicitly.
    """

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl
        return None


_BYOK_PRIVATE_V4 = tuple(
    ipaddress.ip_network(cidr)
    for cidr in ("10.0.0.0/8", "127.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)
_BYOK_PRIVATE_V6 = tuple(ipaddress.ip_network(cidr) for cidr in ("::1/128", "fc00::/7"))
# These legacy ranges can be classified as global by supported Python versions.
# The 6a44 assignment at 192.88.99.2 is not globally reachable either; IPv6
# site-local addresses are outside the deliberately narrow ULA opt-in.
_BYOK_LEGACY_SPECIAL_USE = tuple(
    ipaddress.ip_network(cidr) for cidr in ("192.88.99.0/24", "fec0::/10")
)


def _byok_private_networks_allowed() -> bool:
    """Whether an operator explicitly enabled private BYOK providers."""
    return os.environ.get(BYOK_ALLOW_PRIVATE_NETWORKS_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


class _ByokRequest(urllib.request.Request):
    """Request carrying a BYOK destination-policy snapshot to the wire seam."""

    _fx1_byok_public_only: bool = False
    _fx1_byok_allow_private: bool = False


def _apply_byok_destination_policy(request: _ByokRequest) -> None:
    """Snapshot the process policy onto a request for the transport seam."""
    allow_private = _byok_private_networks_allowed()
    request._fx1_byok_public_only = not allow_private
    request._fx1_byok_allow_private = allow_private


def _normalized_address(raw: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    address = ipaddress.ip_address(raw.split("%", 1)[0])
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address


def _byok_address_allowed(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address, *, allow_private: bool
) -> bool:
    """Conservative BYOK destination policy.

    The opt-in admits ordinary loopback/RFC1918/ULA model servers, but never
    link-local metadata addresses, multicast, unspecified, reserved,
    documentation, benchmarking, carrier-grade NAT, or legacy relay/site-local
    space.
    """
    if any(address in network for network in _BYOK_LEGACY_SPECIAL_USE):
        return False
    if address.is_global and not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    ):
        return True
    if not allow_private:
        return False
    networks = _BYOK_PRIVATE_V4 if isinstance(address, ipaddress.IPv4Address) else _BYOK_PRIVATE_V6
    return any(address in network for network in networks)


def _byok_resolved_addresses(host: str, port: int, *, allow_private: bool) -> tuple[str, ...]:
    """Resolve once, reject a mixed-policy answer set, and return numeric IPs."""
    addresses: list[str] = []
    for family, socktype, _proto, _canonname, sockaddr in socket.getaddrinfo(
        host, port, type=socket.SOCK_STREAM
    ):
        if family not in (socket.AF_INET, socket.AF_INET6) or socktype != socket.SOCK_STREAM:
            continue
        address = _normalized_address(str(sockaddr[0]))
        if not _byok_address_allowed(address, allow_private=allow_private):
            raise ValueError("BYOK endpoint resolved to a prohibited network address")
        normalized = str(address)
        if normalized not in addresses:
            addresses.append(normalized)
    if not addresses:
        raise OSError(f"BYOK endpoint host {host!r} did not resolve")
    return tuple(addresses)


class _ByokPinnedHTTPConnection(http.client.HTTPConnection):
    """HTTP connection pinned to a previously validated numeric address."""

    source_address: tuple[str, int] | None

    def __init__(self, host: str, port: int, address: str, timeout: float) -> None:
        super().__init__(host, port=port, timeout=timeout)
        self._address = address

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self._address, self.port), self.timeout, self.source_address
        )


class _ByokPinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS pinned to a numeric address while authenticating the URL host."""

    source_address: tuple[str, int] | None
    _context: ssl.SSLContext

    def __init__(self, host: str, port: int, address: str, timeout: float) -> None:
        context = ssl.create_default_context()
        # HTTPSConnection only sets HTTP/1.1 ALPN when no context is supplied.
        context.set_alpn_protocols(["http/1.1"])
        super().__init__(host, port=port, timeout=timeout, context=context)
        self._address = address

    def connect(self) -> None:
        raw_sock = socket.create_connection(
            (self._address, self.port), self.timeout, self.source_address
        )
        try:
            self.sock = self._context.wrap_socket(raw_sock, server_hostname=self.host)
        except BaseException:
            raw_sock.close()
            raise


def _open_byok_pinned(
    request: urllib.request.Request, *, timeout_s: float, allow_private: bool
) -> tuple[http.client.HTTPResponse, http.client.HTTPConnection]:
    """Open a direct, DNS-pinned BYOK request; never use proxies or redirects."""
    parsed = urllib.parse.urlparse(request.full_url)
    host = parsed.hostname
    if host is None:  # construction validation should make this unreachable
        raise urllib.error.URLError(ValueError("BYOK endpoint has no host"))
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        addresses = _byok_resolved_addresses(host, port, allow_private=allow_private)
    except (OSError, ValueError) as exc:
        raise urllib.error.URLError(exc) from exc
    target = urllib.parse.urlunparse(("", "", parsed.path or "/", parsed.params, parsed.query, ""))
    # urllib's Request normalizes names such as ``Content-Type`` to
    # ``Content-type`` internally. Restore conventional wire casing so this
    # transport remains behavior-compatible with the prior urllib path.
    headers = {
        "-".join(part.capitalize() for part in name.split("-")): value
        for name, value in request.header_items()
    }
    connection_cls = (
        _ByokPinnedHTTPSConnection if parsed.scheme == "https" else _ByokPinnedHTTPConnection
    )
    # A credentialed POST is never replayed across DNS answers: a provider
    # may have accepted a body before a read-side transport failure. Resolve
    # and validate the whole answer set, then make exactly one attempt.
    connection = connection_cls(host, port, addresses[0], timeout_s)
    try:
        connection.request(request.get_method(), target, body=request.data, headers=headers)
        return connection.getresponse(), connection
    except (OSError, http.client.HTTPException) as exc:
        connection.close()
        raise urllib.error.URLError(exc) from exc


@contextlib.contextmanager
def _openai_urlopen(request: urllib.request.Request, *, timeout_s: float) -> Iterator[Any]:
    """Open one credentialed request with redirects disabled and close it.

    Constructing the opener per call keeps mutable handler state scoped to
    that call when backend instances are shared by concurrent requests.
    ``HTTPError`` is itself a response object; ``OpenerDirector.open`` raises
    it before a caller can enter its response context, so close that error
    response here as well.  Otherwise repeated provider refusals retain their
    sockets until cyclic GC happens to collect the traceback.
    """
    public_only = bool(getattr(request, "_fx1_byok_public_only", False))
    allow_private = bool(getattr(request, "_fx1_byok_allow_private", False))
    if public_only or allow_private:
        response, connection = _open_byok_pinned(
            request, timeout_s=timeout_s, allow_private=allow_private
        )
        if not 200 <= response.status < 300:
            error = urllib.error.HTTPError(
                request.full_url,
                response.status,
                response.reason,
                response.headers,
                response,
            )
            response.close()
            connection.close()
            raise error
        try:
            yield response
        finally:
            response.close()
            connection.close()
        return
    opener = urllib.request.build_opener(_RefuseRedirects())
    try:
        response = opener.open(request, timeout=timeout_s)  # noqa: S310  # nosec B310
    except urllib.error.HTTPError as exc:
        with contextlib.suppress(Exception):
            exc.close()
        raise
    try:
        yield response
    finally:
        response.close()


def _extract_usage(payload: Any) -> dict[str, int] | None:
    """Pull the ``usage`` dict off an OpenAI-compatible response.

    Returns only genuine integer counters (``prompt_tokens``/
    ``completion_tokens``/``total_tokens`` and friends). Booleans,
    strings, and floats are unbillable, including non-finite floats;
    never coerce them into token counts. A missing or wholly malformed
    block is ``None`` rather than fabricated usage."""
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return None
    ints = {str(k): v for k, v in usage.items() if isinstance(v, int) and not isinstance(v, bool)}
    return ints or None


class _UsageTracker:
    """Per-call + cumulative token accounting, shared by the three backends.

    ``last_usage`` is the most recent call's counts; ``total_usage``
    accumulates under a lock so a backend shared across a batch's worker
    threads reports correct totals. Backends whose endpoint omits usage
    keep ``last_usage=None`` and an empty ``total_usage`` forever."""

    def __init__(self) -> None:
        self.last_usage: dict[str, int] | None = None
        self.total_usage: dict[str, int] = {}
        self._usage_lock = threading.Lock()

    def _record_usage(self, usage: dict[str, int] | None) -> None:
        self.last_usage = usage
        if not usage:
            return
        with self._usage_lock:
            for k, v in usage.items():
                self.total_usage[k] = self.total_usage.get(k, 0) + v


def _openai_chat_complete(
    url: str,
    *,
    model: str,
    messages: list[dict[str, Any]],
    timeout_s: float,
    api_key: str | None,
    label: str,
    sampling: SamplingParams | None = None,
    byok: bool = False,
) -> tuple[str, dict[str, int] | None]:
    """POST one OpenAI-compatible chat completion; map errors to RuntimeError.

    Returns ``(content, usage)`` — usage is ``None`` when the endpoint
    omits the block (older servers, local engines)."""
    # temperature pinned to 0 — eval/teacher runs must be deterministic;
    # unpinned sampling makes eval results unreproducible across replays.
    # temperature defaults to 0 — eval/teacher runs must be deterministic;
    # unpinned sampling makes eval results unreproducible across replays.
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            **(sampling or SamplingParams()).body_fields(),
        }
    ).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = _ByokRequest(url, data=body, headers=headers, method="POST")
    if byok:
        _apply_byok_destination_policy(request)
    try:
        with _openai_urlopen(request, timeout_s=timeout_s) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{label} endpoint returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        reason = type(exc.reason).__name__
        raise RuntimeError(f"{label} endpoint unreachable ({reason})") from exc
    except (OSError, http.client.HTTPException) as exc:
        # OSError covers TimeoutError — urlopen does not wrap read-side
        # stalls in URLError, so without this a slow upstream escapes the
        # error envelope as a bare 500.
        raise RuntimeError(f"{label} endpoint transport fault ({type(exc).__name__})") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"malformed {label} completion payload: not JSON") from exc
    except UnicodeError as exc:
        raise RuntimeError(f"{label} endpoint transport fault ({type(exc).__name__})") from exc
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"malformed {label} completion payload: missing choices[0].message.content"
        ) from exc
    if not isinstance(content, str):
        raise RuntimeError(
            f"malformed {label} completion payload: content is {type(content).__name__}, not str"
        )
    return content, _extract_usage(payload)


def _openai_tokenize_count(
    url: str,
    *,
    model: str,
    messages: list[dict[str, Any]],
    timeout_s: float,
    api_key: str | None,
    label: str,
    byok: bool = False,
) -> int:
    """POST one chat-shaped tokenize call; the provider's own count.

    ``url`` is the provider's tokenize route (``{base}/tokenize`` on
    vLLM/SGLang-style engines, ``.../tokenizers/estimate-token-count``
    on Moonshot). The body is the chat shape ``{model, messages,
    add_generation_prompt}`` — the prompt the completion would consume.
    The response's own count wins: ``count`` (vLLM), ``data.total_tokens``
    (Moonshot), or the length of a returned ``tokens``/``token_ids`` list
    (llama.cpp-style). A provider without the route raises
    ``TokenCountUnavailableError`` (HTTP 400/404/405/501 — a refused or
    unknown shape is a capability gap); anything else malformed or
    unreachable raises ``RuntimeError``. Nothing is ever estimated
    harness-side."""
    body = json.dumps(
        {"model": model, "messages": messages, "add_generation_prompt": True}
    ).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = _ByokRequest(url, data=body, headers=headers, method="POST")
    if byok:
        _apply_byok_destination_policy(request)
    try:
        with _openai_urlopen(request, timeout_s=timeout_s) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code in (400, 404, 405, 501):
            raise TokenCountUnavailableError(
                f"{label} endpoint refuses chat tokenization (HTTP {exc.code})"
            ) from exc
        raise RuntimeError(f"{label} tokenize failed: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        reason = type(exc.reason).__name__
        raise RuntimeError(f"{label} endpoint unreachable ({reason})") from exc
    except (OSError, http.client.HTTPException) as exc:
        raise RuntimeError(f"{label} endpoint transport fault ({type(exc).__name__})") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"malformed {label} tokenize payload: not JSON") from exc
    except UnicodeError as exc:
        raise RuntimeError(f"{label} endpoint transport fault ({type(exc).__name__})") from exc
    count = _extract_token_count(payload)
    if count is None:
        raise RuntimeError(
            f"malformed {label} tokenize payload: needs count / data.total_tokens / a tokens list"
        )
    return count


def _extract_token_count(payload: Any) -> int | None:
    """The provider's reported input count — ``count`` (vLLM),
    ``data.total_tokens`` (Moonshot estimate), or the length of a
    ``tokens``/``token_ids`` list. ``None`` when none of those exist."""
    if not isinstance(payload, dict):
        return None
    count = payload.get("count")
    if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
        return count
    data = payload.get("data")
    if isinstance(data, dict):
        total = data.get("total_tokens")
        if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
            return total
    for key in ("tokens", "token_ids"):
        seq = payload.get(key)
        if isinstance(seq, list):
            return len(seq)
    return None


def _tool_call_shape(raw: Any, label: str, i: int) -> dict[str, Any]:
    """Validate one ``tool_calls[]`` entry — fail closed on a malformed
    upstream frame rather than shipping a call fx-1 can't replay."""
    if not isinstance(raw, dict):
        raise RuntimeError(f"malformed {label} tool_calls[{i}]: not an object")
    fn = raw.get("function")
    args = fn.get("arguments") if isinstance(fn, dict) else None
    if (
        not isinstance(raw.get("id"), str)
        or raw.get("type") != "function"
        or not isinstance(fn, dict)
        or not isinstance(fn.get("name"), str)
        or not isinstance(args, str)
    ):
        raise RuntimeError(
            f"malformed {label} tool_calls[{i}]: needs "
            "{id, type: 'function', function: {name, arguments}}"
        )
    return raw


def _openai_chat_complete_tools(
    url: str,
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
    parallel_tool_calls: bool | None,
    timeout_s: float,
    api_key: str | None,
    label: str,
    sampling: SamplingParams | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    byok: bool = False,
) -> tuple[ToolCompletion, dict[str, int] | None]:
    """POST one OpenAI-compatible chat completion carrying ``tools``.

    ``tools``/``tool_choice``/``parallel_tool_calls`` pass through
    verbatim — the provider decides what each means. ``logprobs`` /
    ``top_logprobs`` pass through the same way: the response's
    ``choices[0].logprobs`` is echoed verbatim (``None`` when the
    provider stays silent — provider silence is its own answer, never
    fabricated). The response's ``choices[0]`` is validated fail-closed:
    ``content`` may be ``null`` (pure tool call), ``tool_calls`` entries
    must carry the OpenAI function-call shape, and a non-string
    ``content`` or malformed call raises ``RuntimeError`` — never a
    synthesized message.

    Returns ``(ToolCompletion, usage)`` — usage ``None`` when the
    endpoint omits the block."""
    body_map: dict[str, Any] = {
        "model": model,
        "messages": messages,
        **(sampling or SamplingParams()).body_fields(),
    }
    if tools is not None:
        body_map["tools"] = tools
    if tool_choice is not None:
        body_map["tool_choice"] = tool_choice
    if parallel_tool_calls is not None:
        body_map["parallel_tool_calls"] = parallel_tool_calls
    if logprobs is not None:
        body_map["logprobs"] = logprobs
    if top_logprobs is not None:
        body_map["top_logprobs"] = top_logprobs
    body = json.dumps(body_map).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = _ByokRequest(url, data=body, headers=headers, method="POST")
    if byok:
        _apply_byok_destination_policy(request)
    try:
        with _openai_urlopen(request, timeout_s=timeout_s) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{label} endpoint returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        reason = type(exc.reason).__name__
        raise RuntimeError(f"{label} endpoint unreachable ({reason})") from exc
    except (OSError, http.client.HTTPException) as exc:
        raise RuntimeError(f"{label} endpoint transport fault ({type(exc).__name__})") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"malformed {label} completion payload: not JSON") from exc
    except UnicodeError as exc:
        raise RuntimeError(f"{label} endpoint transport fault ({type(exc).__name__})") from exc
    try:
        choice = payload["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"malformed {label} completion payload: missing choices[0].message"
        ) from exc
    if not isinstance(message, dict):
        raise RuntimeError(
            f"malformed {label} completion payload: message is {type(message).__name__}, not object"
        )
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise RuntimeError(
            f"malformed {label} completion payload: content is "
            f"{type(content).__name__}, not str|null"
        )
    raw_calls = message.get("tool_calls")
    tool_calls: tuple[dict[str, Any], ...] | None = None
    if raw_calls is not None:
        if not isinstance(raw_calls, list):
            raise RuntimeError(
                f"malformed {label} completion payload: tool_calls is "
                f"{type(raw_calls).__name__}, not list"
            )
        tool_calls = tuple(_tool_call_shape(raw, label, i) for i, raw in enumerate(raw_calls))
    finish = choice.get("finish_reason")
    raw_lp = choice.get("logprobs")
    if raw_lp is not None and not isinstance(raw_lp, dict):
        raise RuntimeError(
            f"malformed {label} completion payload: logprobs is "
            f"{type(raw_lp).__name__}, not object|null"
        )
    return (
        ToolCompletion(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish if isinstance(finish, str) else None,
            logprobs=raw_lp,
        ),
        _extract_usage(payload),
    )


def _openai_chat_stream(
    url: str,
    *,
    model: str,
    messages: list[dict[str, Any]],
    timeout_s: float,
    api_key: str | None,
    label: str,
    usage_out: list[dict[str, int]] | None = None,
    sampling: SamplingParams | None = None,
    byok: bool = False,
) -> Iterator[str]:
    """POST one streaming OpenAI-compatible chat completion.

    Sends ``stream: true`` and yields each ``choices[0].delta.content``
    string as it arrives over server-sent events. Empty ``delta`` frames
    (role/handshake chunks) and ``data: [DONE]`` terminate the stream.
    Malformed frames fail closed as ``RuntimeError`` — a stream is never
    silently truncated.

    ``usage_out`` is an out-box: when any chunk carries an OpenAI-style
    ``usage`` dict (e.g. providers that attach per-chunk or final-frame
    usage), the last one lands in ``usage_out``. Never requested via
    ``stream_options`` — strict providers may reject unknown fields, so
    capture stays opportunistic.
    """
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": True,
            **(sampling or SamplingParams()).body_fields(),
        }
    ).encode()
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = _ByokRequest(url, data=body, headers=headers, method="POST")
    if byok:
        _apply_byok_destination_policy(request)
    try:
        with _openai_urlopen(request, timeout_s=timeout_s) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"malformed {label} stream chunk: not JSON ({data[:48]!r})"
                    ) from exc
                u = _extract_usage(chunk)
                if u is not None and usage_out is not None:
                    usage_out.clear()
                    usage_out.append(u)
                choices = chunk.get("choices") if isinstance(chunk, dict) else None
                if choices is None:
                    # A usage-only frame is valid even when its counters
                    # are absent or unbillable. Frame recognition must not
                    # depend on whether the accounting sieve accepted any.
                    if isinstance(chunk, dict) and isinstance(chunk.get("usage"), dict):
                        continue
                    raise RuntimeError(f"malformed {label} stream chunk: missing choices")
                if not choices:
                    continue
                delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
                if not isinstance(delta, dict):
                    raise RuntimeError(f"malformed {label} stream chunk: missing delta")
                content = delta.get("content")
                if content is None:
                    continue
                if not isinstance(content, str):
                    raise RuntimeError(
                        f"malformed {label} stream chunk: content is "
                        f"{type(content).__name__}, not str"
                    )
                yield content
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{label} endpoint returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        reason = type(exc.reason).__name__
        raise RuntimeError(f"{label} endpoint unreachable ({reason})") from exc
    except (OSError, http.client.HTTPException) as exc:
        raise RuntimeError(f"{label} endpoint transport fault ({type(exc).__name__})") from exc
    except UnicodeError as exc:
        raise RuntimeError(f"{label} endpoint transport fault ({type(exc).__name__})") from exc


def _embedding_item_shape(raw: Any, label: str, i: int) -> dict[str, Any]:
    """Validate one ``data[]`` entry — fail closed on a malformed upstream
    frame rather than shipping a vector fx-1 can't attribute."""
    if not isinstance(raw, dict):
        raise RuntimeError(f"malformed {label} embeddings payload: data[{i}] not an object")
    emb = raw.get("embedding")
    emb_ok = isinstance(emb, str) or (
        isinstance(emb, list)
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in emb)
    )
    if (
        raw.get("object") != "embedding"
        or not isinstance(raw.get("index"), int)
        or isinstance(raw.get("index"), bool)
        or not emb_ok
    ):
        raise RuntimeError(
            f"malformed {label} embeddings payload: data[{i}] needs "
            "{object: 'embedding', index: int, embedding: number[]|base64}"
        )
    return raw


def _openai_embeddings_complete(
    url: str,
    *,
    model: str,
    input: Any,  # noqa: A002 — the wire field's own name
    encoding_format: str | None,
    dimensions: int | None,
    user: str | None,
    timeout_s: float,
    api_key: str | None,
    label: str,
    byok: bool = False,
) -> EmbeddingResult:
    """POST one OpenAI-compatible embeddings call; errors → RuntimeError.

    ``input`` forwards verbatim — strings or token arrays both ride the
    wire (the provider's tokenizer contract is its own). ``model`` is the
    request's model verbatim: embedding models name themselves, the
    link's pinned chat model does not apply. The response's ``data`` is
    validated fail-closed: each entry must carry the OpenAI embedding
    shape (``{object: "embedding", index: int, embedding:
    number[]|base64-str}``) — malformed frames raise, nothing is
    synthesized harness-side.
    """
    body_map: dict[str, Any] = {"model": model, "input": input}
    if encoding_format is not None:
        body_map["encoding_format"] = encoding_format
    if dimensions is not None:
        body_map["dimensions"] = dimensions
    if user is not None:
        body_map["user"] = user
    body = json.dumps(body_map).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = _ByokRequest(url, data=body, headers=headers, method="POST")
    if byok:
        _apply_byok_destination_policy(request)
    try:
        with _openai_urlopen(request, timeout_s=timeout_s) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{label} endpoint returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        reason = type(exc.reason).__name__
        raise RuntimeError(f"{label} endpoint unreachable ({reason})") from exc
    except (OSError, http.client.HTTPException) as exc:
        raise RuntimeError(f"{label} endpoint transport fault ({type(exc).__name__})") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"malformed {label} embeddings payload: not JSON") from exc
    except UnicodeError as exc:
        raise RuntimeError(f"{label} endpoint transport fault ({type(exc).__name__})") from exc
    data = payload.get("data")
    if not isinstance(data, list):
        raise RuntimeError(
            f"malformed {label} embeddings payload: data is {type(data).__name__}, not list"
        )
    return EmbeddingResult(
        data=tuple(_embedding_item_shape(raw, label, i) for i, raw in enumerate(data)),
        model=payload.get("model") if isinstance(payload.get("model"), str) else None,
        usage=_extract_usage(payload),
    )


@dataclass(frozen=True)
class EmbeddingResult:
    """A provider's ``/v1/embeddings`` answer.

    ``data`` is the verbatim ``data[]`` list (each validated to the
    OpenAI ``{object: "embedding", index, embedding}`` shape — number
    arrays or base64 strings as the provider sent them); ``model`` and
    ``usage`` echo the provider's own fields (``None`` under provider
    silence — never fabricated).
    """

    data: tuple[dict[str, Any], ...]
    model: str | None
    usage: dict[str, int] | None


@dataclass(frozen=True)
class ToolCompletion:
    """A provider's structured answer when the request carried ``tools``.

    ``content`` is the assistant's text (``None`` when the model went
    straight to a call — OpenAI emits ``content: null`` there);
    ``tool_calls`` is the verbatim ``choices[].message.tool_calls`` list
    (each ``{id, type: \"function\", function: {name, arguments}}``);
    ``finish_reason`` is the upstream's own reason (``tool_calls`` /
    ``stop`` / ``length`` / ...); ``logprobs`` is the verbatim
    ``choices[].logprobs`` payload when the request asked for it
    (``None`` when unrequested or the provider stayed silent). All four
    are provider-reported — nothing is synthesized harness-side.
    """

    content: str | None
    tool_calls: tuple[dict[str, Any], ...] | None
    finish_reason: str | None
    logprobs: dict[str, Any] | None = None


class InferenceBackend(Protocol):
    """Chat-completion interface shared by all fx-1 backends."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
    ) -> str: ...


@runtime_checkable
class ToolBackend(Protocol):
    """Backends with a tool-calling channel.

    ``complete_with_tools`` is the optional second half of the contract:
    consumers capability-check via ``isinstance(b, ToolBackend)`` before
    routing a ``tools`` request, so a resolver-supplied backend without
    the channel fails closed (501) rather than silently dropping the
    caller's tool intent.
    """

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
    ) -> str: ...

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
    ) -> ToolCompletion: ...


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Backends with a ``/v1/embeddings`` channel.

    ``embeddings`` is the optional capability: consumers check
    ``isinstance(b, EmbeddingBackend)`` before routing an embeddings
    request, so a resolver-supplied backend without the channel fails
    closed (501) rather than fabricating vectors. ``input`` is the
    request's own value verbatim — string, string list, token array, or
    token-array list; ``model`` is the request's model verbatim
    (embedding models name themselves on the provider — a link's chat
    pin does not apply).
    """

    def embeddings(
        self,
        input: Any,  # noqa: A002 — the wire field's own name
        *,
        model: str,
        encoding_format: str | None = None,
        dimensions: int | None = None,
        user: str | None = None,
    ) -> EmbeddingResult: ...


@runtime_checkable
class StreamingBackend(Protocol):
    """Backends that additionally emit token deltas over the SSE wire.

    ``stream`` is the optional second half of the contract: consumers check
    ``isinstance(b, StreamingBackend)`` before subscribing, so a resolver-
    supplied backend that can't stream fails closed (501) rather than
    faking chunking.
    """

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
    ) -> str: ...

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
    ) -> Iterator[str]: ...


@runtime_checkable
class TokenCountingBackend(Protocol):
    """Backends with a real input-tokenizer channel.

    ``count_tokens`` is the optional capability: the provider's own
    tokenizer answers the count (vLLM/SGLang's ``/tokenize``, Moonshot's
    ``tokenizers/estimate-token-count``), never a harness-side estimate.
    Consumers capability-check via ``isinstance`` so a resolver-supplied
    backend without the channel fails closed (501) rather than guessing —
    a fabricated token count is worse than a refusal.
    """

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
    ) -> str: ...

    def count_tokens(self, messages: list[dict[str, Any]]) -> int: ...


class HostedK3Backend(_UsageTracker):
    """Hosted Kimi K3 via the Moonshot API (stdlib HTTP; no new deps).

    Endpoint override: ``api_url`` wins, then the ``FX1_BASE_URL``
    environment variable, then the pinned Moonshot URL. The API key comes
    from the ``api_key`` argument or ``MOONSHOT_API_KEY`` — never hardcoded.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "kimi-k3",
        api_url: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("MOONSHOT_API_KEY", "")
        if not self._api_key:
            raise RuntimeError("MOONSHOT_API_KEY is not set; fx-1 never hardcodes credentials")
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s!r}")
        resolved_url = api_url or os.environ.get("FX1_BASE_URL") or MOONSHOT_API_URL
        problem = hosted_api_url_problem(resolved_url)
        if problem is not None:
            raise RuntimeError(f"FX1_BASE_URL {problem}")
        super().__init__()
        self._model = model
        self._api_url = resolved_url
        self._timeout_s = timeout_s

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
    ) -> str:
        # temperature defaults to 0 — eval/teacher runs must be
        # deterministic; unpinned sampling makes eval results
        # unreproducible across replays.
        body = json.dumps(
            {
                "model": self._model,
                "messages": messages,
                **(sampling or SamplingParams()).body_fields(),
            }
        ).encode()
        request = urllib.request.Request(
            self._api_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        try:
            with _openai_urlopen(request, timeout_s=self._timeout_s) as response:
                payload = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"hosted_k3 endpoint returned HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            reason = type(exc.reason).__name__
            raise RuntimeError(f"hosted_k3 endpoint unreachable ({reason})") from exc
        except (OSError, http.client.HTTPException) as exc:
            raise RuntimeError(
                f"hosted_k3 endpoint transport fault ({type(exc).__name__})"
            ) from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("malformed hosted_k3 completion payload: not JSON") from exc
        except UnicodeError as exc:
            raise RuntimeError(
                f"hosted_k3 endpoint transport fault ({type(exc).__name__})"
            ) from exc
        content = payload["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise RuntimeError(
                f"malformed completion payload: content is {type(content).__name__}, not str"
            )
        self._record_usage(_extract_usage(payload))
        return content

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
    ) -> ToolCompletion:
        """Structured completion — Moonshot's wire accepts the OpenAI
        ``tools`` and ``logprobs`` fields; they pass through verbatim."""
        result, usage = _openai_chat_complete_tools(
            self._api_url,
            model=self._model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            timeout_s=self._timeout_s,
            api_key=self._api_key,
            label="hosted_k3",
            sampling=sampling,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
        self._record_usage(usage)
        return result

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
    ) -> Iterator[str]:
        """Stream token deltas; Moonshot's API is OpenAI-SSE-compatible."""
        box: list[dict[str, int]] = []
        for tok in _openai_chat_stream(  # noqa: UP028 — trailer needs the box after exhaustion
            self._api_url,
            model=self._model,
            messages=messages,
            timeout_s=self._timeout_s,
            api_key=self._api_key,
            label="hosted_k3",
            usage_out=box,
            sampling=sampling,
        ):
            yield tok
        if box:
            self._record_usage(box[-1])

    def embeddings(
        self,
        input: Any,  # noqa: A002 — the wire field's own name
        *,
        model: str,
        encoding_format: str | None = None,
        dimensions: int | None = None,
        user: str | None = None,
    ) -> EmbeddingResult:
        """``/v1/embeddings`` against the sibling route of the chat URL —
        the request's ``model`` reaches the wire verbatim (the chat pin
        ``kimi-k3`` is not an embedding model)."""
        result = _openai_embeddings_complete(
            _openai_sibling_url(self._api_url, "embeddings"),
            model=model,
            input=input,
            encoding_format=encoding_format,
            dimensions=dimensions,
            user=user,
            timeout_s=self._timeout_s,
            api_key=self._api_key,
            label="hosted_k3",
        )
        self._record_usage(result.usage)
        return result

    def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Moonshot's own input count — ``tokenizers/estimate-token-count``
        on the sibling route of the chat URL. The provider's tokenizer
        answers; a malformed or refused reply raises, never an estimate."""
        return _openai_tokenize_count(
            _openai_sibling_url(self._api_url, "tokenizers/estimate-token-count"),
            model=self._model,
            messages=messages,
            timeout_s=self._timeout_s,
            api_key=self._api_key,
            label="hosted_k3",
        )


def _plain_http_host_problem(host: str) -> str | None:
    """Why plain HTTP is unsafe for this host — ``None`` for literal loopback."""
    try:
        address = _normalized_address(host)
    except ValueError:
        return "must use HTTPS unless the host is a literal loopback address"
    if not address.is_loopback:
        return "must use HTTPS unless the host is a literal loopback address"
    return None


def hosted_api_url_problem(url: str) -> str | None:
    """Why a credentialed hosted-model URL is unsafe — ``None`` when safe.

    Bearer credentials may cross ordinary HTTPS or a literal loopback HTTP
    address used by a local development server.  Plain HTTP to any external
    or DNS-resolved host is rejected before a request can be constructed.
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return "must be a well-formed HTTPS URL"
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return "must be an HTTPS URL"
    if parsed.username is not None or parsed.password is not None:
        return "must not embed userinfo credentials"
    if parsed.params or parsed.query or parsed.fragment:
        return "must not carry params, query, or fragment"
    try:
        port = parsed.port
    except ValueError:
        return "port must be 1-65535"
    if port is not None and not 0 < port < 65536:
        return "port must be 1-65535"
    host = parsed.hostname
    if not host:
        return "must name a host"
    if parsed.scheme == "http":
        return _plain_http_host_problem(host)
    return None


def byok_base_url_problem(url: str) -> str | None:
    """Why a BYOK endpoint URL is unacceptable — ``None`` when it is.

    A BYOK base names scheme+host+path only: no userinfo (credentials in
    a URL are a leak surface — they ride every redirect and error), no
    params/query/fragment (the harness owns the request shape), a real
    host, a numeric port. Returns reason wording only — the URL itself is
    never echoed into error text, since it may carry pasted secrets.
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return "must be a well-formed http(s) URL"
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return "must be an http(s) URL"
    if parsed.username is not None or parsed.password is not None:
        return "must not embed userinfo credentials"
    if parsed.params or parsed.query or parsed.fragment:
        return "must not carry params, query, or fragment"
    try:
        port = parsed.port
    except ValueError:
        return "port must be 1-65535"
    if port is not None and not 0 < port < 65536:
        return "port must be 1-65535"
    if not parsed.hostname:
        return "must name a host"
    return None


class OpenAICompatBackend(_UsageTracker):
    """BYOK — any OpenAI-compatible chat-completions endpoint.

    Lets the whole fx-1 eval/bench fleet run against user-supplied
    providers (vLLM, SGLang, OpenRouter, Azure OpenAI, Ollama's compat
    shim, ...): every lane then emits the same sealed receipts and
    honesty gates regardless of whose weights sit behind the socket.

    Credentials come from ``FX1_BYOK_API_KEY`` (or the explicit kwarg)
    — never argv and never committed config. The endpoint is
    ``FX1_BYOK_BASE_URL`` + ``/chat/completions`` and the model name is
    ``FX1_BYOK_MODEL``. Explicit kwargs win over env. Fail-closed: a
    missing credential or a malformed payload raises, nothing is
    fabricated.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        *,
        timeout_s: int = 120,
    ) -> None:
        url = base_url if base_url is not None else os.environ.get(BYOK_BASE_URL_ENV, "")
        key = api_key if api_key is not None else os.environ.get(BYOK_API_KEY_ENV, "")
        mdl = model if model is not None else os.environ.get(BYOK_MODEL_ENV, "")
        missing = [
            name
            for name, val in (
                (BYOK_BASE_URL_ENV, url),
                (BYOK_API_KEY_ENV, key),
                (BYOK_MODEL_ENV, mdl),
            )
            if not val
        ]
        if missing:
            raise BackendNotConfiguredError(
                "BYOK backend is not configured; set "
                + ", ".join(missing)
                + " (fx-1 never hardcodes credentials)"
            )
        problem = byok_base_url_problem(url)
        if problem is not None:
            raise RuntimeError(f"{BYOK_BASE_URL_ENV} {problem}")
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s!r}")
        super().__init__()
        self._url = _chat_completions_url(url)
        self._api_key = key
        self._model = mdl
        self._timeout_s = timeout_s

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
    ) -> str:
        content, usage = _openai_chat_complete(
            self._url,
            model=self._model,
            messages=messages,
            timeout_s=self._timeout_s,
            api_key=self._api_key,
            label="BYOK",
            sampling=sampling,
            byok=True,
        )
        self._record_usage(usage)
        return content

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
    ) -> ToolCompletion:
        """Structured completion against the caller-declared endpoint —
        ``tools``/``tool_choice``/``parallel_tool_calls`` and
        ``logprobs``/``top_logprobs`` pass through verbatim; a provider
        that doesn't know them answers honestly (its own 4xx surfaces
        as ``RuntimeError``)."""
        result, usage = _openai_chat_complete_tools(
            self._url,
            model=self._model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            timeout_s=self._timeout_s,
            api_key=self._api_key,
            label="BYOK",
            sampling=sampling,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            byok=True,
        )
        self._record_usage(usage)
        return result

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
    ) -> Iterator[str]:
        """Stream token deltas from the caller-declared endpoint."""
        box: list[dict[str, int]] = []
        for tok in _openai_chat_stream(  # noqa: UP028 — trailer needs the box after exhaustion
            self._url,
            model=self._model,
            messages=messages,
            timeout_s=self._timeout_s,
            api_key=self._api_key,
            label="BYOK",
            usage_out=box,
            sampling=sampling,
            byok=True,
        ):
            yield tok
        if box:
            self._record_usage(box[-1])

    def embeddings(
        self,
        input: Any,  # noqa: A002 — the wire field's own name
        *,
        model: str,
        encoding_format: str | None = None,
        dimensions: int | None = None,
        user: str | None = None,
    ) -> EmbeddingResult:
        """``/v1/embeddings`` on the caller-declared endpoint — the
        request's ``model`` reaches the wire verbatim (the BYOK chat
        pin is not an embedding model); a provider that doesn't know
        the route answers honestly (its own 4xx surfaces)."""
        result = _openai_embeddings_complete(
            _openai_sibling_url(self._url, "embeddings"),
            model=model,
            input=input,
            encoding_format=encoding_format,
            dimensions=dimensions,
            user=user,
            timeout_s=self._timeout_s,
            api_key=self._api_key,
            label="BYOK",
            byok=True,
        )
        self._record_usage(result.usage)
        return result

    def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        """The BYOK provider's own input count via the ``/tokenize``
        sibling route (vLLM/SGLang-style chat shape). A provider without
        the route raises ``TokenCountUnavailableError`` — the honest 501,
        never a guessed count."""
        return _openai_tokenize_count(
            _openai_sibling_url(self._url, "tokenize"),
            model=self._model,
            messages=messages,
            timeout_s=float(self._timeout_s),
            api_key=self._api_key,
            label="BYOK",
            byok=True,
        )


def _free_tcp_port() -> int:
    """An ephemeral 127.0.0.1 port for a per-backend engine spawn."""
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


class LocalFx1Backend(_UsageTracker):
    """A local fx-1 checkpoint served by a local OpenAI-compatible engine.

    fx-1 ships no in-process inference stack — the checkpoint is weights +
    card, and completion goes through an engine that serves those weights
    (vLLM, SGLang, llama.cpp's server, ...). Two ways to reach one:

    - ``FX1_LOCAL_SERVE_URL`` / ``serve_url=`` — attach to an already-running
      engine (e.g. ``http://127.0.0.1:8000/v1``).
    - ``FX1_LOCAL_SERVE_CMD`` / ``serve_cmd=`` — a spawn template for the
      engine, e.g. ``vllm serve \"$checkpoint_dir\" --port 8000``. The
      ``$checkpoint_dir`` placeholder expands to the *verified* checkpoint
      root and ``FX1_CHECKPOINT_DIR`` is exported to the child. The backend
      waits for the attach URL to answer, and ``close()`` terminates the
      child.

    ``serve_url`` is always required to complete — ``serve_cmd`` only decides
    who brings the engine up. Fail-closed: a missing card, a failed ship
    gate, an unsigned release under ``FX1_SIGNING_KEY``, or no engine config
    at all each raise instead of fabricating.

    Attestation enforcement: when ``require_signature=True`` (the default once
    FX1_SIGNING_KEY is set), unsigned or signature-mismatched checkpoints
    refuse to serve — the served model must be the model whose card passed
    the gates, verifiably.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        require_signature: bool | None = None,
        serve_url: str | None = None,
        serve_cmd: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout_s: float | None = None,
        start_timeout_s: float | None = None,
    ) -> None:
        root = Path(checkpoint_dir)
        card_path = root / "modelcard.json"
        if not card_path.exists():
            raise FileNotFoundError(
                f"no model card at {card_path}; an fx-1 checkpoint without a card is not servable"
            )
        self.card = ModelCard.load(card_path)
        if not self.card.eval_delta.ship_eligible:
            raise RuntimeError(
                f"{self.card.version} failed the ship gate "
                "(honesty/domain/general deltas); refusing to serve"
            )
        if require_signature is None:
            require_signature = bool(os.environ.get("FX1_SIGNING_KEY"))
        if require_signature:
            from fx1.serve.signing import verify_release

            if not verify_release(root):
                raise RuntimeError(
                    "checkpoint release signature missing or invalid; "
                    "fx-1 serves only signed releases when FX1_SIGNING_KEY "
                    "is configured"
                )
        self._root = root
        # The weights sha this checkpoint's manifest pins — compared against
        # the sha a live engine advertises before we attach (see
        # ``_verify_served_weights``). None when the dir has no readable
        # manifest (e.g. a raw vLLM checkpoint layout) — then there is
        # nothing to verify against and attach stays a liveness probe.
        self._expected_weights_sha: str | None = None
        manifest_path = root / "weights.manifest.json"
        if manifest_path.is_file():
            try:
                manifest_obj = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                manifest_obj = {}
            sha = manifest_obj.get("weights_sha256") if isinstance(manifest_obj, dict) else None
            if isinstance(sha, str):
                self._expected_weights_sha = sha
        self._verified_attach = False
        url = serve_url if serve_url is not None else os.environ.get(LOCAL_SERVE_URL_ENV, "")
        if url:
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise RuntimeError(f"{LOCAL_SERVE_URL_ENV} must be an http(s) URL, got {url!r}")
        self._url = _chat_completions_url(url) if url else ""
        cmd = serve_cmd if serve_cmd is not None else os.environ.get(LOCAL_SERVE_CMD_ENV, "")
        self._serve_cmd = cmd
        if (
            serve_url is None
            and serve_cmd is None
            and (root / "weights.safetensors").is_file()
            and not self._shares_env_checkpoint(root)
        ):
            # A checkpoint dir that is not the environment's pinned
            # FX1_CHECKPOINT_DIR can never be served by the shared
            # FX1_LOCAL_SERVE_URL engine — that engine holds different
            # weights, so attaching would silently serve the wrong model
            # (``_verify_served_weights`` refuses it). Give this backend a
            # dedicated in-repo engine instead: a free port per instance.
            port = _free_tcp_port()
            self._url = _chat_completions_url(f"http://127.0.0.1:{port}/v1")
            self._serve_cmd = (
                f"$python -m fx1.serve.local_engine --checkpoint-dir $checkpoint_dir --port {port}"
            )
        self._model = (
            model if model is not None else (os.environ.get(LOCAL_MODEL_ENV) or self.card.version)
        )
        self._api_key = api_key if api_key is not None else os.environ.get(LOCAL_API_KEY_ENV, "")
        super().__init__()
        self._timeout_s = _env_float(LOCAL_TIMEOUT_S_ENV, timeout_s, 120.0)
        self._start_timeout_s = _env_float(LOCAL_START_TIMEOUT_S_ENV, start_timeout_s, 60.0)
        self._proc: subprocess.Popen[bytes] | None = None
        # Guards spawn/close so a shared backend is safe under complete_many.
        self._engine_lock = threading.Lock()

    @staticmethod
    def _shares_env_checkpoint(root: Path) -> bool:
        """True when *root* is the checkpoint the env's engine config serves.

        ``FX1_LOCAL_SERVE_URL``/``_CMD`` describe one engine holding one
        checkpoint — the ``FX1_CHECKPOINT_DIR`` pin. When the pin is unset
        the env can't be assumed to serve *this* dir: an unverified attach
        is worth less than a dedicated engine that provably loads *root*.
        """
        env_ckpt = os.environ.get("FX1_CHECKPOINT_DIR")
        if not env_ckpt:
            return False
        try:
            return Path(env_ckpt).resolve() == root.resolve()
        except OSError:
            return False

    def _engine_up(self) -> bool:
        """True once the attach URL's engine answers a models probe."""
        parsed = urllib.parse.urlparse(self._url)
        probe = f"{parsed.scheme}://{parsed.netloc}/v1/models"
        try:
            with urllib.request.urlopen(probe, timeout=1):  # noqa: S310 — declared local engine  # nosec B310
                return True
        except urllib.error.HTTPError:
            return True  # any HTTP response means a server is listening
        except (urllib.error.URLError, OSError):
            return False

    def _verify_served_weights(self) -> None:
        """Refuse to attach when the live engine serves different weights.

        A port answering ``/v1/models`` could be *any* engine — including
        one holding another checkpoint. When the engine advertises the
        fx-1 weights pin (as ``LocalWeightsEngine.describe`` does), it
        must equal the sha256 this backend's checkpoint manifest pins;
        otherwise the attach would silently serve the wrong model.
        Engines that don't advertise the key can't be verified — attach
        proceeds, same contract as before.
        """
        if self._expected_weights_sha is None or not self._url:
            self._verified_attach = True
            return
        parsed = urllib.parse.urlparse(self._url)
        probe = f"{parsed.scheme}://{parsed.netloc}/v1/models"
        try:
            with urllib.request.urlopen(probe, timeout=2) as resp:  # noqa: S310 — declared local engine  # nosec B310
                payload = json.loads(resp.read())
        except (urllib.error.URLError, OSError, ValueError):
            return  # not up / not parseable — liveness is _engine_up's job
        data = payload.get("data") if isinstance(payload, dict) else None
        for item in data or []:
            fx1_meta = item.get("fx1") if isinstance(item, dict) else None
            served_sha = fx1_meta.get("weights_sha256") if isinstance(fx1_meta, dict) else None
            if served_sha is not None and served_sha != self._expected_weights_sha:
                raise RuntimeError(
                    f"local fx-1 engine at {probe} advertises weights_sha256 "
                    f"{str(served_sha)[:16]}… but this backend's checkpoint "
                    f"manifest pins {self._expected_weights_sha[:16]}… — "
                    "refusing to attach to weights it did not verify"
                )
        self._verified_attach = True

    def _ensure_engine(self) -> None:
        """Spawn the engine from the serve template, once, and wait for it."""
        if self._proc is not None or self._verified_attach:
            return
        if not self._serve_cmd:
            if self._engine_up():
                self._verify_served_weights()
            return  # attach-only: the wire call itself is the liveness check
        if self._engine_up():
            self._verify_served_weights()
            return
        proc: subprocess.Popen[bytes] | None = None
        try:
            with self._engine_lock:
                if self._proc is not None:
                    return  # another thread spawned it while we waited
                if self._engine_up():
                    self._verify_served_weights()
                    return
                proc = self._spawn_engine()
                self._wait_for_engine(proc)
                # Publish only a ready child. Until then this call owns
                # cleanup, including interruptions and weight refusals.
                self._proc = proc
        except BaseException as exc:
            # close() reacquires the engine lock and could take ownership of a
            # replacement child. Clean up only our unpublished child, outside
            # the lock, preserving the startup failure if cleanup also fails.
            try:
                self._close_process(proc)
            except Exception as cleanup_exc:
                exc.add_note(f"local fx-1 engine cleanup also failed: {cleanup_exc!r}")
            raise

    def _spawn_engine(self) -> subprocess.Popen[bytes]:
        """Start the declared command; the caller owns the returned child."""
        argv = shlex.split(
            string.Template(self._serve_cmd).substitute(
                checkpoint_dir=str(self._root), python=shlex.quote(sys.executable)
            )
        )
        env = dict(os.environ, FX1_CHECKPOINT_DIR=str(self._root))
        try:
            return subprocess.Popen(  # noqa: S603 — argv list, no shell  # nosec B603
                argv, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except OSError as exc:
            raise RuntimeError(
                f"failed to spawn local fx-1 engine {self._serve_cmd!r}: {exc}"
            ) from exc

    def _wait_for_engine(self, proc: subprocess.Popen[bytes]) -> None:
        """Verify startup, retaining the child's exit status before cleanup."""
        deadline = time.monotonic() + self._start_timeout_s
        while time.monotonic() < deadline:
            returncode = proc.poll()
            if returncode is not None:
                raise RuntimeError(
                    "local fx-1 engine exited during startup "
                    f"(rc={returncode}): {self._serve_cmd!r}"
                )
            if self._engine_up():
                self._verify_served_weights()
                return
            time.sleep(0.1)
        raise RuntimeError(
            "local fx-1 engine did not become ready within "
            f"{self._start_timeout_s:g}s: {self._serve_cmd!r}"
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
    ) -> str:
        if not self._url:
            raise BackendNotConfiguredError(
                "local_fx1 is not configured: set FX1_LOCAL_SERVE_URL to an "
                "OpenAI-compatible engine serving the checkpoint (fx-1 never "
                "hardcodes endpoints); FX1_LOCAL_SERVE_CMD may spawn one"
            )
        self._ensure_engine()
        content, usage = _openai_chat_complete(
            self._url,
            model=self._model,
            messages=messages,
            timeout_s=self._timeout_s,
            api_key=self._api_key or None,
            label="local_fx1",
            sampling=sampling,
        )
        self._record_usage(usage)
        return content

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
    ) -> ToolCompletion:
        """Structured completion — delegated to the serving engine. An
        engine/chat-template without tool or logprobs support answers its
        own error (honest 4xx → RuntimeError); nothing is faked
        harness-side."""
        if not self._url:
            raise BackendNotConfiguredError(
                "local_fx1 is not configured: set FX1_LOCAL_SERVE_URL to an "
                "OpenAI-compatible engine serving the checkpoint (fx-1 never "
                "hardcodes endpoints); FX1_LOCAL_SERVE_CMD may spawn one"
            )
        self._ensure_engine()
        result, usage = _openai_chat_complete_tools(
            self._url,
            model=self._model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            timeout_s=self._timeout_s,
            api_key=self._api_key or None,
            label="local_fx1",
            sampling=sampling,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
        self._record_usage(usage)
        return result

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
    ) -> Iterator[str]:
        """Stream token deltas; the engine is ensured before subscribing."""
        self._ensure_engine()
        box: list[dict[str, int]] = []
        for tok in _openai_chat_stream(  # noqa: UP028 — trailer needs the box after exhaustion
            self._url,
            model=self._model,
            messages=messages,
            timeout_s=self._timeout_s,
            api_key=self._api_key or None,
            label="local_fx1",
            usage_out=box,
            sampling=sampling,
        ):
            yield tok
        if box:
            self._record_usage(box[-1])

    def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        """The serving engine's own input count via its ``/tokenize``
        sibling route — same configured check + spawn contract as
        ``complete``. An engine without the route raises
        ``TokenCountUnavailableError`` (honest 501, never an estimate)."""
        if not self._url:
            raise BackendNotConfiguredError(
                "local_fx1 is not configured: set FX1_LOCAL_SERVE_URL to an "
                "OpenAI-compatible engine serving the checkpoint (fx-1 never "
                "hardcodes endpoints); FX1_LOCAL_SERVE_CMD may spawn one"
            )
        self._ensure_engine()
        return _openai_tokenize_count(
            _openai_sibling_url(self._url, "tokenize"),
            model=self._model,
            messages=messages,
            timeout_s=self._timeout_s,
            api_key=self._api_key or None,
            label="local_fx1",
        )

    def close(self) -> None:
        """Terminate a spawned engine; a no-op when only attaching."""
        with self._engine_lock:
            proc, self._proc = self._proc, None
        self._close_process(proc)

    @staticmethod
    def _close_process(proc: subprocess.Popen[bytes] | None) -> None:
        """Stop and reap one owned child without acquiring the engine lock."""
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            # ``kill`` only sends the signal.  Reap the child as well so a
            # stubborn engine cannot remain as a zombie until this process
            # exits (and so its OS handles are released on Windows).
            proc.wait()

    def __enter__(self) -> LocalFx1Backend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()


def get_backend(kind: str, **kwargs: object) -> InferenceBackend:
    """Backend factory: ``hosted_k3``, ``local_fx1`` or ``byok``. Fail-closed."""
    backends = {
        "hosted_k3": HostedK3Backend,
        "local_fx1": LocalFx1Backend,
        "byok": OpenAICompatBackend,
    }
    if kind not in backends:
        raise KeyError(f"unknown backend {kind!r}; choose from {sorted(backends)}")
    return cast(InferenceBackend, backends[kind](**kwargs))
