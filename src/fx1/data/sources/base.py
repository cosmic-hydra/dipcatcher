"""Core types for the datasource layer.

Honesty contract (mirrors the lab's fail-closed discipline):

1. Adapters execute bundled plugin CLI scripts via subprocess — they never
   fabricate numbers. Any failure produces ``FetchResult(ok=False)`` with
   the real error, not a placeholder.
2. Credentials are resolved from the environment *inside the child process*;
   this layer only checks presence by **name** and never reads, prints, or
   forwards values on a command line.
3. ``as_of`` point-in-time discipline: market/fundamental payloads intended
   for the training corpus must carry the observation date, otherwise the
   ingest gate refuses them (leakage prevention).
"""

from __future__ import annotations

import hashlib
import locale
import os
import re
import subprocess
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

MAX_OUTPUT_CHARS = 200_000
DEFAULT_TIMEOUT_S = 120.0

# glibc provides C.UTF-8. macOS rejects that name and then a child that
# calls setlocale(LC_ALL, "") fails. Probe once, in this order.
_UTF8_LOCALE_CANDIDATES = ("C.UTF-8", "en_US.UTF-8", "UTF-8")
_locale_lock = threading.Lock()
_locale_cached: str | None = None
_locale_ready = False


def _preferred_utf8_locale() -> str | None:
    """A UTF-8 locale this process can activate, or None when none is installed."""
    global _locale_cached, _locale_ready
    if _locale_ready:
        return _locale_cached
    with _locale_lock:
        if _locale_ready:
            return _locale_cached
        chosen: str | None = None
        try:
            previous = locale.setlocale(locale.LC_ALL)
        except locale.Error:
            previous = None
        try:
            for candidate in _UTF8_LOCALE_CANDIDATES:
                try:
                    locale.setlocale(locale.LC_ALL, candidate)
                except locale.Error:
                    continue
                chosen = candidate
                break
        finally:
            if previous is not None:
                with suppress(locale.Error):
                    locale.setlocale(locale.LC_ALL, previous)
        _locale_cached = chosen
        _locale_ready = True
        return chosen


class SourceKind(StrEnum):
    """How a source is invoked."""

    AGENT_GW = "agent_gw"  # standard describe/call --api-name/--params-json
    CUSTOM_CLI = "custom_cli"  # source-specific argument grammar
    SCENARIO_ROUTER = "scenario_router"  # finance_fetch scenario envelope
    MCP = "mcp"  # MCP connector, no local CLI — honestly unavailable offline


class SourceStatus(StrEnum):
    """Availability probe outcome."""

    READY = "ready"  # script found and credentials resolvable (or unneeded)
    READY_UNVERIFIED = "ready_unverified"  # script found; creds auto/unknown
    NO_SCRIPT = "no_script"  # bundled script not found on this host
    NO_CREDENTIALS = "no_credentials"  # script found, required env missing
    MCP_REQUIRED = "mcp_required"  # only reachable via MCP runtime


class LatencyClass(StrEnum):
    """Dominant data cadence — drives PIT discipline."""

    REALTIME = "realtime"
    DAILY = "daily"
    FILINGS = "filings"
    NEWS = "news"
    RESEARCH = "research"
    MACRO = "macro"
    ENTERPRISE = "enterprise"


class FetchRequest(BaseModel):
    """One datasource call."""

    api: str = Field(description="Source-native API/tool name")
    params: dict[str, object] = Field(default_factory=dict)
    as_of: str | None = Field(
        default=None,
        description="Observation date (YYYY-MM-DD). Mandatory for corpus "
        "ingestion of time-stamped market data.",
    )
    timeout_s: float = Field(default=DEFAULT_TIMEOUT_S, gt=0, le=600)

    @field_validator("as_of")
    @classmethod
    def _as_of_is_a_real_date(cls, as_of: str | None) -> str | None:
        # The PIT leakage gate keys on this field; a non-date string would
        # satisfy "present" while pinning the wrong observation date.
        if as_of is not None:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", as_of):
                raise ValueError(f"as_of must be YYYY-MM-DD, got {as_of!r}")
            try:
                date.fromisoformat(as_of)
            except ValueError:
                raise ValueError(f"as_of is not a real date: {as_of!r}") from None
        return as_of


class FetchResult(BaseModel):
    """Honest outcome of a datasource call — success or described failure."""

    ok: bool
    source: str
    api: str
    text: str = Field(default="", description="Assistant-facing result text")
    files: list[str] = Field(default_factory=list)
    error: str | None = None
    status: SourceStatus = SourceStatus.READY
    fetched_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    as_of: str | None = None
    payload_sha256: str = Field(default="", description="SHA-256 of the exact returned text")
    truncated: bool = False
    elapsed_ms: int = 0

    @staticmethod
    def hash_payload(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @classmethod
    def failure(cls, *, source: str, api: str, error: str, status: SourceStatus) -> FetchResult:
        """A described failure. Never carries invented payload text."""
        return cls(ok=False, source=source, api=api, error=error, status=status)


class SourceProbe(BaseModel):
    """Availability report. ``credentials`` lists env var *names* only."""

    name: str
    status: SourceStatus
    script: str | None = None
    detail: str = ""
    credentials: list[str] = Field(default_factory=list)


class RunOutcome(BaseModel):
    """Subprocess result captured by a Runner."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


def default_runner(
    argv: list[str], *, timeout_s: float, extra_env: dict[str, str] | None = None
) -> RunOutcome:
    """Execute *argv* with a scrubbed environment.

    The child receives PATH/HOME/LANG plus the *values* of allowlisted
    credential variables from the parent environment (so gateway SDKs can
    authenticate), but nothing else — and nothing is ever placed on argv.
    """
    env = {
        k: os.environ[k]
        for k in ("PATH", "HOME", "LANG", "LC_ALL", "PYTHONPATH")
        if k in os.environ
    }
    # UTF-8 mode does not require a glibc locale name. Set LANG only when
    # this host actually has a UTF-8 locale; C.UTF-8 is not one of them on macOS.
    env.setdefault("PYTHONUTF8", "1")
    if "LANG" not in env:
        lang = _preferred_utf8_locale()
        if lang is not None:
            env["LANG"] = lang
    if extra_env:
        env.update(extra_env)
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return RunOutcome(
            returncode=124,
            stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
            stderr=f"timeout after {timeout_s}s",
            timed_out=True,
        )
    except OSError as exc:
        return RunOutcome(returncode=127, stdout="", stderr=str(exc))
    truncated = False
    stdout = proc.stdout or ""
    if len(stdout) > MAX_OUTPUT_CHARS:
        stdout = stdout[:MAX_OUTPUT_CHARS]
        truncated = True
    if truncated:
        stdout += "\n[fx1: output truncated]"
    return RunOutcome(returncode=proc.returncode, stdout=stdout, stderr=proc.stderr or "")


Runner = Callable[..., RunOutcome]


class DataSourceAdapter(ABC):
    """Uniform interface over every bundled datasource CLI."""

    def __init__(self, spec: object, runner: Runner | None = None) -> None:
        from fx1.data.sources.registry import SourceSpec

        if not isinstance(spec, SourceSpec):  # pragma: no cover - defensive
            raise TypeError("spec must be a SourceSpec")
        self.spec: SourceSpec = spec
        self._runner = runner or default_runner

    # -- availability ----------------------------------------------------
    def resolve_script(self) -> Path | None:
        """First existing script candidate across plugin roots, else None."""
        for root in self.spec.plugin_roots():
            for candidate in self.spec.script_candidates:
                path = Path(root) / candidate
                if path.exists():
                    return path
        return None

    def credentials_present(self) -> bool | None:
        """True/False when checkable, None when auto-resolved by runtime."""
        if not self.spec.credentials:
            return None
        if any(os.environ.get(name) for name in self.spec.credentials):
            return True
        # agent-gw SDKs also resolve credentials from the runtime config file.
        return (
            "KIMI_API_KEY" in self.spec.credentials
            and (Path.home() / ".kimi" / "agent-gw.json").exists()
        )

    def probe(self) -> SourceProbe:
        """Fail-closed availability check; never leaks secret values."""
        script = self.resolve_script()
        cred_names = list(self.spec.credentials)
        if self.spec.kind is SourceKind.MCP:
            return SourceProbe(
                name=self.spec.name,
                status=SourceStatus.MCP_REQUIRED,
                detail="MCP connector — invoke through the host MCP runtime; "
                "no offline CLI. Degrades honestly rather than faking data.",
                credentials=cred_names,
            )
        if script is None:
            return SourceProbe(
                name=self.spec.name,
                status=SourceStatus.NO_SCRIPT,
                detail="bundled script not found; set FX1_PLUGIN_ROOTS or install the plugin",
                credentials=cred_names,
            )
        creds = self.credentials_present()
        if creds is False:
            return SourceProbe(
                name=self.spec.name,
                status=SourceStatus.NO_CREDENTIALS,
                script=str(script),
                detail="required credential env var(s) absent: " + ", ".join(cred_names),
                credentials=cred_names,
            )
        status = SourceStatus.READY_UNVERIFIED if creds is None else SourceStatus.READY
        detail = (
            "credentials auto-resolved by runtime"
            if creds is None
            else "credentials resolvable (env var or runtime config)"
        )
        return SourceProbe(
            name=self.spec.name,
            status=status,
            script=str(script),
            detail=detail,
            credentials=cred_names,
        )

    # -- operations ------------------------------------------------------
    @abstractmethod
    def describe(self) -> FetchResult:
        """Return the source's own capability documentation."""

    @abstractmethod
    def fetch(self, request: FetchRequest) -> FetchResult:
        """Execute one call. Honest failure on any error."""

    # -- shared helpers --------------------------------------------------
    def _preflight(self, api: str) -> FetchResult | None:
        """Return a failure result when unavailable, else None."""
        probe = self.probe()
        if probe.status in (
            SourceStatus.READY,
            SourceStatus.READY_UNVERIFIED,
        ):
            return None
        return FetchResult.failure(
            source=self.spec.name,
            api=api,
            error=f"source unavailable ({probe.status}): {probe.detail}",
            status=probe.status,
        )

    def _credential_env(self) -> dict[str, str]:
        """Values of declared credential vars, passed via env only."""
        return {name: os.environ[name] for name in self.spec.credentials if os.environ.get(name)}

    def _run(
        self,
        argv: list[str],
        *,
        request: FetchRequest,
        started: datetime,
    ) -> FetchResult:
        import time

        t0 = time.monotonic()
        outcome = self._runner(argv, timeout_s=request.timeout_s, extra_env=self._credential_env())
        elapsed = int((time.monotonic() - t0) * 1000)
        if outcome.timed_out:
            return FetchResult.failure(
                source=self.spec.name,
                api=request.api,
                error=f"timeout after {request.timeout_s}s (kill ≠ no data; "
                "retry with a larger window)",
                status=self.probe().status,
            )
        if outcome.returncode != 0:
            return FetchResult.failure(
                source=self.spec.name,
                api=request.api,
                error=outcome.stderr.strip()[:2000] or f"exit code {outcome.returncode}",
                status=self.probe().status,
            )
        text = outcome.stdout.strip()
        if not text:
            return FetchResult.failure(
                source=self.spec.name,
                api=request.api,
                error="source returned empty output",
                status=self.probe().status,
            )
        return FetchResult(
            ok=True,
            source=self.spec.name,
            api=request.api,
            text=text,
            status=self.probe().status,
            fetched_at=started.isoformat(),
            as_of=request.as_of,
            payload_sha256=FetchResult.hash_payload(text),
            truncated="[fx1: output truncated]" in text,
            elapsed_ms=elapsed,
        )
