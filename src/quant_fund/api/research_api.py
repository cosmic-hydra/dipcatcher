"""Read-only research evidence API.

Serves committed/lab research artifacts over HTTP for local tooling (web app,
typescript client, notebooks). Every route is GET; there are no order, broker,
execution, or live-trading endpoints anywhere in this app — the route audit in
``tests/unit/api/test_research_api.py`` asserts that machine-checkably.

Evidence roots (all read-only, resolved at app construction):

- ``<repo>/receipts`` — committed schema-tagged research receipts (``*.json``).
- ``<repo>/verifier`` — acceptance-criteria versions (``vN/acceptance.md``) and
  run logs (``runs/*.md``).
- ``<repo>/artifacts`` — committed flat research artifacts (``*.json``).
- ``<data_root>/metadata/research/runs`` — per-run research notebooks
  (``<run_id>.json`` / ``<run_id>.md``) verified by
  ``quant_fund.research.verify.verify_research_artifact``.
- ``<data_root>/metadata/<kind>/<name>`` — sealed benchmark run directories
  (manifest.json + validation/test receipts) verified by
  ``quant_fund.research.phase1_verify.verify_phase1_run``.

``<data_root>`` defaults to ``$RESEARCH_API_DATA_ROOT``, then
``$QUANT_DATA_ROOT``, then ``<repo>/data``.

Secure default: ``/health`` is the only unauthenticated route. When
``RESEARCH_API_KEY`` is set every other route requires ``X-API-Key``; when it
is unset only loopback clients are served. The service holds no credentials
and exposes no trading surface, but it must still never be bound to a
non-loopback interface without ``RESEARCH_API_KEY``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from quant_fund import __firm__, __version__

_REPO_ROOT = Path(__file__).resolve().parents[3]

_PUBLIC_PATHS = frozenset({"/health"})
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})
_RUN_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_RECEIPT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_ARTIFACT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./-]{0,255}$")
_RESULT_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_VERIFIER_VERSION_RE = re.compile(r"^v[0-9]+$")
_VERIFIER_RUN_RE = re.compile(r"^([0-9T:\-]+Z)_?(v[0-9]+)?$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_LIST_LIMIT = 500
_VERDICT_RE = re.compile(r"^Verdict:\s*([A-Za-z_]+)", re.MULTILINE)
_TIME_RE = re.compile(r"^-\s*Time:\s*(\S+)", re.MULTILINE)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResearchApiSettings:
    """Filesystem roots + auth for the read-only research API."""

    data_root: Path
    receipts_dir: Path
    verifier_dir: Path
    artifacts_dir: Path
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> ResearchApiSettings:
        data_root_env = os.environ.get("RESEARCH_API_DATA_ROOT") or os.environ.get(
            "QUANT_DATA_ROOT"
        )
        data_root = Path(data_root_env) if data_root_env else _REPO_ROOT / "data"
        return cls(
            data_root=data_root,
            receipts_dir=Path(
                os.environ.get("RESEARCH_API_RECEIPTS_DIR", str(_REPO_ROOT / "receipts"))
            ),
            verifier_dir=Path(
                os.environ.get("RESEARCH_API_VERIFIER_DIR", str(_REPO_ROOT / "verifier"))
            ),
            artifacts_dir=Path(
                os.environ.get("RESEARCH_API_ARTIFACTS_DIR", str(_REPO_ROOT / "artifacts"))
            ),
            api_key=os.environ.get("RESEARCH_API_KEY") or None,
        )

    @property
    def runs_dir(self) -> Path:
        return self.data_root / "metadata" / "research" / "runs"

    @property
    def metadata_dir(self) -> Path:
        return self.data_root / "metadata"


# ---------------------------------------------------------------------------
# Response models (everything served validates against one of these)
# ---------------------------------------------------------------------------


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(_Model):
    status: Literal["ok"] = "ok"
    service: str = "dipcatcher-research-api"
    firm: str = __firm__
    version: str = __version__
    mode: Literal["read_only"] = "read_only"
    data_root_present: bool
    receipts_dir_present: bool
    verifier_dir_present: bool
    artifacts_dir_present: bool
    claim: Literal["research_only"] = "research_only"


class ReadyResponse(_Model):
    """Fail-closed readiness: evidence roots must exist as directories."""

    status: Literal["ready", "not_ready"]
    service: str = "dipcatcher-research-api"
    firm: str = __firm__
    version: str = __version__
    mode: Literal["read_only"] = "read_only"
    checks: dict[str, bool]
    claim: Literal["research_only"] = "research_only"


class RunSummary(_Model):
    """Summary of one ``metadata/research/runs/<run_id>.json`` notebook."""

    run_id: str
    sha256: str
    generated_at: str | None
    schema_version: int | None
    data_source: str | None
    synthetic: bool | None
    claim: str | None
    ranking_target: str | None
    git_revision: str | None
    n_hypotheses: int
    n_rankers: int
    has_markdown: bool
    size_bytes: int
    parse_error: str | None = None


class RunListResponse(_Model):
    items: list[RunSummary]
    total: int
    limit: int
    offset: int
    claim: Literal["research_only"] = "research_only"


class RunDetail(_Model):
    run_id: str
    sha256: str
    receipt: dict[str, Any]


class RunMarkdownDetail(_Model):
    run_id: str
    sha256: str
    markdown: str


class ReceiptSummary(_Model):
    """Summary of one committed ``receipts/*.json`` evidence document."""

    id: str
    filename: str
    sha256: str
    schema_tag: str | None
    generated_at: str | None
    research_only: bool | None
    live_pnl_claim: bool | None
    size_bytes: int
    parse_error: str | None = None


class ReceiptListResponse(_Model):
    items: list[ReceiptSummary]
    total: int
    claim: Literal["research_only"] = "research_only"


class ReceiptDetail(_Model):
    id: str
    filename: str
    sha256: str
    schema_tag: str | None
    receipt: dict[str, Any]


class ResultFile(_Model):
    """One file inside a sealed benchmark run directory."""

    name: str
    sha256: str
    size_bytes: int
    compressed: bool


class ResultSummary(_Model):
    """A ``<data_root>/metadata/<kind>/<name>/`` sealed run directory."""

    kind: str
    name: str
    path: str
    created_at: str | None
    has_manifest: bool
    has_validation: bool
    has_test: bool


class ResultListResponse(_Model):
    items: list[ResultSummary]
    total: int
    claim: Literal["research_only"] = "research_only"


class ResultDetail(_Model):
    kind: str
    name: str
    path: str
    manifest: dict[str, Any] | None
    manifest_sha256: str | None
    files: list[ResultFile]


class VerifierVersionSummary(_Model):
    version: str
    sha256: str
    size_bytes: int


class VerifierVersionListResponse(_Model):
    items: list[VerifierVersionSummary]
    total: int
    claim: Literal["research_only"] = "research_only"


class VerifierVersionDetail(_Model):
    version: str
    sha256: str
    acceptance_markdown: str


class VerifierRunSummary(_Model):
    id: str
    version: str | None
    time: str | None
    verdict: str | None
    sha256: str
    size_bytes: int


class VerifierRunListResponse(_Model):
    items: list[VerifierRunSummary]
    total: int
    claim: Literal["research_only"] = "research_only"


class VerifierRunDetail(_Model):
    id: str
    version: str | None
    time: str | None
    verdict: str | None
    sha256: str
    markdown: str


class ArtifactSummary(_Model):
    """One committed ``artifacts/**/*.json`` research artifact."""

    id: str
    path: str
    sha256: str
    schema_tag: str | None
    size_bytes: int
    parse_error: str | None = None


class ArtifactListResponse(_Model):
    items: list[ArtifactSummary]
    total: int
    claim: Literal["research_only"] = "research_only"


class ArtifactDetail(_Model):
    id: str
    path: str
    sha256: str
    artifact: dict[str, Any]


class VerificationResult(_Model):
    """Outcome of running a fail-closed verifier over an evidence object.

    ``verifier`` names the check that ran so callers can distinguish the full
    research-notebook gate (``verify_research_artifact``) from the sealed-run
    verifier (``verify_phase1_run``) and the lightweight committed-receipt
    honesty check (``receipt_honesty_flags``).
    """

    subject: str
    kind: Literal["research_run", "receipt", "benchmark_run"]
    verifier: str
    valid: bool
    errors: list[str]
    sha256: str | None = None
    run_id: str | None = None
    state: str | None = None
    scorecard_families: int | None = None
    context: str | None = None
    claim: Literal["research_only"] = "research_only"
    live_pnl_claim: Literal[False] = False


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contained(root: Path, candidate: Path) -> Path:
    """Resolve *candidate* and require it to stay under *root* (fail closed)."""
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise HTTPException(400, "path escapes the configured evidence root") from exc
    return resolved


class _JsonDocCache:
    """Content-keyed parse cache; always hash the current evidence bytes."""

    def __init__(self) -> None:
        self._cache: dict[Path, tuple[str, dict[str, Any] | None, str | None]] = {}

    def read(self, path: Path) -> tuple[str, dict[str, Any] | None, str | None]:
        """Return ``(sha256, payload, parse_error)`` for a JSON file.

        ``payload`` is ``None`` and ``parse_error`` set when the bytes are not
        a JSON object; the sha256 is over raw bytes either way.
        """
        raw = path.read_bytes()
        digest = _sha256_bytes(raw)
        cached = self._cache.get(path)
        if cached is not None and cached[0] == digest:
            return cached
        payload: dict[str, Any] | None
        error: str | None
        try:
            parsed = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            payload, error = None, f"invalid_json:{type(exc).__name__}"
        else:
            if isinstance(parsed, dict):
                payload, error = parsed, None
            else:
                payload, error = None, "json_not_object"
        if len(self._cache) >= 1024:
            self._cache.clear()
        self._cache[path] = (digest, payload, error)
        return digest, payload, error


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _bool_or_none(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(settings: ResearchApiSettings | None = None) -> FastAPI:  # noqa: C901
    cfg = settings or ResearchApiSettings.from_env()
    cfg = replace(
        cfg,
        data_root=cfg.data_root.resolve(),
        receipts_dir=cfg.receipts_dir.resolve(),
        verifier_dir=cfg.verifier_dir.resolve(),
        artifacts_dir=cfg.artifacts_dir.resolve(),
    )
    docs = _JsonDocCache()

    app = FastAPI(
        title="dipcatcher research API",
        version=__version__,
        description=(
            "Read-only access to research runs, sealed receipts, verifier "
            "acceptance history, and benchmark results. No trading, order, "
            "broker, or execution endpoints exist in this app."
        ),
    )

    @app.middleware("http")
    async def research_api_auth(request: Request, call_next: Any) -> Any:
        """Loopback-only by default; X-API-Key when RESEARCH_API_KEY is set."""
        if request.url.path in _PUBLIC_PATHS:
            response = await call_next(request)
        elif cfg.api_key:
            provided = request.headers.get("X-API-Key")
            if not provided or not hmac.compare_digest(
                provided.encode("utf-8"), cfg.api_key.encode("utf-8")
            ):
                response = JSONResponse(
                    status_code=401, content={"detail": "invalid or missing X-API-Key"}
                )
            else:
                response = await call_next(request)
        else:
            host = (request.client.host if request.client else "") or ""
            if host not in _LOOPBACK_HOSTS:
                response = JSONResponse(
                    status_code=403,
                    content={
                        "detail": (
                            "RESEARCH_API_KEY is unset; non-localhost clients are "
                            "refused. Set RESEARCH_API_KEY and send X-API-Key, or "
                            "bind to 127.0.0.1 only."
                        )
                    },
                )
            else:
                response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    # -- evidence readers ----------------------------------------------------

    def _runs_dir() -> Path:
        return _contained(cfg.data_root, cfg.runs_dir)

    def _run_path(run_id: str) -> Path:
        if not _RUN_ID_RE.fullmatch(run_id):
            raise HTTPException(400, "run_id must be a 64-char lowercase sha256")
        path = _contained(_runs_dir(), _runs_dir() / f"{run_id}.json")
        if not path.is_file():
            raise HTTPException(404, f"research run not found: {run_id}")
        return path

    def _run_summary(path: Path) -> RunSummary:
        stat = path.stat()
        digest, payload, error = docs.read(path)
        provenance = payload.get("provenance") if isinstance(payload, dict) else None
        if not isinstance(provenance, dict):
            provenance = {}
        hypotheses = payload.get("hypotheses") if isinstance(payload, dict) else None
        rankers = payload.get("rankers") if isinstance(payload, dict) else None
        run_id = provenance.get("run_id")
        return RunSummary(
            run_id=_str_or_none(run_id) or path.stem,
            sha256=digest,
            generated_at=_str_or_none(payload.get("generated_at") if payload else None),
            schema_version=_int_or_none(payload.get("schema_version") if payload else None),
            data_source=_str_or_none(payload.get("data_source") if payload else None),
            synthetic=_bool_or_none(payload.get("synthetic") if payload else None),
            claim=_str_or_none(payload.get("claim") if payload else None),
            ranking_target=_str_or_none(payload.get("ranking_target") if payload else None),
            git_revision=_str_or_none(provenance.get("git_revision")),
            n_hypotheses=len(hypotheses) if isinstance(hypotheses, list) else 0,
            n_rankers=len(rankers) if isinstance(rankers, list) else 0,
            has_markdown=path.with_suffix(".md").is_file(),
            size_bytes=stat.st_size,
            parse_error=error,
        )

    def _receipt_path(receipt_id: str) -> Path:
        if not _RECEIPT_ID_RE.fullmatch(receipt_id):
            raise HTTPException(400, "invalid receipt id")
        root = cfg.receipts_dir
        path = _contained(root, root / f"{receipt_id}.json")
        if not path.is_file():
            raise HTTPException(404, f"receipt not found: {receipt_id}")
        return path

    def _receipt_summary(path: Path) -> ReceiptSummary:
        stat = path.stat()
        digest, payload, error = docs.read(path)
        return ReceiptSummary(
            id=path.stem,
            filename=path.name,
            sha256=digest,
            schema_tag=_str_or_none(payload.get("schema") if payload else None),
            generated_at=_str_or_none(payload.get("generated_at") if payload else None),
            research_only=_bool_or_none(payload.get("research_only") if payload else None),
            live_pnl_claim=_bool_or_none(payload.get("live_pnl_claim") if payload else None),
            size_bytes=stat.st_size,
            parse_error=error,
        )

    def _iter_receipt_files() -> list[Path]:
        root = cfg.receipts_dir
        if not root.is_dir():
            return []
        from quant_fund.utils.receipt import verified_corpus_files

        return sorted(_contained(root, p) for p in verified_corpus_files(root))

    def _iter_result_dirs() -> list[tuple[str, str, Path]]:
        metadata = _contained(cfg.data_root, cfg.metadata_dir)
        found: list[tuple[str, str, Path]] = []
        if not metadata.is_dir():
            return found
        for kind_dir in sorted(p for p in metadata.iterdir() if p.is_dir()):
            kind_dir = _contained(metadata, kind_dir)
            for run_dir in sorted(p for p in kind_dir.iterdir() if p.is_dir()):
                run_dir = _contained(metadata, run_dir)
                if _contained(run_dir, run_dir / "manifest.json").is_file():
                    found.append((kind_dir.name, run_dir.name, run_dir))
        return found

    def _result_path(kind: str, name: str) -> Path:
        if not _RESULT_COMPONENT_RE.fullmatch(kind):
            raise HTTPException(400, "invalid result kind")
        if not _RESULT_COMPONENT_RE.fullmatch(name):
            raise HTTPException(400, "invalid result name")
        metadata = _contained(cfg.data_root, cfg.metadata_dir)
        path = _contained(metadata, metadata / kind / name)
        if not path.is_dir() or not _contained(path, path / "manifest.json").is_file():
            raise HTTPException(404, f"result not found: {kind}/{name}")
        return path

    def _result_files(run_dir: Path) -> list[ResultFile]:
        files: list[ResultFile] = []
        for path in sorted(p for p in run_dir.iterdir() if p.is_file()):
            path = _contained(run_dir, path)
            files.append(
                ResultFile(
                    name=path.name,
                    sha256=_sha256_file(path),
                    size_bytes=path.stat().st_size,
                    compressed=path.suffix == ".gz",
                )
            )
        return files

    def _verifier_run_id_parts(stem: str) -> tuple[str | None, str | None]:
        match = _VERIFIER_RUN_RE.fullmatch(stem)
        if match is None:
            return None, None
        return match.group(1), match.group(2)

    def _verifier_run_path(run_log_id: str) -> Path:
        if not _RECEIPT_ID_RE.fullmatch(run_log_id):
            raise HTTPException(400, "invalid verifier run id")
        root = cfg.verifier_dir / "runs"
        path = _contained(root, root / f"{run_log_id}.md")
        if not path.is_file():
            raise HTTPException(404, f"verifier run not found: {run_log_id}")
        return path

    def _verifier_run_summary(path: Path) -> VerifierRunSummary:
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        time_str, version = _verifier_run_id_parts(path.stem)
        verdict_match = _VERDICT_RE.search(text)
        time_match = _TIME_RE.search(text)
        return VerifierRunSummary(
            id=path.stem,
            version=version,
            time=time_match.group(1) if time_match else time_str,
            verdict=verdict_match.group(1) if verdict_match else None,
            sha256=_sha256_bytes(raw),
            size_bytes=len(raw),
        )

    def _artifact_path(artifact_id: str) -> Path:
        if not _ARTIFACT_ID_RE.fullmatch(artifact_id):
            raise HTTPException(400, "invalid artifact id")
        root = cfg.artifacts_dir
        path = _contained(root, root / f"{artifact_id}.json")
        if not path.is_file():
            raise HTTPException(404, f"artifact not found: {artifact_id}")
        return path

    def _artifact_summary(root: Path, path: Path) -> ArtifactSummary:
        stat = path.stat()
        digest, payload, error = docs.read(path)
        rel = path.relative_to(root).as_posix()
        return ArtifactSummary(
            id=rel[: -len(".json")] if rel.endswith(".json") else rel,
            path=rel,
            sha256=digest,
            schema_tag=_str_or_none(payload.get("schema") if payload else None),
            size_bytes=stat.st_size,
            parse_error=error,
        )

    # -- verification ----------------------------------------------------------

    def _verify_run(run_id: str) -> VerificationResult:
        """Verify a run receipt in the layout its ``artifacts`` block declares.

        Run receipts under ``runs/`` are the immutable twins; their
        ``artifacts.immutable_*`` pointers are relative to the research root
        (``metadata/research/``), so verifying the file in place misreports
        ``artifact_missing`` context errors. We stage the declared layout in a
        temp dir — the receipt bytes as ``latest.json`` plus
        ``runs/<run_id>.{json,md}`` — and run the canonical notebook verifier
        on it. The staging is write-free with respect to the evidence roots.
        """
        import shutil
        import tempfile

        if not _RUN_ID_RE.fullmatch(run_id):
            raise HTTPException(400, "run_id must be a 64-char lowercase sha256")
        path = _run_path(run_id)
        raw = path.read_bytes()
        digest = _sha256_bytes(raw)
        from quant_fund.research.verify import verify_research_artifact

        with tempfile.TemporaryDirectory(prefix="research_api_verify_") as tmp:
            stage = Path(tmp)
            staged_runs = stage / "runs"
            staged_runs.mkdir()
            (staged_runs / f"{run_id}.json").write_bytes(raw)
            markdown = _contained(_runs_dir(), path.with_suffix(".md"))
            if markdown.is_file():
                shutil.copyfile(markdown, staged_runs / f"{run_id}.md")
            (stage / "latest.json").write_bytes(raw)
            # Reject pointers before the canonical verifier can read them.
            # Verification is deliberately uncached: the markdown twin can
            # change independently of the JSON receipt.
            try:
                payload = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                payload = None
            artifacts = payload.get("artifacts") if isinstance(payload, dict) else None
            if isinstance(artifacts, dict):
                for key in ("immutable_json", "immutable_markdown"):
                    pointer = artifacts.get(key)
                    if isinstance(pointer, str):
                        _contained(stage, stage / pointer)
            result = verify_research_artifact(stage / "latest.json")
        outcome = VerificationResult(
            subject=run_id,
            kind="research_run",
            verifier="verify_research_artifact",
            valid=bool(result.get("valid")),
            errors=[str(e) for e in result.get("errors", [])],
            sha256=digest,
            run_id=_str_or_none(result.get("run_id")),
            scorecard_families=_int_or_none(result.get("scorecard_families")),
            context="staged_declared_layout",
        )
        return outcome

    def _verify_receipt(receipt_id: str) -> VerificationResult:
        """Lightweight honesty-flag check for committed receipts/*.json.

        These schema-tagged evidence documents are not research notebooks, so
        the notebook verifier does not apply. The check is intentionally
        narrow: valid JSON object, schema tag present, and the honesty flags
        ``research_only == true`` / ``live_pnl_claim == false``.
        """
        path = _receipt_path(receipt_id)
        digest, payload, error = docs.read(path)
        errors: list[str] = []
        if error is not None or payload is None:
            errors.append(error or "json_not_object")
        else:
            if not _str_or_none(payload.get("schema")):
                errors.append("schema_missing")
            if payload.get("research_only") is not True:
                errors.append("research_only_flag_invalid")
            if payload.get("live_pnl_claim") is not False:
                errors.append("live_pnl_claim_invalid")
        return VerificationResult(
            subject=receipt_id,
            kind="receipt",
            verifier="receipt_honesty_flags",
            valid=not errors,
            errors=errors,
            sha256=digest,
        )

    def _verify_result(kind: str, name: str) -> VerificationResult:
        run_dir = _result_path(kind, name)
        from quant_fund.research.phase1_verify import verify_phase1_run

        _result_files(run_dir)  # reject escaping symlink files before verification
        result = verify_phase1_run(run_dir)
        manifest_sha: str | None = None
        manifest = run_dir / "manifest.json"
        if manifest.is_file():
            manifest_sha = _sha256_file(manifest)
        return VerificationResult(
            subject=f"{kind}/{name}",
            kind="benchmark_run",
            verifier="verify_phase1_run",
            valid=bool(result.get("valid")),
            errors=[str(e) for e in result.get("errors", [])],
            sha256=manifest_sha,
            state=_str_or_none(result.get("state")),
        )

    # -- routes ----------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            data_root_present=cfg.data_root.is_dir(),
            receipts_dir_present=cfg.receipts_dir.is_dir(),
            verifier_dir_present=cfg.verifier_dir.is_dir(),
            artifacts_dir_present=cfg.artifacts_dir.is_dir(),
        )

    @app.get("/ready", response_model=ReadyResponse)
    def ready() -> ReadyResponse | JSONResponse:
        """Readiness probe: evidence directories must be present (fail closed).

        Unlike ``/health`` (always 200 while the process is up), this returns
        503 when a configured root is missing. Auth follows the same
        loopback-or-key gate as other non-public routes; Docker HEALTHCHECK
        continues to use ``/health``.
        """
        checks = {
            "data_root": cfg.data_root.is_dir(),
            "receipts_dir": cfg.receipts_dir.is_dir(),
            "verifier_dir": cfg.verifier_dir.is_dir(),
            "artifacts_dir": cfg.artifacts_dir.is_dir(),
        }
        if all(checks.values()):
            return ReadyResponse(status="ready", checks=checks)
        return JSONResponse(
            status_code=503,
            content=ReadyResponse(status="not_ready", checks=checks).model_dump(),
        )

    @app.get("/runs", response_model=RunListResponse)
    def list_runs(
        limit: int = Query(default=50, ge=1, le=_MAX_LIST_LIMIT),
        offset: int = Query(default=0, ge=0),
    ) -> RunListResponse:
        root = _runs_dir()
        files = (
            sorted(_contained(root, p) for p in root.glob("*.json") if p.is_file())
            if root.is_dir()
            else []
        )
        summaries = [_run_summary(path) for path in files]
        summaries.sort(key=lambda s: (s.generated_at or "", s.run_id), reverse=True)
        window = summaries[offset : offset + limit]
        return RunListResponse(items=window, total=len(summaries), limit=limit, offset=offset)

    @app.get("/runs/latest", response_model=RunDetail)
    def latest_run() -> RunDetail:
        path = _contained(cfg.data_root, cfg.data_root / "metadata" / "research" / "latest.json")
        if not path.is_file():
            raise HTTPException(404, "no latest research receipt at metadata/research/latest.json")
        digest, payload, error = docs.read(path)
        if payload is None:
            raise HTTPException(422, f"latest research receipt unreadable: {error}")
        provenance = payload.get("provenance")
        run_id = provenance.get("run_id") if isinstance(provenance, dict) else None
        return RunDetail(
            run_id=_str_or_none(run_id) or "latest",
            sha256=digest,
            receipt=payload,
        )

    @app.get("/runs/{run_id}", response_model=RunDetail)
    def get_run(run_id: str) -> RunDetail:
        path = _run_path(run_id)
        digest, payload, error = docs.read(path)
        if payload is None:
            raise HTTPException(422, f"research run receipt unreadable: {error}")
        provenance = payload.get("provenance")
        actual = provenance.get("run_id") if isinstance(provenance, dict) else None
        return RunDetail(
            run_id=_str_or_none(actual) or run_id,
            sha256=digest,
            receipt=payload,
        )

    @app.get("/runs/{run_id}/markdown", response_model=RunMarkdownDetail)
    def get_run_markdown(run_id: str) -> RunMarkdownDetail:
        json_path = _run_path(run_id)
        path = _contained(_runs_dir(), json_path.with_suffix(".md"))
        if not path.is_file():
            raise HTTPException(404, f"markdown twin not found for run: {run_id}")
        raw = path.read_bytes()
        return RunMarkdownDetail(
            run_id=run_id,
            sha256=_sha256_bytes(raw),
            markdown=raw.decode("utf-8", errors="replace"),
        )

    @app.get("/runs/{run_id}/verification", response_model=VerificationResult)
    def verify_run(run_id: str) -> VerificationResult:
        return _verify_run(run_id)

    @app.get("/receipts", response_model=ReceiptListResponse)
    def list_receipts() -> ReceiptListResponse:
        items = [_receipt_summary(path) for path in _iter_receipt_files()]
        return ReceiptListResponse(items=items, total=len(items))

    @app.get("/receipts/by-hash/{digest}", response_model=ReceiptDetail)
    def get_receipt_by_hash(digest: str) -> ReceiptDetail:
        if not _SHA256_RE.fullmatch(digest):
            raise HTTPException(400, "digest must be a 64-char lowercase sha256")
        for path in _iter_receipt_files():
            file_digest, payload, error = docs.read(path)
            if file_digest == digest:
                if payload is None:
                    raise HTTPException(422, f"receipt unreadable: {error}")
                return ReceiptDetail(
                    id=path.stem,
                    filename=path.name,
                    sha256=file_digest,
                    schema_tag=_str_or_none(payload.get("schema")),
                    receipt=payload,
                )
        raise HTTPException(404, f"no receipt with sha256: {digest}")

    @app.get("/receipts/{receipt_id}", response_model=ReceiptDetail)
    def get_receipt(receipt_id: str) -> ReceiptDetail:
        path = _receipt_path(receipt_id)
        digest, payload, error = docs.read(path)
        if payload is None:
            raise HTTPException(422, f"receipt unreadable: {error}")
        return ReceiptDetail(
            id=path.stem,
            filename=path.name,
            sha256=digest,
            schema_tag=_str_or_none(payload.get("schema")),
            receipt=payload,
        )

    @app.get("/receipts/{receipt_id}/verification", response_model=VerificationResult)
    def verify_receipt(receipt_id: str) -> VerificationResult:
        return _verify_receipt(receipt_id)

    @app.get("/results", response_model=ResultListResponse)
    def list_results() -> ResultListResponse:
        items: list[ResultSummary] = []
        for kind, name, run_dir in _iter_result_dirs():
            created_at: str | None = None
            _digest, manifest, _error = docs.read(run_dir / "manifest.json")
            if manifest is not None:
                created_at = _str_or_none(manifest.get("created_at"))
            names = {p.name for p in run_dir.iterdir() if p.is_file()}
            items.append(
                ResultSummary(
                    kind=kind,
                    name=name,
                    path=f"metadata/{kind}/{name}",
                    created_at=created_at,
                    has_manifest="manifest.json" in names,
                    has_validation=any(
                        n in names for n in ("validation.json", "validation.json.gz")
                    ),
                    has_test=any(n in names for n in ("test.json", "test.json.gz")),
                )
            )
        return ResultListResponse(items=items, total=len(items))

    @app.get("/results/{kind}/{name}", response_model=ResultDetail)
    def get_result(kind: str, name: str) -> ResultDetail:
        run_dir = _result_path(kind, name)
        digest, manifest, _error = docs.read(run_dir / "manifest.json")
        return ResultDetail(
            kind=kind,
            name=name,
            path=f"metadata/{kind}/{name}",
            manifest=manifest,
            manifest_sha256=digest,
            files=_result_files(run_dir),
        )

    @app.get("/results/{kind}/{name}/verification", response_model=VerificationResult)
    def verify_result(kind: str, name: str) -> VerificationResult:
        return _verify_result(kind, name)

    @app.get("/verifier/versions", response_model=VerifierVersionListResponse)
    def list_verifier_versions() -> VerifierVersionListResponse:
        root = cfg.verifier_dir
        items: list[VerifierVersionSummary] = []
        if root.is_dir():
            for version_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                if not _VERIFIER_VERSION_RE.fullmatch(version_dir.name):
                    continue
                acceptance = _contained(root, version_dir / "acceptance.md")
                if not acceptance.is_file():
                    continue
                raw = acceptance.read_bytes()
                items.append(
                    VerifierVersionSummary(
                        version=version_dir.name,
                        sha256=_sha256_bytes(raw),
                        size_bytes=len(raw),
                    )
                )
        return VerifierVersionListResponse(items=items, total=len(items))

    @app.get("/verifier/versions/{version}", response_model=VerifierVersionDetail)
    def get_verifier_version(version: str) -> VerifierVersionDetail:
        if not _VERIFIER_VERSION_RE.fullmatch(version):
            raise HTTPException(400, "version must look like 'vN'")
        path = _contained(cfg.verifier_dir, cfg.verifier_dir / version / "acceptance.md")
        if not path.is_file():
            raise HTTPException(404, f"verifier version not found: {version}")
        raw = path.read_bytes()
        return VerifierVersionDetail(
            version=version,
            sha256=_sha256_bytes(raw),
            acceptance_markdown=raw.decode("utf-8", errors="replace"),
        )

    @app.get("/verifier/runs", response_model=VerifierRunListResponse)
    def list_verifier_runs() -> VerifierRunListResponse:
        root = cfg.verifier_dir / "runs"
        items = [
            _verifier_run_summary(_contained(cfg.verifier_dir, path))
            for path in sorted(root.glob("*.md"))
            if path.is_file()
        ]
        return VerifierRunListResponse(items=items, total=len(items))

    @app.get("/verifier/runs/{run_log_id}", response_model=VerifierRunDetail)
    def get_verifier_run(run_log_id: str) -> VerifierRunDetail:
        path = _verifier_run_path(run_log_id)
        raw = path.read_bytes()
        summary = _verifier_run_summary(path)
        return VerifierRunDetail(
            id=summary.id,
            version=summary.version,
            time=summary.time,
            verdict=summary.verdict,
            sha256=summary.sha256,
            markdown=raw.decode("utf-8", errors="replace"),
        )

    @app.get("/artifacts", response_model=ArtifactListResponse)
    def list_artifacts() -> ArtifactListResponse:
        root = cfg.artifacts_dir
        items: list[ArtifactSummary] = []
        if root.is_dir():
            for path in sorted(root.rglob("*.json")):
                if not path.is_file():
                    continue
                items.append(_artifact_summary(root, _contained(root, path)))
        return ArtifactListResponse(items=items, total=len(items))

    @app.get("/artifacts/{artifact_id:path}", response_model=ArtifactDetail)
    def get_artifact(artifact_id: str) -> ArtifactDetail:
        path = _artifact_path(artifact_id)
        digest, payload, error = docs.read(path)
        if payload is None:
            raise HTTPException(422, f"artifact unreadable: {error}")
        rel = path.relative_to(cfg.artifacts_dir.resolve()).as_posix()
        return ArtifactDetail(
            id=rel[: -len(".json")] if rel.endswith(".json") else rel,
            path=rel,
            sha256=digest,
            artifact=payload,
        )

    return app


app = create_app()


if __name__ == "__main__":
    # `python -m quant_fund.api.research_api` — same loopback guard as
    # `dipcatcher api`: refuse non-loopback binds without RESEARCH_API_KEY.
    import uvicorn

    host = os.environ.get("RESEARCH_API_HOST", "127.0.0.1")
    port = int(os.environ.get("RESEARCH_API_PORT", "8010"))
    if host not in {"127.0.0.1", "::1", "localhost"} and not os.environ.get("RESEARCH_API_KEY"):
        raise SystemExit(
            "non-loopback binding requires RESEARCH_API_KEY; refusing unauthenticated exposure"
        )
    uvicorn.run(app, host=host, port=port, reload=False)
