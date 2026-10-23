"""Unified ``receipt.v2`` envelope for sealed research evidence (P7.2 + P7.4).

Every research lane (eval, incumbent, carry, paper) seals the same envelope:
dataset/code/params digests, an environment fingerprint (Python, NumPy,
Polars, SciPy plus the BLAS/LAPACK build actually loaded), a mandatory
``live_pnl_claim`` flag fixed ``false``, a verdict, and a canonical SHA-256
seal. ``receipt_v2.schema.json`` beside this module is the published
contract; the pydantic models here are its executable form and stay in sync
with it (test_receipt_v2 checks the drift).

``verify_receipt_file`` validates either generation structurally and checks
hash consistency. v1 is verified under both digests conventions in the repo:
the canonical ``canonical_json_bytes`` form and the strict ``json.dumps``
form used by ``real_benchmark``.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.research.evalue_contracts import EVALUE_FAMILY_KINDS
from quant_fund.research.impossible_fit import impossible_fit_scan
from quant_fund.research.quantile_ladder import QUANTILE_LADDER_KINDS
from quant_fund.utils.hashing import SHA256_HEX_LENGTH, canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

RECEIPT_V2_SCHEMA = "receipt.v2"
RECEIPT_V2_SCHEMA_VERSION: Literal[2] = 2
RECEIPT_V2_VERDICTS = ("pass", "fail", "blocked")
RECEIPT_V2_SCHEMA_FILE = "receipt_v2.schema.json"

# The numeric stack a determinism sweep (P7.4) compares across machines.
REQUIRED_ENV_PACKAGES = frozenset({"numpy", "polars", "scipy"})

Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


def _is_sha256(value: object) -> bool:
    """Accept only lowercase hexadecimal SHA-256 digests."""
    return (
        isinstance(value, str)
        and len(value) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _timestamp_valid(value: object) -> bool:
    """Require an ISO-8601 timestamp carrying an explicit timezone."""
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


class BlasDependency(BaseModel):
    """One BLAS/LAPACK build dependency as recorded by ``np.__config__.CONFIG``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    found: bool
    version: str | None = None


class ThreadpoolInfo(BaseModel):
    """One loaded BLAS threadpool (``threadpoolctl.threadpool_info`` row)."""

    model_config = ConfigDict(extra="forbid")

    prefix: str = Field(min_length=1)
    user_api: str = Field(min_length=1)
    internal_api: str = Field(min_length=1)
    version: str | None = None
    num_threads: int = Field(ge=1)
    threading_layer: str | None = None
    architecture: str | None = None


class ReceiptEnvironment(BaseModel):
    """Per-receipt environment block: interpreter, numeric stack, BLAS/LAPACK."""

    model_config = ConfigDict(extra="forbid")

    python: str = Field(min_length=1)
    implementation: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    machine: str = Field(min_length=1)
    byteorder: Literal["little", "big"]
    packages: dict[str, str]
    blas: BlasDependency
    lapack: BlasDependency
    threadpools: list[ThreadpoolInfo]
    fingerprint_sha256: Sha256Hex

    @model_validator(mode="after")
    def packages_cover_numeric_stack(self) -> ReceiptEnvironment:
        missing = sorted(REQUIRED_ENV_PACKAGES - self.packages.keys())
        if missing:
            raise ValueError(f"environment.packages missing: {','.join(missing)}")
        if any(not name.strip() for name in self.packages):
            raise ValueError("environment.packages names must be nonempty")
        if any(not value.strip() for value in self.packages.values()):
            raise ValueError("environment.packages versions must be nonempty")
        return self


class ReceiptV2(BaseModel):
    """The unified sealed-receipt envelope.

    ``payload`` holds the lane-specific body (e.g. the ``fleet_eval.v1``
    receipt verbatim). ``receipt_sha256`` is added by the writer's seal step
    and verified, not produced, by this model. The JSON key is ``schema``;
    the attribute is ``schema_id`` because ``schema`` shadows a BaseModel
    member.
    """

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["receipt.v2"] = Field(alias="schema")
    schema_version: Literal[2]
    kind: str = Field(min_length=1)
    data_label: str = Field(min_length=1)
    generated_at: str
    git_revision: str = Field(min_length=1)
    dataset_hash: Sha256Hex
    params_hash: Sha256Hex
    code_sha256: Sha256Hex
    code_files: dict[str, Sha256Hex] = Field(min_length=1)
    environment: ReceiptEnvironment
    live_pnl_claim: Literal[False]
    verdict: Literal["pass", "fail", "blocked"]
    payload: dict[str, Any]
    receipt_sha256: Sha256Hex | None = None

    @model_validator(mode="after")
    def generated_at_has_timezone(self) -> ReceiptV2:
        if not _timestamp_valid(self.generated_at):
            raise ValueError("generated_at must be an ISO-8601 timestamp with timezone")
        if not self.payload:
            raise ValueError("payload must be a nonempty object")
        return self


def _package_versions(names: tuple[str, ...]) -> dict[str, str]:
    packages: dict[str, str] = {}
    for name in names:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "UNAVAILABLE"
    return packages


def _blas_dependency(kind: Literal["blas", "lapack"]) -> dict[str, Any]:
    """Normalize ``np.__config__.CONFIG['Build Dependencies'][kind]``."""
    config = getattr(np.__config__, "CONFIG", None)
    raw: Mapping[str, Any] = {}
    if isinstance(config, Mapping):
        deps = config.get("Build Dependencies")
        if isinstance(deps, Mapping):
            candidate = deps.get(kind)
            if isinstance(candidate, Mapping):
                raw = candidate
    raw_version = raw.get("version")
    return {
        "name": str(raw.get("name") or "unknown"),
        "found": bool(raw.get("found", False)),
        "version": None if raw_version in (None, "unknown") else str(raw_version),
    }


def _blas_threadpools() -> list[dict[str, Any]]:
    """Loaded BLAS pools via threadpoolctl when it is installed (sklearn dep).

    The fingerprint records the library actually mapped into the process and
    its thread count — the two facts a cross-machine determinism sweep needs.
    """
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return []
    pools: list[dict[str, Any]] = []
    for info in threadpool_info():
        if not isinstance(info, Mapping) or info.get("user_api") != "blas":
            continue
        num_threads = info.get("num_threads")
        if not isinstance(num_threads, int) or isinstance(num_threads, bool) or num_threads < 1:
            continue
        row: dict[str, Any] = {
            "prefix": str(info.get("prefix") or "unknown"),
            "user_api": "blas",
            "internal_api": str(info.get("internal_api") or "unknown"),
            "num_threads": num_threads,
        }
        for key in ("version", "threading_layer", "architecture"):
            value = info.get(key)
            if isinstance(value, str) and value:
                row[key] = value
        pools.append(row)
    pools.sort(key=lambda row: (str(row["prefix"]), int(row["num_threads"])))
    return pools


def environment_fingerprint() -> dict[str, Any]:
    """Capture the runtime numeric stack; embed its own canonical digest.

    ``fingerprint_sha256`` covers every other field in the block, so a
    determinism sweep compares one digest instead of diffing nested config.
    """
    environment: dict[str, Any] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "packages": _package_versions(("numpy", "polars", "scipy")),
        "blas": _blas_dependency("blas"),
        "lapack": _blas_dependency("lapack"),
        "threadpools": _blas_threadpools(),
    }
    environment["fingerprint_sha256"] = hash_bytes(canonical_json_bytes(environment))
    return environment


def code_fingerprint(paths: tuple[Path, ...] | list[Path]) -> dict[str, Any]:
    """Map each writer source file to its SHA-256 and digest the map.

    Returns ``{"files": {name: sha256}, "code_sha256": digest-of-map}`` so a
    receipt can carry both the audit trail and one bound digest.
    """
    files: dict[str, str] = {}
    for raw in paths:
        path = Path(raw)
        if path.name in files:
            raise ValueError(f"duplicate code file name: {path.name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not files:
        raise ValueError("code_fingerprint requires at least one file")
    return {"files": files, "code_sha256": hash_bytes(canonical_json_bytes(files))}


def build_receipt_v2(
    *,
    kind: str,
    data_label: str,
    dataset: Mapping[str, Any],
    params: Mapping[str, Any],
    code_files: tuple[Path, ...] | list[Path],
    verdict: str,
    payload: Mapping[str, Any],
    generated_at: str | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    """Build the unsealed ``receipt.v2`` envelope and validate it fail-closed.

    ``dataset``/``params`` are canonicalized to digests — callers pass the
    identity of what was evaluated (e.g. per-shard content hashes) and the
    run parameters, not the raw arrays.
    """
    code = code_fingerprint(list(code_files))
    stamped = generated_at if generated_at is not None else datetime.now(UTC).isoformat()
    receipt: dict[str, Any] = {
        "schema": RECEIPT_V2_SCHEMA,
        "schema_version": RECEIPT_V2_SCHEMA_VERSION,
        "kind": str(kind),
        "data_label": str(data_label),
        "generated_at": stamped,
        "git_revision": revision if revision is not None else git_revision(),
        "dataset_hash": hash_bytes(canonical_json_bytes(dataset)),
        "params_hash": hash_bytes(canonical_json_bytes(params)),
        "code_sha256": code["code_sha256"],
        "code_files": code["files"],
        "environment": environment_fingerprint(),
        "live_pnl_claim": False,
        "verdict": verdict,
        "payload": json.loads(canonical_json_bytes(dict(payload))),
    }
    ReceiptV2.model_validate(receipt)
    return receipt


def seal_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize a receipt body and stamp ``receipt_sha256`` over it."""
    canonical = json.loads(canonical_json_bytes(dict(receipt)))
    canonical.pop("receipt_sha256", None)
    return {**canonical, "receipt_sha256": hash_bytes(canonical_json_bytes(canonical))}


def _canonical_digest(body: Mapping[str, Any]) -> str:
    return hash_bytes(canonical_json_bytes(dict(body)))


def _strict_digest(body: Mapping[str, Any]) -> str:
    """The ``real_benchmark`` seal: sorted compact JSON, ASCII-escaped."""
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class ReceiptVerification(TypedDict):
    """``verify-receipt`` result. ``kind`` is untrusted until errors is empty.

    ``warnings`` are informational statistical-leakage canaries from
    ``impossible_fit_scan``: scores too good to be honest. They never affect
    ``valid`` — a degenerate synthetic shard can legitimately trip them.
    """

    valid: bool
    path: str
    schema: str
    kind: object
    verdict: object
    digest_convention: str | None
    errors: list[str]
    warnings: list[str]


def _result(
    path: Path, payload: object, digest_convention: str | None, errors: list[str]
) -> ReceiptVerification:
    body = payload if isinstance(payload, dict) else {}
    schema_field = body.get("schema", body.get("schema_version", "unknown"))
    return {
        "valid": not errors,
        "path": str(path),
        "schema": str(schema_field),
        "kind": body.get("kind"),
        "verdict": body.get("verdict"),
        "digest_convention": digest_convention,
        "errors": errors,
        "warnings": impossible_fit_scan(payload),
    }


def wrap_receipt_v2(
    receipt: Mapping[str, Any],
    *,
    code_files: tuple[Path, ...] | list[Path],
    verdict: str,
    kind: str | None = None,
    data_label: str | None = None,
    dataset: Mapping[str, Any] | None = None,
    params: Mapping[str, Any] | None = None,
    generated_at: str | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    """Wrap a lane's own v1 receipt body in the ``receipt.v2`` envelope.

    Shared by every v1 writer opting into ``receipt_version=2``: the envelope
    bindings come from the payload's own identity fields — the lane's
    ``inputs_sha256``/``dataset_sha256``/``weights_sha256`` digests seed
    ``dataset_hash`` (a digest of the payload itself when absent), a
    ``params`` mapping seeds ``params_hash``, and the lane's recorded
    ``generated_at``/``git_revision``/``code_revision``/``generated_at_commit``
    stamp the envelope. Callers may override any binding explicitly when the
    lane's identity lives in differently named fields.

    A top-level ``meta`` block is volatile provenance (wall-clock stamps and
    the checked-out revision): replay-declared lanes keep it out of the
    sealed artifact so argv-produced bytes are byte-identical across runs.
    The envelope still stamps it here — ``meta`` never enters the embedded
    payload.
    """
    meta = receipt.get("meta")
    meta_map: Mapping[str, Any] = meta if isinstance(meta, Mapping) else {}
    sealed_body = {key: value for key, value in receipt.items() if key != "meta"}
    if dataset is not None:
        bound_dataset: Mapping[str, Any] = dataset
    else:
        bound_dataset = {
            key: sealed_body[key]
            for key in ("inputs_sha256", "dataset_sha256", "weights_sha256")
            if key in sealed_body
        }
        if not bound_dataset:
            bound_dataset = {"payload_sha256": hash_bytes(canonical_json_bytes(dict(sealed_body)))}
    lane_params = receipt.get("params")
    bound_params = (
        params if params is not None else lane_params if isinstance(lane_params, Mapping) else {}
    )
    revision = (
        revision
        or meta_map.get("git_revision")
        or meta_map.get("code_revision")
        or receipt.get("git_revision")
        or receipt.get("code_revision")
        or receipt.get("generated_at_commit")
    )
    generated_at = generated_at or meta_map.get("generated_at") or receipt.get("generated_at")
    return build_receipt_v2(
        kind=str(kind or receipt.get("kind") or receipt.get("schema") or "receipt"),
        data_label=str(data_label or receipt.get("data_label") or "UNKNOWN"),
        dataset=bound_dataset,
        params=bound_params,
        code_files=code_files,
        verdict=verdict,
        payload=dict(sealed_body),
        generated_at=str(generated_at) if generated_at is not None else None,
        revision=str(revision) if revision is not None else None,
    )


def _digest_or_none(
    body: Mapping[str, Any], digest: Callable[[Mapping[str, Any]], str]
) -> str | None:
    """Hash a body that may be unhashable (NaN, unserializable, deep nest).

    ``json.loads`` accepts literals canonical digests reject — NaN floats,
    >4300-digit ints already fail at load, but a NaN *inside* a parsed body
    reaches the digester, where ``allow_nan=False`` raises. The verifier must
    degrade to a verdict, never crash on hostile input.
    """
    try:
        return digest(body)
    except (ValueError, RecursionError, TypeError):
        return None


def _seal_errors(payload: Mapping[str, Any]) -> tuple[str | None, list[str]]:
    """Check ``receipt_sha256``; report which digest convention matched.

    A convention that cannot digest the body at all (e.g. ``json.dumps``
    refusing a non-finite float that survived into a programmatic payload)
    simply does not match — uncomputable digests are an error, not a crash.
    """
    seal = payload.get("receipt_sha256")
    if not _is_sha256(seal):
        return None, ["receipt_sha256_missing_or_invalid"]
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    computable = False
    for name, digest in (("canonical_json", _canonical_digest), ("strict_json", _strict_digest)):
        try:
            actual = digest(body)
        except (TypeError, ValueError):
            continue
        computable = True
        if actual == seal:
            return name, []
    return None, ["receipt_sha256_mismatch" if computable else "receipt_sha256_uncomputable"]


def _env_fingerprint_errors(environment: object) -> list[str]:
    if not isinstance(environment, dict):
        return ["environment_invalid"]
    stamp = environment.get("fingerprint_sha256")
    if not _is_sha256(stamp):
        return ["environment_fingerprint_missing"]
    body = {key: value for key, value in environment.items() if key != "fingerprint_sha256"}
    actual = _digest_or_none(body, _canonical_digest)
    if actual is None:
        return ["environment_fingerprint_uncomputable"]
    if actual != stamp:
        return ["environment_fingerprint_mismatch"]
    return []


def _code_consistency_errors(payload: Mapping[str, Any]) -> list[str]:
    files = payload.get("code_files")
    if not isinstance(files, dict) or not files:
        return ["code_files_invalid"]
    actual = _digest_or_none(files, _canonical_digest)
    if actual is None:
        return ["code_sha256_uncomputable"]
    if actual != payload.get("code_sha256"):
        return ["code_sha256_mismatch"]
    return []


def _looks_like_fleet_eval(body: object) -> bool:
    """Structural fingerprint of a fleet_eval receipt — schema/kind agnostic.

    Dispatch on content, not the claimed ``schema``/``kind``: those fields are
    attacker-controlled, and a reseal costs nothing (the digest is a public
    sha256), so renaming them must not evade the deep contract checks.
    """
    if not (
        isinstance(body, Mapping)
        and isinstance(body.get("results"), list)
        and isinstance(body.get("models"), list)
        and isinstance(body.get("shards"), dict)
        and "n_eval" in body
    ):
        return False
    # Adjacent eval lanes (calibration_eval.v1, coherence_eval.v1,
    # hstep_bench.v1) share the results/models/shards envelope — the
    # discriminating signature is the scored grid itself: fleet rows carry
    # per-tau pinball cells and crps. Multi-horizon lanes (hstep, multih)
    # carry a per-row ``horizon`` / top-level ``horizons`` — fleet_eval is
    # single-horizon by construction, so those are not fleet receipts.
    if "horizons" in body:
        return False
    scored = [
        row
        for row in body["results"]
        if isinstance(row, Mapping)
        and row.get("status") is not None
        and ("crps" in row or any(key.startswith("pinball_") for key in row))
    ]
    return bool(scored) and all("horizon" not in row for row in scored)


def _looks_like_hstep_eval(body: object) -> bool:
    """Structural fingerprint of an hstep_bench receipt — schema/kind agnostic.

    Same rename-resilience contract as ``_looks_like_fleet_eval``: the
    multi-horizon scored grid (``horizons`` top-level + per-row ``horizon``)
    is the signature a bare schema strip cannot shed.
    """
    if not (
        isinstance(body, Mapping)
        and isinstance(body.get("results"), list)
        and isinstance(body.get("models"), list)
        and isinstance(body.get("shards"), dict)
        and isinstance(body.get("horizons"), list)
        and "n_eval" in body
    ):
        return False
    return any(
        isinstance(row, Mapping)
        and "horizon" in row
        and ("crps" in row or any(key.startswith("pinball_") for key in row))
        for row in body["results"]
    )


def _looks_like_v2_envelope(payload: Mapping[str, Any]) -> bool:
    """Structural fingerprint of the receipt.v2 envelope."""
    return (
        isinstance(payload.get("payload"), Mapping)
        and isinstance(payload.get("environment"), Mapping)
        and isinstance(payload.get("code_files"), dict)
        and isinstance(payload.get("dataset_hash"), str)
        and isinstance(payload.get("params_hash"), str)
    )


_AUDITOR_ERRORS = (ValueError, TypeError, KeyError, RecursionError, AttributeError)


def _guarded(
    audit: Callable[[Mapping[str, Any]], list[str]], label: str
) -> Callable[[Mapping[str, Any]], list[str]]:
    """A payload that crashes a kind auditor fails that audit — never the gate."""

    def _run(payload: Mapping[str, Any]) -> list[str]:
        try:
            return audit(payload)
        except _AUDITOR_ERRORS:
            return [f"{label}_audit_crash"]

    return _run


def _forbidden_scan_clean(blob: Mapping[str, Any]) -> bool:
    """Key scan must fail closed, not crash, on pathological nesting."""
    try:
        return family_blob_forbidden_metrics_absent(blob)
    except _AUDITOR_ERRORS:
        return False


#: v1 receipt kinds exempt from the blanket forbidden-metric scan — the
#: paper/simulation lanes embed nav/sharpe diagnostics in bodies already
#: gated by their own honesty contract (sim_live_contract_errors).
_PAPER_SCAN_EXEMPT_KINDS = frozenset({"sim_live_receipt", "sim_live_bench_receipt"})


def _inner_claimed_kinds(payload: Mapping[str, Any]) -> set[str]:
    """Kind strings the sealed *inner* payload claims about itself.

    The envelope ``kind`` is attacker-renameable at zero cost; the inner
    body's own ``kind``/``schema`` are inside the seal, so a renamed outer
    kind must not strip the deep checks. ``<base>.vN`` schemas map to both
    ``<base>`` and ``<base>_eval`` (the lane naming convention).
    """
    inner = payload.get("payload")
    claims: set[str] = set()
    if not isinstance(inner, Mapping):
        return claims
    for key in ("kind", "schema"):
        value = inner.get(key)
        if isinstance(value, str) and value:
            claims.add(value)
            base, sep, suffix = value.rpartition(".v")
            if sep and suffix.isdigit() and base:
                claims.add(base)
                claims.add(f"{base}_eval")
    return claims


# Lanes dispatched on the outer ``kind`` OR on a sealed inner claim. An inner
# claim that disagrees with the outer kind is additionally flagged
# ``kind_fingerprint_mismatch`` — resealing an envelope is free, so renaming the
# outer kind must not shed a lane's deep checks.
_LANE_CONSISTENCY: dict[str, str] = {
    "distribution_fleet_eval": "quant_fund.research.fleet_eval.fleet_v2_consistency_errors",
    "capacity_overlay_eval": "quant_fund.research.capacity_overlay.capacity_v2_consistency_errors",
    "cross_sectional_rankic_eval": "quant_fund.research.cross_sectional.rankic_v2_consistency_errors",
    "vol_bench": "quant_fund.research.vol_bench.vol_bench_v2_consistency_errors",
    "basis_carry": "quant_fund.research.basis_carry.basis_carry_v2_consistency_errors",
    "basis_carry_eval": "quant_fund.research.basis_carry.basis_carry_v2_consistency_errors",
    "crossvenue_basis": "quant_fund.research.crossvenue_basis.crossvenue_basis_v2_consistency_errors",
    "crossvenue_basis_eval": "quant_fund.research.crossvenue_basis.crossvenue_basis_v2_consistency_errors",
}

# Lane contracts declared ahead of the module that implements them. A receipt
# claiming one of these kinds fails closed with ``<label>_lane_missing`` —
# never "verified". Each entry requires a non-empty reason; the resolve gate in
# tests/unit/research/test_receipt_v2_lane_targets_resolve.py fails on an
# undocumented pending lane and on a pending lane whose module has since
# landed (shrink-only, so this dict cannot rot).
PENDING_LANE_CONSISTENCY: dict[str, tuple[tuple[str, str, str], str]] = {
    "hstep_bench": (
        (
            "quant_fund.research.hstep_bench",
            "hstep_bench_v2_consistency_errors",
            "hstep_bench_v2_consistency",
        ),
        "multi-horizon h-step lane contract; the scored grid is served today by "
        "research.multih_fleet (kind multih_fleet_eval) and _looks_like_hstep_eval "
        "still fingerprints hstep_bench.v1 envelopes",
    ),
    "fleet_significance_eval": (
        (
            "quant_fund.research.fleet_significance",
            "fleet_significance_v2_consistency_errors",
            "fleet_significance_v2_consistency",
        ),
        "paired DM-matrix significance lane; research.concordance:130-134 mirrors "
        "its ordered pairwise convention pending the module landing on main",
    ),
    "mixture_stability_eval": (
        (
            "quant_fund.research.mixture_stability",
            "mixture_stability_consistency_errors",
            "mixture_stability",
        ),
        "mixture-stability lane (weight-path stability across refits); distinct "
        "from kind expert_mixture_eval, which research.expert_mixture serves and "
        "whose regret identities its own checker re-derives",
    ),
}

# The v1 schema contract for the same missing h-step lane, dispatched from
# ``_verify_v1`` on ``schema == "hstep_bench.v1"`` or the structural
# fingerprint. Same fail-closed rule; enumerated here so the resolve gate covers
# every declared lane target, v1 and v2 alike.
PENDING_V1_CONTRACT: tuple[str, str, str] = (
    "quant_fund.research.hstep_bench",
    "hstep_bench_v1_contract_errors",
    "hstep_bench_v1_contract",
)

# Lanes dispatched on the outer ``kind`` alone, reached only once the
# inner-claim pass above is clean. Value is (module, checker, label); the label
# names the ``<label>_lane_missing`` error emitted when the module is absent.
_LANE_CONSISTENCY_OUTER_KIND: dict[str, tuple[str, str, str]] = {
    "identity_sweep": (
        "quant_fund.research.identity_sweep",
        "identity_v2_consistency_errors",
        "identity_v2_consistency",
    ),
    "coherence_eval": (
        "quant_fund.research.coherence",
        "coherence_v2_consistency_errors",
        "coherence_v2",
    ),
    "selection_concordance": (
        "quant_fund.research.concordance",
        "concordance_consistency_errors",
        "concordance",
    ),
    "evidence_audit": (
        "quant_fund.research.evidence_audit",
        "evidence_audit_consistency_errors",
        "evidence_audit",
    ),
    "nautilus_conformance": (
        "quant_fund.backtest.nautilus_conformance",
        "nautilus_conformance_consistency_errors",
        "nautilus_conformance",
    ),
    "multih_fleet_eval": (
        "quant_fund.research.multih_fleet",
        "multih_fleet_consistency_errors",
        "multih_fleet",
    ),
    "expert_mixture_eval": (
        "quant_fund.research.expert_mixture",
        "expert_mixture_consistency_errors",
        "expert_mixture",
    ),
    "calibration_eval": (
        "quant_fund.research.calibration_eval",
        "calibration_v2_consistency_errors",
        "calibration_v2",
    ),
    **{kind: target for kind, (target, _reason) in PENDING_LANE_CONSISTENCY.items()},
}


def _lane_checker(
    module_path: str, func_name: str, label: str
) -> Callable[[Mapping[str, Any]], list[str]]:
    """Resolve a lane consistency checker, tolerating lanes whose module is not
    yet merged — a receipt claiming an absent lane's kind fails closed rather
    than crashing the sweep (forward-compat for lanes that land later)."""
    try:
        func = getattr(importlib.import_module(module_path), func_name)
    except ImportError:
        return lambda _payload: [f"{label}_lane_missing"]
    return _guarded(func, label)


def _kind_consistency_errors(payload: Mapping[str, Any]) -> list[str]:
    """Lane-specific re-derivation of the bound digests, where defined.

    Dispatches on the envelope ``kind``, on the sealed inner payload's claimed
    ``kind``/``schema`` (renaming the outer kind cannot strip a lane's deep
    checks — the mismatch is flagged), and on the structural fingerprint of
    the fleet/hstep scored grids (a bare schema strip cannot shed them).
    """
    kind = payload.get("kind")
    inner = payload.get("payload")
    looks_fleet = _looks_like_fleet_eval(inner)
    errors: list[str] = []
    # Outer kind or sealed inner claims route through the shared lane map.
    for claimed in sorted({kind, *_inner_claimed_kinds(payload)}, key=str):
        if claimed in _LANE_CONSISTENCY:
            path = _LANE_CONSISTENCY[claimed]
            module, _, func = path.rpartition(".")
            errors.extend(_lane_checker(module, func, f"{claimed}_consistency")(payload))
            if claimed != kind:
                errors.append("kind_fingerprint_mismatch")
            break  # one lane contract per envelope
    if looks_fleet and kind != "distribution_fleet_eval" and not errors:
        errors.extend(
            _lane_checker(
                "quant_fund.research.fleet_eval",
                "fleet_v2_consistency_errors",
                "fleet_v2_consistency",
            )(payload)
        )
        errors.append("kind_fingerprint_mismatch")
    if errors:
        return errors
    # One lane contract per envelope: the kinds are distinct, so a lookup is
    # equivalent to the former ``if kind == ...`` chain. Lanes whose module has
    # not landed are in PENDING_LANE_CONSISTENCY and still fail closed.
    if isinstance(kind, str):
        target = _LANE_CONSISTENCY_OUTER_KIND.get(kind)
        if target is not None:
            module, func, label = target
            return _lane_checker(module, func, label)(payload)
    return []


def _verify_v2(path: Path, payload: Mapping[str, Any]) -> ReceiptVerification:
    errors: list[str] = []
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    try:
        ReceiptV2.model_validate(payload)
    except RecursionError:
        return _result(path, payload, None, ["receipt_v2_schema:nesting_depth"])
    except ValidationError as exc:
        for issue in exc.errors():
            location = ".".join(str(part) for part in issue["loc"]) or "envelope"
            errors.append(f"receipt_v2_schema:{location}:{issue['msg']}")
        return _result(path, payload, None, errors)
    convention, seal_errors = _seal_errors(payload)
    errors.extend(seal_errors)
    errors.extend(_env_fingerprint_errors(body.get("environment")))
    errors.extend(_code_consistency_errors(body))
    payload_body = body.get("payload")
    if isinstance(payload_body, Mapping):
        # Same exemption as the writers: the envelope honesty flag carries a
        # forbidden token but is required, so it is excluded from the scan.
        scanned = {key: value for key, value in payload_body.items() if key != "live_pnl_claim"}
        if not _forbidden_scan_clean(scanned):
            errors.append("payload_forbidden_metrics")
        # Envelope/payload agreement: a payload that echoes either honesty
        # field must not contradict the sealed envelope under a fresh seal.
        inner_label = payload_body.get("data_label")
        if inner_label is not None and inner_label != body.get("data_label"):
            errors.append("payload_data_label_mismatch")
        inner_claim = payload_body.get("live_pnl_claim")
        if inner_claim is not None and inner_claim is not False:
            errors.append("payload_live_pnl_claim_not_false")
        # An inner body that carries receipt_sha256 asserts it binds this
        # payload — a stale or forged inner seal must not ride inside a
        # valid envelope (absent is fine: wrapped bodies are unsigned).
        if "receipt_sha256" in payload_body:
            _inner_conv, inner_seal_errors = _seal_errors(payload_body)
            errors.extend(f"inner_{e}" for e in inner_seal_errors)
        # The inner body claims its own schema/kind — lane contracts apply
        # regardless of what the envelope's ``kind`` was renamed to.
        from quant_fund.research.lane_contracts import lane_contract_errors

        errors.extend(lane_contract_errors(payload_body))
        errors.extend(_tape_binding_errors(payload_body))
    errors.extend(_tape_binding_errors(body))
    errors.extend(_kind_consistency_errors(body))
    return _result(path, payload, convention, errors)


def _tape_binding_errors(payload: Mapping[str, Any]) -> list[str]:
    """A declared tape binding must resolve to a committed ``data/manifests`` pin.

    Ratchet-in contract: receipts declaring ``tape_manifest_sha256`` or
    ``dataset_sha256`` under a non-synthetic ``data_label`` fail closed when
    the digest is unknown to the tape registry; bodies without bindings
    stay admissible.
    """
    from quant_fund.research.tape_registry import tape_binding_errors

    return tape_binding_errors(payload)


def _carries_v2_evidence(payload: Mapping[str, Any]) -> bool:
    """Whether a v1-dispatched blob still carries the v2 envelope's bound fields.

    A ``receipt.v2`` envelope with ``schema``/``schema_version`` stripped and
    re-sealed digests fine under the weaker v1 contract while silently losing
    its environment/code/dataset verification. Legit v1 receipts never carry
    this joint signature (checked against every committed receipt).
    """
    return (
        isinstance(payload.get("payload"), Mapping)
        and isinstance(payload.get("code_files"), Mapping)
        and _is_sha256(payload.get("dataset_hash"))
        and _is_sha256(payload.get("params_hash"))
    )


def _forward_prereg_honesty_errors(payload: Mapping[str, Any]) -> list[str]:
    """Honesty-flag failures on a forward-record pre-registration."""
    errors: list[str] = []
    honesty = payload.get("honesty")
    if not isinstance(honesty, Mapping) or honesty.get("research_only") is not True:
        errors.append("research_only_not_true")
    if not isinstance(honesty, Mapping) or honesty.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    if isinstance(honesty, Mapping) and honesty.get("live_trading_claim") is not False:
        errors.append("live_trading_claim_not_false")
    return errors


def _extend_custody_release_errors(errors: list[str], payload: Mapping[str, Any]) -> None:
    if payload.get("schema") == "custody_proof.v1":
        from quant_fund.research.custody import custody_contract_errors

        errors.extend(custody_contract_errors(payload))
    if payload.get("schema") == "release_attestation.v1":
        from quant_fund.research.release_attestation import release_contract_errors

        errors.extend(release_contract_errors(payload))


def _prereg_document_lanes(payload: Mapping[str, Any]) -> tuple[bool, bool]:
    """Classify a body as a forward preregistration and/or a sidecar lane."""
    schema = payload.get("schema")
    embedded = payload.get("_seal")
    is_forward = schema == "forward_record_preregistration.v1" or (
        isinstance(embedded, Mapping)
        and embedded.get("schema") == "prereg_seal.v1"
        and schema != "prereg_seal.v1"
    )
    is_sidecar_lane = schema == "lane_receipt.v1" and not _is_sha256(payload.get("receipt_sha256"))
    return bool(is_forward), bool(is_sidecar_lane)


def _sidecar_lane_honesty_errors(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("research_only") is not True:
        errors.append("research_only_not_true")
    if payload.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    return errors


def _sealed_bytes_match_errors(path: Path, payload: Mapping[str, Any]) -> list[str]:
    """Reject a mutated object presented with the committed file's path.

    File seals bind bytes on disk. The mutation fuzz passes the mutated
    payload with the original path, so the object must equal the file.
    Forward records embed the content hash inside ``_seal``, which excludes
    the seal block, so a seal-metadata edit still matches the untouched sidecar.
    """
    try:
        on_disk = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ["sealed_file_unreadable"]
    if on_disk != payload:
        return ["payload_file_mismatch"]
    return []


def _prereg_documented_errors(path: Path, payload: Mapping[str, Any]) -> list[str] | None:
    """Errors for documents sealed by ``prereg_seal.v1`` rather than ``receipt_sha256``.

    Returns ``None`` when the body is an ordinary research receipt (caller
    falls through to the ``receipt_sha256`` check). Forward-record
    pre-registrations embed ``_seal.content_sha256``; the cost-aware lane
    receipt binds the file bytes with a sibling ``.seal.json``. Both verify
    through ``quant_fund.data.prereg_seal``. A sidecar file itself is not a
    receipt and returns ``None`` so scanners that skip ``*.seal.json`` stay
    the only callers that should see one.
    """
    is_forward, is_sidecar_lane = _prereg_document_lanes(payload)
    if not is_forward and not is_sidecar_lane:
        return None

    from quant_fund.data.prereg_seal import verify_payload, verify_seal

    errors: list[str] = []
    if is_forward:
        if not isinstance(payload, dict):
            return ["seal_missing"]
        checked = verify_payload(dict(payload))
        errors.extend(str(item) for item in checked.get("errors", []))
        errors.extend(_forward_prereg_honesty_errors(payload))
    else:
        errors.extend(_sidecar_lane_honesty_errors(payload))

    if path.is_file():
        side = verify_seal(path)
        errors.extend(str(item) for item in side.get("errors", []))
        errors.extend(_sealed_bytes_match_errors(path, payload))
    elif is_sidecar_lane:
        errors.append("seal_sidecar_missing")
    return errors


def _verify_v1(path: Path, payload: Mapping[str, Any]) -> ReceiptVerification:
    errors: list[str] = []
    alternate = _prereg_documented_errors(path, payload)
    if alternate is None:
        convention, seal_errors = _seal_errors(payload)
        errors.extend(seal_errors)
    else:
        convention = "prereg_seal"
        errors.extend(alternate)
    if _carries_v2_evidence(payload):
        errors.append("possible_v2_downgrade")
    claim = payload.get("live_pnl_claim")
    if claim is not None and claim is not False:
        errors.append("live_pnl_claim_not_false")
    # The honesty scan applies to research-lane receipts — a v1 payload naming
    # a forbidden headline metric must not verify clean. Paper/simulation lanes
    # legitimately carry nav_*/sharpe_simulated diagnostics under their own
    # contract (sim_live_contract_errors still gates the honesty flags).
    if payload.get("kind") not in _PAPER_SCAN_EXEMPT_KINDS:
        scanned = {key: value for key, value in payload.items() if key != "live_pnl_claim"}
        if not _forbidden_scan_clean(scanned):
            errors.append("forbidden_metric_keys")
    if payload.get("schema") == "fleet_eval.v1" or _looks_like_fleet_eval(payload):
        # Claimed-schema OR structural fingerprint: a payload that claims the
        # schema but is too malformed to match the fingerprint still gets the
        # contract check (which is what flags the malformation), and a payload
        # structurally shaped like a fleet receipt cannot evade it by renaming.
        from quant_fund.research.fleet_eval import (
            fleet_v1_audit_errors,
            fleet_v1_contract_errors,
        )

        errors.extend(_guarded(fleet_v1_contract_errors, "fleet_v1_contract")(payload))
        errors.extend(_guarded(fleet_v1_audit_errors, "fleet_v1")(payload))
    schema = payload.get("schema")
    if schema == "vol_bench.v1":
        from quant_fund.research.vol_bench import vol_bench_contract_errors

        errors.extend(vol_bench_contract_errors(payload))
    if payload.get("kind") in QUANTILE_LADDER_KINDS:
        from quant_fund.research.quantile_ladder import _quantile_ladder_errors

        errors.extend(_quantile_ladder_errors(payload))
    if payload.get("kind") in EVALUE_FAMILY_KINDS:
        from quant_fund.research.evalue_contracts import evalue_family_contract_errors

        errors.extend(evalue_family_contract_errors(payload))
    elif schema == "capacity_overlay.v1":
        from quant_fund.research.capacity_overlay import (
            capacity_contract_errors,
            capacity_v1_audit_errors,
        )

        errors.extend(capacity_contract_errors(payload))
        errors.extend(_guarded(capacity_v1_audit_errors, "capacity_v1")(payload))
    elif schema == "cross_sectional_rankic.v1":
        from quant_fund.research.cross_sectional import (
            rankic_contract_errors,
            rankic_v1_audit_errors,
        )

        errors.extend(rankic_contract_errors(payload))
        errors.extend(_guarded(rankic_v1_audit_errors, "rankic_v1")(payload))
    elif payload.get("kind") == "ranker_probability_experiment":
        from quant_fund.research.ranker_probability import ranker_prob_contract_errors

        errors.extend(ranker_prob_contract_errors(payload))
    from quant_fund.research.lane_contracts import lane_contract_errors

    errors.extend(lane_contract_errors(payload))
    from quant_fund.research.script_receipts import script_receipt_contract_errors

    errors.extend(script_receipt_contract_errors(schema, payload))
    if payload.get("catalog") == "hedge_lab_analytics":
        from quant_fund.hedge_lab._receipt import lane_receipt_contract_errors

        errors.extend(lane_receipt_contract_errors(payload))
    if payload.get("schema_version") == 1 and isinstance(payload.get("artifacts"), dict):
        from quant_fund.data.ingest import data_manifest_contract_errors

        errors.extend(data_manifest_contract_errors(payload))
    if payload.get("schema") == "hstep_bench.v1" or _looks_like_hstep_eval(payload):
        errors.extend(_lane_checker(*PENDING_V1_CONTRACT)(payload))
    elif payload.get("schema") == "calibration_eval.v1":
        from quant_fund.research.calibration_eval import calibration_contract_errors

        errors.extend(calibration_contract_errors(payload))
    if payload.get("schema") == "cost_calibration.v1":
        from quant_fund.research.cost_calibration import (
            cost_calibration_contract_errors,
        )

        errors.extend(cost_calibration_contract_errors(payload))
    errors.extend(_tape_binding_errors(payload))
    _extend_custody_release_errors(errors, payload)
    return _result(path, payload, convention, errors)


def verify_receipt_payload(
    payload: object, path: Path | str = Path("<memory>")
) -> ReceiptVerification:
    """Verify one parsed receipt object: structure plus hash consistency.

    ``receipt.v2`` envelopes are validated against ``ReceiptV2`` and their
    sealed digest, environment fingerprint, code-map digest, and (for known
    kinds) dataset/params digests are re-derived. Older receipts are checked
    for a consistent ``receipt_sha256`` seal under either repo convention;
    ``fleet_eval.v1`` payloads additionally get their writer's contract, and
    the committed ``scripts/`` lane schemas get internal-consistency
    re-derivation via ``script_receipts``.
    """
    path = Path(path)
    if not isinstance(payload, dict):
        return _result(path, payload, None, ["receipt_not_object"])
    try:
        # The v2 marker is the `schema: "receipt.v2"` tag or the structural
        # envelope fingerprint — `schema_version` is a per-format counter
        # (data-source receipts use 2 without being receipt.v2 envelopes), so
        # it cannot dispatch on its own.
        if payload.get("schema") == RECEIPT_V2_SCHEMA or _looks_like_v2_envelope(payload):
            return _verify_v2(path, payload)
        return _verify_v1(path, payload)
    except (TypeError, ValueError, RecursionError) as exc:
        # Non-finite floats and pathological nesting make the canonical digest
        # raise; a malformed receipt must fail closed, not crash a sweep.
        return _result(path, payload, None, [f"receipt_undigestable:{exc.__class__.__name__}"])


def _reject_json_constant(value: str) -> Any:
    """``NaN``/``Infinity`` are not JSON literals; a receipt containing one is malformed."""
    raise ValueError(f"nonstandard JSON constant in receipt: {value}")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate_json_key:{key}")
        seen.add(key)
        obj[key] = value
    return obj


def verify_receipt_bytes(data: bytes, path: Path | str = Path("<memory>")) -> ReceiptVerification:
    """Parse and verify one immutable receipt byte string.

    Duplicate object keys are rejected: ``{"k": 1, "k": 2}`` parses to ``2``
    in Python but would let a byte string carry two readable claims while only
    one is sealed — the bytes must determine a unique payload.
    """
    receipt_path = Path(path)
    try:
        payload: object = json.loads(
            data,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_no_duplicate_keys,
        )
    except ValueError as exc:
        # ValueError covers JSONDecodeError, the >4300-digit integer limit,
        # and the duplicate_json_key marker raised by the pairs hook.
        if str(exc).startswith("duplicate_json_key:"):
            return _result(receipt_path, {}, None, [str(exc)])
        return _result(receipt_path, {}, None, [f"receipt_unreadable:{exc.__class__.__name__}"])
    except (UnicodeError, RecursionError) as exc:
        # RecursionError covers pathological nesting depth. Corrupt input must
        # degrade to a verdict, never crash the gate.
        return _result(receipt_path, {}, None, [f"receipt_unreadable:{exc.__class__.__name__}"])
    return verify_receipt_payload(payload, receipt_path)


def verify_receipt_file(path: Path | str) -> ReceiptVerification:
    """Read a receipt JSON file once and verify it. Fails closed on unreadable input."""
    file_path = Path(path)
    try:
        data = file_path.read_bytes()
    except OSError as exc:
        return _result(file_path, {}, None, [f"receipt_unreadable:{exc.__class__.__name__}"])
    return verify_receipt_bytes(data, file_path)


def receipt_v2_json_schema() -> dict[str, Any]:
    """Load the published JSON-schema contract shipped beside this module."""
    from importlib.resources import files

    resource = files("quant_fund.research").joinpath(RECEIPT_V2_SCHEMA_FILE)
    return json.loads(resource.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
