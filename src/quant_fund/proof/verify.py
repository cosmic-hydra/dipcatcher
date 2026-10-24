"""Proof bundle verifier (DESIGN.md §5.5) — fixes A1 F2.

Fail-closed, ordered checks; every failed check appends a machine-readable
reason and flips ``ok`` to False. The verifier never trusts the bundle's
own metric claims: headline metrics are independently recomputed from the
trade-log bytes via ``metrics/returns.py`` and compared against
``metrics_recompute`` at rtol=1e-9 / atol=1e-12 (§8.2).
"""

from __future__ import annotations

import io
import json
import math
import re
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ValidationError

from quant_fund.proof.bundle import (
    METRIC_KEYS,
    chain_head,
    compute_bundle_id,
    env_fingerprint,
    load_chain,
    recompute_headline_metrics,
    sidecar_paths,
    signing_payload_bytes,
)
from quant_fund.proof.recorder import read_leaf_hash
from quant_fund.proof.sign import HmacSha256Signer
from quant_fund.proofcore.contracts import (
    GENESIS_HASH,
    SCHEMA_VERSION,
    ProofBundleV1,
    SignatureUnavailableError,
    canonical_json_bytes,
    merkle_root_hex,
    sha256_hex_bytes,
)

__all__ = ["RECOMPUTE_RTOL", "RECOMPUTE_ATOL", "VerificationResult", "verify_bundle"]

#: §8.2 — same machine, deterministic engine; anything looser hides tampering.
RECOMPUTE_RTOL = 1e-9
RECOMPUTE_ATOL = 1e-12


class VerificationResult(BaseModel):
    """Machine-readable verification outcome (DESIGN.md §5.5)."""

    ok: bool
    reasons: list[str]
    env_mismatch: bool
    metrics_match: dict[str, bool]


def _metric_close(a: float, b: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isnan(a) or math.isnan(b):
        return False
    return math.isclose(a, b, rel_tol=RECOMPUTE_RTOL, abs_tol=RECOMPUTE_ATOL)


def _infer_bundle_dir(bundle_path: Path) -> Path:
    parent = bundle_path.parent
    if parent.name == "bundles":
        return parent.parent
    return parent


def _check_chain(payload: dict[str, Any], bundle_dir: Path, reasons: list[str]) -> None:
    """Check 3: prev_bundle_hash links into the bundles.jsonl head."""
    try:
        chain = load_chain(bundle_dir)
    except Exception as exc:  # untrusted chain input must yield a failed verdict
        reasons.append(f"chain:unreadable:{exc.__class__.__name__}")
        return
    for index, prior in enumerate(chain):
        expected_prev = chain[index - 1].bundle_id if index else GENESIS_HASH
        if prior.prev_bundle_hash != expected_prev:
            reasons.append(f"chain:link_broken_at_index:{index}")
        if compute_bundle_id(prior.model_dump(mode="json")) != prior.bundle_id:
            reasons.append(f"chain:self_hash_mismatch_at_index:{index}")
    head = chain[-1].bundle_id if chain else GENESIS_HASH
    bundle_id = str(payload.get("bundle_id", ""))
    if bundle_id in {entry.bundle_id for entry in chain}:
        position = next(i for i, entry in enumerate(chain) if entry.bundle_id == bundle_id)
        expected_prev = chain[position - 1].bundle_id if position else GENESIS_HASH
        if payload.get("prev_bundle_hash") != expected_prev:
            reasons.append("chain:bundle_prev_mismatch")
        if canonical_json_bytes(chain[position].model_dump(mode="json")) != canonical_json_bytes(
            payload
        ):
            reasons.append("chain:bundle_file_mismatch")
    else:
        reasons.append("chain:bundle_missing")
        if payload.get("prev_bundle_hash") != head:
            reasons.append("chain:prev_not_head")


def _check_signature(payload: dict[str, Any], strict: bool, reasons: list[str]) -> None:
    """Check 4: HMAC verify; strict mode rejects unsigned bundles."""
    signature = payload.get("signature") or {}
    scheme = signature.get("scheme")
    if scheme == "none":
        if strict:
            reasons.append("signature:unsigned_strict")
        return
    if scheme != "hmac-sha256":
        reasons.append(f"signature:unknown_scheme:{scheme}")
        return
    try:
        signer = HmacSha256Signer.from_env()
    except SignatureUnavailableError:
        reasons.append("signature:key_unavailable")
        return
    if signature.get("key_id") != signer.key_id:
        reasons.append("signature:key_id_mismatch")
        return
    if not signer.verify(signing_payload_bytes(payload), str(signature.get("value", ""))):
        reasons.append("signature:hmac_mismatch")


def _check_data_manifest(bundle: ProofBundleV1, reasons: list[str]) -> None:
    """Check 5: recompute per-read leaf hashes + Merkle root."""
    manifest = bundle.data_manifest
    try:
        recomputed = merkle_root_hex([read_leaf_hash(read) for read in manifest.reads])
    except Exception as exc:  # malformed leaf -> failed check, never a raise
        reasons.append(f"data_manifest:leaf_error:{exc.__class__.__name__}")
        return
    if recomputed != manifest.merkle_root:
        reasons.append("data_manifest:merkle_root_mismatch")


def _check_sidecars(
    bundle: ProofBundleV1, bundle_dir: Path, reasons: list[str]
) -> dict[str, bytes]:
    """Check 6: re-hash signal_log/trade_log/metrics/config sidecar files.

    ADVERSARIAL R2 §2-CFG: the config sidecar is hash-checked too — a
    tampered recorded config previously passed plain ``verify_bundle`` and
    was only caught by replay. Fail closed with
    ``sidecar:config_sha256_mismatch``.
    """
    sidecars = sidecar_paths(bundle_dir, bundle.bundle_id)
    expected = {
        "signal_log": bundle.signal_log_sha256,
        "trade_log": bundle.trade_log_sha256,
        "metrics": bundle.metrics_sha256,
        "config": bundle.config_sha256,
    }
    contents: dict[str, bytes] = {}
    for kind, path in sidecars.items():
        if path.is_symlink():
            reasons.append(f"sidecar:{kind}:unsafe_symlink")
            continue
        if not path.exists():
            reasons.append(f"sidecar:{kind}:missing")
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            # Windows read_bytes on a directory raises PermissionError.
            # The contract name is the POSIX error a directory read produces.
            label = {True: "IsADirectoryError"}.get(path.is_dir(), exc.__class__.__name__)
            reasons.append(f"sidecar:{kind}:unreadable:{label}")
            continue
        if sha256_hex_bytes(data) != expected[kind]:
            reasons.append(f"sidecar:{kind}_sha256_mismatch")
        contents[kind] = data
    _check_committed_sidecars(bundle, bundle_dir, contents.get("config"), reasons)
    return contents


_COMMITTED_KIND_RE = re.compile(r"[a-z0-9_]+\Z")


def _check_committed_sidecars(
    bundle: ProofBundleV1,
    bundle_dir: Path,
    config_bytes: bytes | None,
    reasons: list[str],
) -> None:
    """Check 6b (wave 2, additive): hash-check sidecars committed in config.

    The wave-2 runner commits ``config["sidecars"]["<kind>_sha256"]`` for the
    trace/env/seeds sidecars (WAVE2.md §2.5); the commitment chain is
    bundle -> config (check 6) -> these files, mirroring the wave-1 config
    sidecar pattern (``sidecar:<kind>_sha256_mismatch``). Wave-1 bundles carry
    no ``sidecars`` map and skip this check entirely. Commitment keys are
    restricted to ``[a-z0-9_]`` so a hostile bundle cannot turn a commitment
    into a path traversal outside ``bundle_dir``.
    """
    if config_bytes is None:
        return
    try:
        config_doc: Any = json.loads(config_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError):
        reasons.append("sidecar:config_invalid_json")
        return
    if not isinstance(config_doc, dict):
        return
    commitments = config_doc.get("sidecars")
    if not isinstance(commitments, dict):
        return
    for key in sorted(commitments):
        kind = key[: -len("_sha256")] if key.endswith("_sha256") else key
        expected = commitments[key]
        if not _COMMITTED_KIND_RE.fullmatch(kind) or not isinstance(expected, str):
            reasons.append(f"sidecar:{kind}:invalid_commitment")
            continue
        path = bundle_dir / f"{bundle.bundle_id}.{kind}.json"
        if path.is_symlink():
            reasons.append(f"sidecar:{kind}:unsafe_symlink")
            continue
        if not path.exists():
            reasons.append(f"sidecar:{kind}:missing")
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            label = {True: "IsADirectoryError"}.get(path.is_dir(), exc.__class__.__name__)
            reasons.append(f"sidecar:{kind}:unreadable:{label}")
            continue
        if sha256_hex_bytes(data) != expected:
            reasons.append(f"sidecar:{kind}_sha256_mismatch")


def _check_metrics_recompute(
    bundle: ProofBundleV1,
    trade_log_bytes: bytes | None,
    reasons: list[str],
) -> dict[str, bool]:
    """Check 7 (A1 F2): RECOMPUTE metrics from trade log bytes and compare."""
    metrics_match: dict[str, bool] = {}
    recorded = bundle.metrics_recompute
    if trade_log_bytes is None:
        for key in METRIC_KEYS:
            metrics_match[key] = False
        reasons.append("metrics_recompute:no_trade_log")
        return metrics_match
    try:
        trade_log = pl.read_parquet(io.BytesIO(trade_log_bytes))
        recomputed = recompute_headline_metrics(trade_log)
    except Exception as exc:
        for key in METRIC_KEYS:
            metrics_match[key] = False
        reasons.append(f"metrics_recompute:recompute_error:{exc.__class__.__name__}")
        return metrics_match
    for key in METRIC_KEYS:
        if key not in recorded:
            metrics_match[key] = False
            reasons.append(f"metrics_recompute:missing_key:{key}")
            continue
        match = _metric_close(float(recorded[key]), float(recomputed[key]))
        metrics_match[key] = match
        if not match:
            reasons.append(f"metrics_recompute:{key}")
    extra = sorted(set(recorded) - set(METRIC_KEYS))
    if extra:
        reasons.append(f"metrics_recompute:unexpected_keys:{','.join(extra)}")
    return metrics_match


def verify_bundle(
    bundle_path: Path,
    *,
    bundle_dir: Path | None = None,
    replay: bool = False,
    strict_signature: bool = True,
    trusted_head: str | None = None,
    pit_root: Path | None = None,
) -> VerificationResult:
    """Verify one proof bundle with fail-closed ordered checks (§5.5).

    1. schema validation (pydantic) + schema_version == 'proofcore/1'
    2. bundle_id self-hash recompute
    3. chain: bundle is present in bundles.jsonl with valid self-hashes and links
    4. signature: HMAC verify (strict) or report scheme='none' as reason
    5. data_manifest: recompute per-read leaf hashes + Merkle root
    6. re-hash signal_log/trade_log/metrics/config sidecar files
    6b. re-hash wave-2 sidecars committed in config["sidecars"] (trace/env/seeds)
    7. RECOMPUTE metrics from trade log bytes and compare (rtol 1e-9, atol 1e-12)
    8. env_fingerprint comparison -> env_mismatch warning (A3 F5.2, non-fatal)
    9. optional replay: bit-exact re-execution via proof.replay (wave 2)
    """
    reasons: list[str] = []
    env_mismatch = False
    metrics_match: dict[str, bool] = {}
    bundle_path = Path(bundle_path)
    if bundle_dir is None:
        bundle_dir = _infer_bundle_dir(bundle_path)
    bundle_dir = Path(bundle_dir)

    # Check 1 — schema validation.
    try:
        raw: dict[str, Any] = json.loads(bundle_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return VerificationResult(
            ok=False,
            reasons=[f"schema:unreadable:{exc.__class__.__name__}"],
            env_mismatch=False,
            metrics_match={},
        )
    if not isinstance(raw, dict):
        raw = {}
    if raw.get("schema_version") != SCHEMA_VERSION:
        reasons.append(f"schema:version:{raw.get('schema_version')!r}")
    try:
        bundle = ProofBundleV1.model_validate(raw)
    except ValidationError as exc:
        return VerificationResult(
            ok=False,
            reasons=[*reasons, f"schema:invalid:{exc.error_count()}errors"],
            env_mismatch=False,
            metrics_match={},
        )

    # Check 2 — self-hash.
    if compute_bundle_id(raw) != bundle.bundle_id:
        reasons.append("bundle_id:self_hash_mismatch")

    # Check 3 — chain linkage. A missing chain cannot establish membership.
    chain_path = bundle_dir / "bundles.jsonl"
    if chain_path.exists():
        _check_chain(raw, bundle_dir, reasons)
    else:
        reasons.append("chain:missing")

    # §5.6 — trusted head: the stored chain must terminate at trusted_head.
    if trusted_head is not None and chain_path.exists():
        try:
            if chain_head(bundle_dir) != trusted_head:
                reasons.append("chain:trusted_head_mismatch")
        except Exception as exc:  # untrusted chain input must yield a failed verdict
            reasons.append(f"chain:trusted_head_unreadable:{exc.__class__.__name__}")

    # Check 4 — signature.
    _check_signature(raw, strict_signature, reasons)

    # Check 5 — data manifest Merkle recompute.
    _check_data_manifest(bundle, reasons)

    # Check 6 — sidecar hashes.
    sidecar_bytes = _check_sidecars(bundle, bundle_dir, reasons)

    # Check 7 — metric recomputation from trade-log bytes (A1 F2).
    metrics_match = _check_metrics_recompute(bundle, sidecar_bytes.get("trade_log"), reasons)

    # Check 8 — env fingerprint (warning only, A3 F5.2).
    current_env = env_fingerprint().model_dump(mode="json")
    if current_env != bundle.env.model_dump(mode="json"):
        env_mismatch = True

    # Check 9 — optional deterministic replay.
    if replay:
        from quant_fund.proof.replay import replay_bundle

        replay_ok, replay_detail = replay_bundle(
            bundle_path,
            bundle_dir=bundle_dir,
            pit_root=pit_root,
        )
        if not replay_ok:
            reasons.append(f"replay:{replay_detail}")

    return VerificationResult(
        ok=not reasons,
        reasons=reasons,
        env_mismatch=env_mismatch,
        metrics_match=metrics_match,
    )
