"""Tape manifests: a committed registry pinning the bytes behind receipts.

Every real-tape receipt attests its input only by self-declared digest
(``inputs_sha256``, ``dataset_sha256``) — nothing committed proves which
bytes those digests name, so a lane could seal any number and still verify.
A ``tape_manifest.v1`` file under ``data/manifests/`` closes that
self-attestation gap: it seals the tape files' own sha256/byte counts, the
frame's row/name/window profile, and the canonical CSV digest the lanes
hash (``hash_bytes(frame.write_csv().encode("utf-8"))``, the
``coverage_watch`` inputs convention), plus an optional link to the
``bar_promotion.v1`` receipt that published the tape.

``verify-receipt`` consults the same registry: a receipt whose body declares
a non-synthetic ``data_label`` and binds a ``tape_manifest_sha256`` or
``dataset_sha256`` must resolve to a committed manifest's seal or tape
digests, else it fails closed with ``tape_manifest_unknown``. Tapes stay
gitignored — the manifest attests bytes, never ships them; a missing or
tampered tape fails ``tape-verify``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_fund.utils.hashing import SHA256_HEX_LENGTH, canonical_json_bytes, hash_bytes

TAPE_MANIFEST_SCHEMA = "tape_manifest.v1"
MANIFESTS_DIR = Path("data") / "manifests"

#: Identity column candidates, in preference order — ``n_names`` counts the
#: first one present.
_IDENTITY_COLUMNS = ("security_id", "symbol", "name", "ticker")
#: Timestamp column candidates for the manifest's ``window``.
_TIME_COLUMNS = ("event_time", "timestamp", "time", "date")
#: Receipt body keys that declare a binding to a pinned tape.
TAPE_BINDING_KEYS = ("tape_manifest_sha256", "dataset_sha256")
#: Labels exempt from the tape-binding ratchet — synthetic/corpus lanes
#: generate their inputs, so there is no external tape to pin.
SYNTHETIC_LABELS = frozenset({"SYNTHETIC", "CORPUS"})


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def frame_csv_sha256(frame: pl.DataFrame) -> str:
    """Canonical tape digest — the same bytes a lane hashes as ``inputs_sha256``."""
    return hash_bytes(frame.write_csv().encode("utf-8"))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _repo_relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        raise ValueError(f"tape path is not repo-relative to {root}: {path}") from None


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _load_frames(
    paths: Sequence[Path], frame_loader: Callable[[Path], pl.DataFrame]
) -> pl.DataFrame:
    frames = [frame_loader(path) for path in paths]
    if len(frames) == 1:
        return frames[0]
    return pl.concat(frames, how="vertical")


def _profile(frame: pl.DataFrame) -> dict[str, Any]:
    names_column = next((column for column in _IDENTITY_COLUMNS if column in frame.columns), None)
    time_column = next((column for column in _TIME_COLUMNS if column in frame.columns), None)
    window = None
    if time_column is not None:
        window = {
            "start": str(frame.get_column(time_column).min()),
            "end": str(frame.get_column(time_column).max()),
        }
    return {
        "n_rows": frame.height,
        "n_names": None if names_column is None else frame.get_column(names_column).n_unique(),
        "window": window,
    }


def pin_tape(
    source_label: str,
    tape_path: str | Path | Sequence[str | Path],
    *,
    frame_loader: Callable[[Path], pl.DataFrame] | None = None,
    out_dir: Path = Path("data") / "manifests",
    promotion_receipt: str | Path | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Seal a ``tape_manifest.v1`` for ``tape_path`` into ``out_dir``.

    ``tape_path`` is one parquet or an ordered sequence (concatenated for the
    CSV digest). ``frame_loader`` overrides the canonical
    ``pl.read_parquet`` read for callers whose tape is not parquet; the
    pinned ``frame_csv_sha256`` still follows the lane convention.
    """
    label = str(source_label).strip()
    if not label or label in {".", ".."} or "/" in label or "\\" in label:
        raise ValueError("source_label must be a path-safe non-empty label")
    paths = (
        [Path(tape_path)]
        if isinstance(tape_path, (str, Path))
        else [Path(item) for item in tape_path]
    )
    if not paths:
        raise ValueError("pin_tape requires at least one tape file")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"tape file not found: {path}")
    root = Path.cwd() if root is None else Path(root)
    loader = pl.read_parquet if frame_loader is None else frame_loader
    frame = _load_frames(paths, loader)
    if frame.is_empty():
        raise ValueError("tape frame has no rows")

    tape_files = [
        {
            "path": _repo_relative(path, root),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "n_bytes": path.stat().st_size,
        }
        for path in paths
    ]
    promotion_sha: str | None = None
    if promotion_receipt is not None:
        receipt_path = Path(promotion_receipt)
        if not receipt_path.is_file():
            raise FileNotFoundError(f"promotion receipt not found: {receipt_path}")
        promotion_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()

    body: dict[str, Any] = {
        "schema": TAPE_MANIFEST_SCHEMA,
        "source_label": label,
        "tape_files": tape_files,
        "frame_csv_sha256": frame_csv_sha256(frame),
        "collected_at": datetime.now(UTC).isoformat(),
        "promotion_receipt_sha256": promotion_sha,
        **_profile(frame),
    }
    manifest = {**body, "receipt_sha256": hash_bytes(canonical_json_bytes(body))}
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{label}.json"
    _write_atomic(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return {"path": manifest_path, "manifest": manifest}


def load_registry(root: Path | str) -> dict[str, dict[str, Any]]:
    """Map every digest a committed manifest attests to that manifest.

    Scans ``<root>/data/manifests/*.json``; only seal-valid
    ``tape_manifest.v1`` files are indexed — a manifest that fails its own
    seal attests nothing. Indexed digests: the manifest's
    ``receipt_sha256``, each ``tape_files[].sha256``, and
    ``frame_csv_sha256``.
    """
    registry: dict[str, dict[str, Any]] = {}
    manifests_dir = Path(root) / MANIFESTS_DIR
    if not manifests_dir.is_dir():
        return registry
    for path in sorted(manifests_dir.glob("*.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(manifest, dict) or manifest.get("schema") != TAPE_MANIFEST_SCHEMA:
            continue
        seal = manifest.get("receipt_sha256")
        if not _is_sha256(seal):
            continue
        body = {key: value for key, value in manifest.items() if key != "receipt_sha256"}
        if hash_bytes(canonical_json_bytes(body)) != seal:
            continue
        registry[str(seal)] = manifest
        tape_files = manifest.get("tape_files")
        if isinstance(tape_files, list):
            for entry in tape_files:
                if isinstance(entry, Mapping) and _is_sha256(entry.get("sha256")):
                    registry[str(entry["sha256"])] = manifest
        csv_digest = manifest.get("frame_csv_sha256")
        if _is_sha256(csv_digest):
            registry[str(csv_digest)] = manifest
        # Evaluated-stream digests: a lane's ``dataset_sha256`` binds the
        # exact stream it scored — manifests attest which digests a tape
        # legitimately produces.
        for entry in manifest.get("eval_streams") or []:
            if isinstance(entry, Mapping) and _is_sha256(entry.get("dataset_sha256")):
                registry[str(entry["dataset_sha256"])] = manifest
    return registry


def lookup_digest(registry: Mapping[str, dict[str, Any]], digest: object) -> dict[str, Any] | None:
    """Resolve a claimed tape digest to its manifest; ``None`` when unknown."""
    if not isinstance(digest, str):
        return None
    return registry.get(digest)


def verify_manifest(
    path: Path | str,
    root: Path | str,
    *,
    frame_loader: Callable[[Path], pl.DataFrame] | None = None,
) -> list[str]:
    """Re-derive a pinned manifest from tape bytes; ``[]`` means intact.

    Fails closed on every drift: unreadable file, wrong schema, stale seal,
    missing/tampered tape bytes, or a profile/CSV digest that no longer
    re-derives from the pinned tapes.
    """
    manifest_path = Path(path)
    try:
        manifest: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"manifest_unreadable:{exc.__class__.__name__}"]
    if not isinstance(manifest, dict):
        return ["manifest_not_object"]
    if manifest.get("schema") != TAPE_MANIFEST_SCHEMA:
        return ["schema_mismatch"]
    errors: list[str] = []
    seal = manifest.get("receipt_sha256")
    if not _is_sha256(seal):
        errors.append("receipt_sha256_missing_or_invalid")
    else:
        body = {key: value for key, value in manifest.items() if key != "receipt_sha256"}
        if hash_bytes(canonical_json_bytes(body)) != seal:
            errors.append("receipt_sha256_mismatch")

    tape_files = manifest.get("tape_files")
    if not isinstance(tape_files, list) or not tape_files:
        errors.append("tape_files_missing")
        return errors
    label = manifest.get("source_label")
    if not isinstance(label, str) or not label.strip():
        errors.append("source_label_invalid")
    if not _is_sha256(manifest.get("frame_csv_sha256")):
        errors.append("frame_csv_sha256_invalid")

    root_path = Path(root)
    tape_paths: list[Path] = []
    for index, entry in enumerate(tape_files):
        if not isinstance(entry, Mapping):
            errors.append(f"tape_files[{index}]_not_object")
            continue
        rel = entry.get("path")
        if not isinstance(rel, str) or not rel or rel.startswith("/") or ".." in rel.split("/"):
            errors.append(f"tape_files[{index}]_path_invalid")
            continue
        tape_path = root_path / rel
        tape_paths.append(tape_path)
        if not tape_path.is_file():
            errors.append(f"tape_missing:{rel}")
            continue
        raw = tape_path.read_bytes()
        if not _is_sha256(entry.get("sha256")) or hashlib.sha256(raw).hexdigest() != entry.get(
            "sha256"
        ):
            errors.append(f"tape_sha256_mismatch:{rel}")
        if entry.get("n_bytes") != len(raw):
            errors.append(f"tape_n_bytes_mismatch:{rel}")
    if errors or not tape_paths:
        return errors

    loader = pl.read_parquet if frame_loader is None else frame_loader
    try:
        frame = _load_frames(tape_paths, loader)
    except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
        return [*errors, f"frame_load_failed:{exc.__class__.__name__}"]
    if frame_csv_sha256(frame) != manifest.get("frame_csv_sha256"):
        errors.append("frame_csv_sha256_mismatch")
    profile = _profile(frame)
    for key in ("n_rows", "n_names", "window"):
        if manifest.get(key) != profile[key]:
            errors.append(f"{key}_mismatch")
    return errors


def _rederived_dataset_sha256(body: Mapping[str, Any]) -> str | None:
    """Composite ``dataset_sha256`` the basis lanes re-derive from input digests.

    ``basis_carry`` / ``crossvenue_basis`` hash the embedded series digests.
    That field is an internal consistency check (the lane contract recomputes
    it and flags ``dataset_sha256_mismatch``). It is not a tape-file,
    frame-CSV, or manifest seal, so the tape registry cannot attest it unless
    someone later pins the underlying venue tapes. Treating the composite as
    a tape binding fails closed on receipts whose own contract already binds
    the digest — a false ``tape_manifest_unknown``.
    """
    schema = body.get("schema") or body.get("kind")
    inputs = body.get("inputs")
    if not isinstance(inputs, Mapping):
        return None
    try:
        if schema in {"basis_carry.v1", "basis_carry"}:
            spot = inputs.get("spot")
            contracts = inputs.get("contracts")
            if not isinstance(spot, Mapping) or not isinstance(contracts, Mapping):
                return None
            material: Mapping[str, Any] = {
                "spot": spot.get("sha256"),
                "contracts": {
                    str(sym): meta.get("sha256")
                    for sym, meta in contracts.items()
                    if isinstance(meta, Mapping)
                },
            }
        elif schema in {"crossvenue_basis.v1", "crossvenue_basis"}:
            legs = inputs.get("legs")
            if not isinstance(legs, Mapping):
                return None
            material = {
                venue: {
                    field: (meta.get(field) or {}).get("sha256")
                    if meta.get(field) is not None
                    else None
                    for field in ("spot", "mark", "funding")
                }
                for venue, meta in legs.items()
                if isinstance(meta, Mapping)
            }
        else:
            return None
        return hash_bytes(canonical_json_bytes(dict(material)))
    except (TypeError, ValueError, AttributeError):
        return None


def tape_binding_errors(body: Mapping[str, Any], *, root: Path | str | None = None) -> list[str]:
    """A non-synthetic receipt binding a tape digest must resolve to a manifest.

    Ratchet-in: bodies that declare no ``tape_manifest_sha256`` /
    ``dataset_sha256`` stay admissible, and SYNTHETIC/CORPUS lanes are
    exempt. Any declared binding that no committed ``tape_manifest.v1``
    attests fails closed as ``tape_manifest_unknown``.

    A ``dataset_sha256`` that recomputes as the lane's composite of embedded
    series digests is not a tape pin (see ``_rederived_dataset_sha256``).
    ``tape_manifest_sha256`` is always a tape pin.
    """
    label = body.get("data_label")
    # A missing or blank label is an honesty-quarantine finding, not a
    # non-synthetic tape claim. Enforcing the registry here turns that
    # quarantine into a seal rejection and hides the honesty verdict.
    if not isinstance(label, str) or not label.strip():
        return []
    if label.upper() in SYNTHETIC_LABELS:
        return []
    # Corpus lanes evaluate the committed receipt corpus itself — pinned by
    # the corpus epoch chains (verify-repo), not by a tape manifest.
    if str(body.get("kind")) in {"corpus_inference.v1", "suite_health"}:
        return []
    composite = _rederived_dataset_sha256(body)
    bound: list[str] = []
    for key in TAPE_BINDING_KEYS:
        value = body.get(key)
        if not isinstance(value, str):
            continue
        if key == "dataset_sha256" and composite is not None and value == composite:
            continue
        bound.append(value)
    if not bound:
        return []
    registry = load_registry(_repo_root() if root is None else root)
    if any(lookup_digest(registry, digest) is None for digest in bound):
        return ["tape_manifest_unknown"]
    return []
