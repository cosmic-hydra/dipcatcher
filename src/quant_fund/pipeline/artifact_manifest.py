"""Artifact manifests that make dataset identity impossible to omit silently.

Every training dispatch persists its payload through
:func:`save_training_artifact`, which *requires* a complete
:class:`ArtifactIdentity` — dataset identity (materialized-panel hash, source
manifest hash, row/column counts, time range, label/horizon, data source),
feature identity, label/horizon identity, canonical config hash, git revision
and dirty-worktree hash. The identity is written as an additive ``identity``
block inside the standard ``model_artifact.v1`` manifest, so artifacts remain
loadable through ``models.base`` unchanged.

Fail-closed at every layer: a missing identity parameter is a ``TypeError``, a
partial identity block is an :class:`ArtifactManifestError` on verification,
and :mod:`quant_fund.proof.promotion_receipt` refuses to compose a promotion
receipt without a verified identity. Dataset identity cannot be omitted
silently by accident and cannot be asserted without the materialized sources.

The "dataset" is the gold panel a dispatch consumed (the frame passed to
training/fitting). For unsupervised artifacts the label identity records
:data:`UNSUPERVISED_LABEL` and the declared forecast grid only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Any, Final, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
)

from quant_fund.features.metadata import FEATURE_SET_VERSION
from quant_fund.models.base import save_joblib_artifact
from quant_fund.utils.hashing import (
    canonical_frame_fingerprint,
    canonical_json_bytes,
    hash_bytes,
    hash_file,
)
from quant_fund.utils.reproducibility import git_revision, git_worktree_sha256

MODEL_ARTIFACT_SCHEMA: Final = "model_artifact.v1"
ARTIFACT_IDENTITY_SCHEMA: Final = "artifact_identity.v1"
MANIFEST_SUFFIX: Final = ".manifest.json"
IDENTITY_BLOCK_KEY: Final = "identity"
# Label identity for artifacts fitted without a supervised target.
UNSUPERVISED_LABEL: Final = "unsupervised"
# Materialized inputs of a gold panel; the source manifest binds their bytes.
SOURCE_PANEL_PARTS: Final[tuple[str, ...]] = (
    "gold/features.parquet",
    "gold/labels.parquet",
    "silver/universe.parquet",
)

Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class ArtifactManifestError(RuntimeError):
    """Raised when an artifact manifest is absent, malformed or unbound."""


class DatasetIdentityError(RuntimeError):
    """Raised when dataset identity cannot be derived fail-closed."""


class DatasetIdentity(BaseModel):
    """What data produced the artifact — never optional, never implicit."""

    model_config = ConfigDict(extra="forbid")

    materialized_panel_sha256: Sha256Hex
    source_manifest_sha256: Sha256Hex
    row_count: int = Field(ge=1)
    column_count: int = Field(ge=1)
    time_start: str = Field(min_length=1)
    time_end: str = Field(min_length=1)
    label: str = Field(min_length=1)
    label_horizon_bars: int = Field(ge=1)
    data_source: str = Field(min_length=1)

    @field_validator("time_start", "time_end")
    @classmethod
    def _parseable_timestamp(cls, value: str) -> str:
        from datetime import datetime

        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"not an ISO-8601 timestamp: {value!r}") from exc
        return value


class FeatureIdentity(BaseModel):
    """The exact feature set the artifact was fitted on."""

    model_config = ConfigDict(extra="forbid")

    features: list[str] = Field(min_length=1)
    feature_set_version: str = Field(min_length=1)
    feature_set_sha256: Sha256Hex


class LabelIdentity(BaseModel):
    """Supervised target identity (label column + trailing horizon bars)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    horizon_bars: int = Field(ge=1)


class HorizonIdentity(BaseModel):
    """Declared forecast horizon grid the artifact is aligned to."""

    model_config = ConfigDict(extra="forbid")

    bars: list[int] = Field(min_length=1)
    names: list[str] = Field(min_length=1)


class ArtifactIdentity(BaseModel):
    """Full identity block bound into ``model_artifact.v1`` under ``identity``."""

    model_config = ConfigDict(extra="forbid")

    identity_schema: Literal["artifact_identity.v1"] = ARTIFACT_IDENTITY_SCHEMA
    dataset: DatasetIdentity
    features: FeatureIdentity
    label: LabelIdentity
    horizon: HorizonIdentity
    config_sha256: Sha256Hex
    git_revision: str = Field(min_length=1)
    git_worktree_sha256: str = Field(min_length=1)


def manifest_path(artifact: Path) -> Path:
    """Return the ``model_artifact.v1`` manifest path for an artifact."""
    return Path(artifact).with_name(f"{Path(artifact).name}{MANIFEST_SUFFIX}")


def _source_manifest_hash(data_root: Path) -> str:
    """Hash the materialized source manifest (gold/silver panel inputs).

    The manifest binds the bytes of every panel input under ``data_root`` so a
    later swap of source data is detectable even when the panel columns match.
    """
    digest = hash_bytes(b"dataset-source-manifest.v1")
    missing: list[str] = []
    for relative in SOURCE_PANEL_PARTS:
        part = Path(data_root) / relative
        if not part.is_file():
            missing.append(relative)
            continue
        digest = hash_bytes(digest.encode() + relative.encode() + hash_file(part).encode())
    if missing:
        raise DatasetIdentityError(
            "dataset identity requires materialized panel sources: " + ", ".join(missing)
        )
    return digest


def build_dataset_identity(
    frame: Any,
    *,
    data_root: Path,
    data_source: str,
    label: str,
    label_horizon_bars: int,
) -> DatasetIdentity:
    """Derive fail-closed dataset identity from a materialized panel.

    Raises :class:`DatasetIdentityError` when the panel is empty, the label is
    absent, timestamps are missing, or the materialized sources do not exist.
    """
    if frame is None or int(frame.height) < 1:
        raise DatasetIdentityError("dataset identity requires a non-empty materialized panel")
    columns = [str(name) for name in frame.columns]
    if "event_time" not in columns:
        raise DatasetIdentityError("dataset identity requires an event_time column")
    if label != UNSUPERVISED_LABEL and label not in columns:
        raise DatasetIdentityError(f"label {label!r} is not present in the materialized panel")
    if not str(data_source).strip():
        raise DatasetIdentityError("dataset identity requires a non-empty data_source")
    if int(label_horizon_bars) < 1:
        raise DatasetIdentityError("dataset identity requires label_horizon_bars >= 1")
    times = frame.select("event_time").drop_nulls().sort("event_time")
    if int(times.height) < 1:
        raise DatasetIdentityError("dataset identity requires non-null event_time values")
    return DatasetIdentity(
        materialized_panel_sha256=canonical_frame_fingerprint(frame),
        source_manifest_sha256=_source_manifest_hash(Path(data_root)),
        row_count=int(frame.height),
        column_count=len(columns),
        time_start=str(times["event_time"][0]),
        time_end=str(times["event_time"][-1]),
        label=label,
        label_horizon_bars=int(label_horizon_bars),
        data_source=str(data_source),
    )


def build_artifact_identity(
    *,
    dataset: DatasetIdentity,
    features: list[str],
    label: str,
    label_horizon_bars: int,
    horizon_bars: list[int],
    horizon_names: list[str],
    config: BaseModel,
) -> ArtifactIdentity:
    """Assemble the complete identity block for one training dispatch."""
    feature_list = [str(name) for name in features]
    if not feature_list or any(not name.strip() for name in feature_list):
        raise ArtifactManifestError("artifact identity requires a non-empty feature list")
    if len(set(feature_list)) != len(feature_list):
        raise ArtifactManifestError("artifact identity feature list contains duplicates")
    if not horizon_names or len(horizon_names) != len(horizon_bars):
        raise ArtifactManifestError("artifact identity horizon grid is inconsistent")
    return ArtifactIdentity(
        dataset=dataset,
        features=FeatureIdentity(
            features=feature_list,
            feature_set_version=str(FEATURE_SET_VERSION),
            feature_set_sha256=hash_bytes(
                canonical_json_bytes(
                    {"features": sorted(feature_list), "version": FEATURE_SET_VERSION}
                )
            ),
        ),
        label=LabelIdentity(name=str(label), horizon_bars=int(label_horizon_bars)),
        horizon=HorizonIdentity(
            bars=[int(bar) for bar in horizon_bars],
            names=[str(name) for name in horizon_names],
        ),
        config_sha256=hash_bytes(canonical_json_bytes(config.model_dump(mode="json"))),
        git_revision=git_revision(),
        git_worktree_sha256=git_worktree_sha256(),
    )


def bind_artifact_identity(artifact: Path, identity: ArtifactIdentity) -> dict[str, Any]:
    """Attach the identity block to an existing ``model_artifact.v1`` manifest."""
    target = manifest_path(artifact)
    if not Path(artifact).is_file():
        raise ArtifactManifestError(f"artifact payload does not exist: {artifact}")
    if not target.is_file():
        raise ArtifactManifestError(f"artifact manifest does not exist: {target}")
    try:
        record = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactManifestError(f"artifact manifest is unreadable: {target}") from exc
    if not isinstance(record, dict):
        raise ArtifactManifestError(f"artifact manifest is not a JSON object: {target}")
    if record.get("schema") != MODEL_ARTIFACT_SCHEMA:
        raise ArtifactManifestError(f"artifact manifest schema mismatch: {target}")
    record[IDENTITY_BLOCK_KEY] = identity.model_dump(mode="json")
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return record


def _verify_identity_block(record: dict[str, Any]) -> ArtifactIdentity:
    block = record.get(IDENTITY_BLOCK_KEY)
    if not isinstance(block, dict):
        raise ArtifactManifestError("artifact manifest carries no dataset identity block")
    try:
        identity = ArtifactIdentity.model_validate(block)
    except ValidationError as exc:  # fail closed on any gap
        raise ArtifactManifestError(f"artifact identity block is incomplete: {exc}") from exc
    return identity


def verify_artifact_manifest(artifact: Path) -> ArtifactIdentity:
    """Fail-closed verification of an artifact's bound identity.

    Returns the verified :class:`ArtifactIdentity`; raises
    :class:`ArtifactManifestError` when the manifest or its identity block is
    absent, malformed, partial, or does not bind the current payload bytes.
    """
    artifact = Path(artifact)
    target = manifest_path(artifact)
    if not artifact.is_file():
        raise ArtifactManifestError(f"artifact payload does not exist: {artifact}")
    if not target.is_file():
        raise ArtifactManifestError(f"artifact manifest does not exist: {target}")
    try:
        record = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactManifestError(f"artifact manifest is unreadable: {target}") from exc
    if not isinstance(record, dict):
        raise ArtifactManifestError(f"artifact manifest is not a JSON object: {target}")
    if record.get("schema") != MODEL_ARTIFACT_SCHEMA:
        raise ArtifactManifestError(f"artifact manifest schema mismatch: {target}")
    if record.get("sha256") != hash_file(artifact):
        raise ArtifactManifestError("artifact manifest does not bind the current payload bytes")
    return _verify_identity_block(record)


def save_training_artifact(payload: Any, path: Path, *, identity: ArtifactIdentity) -> Path:
    """Persist a training payload with its identity bound into the manifest.

    ``identity`` is required and keyword-only: a training dispatch that does
    not declare what data produced the artifact cannot save it at all.
    """
    if not isinstance(identity, ArtifactIdentity):
        raise TypeError("save_training_artifact requires an ArtifactIdentity")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    save_joblib_artifact(payload, target)
    try:
        bind_artifact_identity(target, identity)
    except ArtifactManifestError:
        # Fail closed: never leave a payload behind with no identity binding.
        target.unlink(missing_ok=True)
        manifest_path(target).unlink(missing_ok=True)
        raise
    return target


def identity_for_training(
    frame: Any,
    *,
    config: Any,
    label: str,
    features: list[str],
    label_horizon_bars: int,
) -> ArtifactIdentity:
    """Build the complete identity block for one training dispatch.

    ``config`` is an ``AppConfig``. ``label`` is the supervised target column,
    or :data:`UNSUPERVISED_LABEL` for artifacts fitted without one (the
    horizon then records the declared forecast grid only).
    """
    dataset = build_dataset_identity(
        frame,
        data_root=Path(config.data.root),
        data_source=str(config.data.source),
        label=label,
        label_horizon_bars=label_horizon_bars,
    )
    return build_artifact_identity(
        dataset=dataset,
        features=features,
        label=label,
        label_horizon_bars=label_horizon_bars,
        horizon_bars=[int(bar) for bar in config.horizons.bars],
        horizon_names=[str(name) for name in config.horizons.names],
        config=config,
    )


def identity_from_artifact(artifact: Path) -> ArtifactIdentity:
    """Return the verified identity of a persisted artifact (alias)."""
    return verify_artifact_manifest(artifact)


def dataset_identity_from_artifact(artifact: Path) -> DatasetIdentity:
    """Return only the dataset identity of a persisted artifact."""
    return verify_artifact_manifest(artifact).dataset


__all__ = [
    "ARTIFACT_IDENTITY_SCHEMA",
    "ArtifactIdentity",
    "ArtifactManifestError",
    "DatasetIdentity",
    "DatasetIdentityError",
    "FeatureIdentity",
    "HorizonIdentity",
    "IDENTITY_BLOCK_KEY",
    "LabelIdentity",
    "MANIFEST_SUFFIX",
    "MODEL_ARTIFACT_SCHEMA",
    "UNSUPERVISED_LABEL",
    "bind_artifact_identity",
    "build_artifact_identity",
    "build_dataset_identity",
    "dataset_identity_from_artifact",
    "identity_for_training",
    "identity_from_artifact",
    "manifest_path",
    "save_training_artifact",
    "verify_artifact_manifest",
]
