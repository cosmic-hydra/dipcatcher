"""Fail-closed composition of immutable promotion receipts (``promotion_receipt.v1``).

A promotion receipt is one composed, immutable record binding:

- artifact identity — payload hash, manifest hash, artifact class, feature
  identity, label/horizon identity and full **dataset identity** (required;
  missing or unverified dataset identity blocks promotion),
- the ``evidence_report.v1`` bytes by hash (with its sidecar),
- the ``promotion.v1`` decision, its input metrics and every gate result,
- the approving identity.

Composition is fail-closed: every precondition raises
:class:`PromotionCompositionError` — approved decision only
(``promotion_is_approved``), non-synthetic evidence only, evidence report
complete and hash-bound, all required gate results present and passing, an
honest named approver with a timestamped decision, and a verified artifact
manifest. The receipt is written exactly once (exclusive create) with a
``.sha256`` sidecar and is never rewritten.

Verification is an independent implementation in
:mod:`quant_fund.research.verify` (``verify-research``): composition-side
gating and verification-side checking share no code, so a bug in one cannot
vouch for the other.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from quant_fund.utils.hashing import hash_file


class _DatasetView(Protocol):
    """Attributes promotion reads off a verified dataset identity."""

    materialized_panel_sha256: str
    source_manifest_sha256: str
    row_count: int
    column_count: int
    data_source: str
    label: str


class _IdentityView(Protocol):
    """Attributes promotion reads off a verified artifact identity.

    The concrete type lives in ``pipeline.artifact_manifest``. Proof reaches
    it only inside the functions that verify a manifest, so this package does
    not import the pipeline at module scope.
    """

    dataset: _DatasetView

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]: ...


PROMOTION_RECEIPT_SCHEMA = "promotion_receipt.v1"
PROMOTION_RECEIPT_KIND = "promotion_receipt"
EVIDENCE_REPORT_SCHEMA = "evidence_report.v1"
REQUIRED_PROMOTION_GATES: tuple[str, ...] = (
    "artifact_manifest",
    "dataset_identity",
    "evidence_report",
    "research_receipt",
    "leakage",
)
DISHONEST_APPROVER_NAMES: frozenset[str] = frozenset(
    {"", "unknown", "anonymous", "none", "null", "n/a", "na", "tbd", "unspecified", "someone"}
)
# Stage-aware completeness (scoping, never deleting): the evidence report is
# written at TRAINING time, before this receipt exists. A report whose only
# warning is the stage-expected ``promotion_receipt_missing`` is resolved BY
# this receipt — the receipt records that resolution. Every other warning
# (notably ``health_report_missing``) remains blocking, and the report's own
# ``status: complete`` rule is untouched (nothing is made easier to reach).
STAGE_EXPECTED_REPORT_WARNINGS: frozenset[str] = frozenset({"promotion_receipt_missing"})


class PromotionCompositionError(RuntimeError):
    """Raised whenever a promotion receipt cannot be composed fail-closed."""


class Approver(BaseModel):
    """The approving identity — a named human/service, never anonymous."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    role: str = Field(min_length=1)
    decided_at: str = Field(min_length=1)

    @field_validator("name", "role")
    @classmethod
    def _honest_identity(cls, value: str) -> str:
        cleaned = value.strip()
        if cleaned.lower() in DISHONEST_APPROVER_NAMES:
            raise ValueError(f"approver identity is missing or dishonest: {value!r}")
        return cleaned

    @field_validator("decided_at")
    @classmethod
    def _timestamped(cls, value: str) -> str:
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"approver decided_at is not ISO-8601: {value!r}") from exc
        return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PromotionCompositionError(message)


def _load_evidence_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromotionCompositionError(f"evidence report is unreadable: {path}") from exc
    if not isinstance(report, dict):
        raise PromotionCompositionError(f"evidence report is not a JSON object: {path}")
    return report


def _require_dataset_identity(identity: _IdentityView) -> None:
    dataset = identity.dataset
    _require(
        bool(dataset.materialized_panel_sha256) and bool(dataset.source_manifest_sha256),
        "dataset identity is missing; promotion is blocked",
    )
    _require(
        int(dataset.row_count) >= 1 and int(dataset.column_count) >= 1,
        "dataset identity is empty; promotion is blocked",
    )
    _require(
        bool(str(dataset.data_source).strip()) and bool(str(dataset.label).strip()),
        "dataset identity is unlabeled; promotion is blocked",
    )


def _gate_results_checked(gates: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    _require(isinstance(gates, Mapping) and bool(gates), "gate results are missing")
    checked = {str(name): dict(row) for name, row in gates.items() if isinstance(row, Mapping)}
    _require(len(checked) == len(gates), "gate results must all be objects")
    for name in REQUIRED_PROMOTION_GATES:
        _require(name in checked, f"gate result is missing: {name}")
    for name, row in checked.items():
        _require(row.get("status") == "pass", f"gate result is not passing: {name}")
    return checked


def _artifact_identity_checked(
    artifact: Path,
) -> tuple[_IdentityView, Path, Path, str, str, object]:
    """Step 1: verified artifact manifest with mandated dataset identity."""
    from quant_fund.pipeline.artifact_manifest import (
        ArtifactManifestError,
        manifest_path,
        verify_artifact_manifest,
    )

    artifact_path = Path(artifact)
    try:
        identity = verify_artifact_manifest(artifact_path)
    except ArtifactManifestError as exc:
        raise PromotionCompositionError(
            "artifact manifest is absent or unverified (dataset identity "
            f"cannot be established); promotion is blocked: {exc}"
        ) from exc
    # ArtifactIdentity satisfies this view at runtime. Its pydantic
    # model_dump signature is wider than the Protocol, so the cast is the
    # type boundary; the checks below still run on the live object.
    checked = cast(_IdentityView, identity)
    _require_dataset_identity(checked)
    artifact_sha = hash_file(artifact_path)
    manifest_file = manifest_path(artifact_path)
    manifest_sha = hash_file(manifest_file)
    try:
        manifest_record = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromotionCompositionError(
            f"artifact manifest is unreadable: {manifest_file}"
        ) from exc
    artifact_class = manifest_record.get("class") if isinstance(manifest_record, dict) else None
    return checked, artifact_path, manifest_file, artifact_sha, manifest_sha, artifact_class


def _decision_checked(decision: Mapping[str, Any], artifact_sha: str) -> str:
    """Step 2: approved, non-synthetic decision bound to this artifact."""
    from quant_fund.registry.mlflow_store import promotion_is_approved

    _require(
        promotion_is_approved(dict(decision)),
        "promotion decision is not an approved fail-closed promotion.v1 decision",
    )
    decision_source = str(decision.get("data_source", ""))
    _require(
        bool(decision_source.strip()) and decision_source.strip().upper() != "SYNTHETIC",
        "synthetic evidence cannot be presented for promotion",
    )
    _require(
        decision.get("artifact_sha256") == artifact_sha,
        "promotion decision binds a different artifact",
    )
    _require(bool(str(decision.get("run_id", "")).strip()), "promotion decision has no run_id")
    return decision_source


def _metrics_checked(
    input_metrics: Mapping[str, Any],
    artifact_sha: str,
    identity: _IdentityView,
    decision_source: str,
) -> None:
    """Step 3: metrics bind the same artifact and dataset (staleness)."""
    _require(
        isinstance(input_metrics, Mapping) and bool(input_metrics), "input metrics are missing"
    )
    metrics_source = str(input_metrics.get("data_source", ""))
    _require(
        bool(metrics_source.strip()) and metrics_source.strip().upper() != "SYNTHETIC",
        "synthetic evidence cannot be presented for promotion",
    )
    _require(
        metrics_source == decision_source,
        "input metrics and promotion decision disagree on data source",
    )
    _require(input_metrics.get("synthetic") is not True, "synthetic evidence cannot be promoted")
    _require(
        input_metrics.get("artifact_sha256") == artifact_sha,
        "input metrics bind a different artifact",
    )
    _require(
        input_metrics.get("dataset_content_sha256") == identity.dataset.materialized_panel_sha256,
        "input metrics dataset identity is stale or absent",
    )


def _approver_checked(approver: Mapping[str, Any]) -> Approver:
    """Step 5: approving identity — named, honest, timestamped."""
    try:
        approver_record = Approver.model_validate(dict(approver))
    except Exception as exc:
        raise PromotionCompositionError(
            f"approver identity is missing or dishonest: {exc}"
        ) from exc
    return approver_record


def _report_stage_warnings(report: Mapping[str, Any]) -> list[str]:
    """Stage-aware completeness scoping — resolution, never loosening."""
    warnings = report.get("warnings")
    report_status = report.get("status")
    if report_status == "complete":
        return []
    if (
        report_status == "insufficient_evidence"
        and isinstance(warnings, list)
        and set(warnings) <= STAGE_EXPECTED_REPORT_WARNINGS
    ):
        # Training-time report: the only gap is this receipt's own absence,
        # which composition is resolving right now.
        return sorted(set(warnings))
    raise PromotionCompositionError("evidence report is incomplete; promotion is blocked")


def _report_checked(
    report_path: Path, artifact_sha: str, identity: _IdentityView
) -> tuple[str, str, list[str]]:
    """Step 6: report bytes-bound, complete-or-resolved, research-only."""
    if not report_path.is_file():
        raise PromotionCompositionError(f"evidence report does not exist: {report_path}")
    report = _load_evidence_report(report_path)
    report_sha = hash_file(report_path)
    sidecar = report_path.with_name(f"{report_path.name}.sha256")
    if sidecar.is_file():
        expected = sidecar.read_text(encoding="ascii").strip()
        _require(expected == report_sha, "evidence report hash sidecar mismatch")
    _require(
        report.get("schema") == EVIDENCE_REPORT_SCHEMA,
        "evidence report schema mismatch",
    )
    _require(report.get("research_only") is True, "evidence report is not research-only")
    _require(report.get("live_pnl_claim") is False, "evidence report claims live P&L")
    warnings = report.get("warnings")
    _require(
        not (isinstance(warnings, list) and "synthetic_evidence_not_promotable" in warnings),
        "synthetic evidence is not promotable",
    )
    stage_warnings = _report_stage_warnings(report)
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        raise PromotionCompositionError("evidence report has no provenance")
    _require(
        provenance.get("artifact_sha256") == artifact_sha,
        "evidence report artifact hash does not match the artifact",
    )
    _require(
        provenance.get("dataset_content_sha256") == identity.dataset.materialized_panel_sha256,
        "evidence report dataset identity is stale or absent",
    )
    _require(
        provenance.get("manifest_valid") is True,
        "evidence report does not bind a verified artifact manifest",
    )
    return report_sha, str(report.get("status")), stage_warnings


def _write_receipt_exclusive(destination: Path, sealed: dict[str, Any]) -> None:
    """Immutable exclusive-create write plus sha256 sidecar."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(destination, "x", encoding="utf-8") as handle:
            handle.write(json.dumps(sealed, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise PromotionCompositionError(
            f"promotion receipts are immutable; refusing to overwrite: {destination}"
        ) from exc

    file_sha = hash_file(destination)
    sidecar = destination.with_name(f"{destination.name}.sha256")
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=sidecar.parent,
            prefix=f".{sidecar.name}.",
            suffix=".tmp",
            mode="w",
            encoding="ascii",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(file_sha + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, sidecar)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def compose_promotion_receipt(
    *,
    artifact: Path,
    evidence_report: Path,
    decision: Mapping[str, Any],
    input_metrics: Mapping[str, Any],
    gate_results: Mapping[str, Any],
    approver: Mapping[str, Any],
    out_path: Path,
) -> dict[str, Any]:
    """Compose one immutable ``promotion_receipt.v1`` — fail-closed.

    Raises :class:`PromotionCompositionError` on any missing, mismatched,
    stale, synthetic or dishonest input. The written receipt is never
    overwritten; a second composition to the same ``out_path`` raises.
    """
    from quant_fund.research.receipt_v2 import build_receipt_v2, seal_receipt

    identity, artifact_path, manifest_file, artifact_sha, manifest_sha, artifact_class = (
        _artifact_identity_checked(artifact)
    )
    decision_source = _decision_checked(decision, artifact_sha)
    _metrics_checked(input_metrics, artifact_sha, identity, decision_source)
    gates = _gate_results_checked(gate_results)
    approver_record = _approver_checked(approver)
    report_path = Path(evidence_report)
    report_sha, report_status, stage_warnings = _report_checked(report_path, artifact_sha, identity)

    payload: dict[str, Any] = {
        "schema": PROMOTION_RECEIPT_SCHEMA,
        "claim": "research_only",
        "execution_claim": "research_only",
        "research_only": True,
        "live_pnl_claim": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "artifact_identity": {
            "path": str(artifact_path),
            "artifact_sha256": artifact_sha,
            "manifest_path": str(manifest_file),
            "manifest_sha256": manifest_sha,
            "artifact_class": artifact_class,
            "identity": identity.model_dump(mode="json"),
        },
        "evidence_report": {
            "path": str(report_path),
            "sha256": report_sha,
            "schema": EVIDENCE_REPORT_SCHEMA,
            "status_at_composition": report_status,
            "stage_warnings": stage_warnings,
            "resolved_by": PROMOTION_RECEIPT_SCHEMA,
        },
        "promotion_decision": dict(decision),
        "input_metrics": dict(input_metrics),
        "gate_results": gates,
        "approver": approver_record.model_dump(mode="json"),
    }
    envelope = build_receipt_v2(
        kind=PROMOTION_RECEIPT_KIND,
        data_label=PROMOTION_RECEIPT_SCHEMA,
        dataset={
            "artifact_sha256": artifact_sha,
            "materialized_panel_sha256": identity.dataset.materialized_panel_sha256,
            "evidence_report_sha256": report_sha,
        },
        params={
            "run_id": str(decision.get("run_id", "")),
            "approver": approver_record.name,
            "gates": sorted(gates),
        },
        code_files=(Path(__file__),),
        verdict="pass",
        payload=payload,
    )
    sealed = seal_receipt(envelope)
    _write_receipt_exclusive(Path(out_path), sealed)
    return sealed
