"""Cross-sectional rank-IC evaluation lane (ULTRAPLAN P3.4).

Seeded multi-asset SYNTHETIC panels with a planted cross-sectional signal
are scored per date by Spearman rank-IC between a challenger transform of
the signal and h-step forward returns; the per-date IC series is then
summarized with a Newey-West mean-IC t-stat via
``metrics.cross_section.date_ic_series``. Reporting stays in proper-score
framing (rank-IC distribution and HAC significance), never P&L.

``write_rankic_receipt`` publishes the sealed JSON evidence under
``receipts/`` (canonical payload, sha256 filename, atomic write). All
output is SYNTHETIC correctness evidence, not market data.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_fund.metrics.cross_section import date_ic_series
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.research.fleet_eval import _atomic_write_text
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

Array = NDArray[np.float64]

RANKIC_SCHEMA = "cross_sectional_rankic.v1"

DEFAULT_HORIZONS = (1, 5, 20)
_MIN_ASSETS = 8
_MIN_DATES = 30


@dataclass(frozen=True)
class CrossSectionalPanel:
    """A planted-signal synthetic panel: ``signal[t, i]`` then ``fwd[t, i, h]``.

    ``signal`` is observed at date ``t`` before the horizon-``h`` forward
    return ``fwd[t, i, h]`` is realized — no look-ahead inside a shard.

    ``data_label`` declares the provenance the receipt will carry —
    constructors must name their source; the bench refuses a mixed corpus.
    """

    dates: NDArray[Any]
    asset_ids: tuple[str, ...]
    signal: Array
    forward: dict[int, Array]
    description: str
    data_label: str

    def __post_init__(self) -> None:
        if not str(self.data_label).strip():
            raise ValueError("data_label must be a nonempty string")


PanelGenerator = Callable[[int, int, int, Sequence[int]], CrossSectionalPanel]


def _require_dims(n_assets: int, n_dates: int) -> tuple[int, int]:
    n_assets, n_dates = int(n_assets), int(n_dates)
    if n_assets < _MIN_ASSETS:
        raise ValueError(f"need at least {_MIN_ASSETS} assets per date, got {n_assets}")
    if n_dates < _MIN_DATES + max(DEFAULT_HORIZONS) + 1:
        raise ValueError(
            f"need at least {_MIN_DATES + max(DEFAULT_HORIZONS) + 1} dates, got {n_dates}"
        )
    return n_assets, n_dates


def _dates(n: int) -> NDArray[Any]:
    base = np.datetime64("2020-01-01")
    return base + np.arange(n)


def _forward_returns(
    rng: np.random.Generator,
    n_dates: int,
    n_assets: int,
    horizons: Sequence[int],
    sigma: float = 0.01,
) -> dict[int, Array]:
    """Independent Gaussian daily returns aggregated to h-step forwards."""
    daily = rng.normal(0.0, sigma, size=(n_dates, n_assets))
    out: dict[int, Array] = {}
    for h in horizons:
        h = int(h)
        if h < 1:
            raise ValueError("horizons must be >= 1")
        fwd = np.full((n_dates, n_assets), np.nan)
        cum = np.cumsum(daily, axis=0)
        fwd[h:] = cum[h:] - cum[:-h]
        out[h] = fwd
    return out


def _panel(
    name: str,
    description: str,
    n_dates: int,
    n_assets: int,
    seed: int,
    horizons: Sequence[int],
    beta: float,
    sigma: float,
    map_fn: Callable[[Array], Array] | None = None,
    flip_after: int | None = None,
) -> CrossSectionalPanel:
    rng = np.random.default_rng(seed)
    signal = rng.normal(0.0, 1.0, size=(n_dates, n_assets))
    mapped = signal if map_fn is None else map_fn(signal)
    rng_fwd = np.random.default_rng(seed + 7919)
    forward = _forward_returns(rng_fwd, n_dates, n_assets, horizons, sigma)
    loading = rng.normal(beta, 0.002, size=(n_assets,))
    for h, fwd in forward.items():
        effect = mapped * loading[None, :] * math.sqrt(h)
        if flip_after is not None:
            sign = np.ones(n_dates)
            sign[flip_after:] = -1.0
            effect = effect * sign[:, None]
        forward[h] = fwd + effect
    from quant_fund.utils.numeric import stable_decimal_array

    # Same libm-ulp collapse as the capacity books: the receipt hashes these
    # arrays with tobytes(), so the pin is a byte identity.
    signal = stable_decimal_array(signal)
    forward = {h: stable_decimal_array(fwd) for h, fwd in forward.items()}
    return CrossSectionalPanel(
        dates=_dates(n_dates),
        asset_ids=tuple(f"S{i:03d}" for i in range(n_assets)),
        signal=signal,
        forward=forward,
        description=f"{name}: {description}",
        data_label="SYNTHETIC",
    )


def linear_signal(
    n_dates: int, n_assets: int, seed: int, horizons: Sequence[int]
) -> CrossSectionalPanel:
    """Rank signal maps linearly to forward returns (positive planted IC)."""
    return _panel(
        "linear_signal",
        "fwd = beta*signal + eps",
        n_dates,
        n_assets,
        seed,
        horizons,
        beta=0.02,
        sigma=0.05,
    )


def monotone_cubic(
    n_dates: int, n_assets: int, seed: int, horizons: Sequence[int]
) -> CrossSectionalPanel:
    """fwd = beta*signal^3 — rank-preserving: high Spearman, weaker Pearson."""
    return _panel(
        "monotone_cubic",
        "fwd = beta*signal^3 + eps",
        n_dates,
        n_assets,
        seed,
        horizons,
        beta=0.004,
        sigma=0.05,
        map_fn=lambda s: s**3,
    )


def pure_noise(
    n_dates: int, n_assets: int, seed: int, horizons: Sequence[int]
) -> CrossSectionalPanel:
    """Signal and returns independent — the zero-IC null."""
    return _panel(
        "pure_noise",
        "fwd independent of signal",
        n_dates,
        n_assets,
        seed,
        horizons,
        beta=0.0,
        sigma=0.05,
    )


def regime_flip(
    n_dates: int, n_assets: int, seed: int, horizons: Sequence[int]
) -> CrossSectionalPanel:
    """Loading sign flips mid-sample: |IC| large, mean IC near zero."""
    return _panel(
        "regime_flip",
        "sign(beta) flips at mid-sample",
        n_dates,
        n_assets,
        seed,
        horizons,
        beta=0.03,
        sigma=0.05,
        flip_after=n_dates // 2,
    )


def weak_signal(
    n_dates: int, n_assets: int, seed: int, horizons: Sequence[int]
) -> CrossSectionalPanel:
    """Small beta — the harness must call it borderline, not oversell."""
    return _panel(
        "weak_signal",
        "fwd = tiny*signal + eps",
        n_dates,
        n_assets,
        seed,
        horizons,
        beta=0.004,
        sigma=0.05,
    )


PANEL_GENERATORS: dict[str, PanelGenerator] = {
    "linear_signal": linear_signal,
    "monotone_cubic": monotone_cubic,
    "pure_noise": pure_noise,
    "regime_flip": regime_flip,
    "weak_signal": weak_signal,
}


def resolve_panels(names: Sequence[str] | None = None) -> dict[str, PanelGenerator]:
    if names is None:
        return dict(PANEL_GENERATORS)
    resolved: dict[str, PanelGenerator] = {}
    for name in names:
        if name not in PANEL_GENERATORS:
            raise ValueError(f"unknown panel {name!r}; have {sorted(PANEL_GENERATORS)}")
        resolved[name] = PANEL_GENERATORS[name]
    return resolved


Challenger = Callable[[Array, np.random.Generator], Array]


def _challenger_identity(signal: Array, rng: np.random.Generator) -> Array:
    return signal


def _challenger_noisy(signal: Array, rng: np.random.Generator) -> Array:
    return signal + rng.normal(0.0, 3.0, size=signal.shape)


def _challenger_lagged(signal: Array, rng: np.random.Generator) -> Array:
    out = np.full(signal.shape, np.nan)
    out[1:] = signal[:-1]
    return out


def _challenger_shuffled(signal: Array, rng: np.random.Generator) -> Array:
    out = np.empty(signal.shape)
    for t in range(signal.shape[0]):
        out[t] = signal[t, rng.permutation(signal.shape[1])]
    return out


def _challenger_inverted(signal: Array, rng: np.random.Generator) -> Array:
    return -signal


CHALLENGERS: dict[str, Challenger] = {
    "identity": _challenger_identity,
    "noisy": _challenger_noisy,
    "lagged": _challenger_lagged,
    "shuffled": _challenger_shuffled,
    "inverted": _challenger_inverted,
}


def _to_flat(score: Array, target: Array) -> tuple[Array, Array, Array]:
    mask = np.isfinite(score) & np.isfinite(target)
    n_dates = score.shape[0]
    return (
        score[mask],
        target[mask],
        np.repeat(np.arange(n_dates), mask.sum(axis=1)),
    )


def _stable_metric(value: float) -> float:
    from quant_fund.utils.numeric import stable_decimal

    return stable_decimal(float(value))


def _stable_t_p(t_value: float, n_dates: int) -> tuple[float, float]:
    """Round the t-stat, then the two-sided Student-t tail of that rounded t.

    The lane contract re-derives ``p`` from ``t``. Rounding them independently
    breaks that check, and an unrounded tail still moves with libm.
    """
    from scipy.stats import t as t_dist

    t_rounded = _stable_metric(t_value)
    if n_dates < 2 or not math.isfinite(t_rounded):
        return t_rounded, _stable_metric(float("nan"))
    p_value = 2.0 * float(t_dist.sf(abs(t_rounded), df=n_dates - 1))
    return t_rounded, _stable_metric(p_value)


def run_cross_sectional_bench(
    panels: Mapping[str, PanelGenerator] | None = None,
    challengers: Sequence[str] | None = None,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    n_assets: int = 32,
    n_dates: int = 96,
    seed: int = 11,
    min_names: int = 5,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Score each challenger transform per shard and horizon; return frame + receipt.

    Proper-score framing only: rank-IC distribution and HAC t-stat, never P&L.
    """
    n_assets, n_dates = _require_dims(n_assets, n_dates)
    horizons = tuple(int(h) for h in horizons)
    if not horizons or any(h < 1 for h in horizons):
        raise ValueError("horizons must be a nonempty tuple of ints >= 1")
    resolved_panels = dict(PANEL_GENERATORS) if panels is None else dict(panels)
    challenger_names = list(CHALLENGERS) if challengers is None else list(challengers)
    for name in challenger_names:
        if name not in CHALLENGERS:
            raise ValueError(f"unknown challenger {name!r}; have {sorted(CHALLENGERS)}")

    rows: list[dict[str, Any]] = []
    panel_meta: dict[str, Any] = {}
    for shard_index, (name, generator) in enumerate(resolved_panels.items()):
        shard_seed = int(seed) + 104729 * shard_index
        panel = generator(n_dates, n_assets, shard_seed, horizons)
        panel_meta[name] = {
            "data_label": str(panel.data_label),
            "n_dates": n_dates,
            "n_assets": n_assets,
            "seed": shard_seed,
            "signal_sha256": hash_bytes(panel.signal.tobytes()),
            "forward_sha256": {
                str(h): hash_bytes(fwd.tobytes()) for h, fwd in sorted(panel.forward.items())
            },
            "description": panel.description,
        }
        for chal_index, chal_name in enumerate(challenger_names):
            rng = np.random.default_rng(shard_seed + 811 + 37 * chal_index)
            transformed = CHALLENGERS[chal_name](panel.signal, rng)
            for h in horizons:
                fwd = panel.forward[h]
                scores_flat, target_flat, date_idx = _to_flat(transformed, fwd)
                if scores_flat.size == 0:
                    rows.append(
                        {
                            "shard": name,
                            "challenger": chal_name,
                            "horizon": h,
                            "status": "error",
                            "error": "no finite score/target pairs",
                            "n_dates": 0,
                        }
                    )
                    continue
                result = date_ic_series(
                    scores_flat,
                    target_flat,
                    date_idx,
                    min_names=min_names,
                )
                t_spearman, p_spearman = _stable_t_p(result.t_spearman, result.n_dates)
                rows.append(
                    {
                        "shard": name,
                        "challenger": chal_name,
                        "horizon": h,
                        "status": "ok",
                        "error": "",
                        "n_dates": result.n_dates,
                        "mean_spearman": _stable_metric(result.mean_spearman),
                        "t_spearman": t_spearman,
                        "p_spearman": p_spearman,
                        "mean_pearson": _stable_metric(result.mean_pearson),
                        "t_pearson": _stable_metric(result.t_pearson),
                        "icir_pearson": _stable_metric(result.icir_pearson),
                        "icir_ann_pearson": _stable_metric(result.icir_ann_pearson),
                    }
                )

    columns = [
        "shard",
        "challenger",
        "horizon",
        "status",
        "error",
        "n_dates",
        "mean_spearman",
        "t_spearman",
        "p_spearman",
        "mean_pearson",
        "t_pearson",
        "icir_pearson",
        "icir_ann_pearson",
    ]
    schema = {
        **{k: pl.String for k in ("shard", "challenger", "status", "error")},
        **{k: pl.Int64 for k in ("horizon", "n_dates")},
        **{
            k: pl.Float64
            for k in columns
            if k not in {"shard", "challenger", "status", "error", "horizon", "n_dates"}
        },
    }
    frame = pl.DataFrame(rows, schema=schema, orient="row").select(columns)

    labels = {str(meta["data_label"]) for meta in panel_meta.values()}
    if len(labels) > 1:
        raise ValueError(
            "panels carry mixed data_label values "
            f"{sorted(labels)}; run mixed corpora as separate receipts"
        )
    data_label = next(iter(labels)) if labels else "UNKNOWN"
    panel_digests = {
        name: {
            "signal_sha256": meta["signal_sha256"],
            "forward_sha256": meta["forward_sha256"],
        }
        for name, meta in panel_meta.items()
    }
    inputs_sha256 = hash_bytes(
        canonical_json_bytes(
            {
                "panels": panel_digests,
                "challengers": challenger_names,
                "horizons": list(horizons),
                "n_assets": n_assets,
                "n_dates": n_dates,
                "seed": int(seed),
            }
        )
    )
    # Corpus-level fingerprint: digest over the evaluated panel content only —
    # receipts across lanes that evaluated the same panels agree on it,
    # which is what the cross-receipt lattice edges on.
    dataset_sha256 = hash_bytes(canonical_json_bytes({"shards": panel_digests}))
    receipt: dict[str, Any] = {
        "schema": RANKIC_SCHEMA,
        "kind": "cross_sectional_rankic_eval",
        "data_label": data_label,
        "live_pnl_claim": False,
        "meta": {
            "generated_at": datetime.now(UTC).isoformat(),
            "git_revision": git_revision(),
        },
        "seed": int(seed),
        "n_assets": n_assets,
        "n_dates": n_dates,
        "horizons": list(horizons),
        "challengers": challenger_names,
        "panels": panel_meta,
        "inputs_sha256": inputs_sha256,
        "dataset_sha256": dataset_sha256,
        "n_rows": len(rows),
        "n_error_rows": sum(1 for row in rows if row["status"] != "ok"),
        "results": rows,
    }
    return frame, receipt


def _is_hex64_rk(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def rankic_v1_audit_errors(receipt: Mapping[str, Any]) -> list[str]:
    """Deep audit of a ``cross_sectional_rankic.v1`` payload's result cells.

    Re-derives what the sealed claims assert: the results grid is complete
    over declared challengers x horizons x panels, counts recount, IC
    magnitudes obey correlation bounds (|IC| <= 1, p-values in [0, 1],
    finite t/ICIR), and panel digests are well-formed. Verifier-only.
    """
    errors: list[str] = []
    results = receipt.get("results")
    challengers = receipt.get("challengers")
    horizons = receipt.get("horizons")
    panels = receipt.get("panels")
    if not (
        isinstance(results, list)
        and isinstance(challengers, list)
        and isinstance(horizons, list)
        and isinstance(panels, Mapping)
    ):
        return ["audit_inputs_missing"]

    challenger_set = {str(c) for c in challengers}
    horizon_set = {int(h) for h in horizons if isinstance(h, int) and not isinstance(h, bool)}
    panel_set = {str(p) for p in panels}
    seen: set[tuple[str, str, int]] = set()
    n_error_rows = 0
    n_dates_declared = receipt.get("n_dates")

    for row in results:
        if not isinstance(row, Mapping):
            errors.append("row_not_object")
            continue
        shard = row.get("shard")
        challenger = row.get("challenger")
        horizon = row.get("horizon")
        if not isinstance(shard, str) or not isinstance(challenger, str):
            errors.append("row_identity_missing")
            continue
        key = (shard, challenger, horizon) if isinstance(horizon, int) else None
        if shard not in panel_set or challenger not in challenger_set or horizon not in horizon_set:
            errors.append(f"row_outside_grid:{shard}:{challenger}:{horizon}")
        elif key is not None:
            if key in seen:
                errors.append(f"row_duplicate:{shard}:{challenger}:{horizon}")
            seen.add(key)
        status = row.get("status")
        if status == "ok":
            if row.get("error") != "":
                errors.append(f"row_ok_with_error:{shard}:{challenger}:{horizon}")
        elif status == "error":
            n_error_rows += 1
            if not row.get("error"):
                errors.append(f"row_error_without_error:{shard}:{challenger}:{horizon}")
            continue
        else:
            errors.append(f"row_status_unknown:{shard}:{challenger}:{horizon}")
            continue
        for name in ("mean_spearman", "mean_pearson"):
            value = row.get(name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or not -1.0 <= value <= 1.0
            ):
                errors.append(f"row_{name}_invalid:{shard}:{challenger}:{horizon}")
        p = row.get("p_spearman")
        if (
            not isinstance(p, (int, float))
            or isinstance(p, bool)
            or not math.isfinite(p)
            or not 0.0 <= p <= 1.0
        ):
            errors.append(f"row_p_spearman_invalid:{shard}:{challenger}:{horizon}")
        for name in ("t_spearman", "t_pearson", "icir_pearson", "icir_ann_pearson"):
            value = row.get(name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                errors.append(f"row_{name}_invalid:{shard}:{challenger}:{horizon}")
        n_dates = row.get("n_dates")
        if not isinstance(n_dates, int) or isinstance(n_dates, bool) or n_dates <= 0:
            errors.append(f"row_n_dates_invalid:{shard}:{challenger}:{horizon}")
        elif isinstance(n_dates_declared, int) and n_dates > n_dates_declared:
            errors.append(f"row_n_dates_exceeds_panel:{shard}:{challenger}:{horizon}")

    expected = {(s, c, h) for s in panel_set for c in challenger_set for h in horizon_set}
    if seen != expected:
        errors.append("results_grid_incomplete")
    if receipt.get("n_rows") != len(results):
        errors.append("n_rows_mismatch")
    if receipt.get("n_error_rows") != n_error_rows:
        errors.append("n_error_rows_mismatch")
    for name, meta in panels.items():
        if not isinstance(meta, Mapping):
            errors.append(f"panel_meta_invalid:{name}")
            continue
        if not _is_hex64_rk(meta.get("signal_sha256")):
            errors.append(f"panel_digest_invalid:{name}:signal_sha256")
        fwd = meta.get("forward_sha256")
        if not isinstance(fwd, Mapping):
            errors.append(f"panel_digest_invalid:{name}:forward_sha256")
        else:
            if {int(k) for k in fwd if str(k).isdigit()} != horizon_set:
                errors.append(f"panel_horizon_mismatch:{name}")
            for h_key, digest in fwd.items():
                if not _is_hex64_rk(digest):
                    errors.append(f"panel_digest_invalid:{name}:{h_key}")
        if meta.get("n_assets") != receipt.get("n_assets"):
            errors.append(f"panel_n_assets_mismatch:{name}")
        if meta.get("n_dates") != n_dates_declared:
            errors.append(f"panel_n_dates_mismatch:{name}")
    return errors


def rankic_contract_errors(receipt: Mapping[str, Any]) -> list[str]:
    """Fail-closed contract for a ``cross_sectional_rankic.v1`` payload."""
    research_blob = {key: value for key, value in receipt.items() if key != "live_pnl_claim"}
    errors: list[str] = []
    if receipt.get("schema") != RANKIC_SCHEMA:
        errors.append("schema_not_rankic_v1")
    if receipt.get("data_label") != "SYNTHETIC":
        errors.append("data_label_not_synthetic")
    if receipt.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    if not isinstance(receipt.get("results"), list) or not receipt["results"]:
        errors.append("results_missing_or_empty")
    if not family_blob_forbidden_metrics_absent(research_blob):
        errors.append("forbidden_metric_keys")
    results = receipt.get("results")
    if isinstance(results, list):
        if receipt.get("n_rows") is not None and receipt.get("n_rows") != len(results):
            errors.append("n_rows_mismatch")
        error_rows = sum(
            1 for row in results if not isinstance(row, Mapping) or row.get("status") != "ok"
        )
        if receipt.get("n_error_rows") is not None and receipt.get("n_error_rows") != error_rows:
            errors.append("n_error_rows_mismatch")
    return errors


def rankic_dataset_identity(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Panel content digests bound by a v2 ``dataset_hash``."""
    panels = receipt.get("panels")
    if not isinstance(panels, Mapping):
        raise ValueError("rank-IC receipt has no panels block")
    dataset: dict[str, Any] = {}
    for name, meta in panels.items():
        if not isinstance(meta, Mapping):
            raise ValueError(f"rank-IC panel {name!r} metadata is not an object")
        dataset[str(name)] = {
            "signal_sha256": meta.get("signal_sha256"),
            "forward_sha256": meta.get("forward_sha256"),
        }
    return dataset


def rankic_params(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """The run parameters bound by a v2 ``params_hash``."""
    return {
        "seed": receipt.get("seed"),
        "n_assets": receipt.get("n_assets"),
        "n_dates": receipt.get("n_dates"),
        "horizons": receipt.get("horizons"),
        "challengers": receipt.get("challengers"),
    }


def rankic_verdict(receipt: Mapping[str, Any]) -> str:
    """pass iff every row scored without error; a recorded error is a fail."""
    n_error_rows = receipt.get("n_error_rows")
    return "pass" if n_error_rows == 0 else "fail"


def rankic_receipt_v2(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap a ``cross_sectional_rankic.v1`` payload in the ``receipt.v2`` envelope.

    The v1 payload is embedded verbatim under ``payload``; the envelope binds
    the panel content digests, run params, this module's source hash, and the
    loaded numeric stack. Validates the v1 contract first — a malformed v1
    receipt is never wrapped.
    """
    from quant_fund.research.receipt_v2 import build_receipt_v2

    body = {key: value for key, value in receipt.items() if key != "meta"}
    if rankic_contract_errors(body):
        raise ValueError("rank-IC receipt violates its synthetic research contract")
    meta = receipt.get("meta")
    meta_map: Mapping[str, Any] = meta if isinstance(meta, Mapping) else {}
    generated_at = meta_map.get("generated_at") or receipt.get("generated_at")
    revision = (
        meta_map.get("git_revision")
        or meta_map.get("code_revision")
        or receipt.get("git_revision")
        or receipt.get("code_revision")
    )
    return build_receipt_v2(
        kind=str(receipt["kind"]),
        data_label=str(receipt["data_label"]),
        dataset=rankic_dataset_identity(body),
        params=rankic_params(body),
        code_files=(Path(__file__),),
        verdict=rankic_verdict(body),
        payload=dict(body),
        generated_at=None if generated_at is None else str(generated_at),
        revision=None if revision is None else str(revision),
    )


def rankic_v2_consistency_errors(envelope: Mapping[str, Any]) -> list[str]:
    """Re-derive a rank-IC receipt.v2 envelope's bound digests from its payload."""
    errors: list[str] = []
    payload = envelope.get("payload")
    if not isinstance(payload, Mapping):
        return ["payload_not_object"]
    contract_errors = rankic_contract_errors(payload)
    errors.extend(f"payload_{name}" for name in contract_errors)
    if contract_errors:
        return errors
    try:
        dataset = rankic_dataset_identity(payload)
    except ValueError as exc:
        return [*errors, f"payload_{exc}"]
    if hash_bytes(canonical_json_bytes(dataset)) != envelope.get("dataset_hash"):
        errors.append("dataset_hash_mismatch")
    if hash_bytes(canonical_json_bytes(rankic_params(payload))) != envelope.get("params_hash"):
        errors.append("params_hash_mismatch")
    if rankic_verdict(payload) != envelope.get("verdict"):
        errors.append("verdict_mismatch")
    return errors


def write_rankic_receipt(
    receipt: Mapping[str, Any],
    receipts_dir: Path | str = Path("receipts"),
    *,
    receipt_version: int = 1,
) -> Path:
    """Seal a rank-IC receipt as ``receipts/rankic_eval_<hash>.json`` (atomic).

    ``receipt_version=2`` wraps the v1 payload in the unified ``receipt.v2``
    envelope before sealing.
    """
    from quant_fund.research.receipt_v2 import seal_receipt

    if receipt_version == 1:
        body = {key: value for key, value in receipt.items() if key != "meta"}
        if rankic_contract_errors(body):
            raise ValueError("rank-IC receipt violates its synthetic research contract")
    elif receipt_version == 2:
        body = rankic_receipt_v2(receipt)
    else:
        raise ValueError(f"receipt_version must be 1 or 2, got {receipt_version!r}")
    payload = seal_receipt(body)
    path = Path(receipts_dir) / f"rankic_eval_{payload['receipt_sha256'][:16]}.json"
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def format_rankic_table(frame: pl.DataFrame) -> str:
    """Compact text table for the CLI echo."""
    ok = frame.filter(pl.col("status") == "ok")
    lines = ["shard | challenger | h | n_dates | mean_rankIC | t_NW | p"]
    for row in ok.iter_rows(named=True):
        lines.append(
            f"{row['shard']} | {row['challenger']} | {row['horizon']} | "
            f"{row['n_dates']} | {row['mean_spearman']:+.4f} | "
            f"{row['t_spearman']:+.2f} | {row['p_spearman']:.3g}"
        )
    return "\n".join(lines)
