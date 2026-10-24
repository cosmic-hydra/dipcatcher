"""Volatility-forecast bench on seeded SYNTHETIC vol shards (ULTRAPLAN P3.5).

Each shard generator draws a latent variance path — GJR-GARCH clustering,
rough fOU log-vol, or a planted structural break — then fills every bar with
intraday increments so each bar carries a close-to-close return, a realized
variance (sum of squared increments), and a Parkinson-style range measure.
Origins index the first unobserved bar (the ``garch_benchmark`` protocol):
models refit on observed history only and forecast cumulative realized
variance over the next ``h`` bars. Scores are proper rules only — mean QLIKE
(Patton 2011, robust to proxy noise) and MSE — never headline P&L metrics.
``write_vol_bench_receipt`` seals hash-stamped JSON evidence under
``receipts/`` (the ``fleet_eval`` convention). All output is SYNTHETIC
correctness evidence, not market data.
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_fund.metrics.vol_eval import mse as mse_loss
from quant_fund.metrics.vol_eval import qlike as qlike_loss
from quant_fund.metrics.vol_eval import vol_loss_diff
from quant_fund.models.har import har_forecast, har_rv_fit
from quant_fund.models.realized_garch import RealizedGARCHVol
from quant_fund.models.rough_vol import simulate_fou
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.research.evalues import LossEProcess
from quant_fund.research.garch_benchmark import build_origins
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

Array = NDArray[np.float64]

VOL_BENCH_SCHEMA = "vol_bench.v1"
DEFAULT_HORIZONS: tuple[int, ...] = (1, 5)
DM_REFERENCE = "har"

_VARIANCE_FLOOR = 1e-12
_EPROMOTION_INIT_SCALE = 1e-4
_INTRADAY_STEPS = 24
_LN2 = float(np.log(2.0))
_EWMA_LAMBDA = 0.94


@dataclass(frozen=True)
class VolShard:
    """One seeded SYNTHETIC vol shard: returns + realized-variance proxies.

    ``returns`` are bar close-to-close returns, ``rv`` the per-bar realized
    variance (sum of squared intraday increments — the scored target), and
    ``parkinson`` the per-bar Parkinson range measure consumed by
    ``RealizedGARCHVol``.
    """

    name: str
    returns: Array
    rv: Array
    parkinson: Array
    config: dict[str, Any]


VolShardGenerator = Callable[[int, int], VolShard]
# (returns_hist, rv_hist, parkinson_hist, horizon, seed) -> cumulative
# variance forecast over the next ``horizon`` bars.
VolForecaster = Callable[[Array, Array, Array, int, int], float]


def _require_n(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("shard size n must be a positive integer")
    return int(n)


def _fill_bars(cond_var: Array, rng: np.random.Generator) -> tuple[Array, Array, Array]:
    """Fill each bar with intraday increments drawn at its latent variance.

    Returns (close-to-close return, realized variance, Parkinson range).
    The Parkinson measure uses the intraday cumulative path's high-low
    range: ``(max - min)^2 / (4 ln 2)``. Fail-closed on a degenerate
    non-positive draw.
    """
    var = np.asarray(cond_var, dtype=float).ravel()
    if var.size == 0 or not np.isfinite(var).all() or (var <= 0.0).any():
        raise ValueError("cond_var must be finite and positive")
    increments = (
        rng.normal(0.0, 1.0, (var.size, _INTRADAY_STEPS)) * np.sqrt(var / _INTRADAY_STEPS)[:, None]
    )
    returns = increments.sum(axis=1)
    rv = (increments**2).sum(axis=1)
    path = np.concatenate([np.zeros((var.size, 1)), np.cumsum(increments, axis=1)], axis=1)
    parkinson = (path.max(axis=1) - path.min(axis=1)) ** 2 / (4.0 * _LN2)
    if (rv <= 0.0).any() or (parkinson <= 0.0).any():
        raise ValueError("degenerate intraday fill produced a non-positive measure")
    return (
        np.asarray(returns, dtype=float),
        np.asarray(rv, dtype=float),
        np.asarray(parkinson, dtype=float),
    )


def garch_vol(n: int, seed: int) -> VolShard:
    """GJR-GARCH(1,1) latent vol path with intraday fill (SYNTHETIC).

    The variance recursion is driven by the process's own innovations —
    emitted bar returns are intraday sums at that variance, mirroring how
    real RV proxies observe latent vol with measurement noise.
    """
    rng = np.random.default_rng(seed)
    config: dict[str, Any] = {
        "data_label": "SYNTHETIC",
        "seed": int(seed),
        "process": "gjr_garch_1_1",
        "omega": 4e-6,
        "alpha": 0.05,
        "gamma": 0.08,
        "beta": 0.9,
        "burn": 128,
        "intraday_steps": _INTRADAY_STEPS,
    }
    n = _require_n(n)
    omega = float(config["omega"])
    alpha = float(config["alpha"])
    gamma = float(config["gamma"])
    beta = float(config["beta"])
    persistence = alpha + gamma / 2.0 + beta
    if not 0.0 < persistence < 1.0:
        raise ValueError("GJR-GARCH persistence must lie in (0, 1)")
    burn = int(config["burn"])
    total = n + burn
    z = rng.normal(0.0, 1.0, total)
    eps = np.empty(total, dtype=float)
    var = np.empty(total, dtype=float)
    var[0] = omega / (1.0 - persistence)
    eps[0] = np.sqrt(var[0]) * z[0]
    for t in range(1, total):
        var[t] = omega + (alpha + gamma * (eps[t - 1] < 0.0)) * eps[t - 1] ** 2 + beta * var[t - 1]
        eps[t] = np.sqrt(var[t]) * z[t]
    returns, rv, parkinson = _fill_bars(var[burn:], rng)
    config["persistence"] = float(persistence)
    return VolShard("garch_vol", returns, rv, parkinson, config)


def rough_vol(n: int, seed: int) -> VolShard:
    """Rough-vol shard: log-vol follows an fOU path (H=0.1; SYNTHETIC).

    ``simulate_fou`` draws the centered log-vol factor; it is standardized
    to unit scale and mapped to variance as ``base_var * exp(vol_of_logvol
    * lv)`` so the path stays bounded and seeded-reproducible.
    """
    rng = np.random.default_rng(seed)
    config: dict[str, Any] = {
        "data_label": "SYNTHETIC",
        "seed": int(seed),
        "process": "rough_fou_logvol",
        "hurst": 0.1,
        "base_var": 4e-4,
        "vol_of_logvol": 0.75,
        "burn": 128,
        "intraday_steps": _INTRADAY_STEPS,
    }
    n = _require_n(n)
    burn = int(config["burn"])
    total = n + burn
    if total < 50:
        raise ValueError("rough-vol shard needs n + burn >= 50")
    logvol = simulate_fou(total, float(config["hurst"]), seed=int(seed))
    logvol = (logvol - logvol.mean()) / max(float(logvol.std()), 1e-12)
    cond_var = float(config["base_var"]) * np.exp(float(config["vol_of_logvol"]) * logvol)
    cond_var = np.clip(cond_var, _VARIANCE_FLOOR, None)
    returns, rv, parkinson = _fill_bars(cond_var[burn:], rng)
    return VolShard("rough_vol", returns, rv, parkinson, config)


def break_vol(n: int, seed: int) -> VolShard:
    """Structural-break shard: piecewise-constant variance levels (SYNTHETIC).

    Two planted breaks split the path into thirds with low, crisis-high and
    settled variance regimes — the persistent-break case where cascade
    models must relocate the level.
    """
    rng = np.random.default_rng(seed)
    config: dict[str, Any] = {
        "data_label": "SYNTHETIC",
        "seed": int(seed),
        "process": "piecewise_variance_break",
        "levels": (4e-4, 2.2e-3, 6e-4),
        "breaks": (0.4, 0.75),
        "intraday_steps": _INTRADAY_STEPS,
    }
    n = _require_n(n)
    levels = np.asarray(config["levels"], dtype=float)
    if levels.ndim != 1 or levels.size != 3 or (levels <= 0.0).any():
        raise ValueError("break shard requires three positive variance levels")
    b1, b2 = (int(round(float(f) * n)) for f in config["breaks"])
    if not (0 < b1 < b2 < n):
        raise ValueError("breaks must land strictly inside the shard")
    cond_var = np.empty(n, dtype=float)
    cond_var[:b1] = levels[0]
    cond_var[b1:b2] = levels[1]
    cond_var[b2:] = levels[2]
    returns, rv, parkinson = _fill_bars(cond_var, rng)
    config["break_bars"] = (int(b1), int(b2))
    return VolShard("break_vol", returns, rv, parkinson, config)


VOL_SHARD_GENERATORS: dict[str, VolShardGenerator] = {
    "garch_vol": garch_vol,
    "rough_vol": rough_vol,
    "break_vol": break_vol,
}


def resolve_vol_shard_generators(
    names: Iterable[str] | None = None,
) -> dict[str, VolShardGenerator]:
    """Resolve shard names against ``VOL_SHARD_GENERATORS`` (default: all)."""
    if names is None:
        return dict(VOL_SHARD_GENERATORS)
    resolved: dict[str, VolShardGenerator] = {}
    for raw in names:
        name = str(raw).strip()
        if not name:
            continue
        if name not in VOL_SHARD_GENERATORS:
            raise ValueError(f"unknown vol shard {name!r}")
        resolved[name] = VOL_SHARD_GENERATORS[name]
    if not resolved:
        raise ValueError("vol bench requires at least one shard")
    return resolved


def _forecast_rv_roll(rets: Array, rv: Array, park: Array, h: int, seed: int) -> float:
    """Baseline: trailing-20 mean RV scaled to the horizon."""
    del rets, park, seed
    return max(h * float(np.mean(rv[-20:])), _VARIANCE_FLOOR)


def _forecast_rv_ewma(rets: Array, rv: Array, park: Array, h: int, seed: int) -> float:
    """Baseline: EWMA (lambda=0.94) of the RV series scaled to the horizon."""
    del rets, park, seed
    ewma = float(rv[0])
    for value in rv[1:]:
        ewma = _EWMA_LAMBDA * ewma + (1.0 - _EWMA_LAMBDA) * float(value)
    return max(h * ewma, _VARIANCE_FLOOR)


def _forecast_har(rets: Array, rv: Array, park: Array, h: int, seed: int) -> float:
    """HAR-RV (Corsi 2009) via ``models.har``; recursive rollout for h > 1.

    Zero-return days (rv == 0 on real EOD tape) are floored at
    ``_VARIANCE_FLOOR`` before the fit — a zero rv is unobserved variance,
    not literal zero; the model's own strict-positivity contract stays.
    """
    del rets, park, seed
    floored = np.maximum(np.asarray(rv, dtype=float), _VARIANCE_FLOOR)
    fit = har_rv_fit(floored)
    history = list(floored)
    total = 0.0
    for _ in range(h):
        step = har_forecast(fit, np.asarray(history, dtype=float))
        step = max(float(step), _VARIANCE_FLOOR)
        total += step
        history.append(step)
    return max(total, _VARIANCE_FLOOR)


def _forecast_realized_garch(rets: Array, rv: Array, park: Array, h: int, seed: int) -> float:
    """Realized-GARCH log-linear fit on returns + daily Parkinson measure."""
    del rv, seed
    model = RealizedGARCHVol(mean="Zero")
    model.fit_returns(np.asarray(rets, dtype=float), np.asarray(park, dtype=float))
    forecast = model.forecast(horizon=h)
    variance = np.asarray(forecast["variance"], dtype=float)
    if variance.shape != (h,) or not np.isfinite(variance).all():
        raise ValueError("realized_garch returned a degenerate variance path")
    return max(float(np.sum(variance)), _VARIANCE_FLOOR)


def _forecast_dip_garch_t(rets: Array, rv: Array, park: Array, h: int, seed: int) -> float:
    """``dip_garch_t`` — GARCH(1,1)-t (Constant mean), the quantile_signals spec."""
    del rv, park, seed
    from arch import arch_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = arch_model(
            np.asarray(rets, dtype=float) * 100.0,
            mean="Constant",
            vol="GARCH",
            p=1,
            o=0,
            q=1,
            dist="t",
            rescale=False,
        ).fit(disp="off", show_warning=False, options={"maxiter": 300})
    if not np.isfinite(fit.params).all():
        raise ValueError("garch_t fit produced non-finite parameters")
    forecast = fit.forecast(horizon=h, method="analytic", reindex=False)
    variance = np.asarray(forecast.variance, dtype=float)[-1] / 100.0**2
    if variance.shape != (h,) or not np.isfinite(variance).all() or (variance <= 0.0).any():
        raise ValueError("garch_t returned a degenerate variance path")
    return max(float(np.sum(variance)), _VARIANCE_FLOOR)


VOL_MODEL_REGISTRY: dict[str, VolForecaster] = {
    "rv_roll": _forecast_rv_roll,
    "rv_ewma": _forecast_rv_ewma,
    "har": _forecast_har,
    "realized_garch": _forecast_realized_garch,
    "dip_garch_t": _forecast_dip_garch_t,
}


def resolve_vol_models(names: Iterable[str] | None = None) -> dict[str, VolForecaster]:
    """Resolve model names against ``VOL_MODEL_REGISTRY`` (default: all)."""
    if names is None:
        return dict(VOL_MODEL_REGISTRY)
    resolved: dict[str, VolForecaster] = {}
    for raw in names:
        name = str(raw).strip()
        if not name:
            continue
        if name not in VOL_MODEL_REGISTRY:
            raise ValueError(f"unknown vol model {name!r}")
        resolved[name] = VOL_MODEL_REGISTRY[name]
    if not resolved:
        raise ValueError("vol bench requires at least one model")
    return resolved


def _check_int(value: int, name: str, lo: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if int(value) < lo:
        raise ValueError(f"{name} must be >= {lo}")
    return int(value)


def _check_horizons(horizons: Sequence[int]) -> tuple[int, ...]:
    resolved = tuple(sorted({_check_int(h, "horizons", 1) for h in horizons}))
    if not resolved:
        raise ValueError("horizons must be a nonempty set of positive integers")
    return resolved


def _stable_metric(value: float) -> float:
    from quant_fund.utils.numeric import stable_decimal

    # Eight significant figures match the committed vol-bench replay carrier.
    # The in-memory frame keeps full precision for the closed-form check;
    # only the sealed results use this round.
    return stable_decimal(value)


_SEALED_METRIC_KEYS = (
    "qlike",
    "mse",
    "dm_qlike_mean",
    "dm_qlike_se",
    "dm_qlike_t",
    "epromotion_evalue",
    "epromotion_anytime_p",
)


def _sealed_result_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Copy result rows with libm-sensitive metrics rounded for the seal."""
    sealed: list[dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        for key in _SEALED_METRIC_KEYS:
            value = copy.get(key)
            if isinstance(value, float):
                copy[key] = _stable_metric(value)
        sealed.append(copy)
    return sealed


def _validate_shard(shard: VolShard, shard_name: str, n_bars: int) -> VolShard:
    if not isinstance(shard, VolShard):
        raise ValueError(f"shard {shard_name!r} did not return a VolShard")
    returns = np.asarray(shard.returns, dtype=float).reshape(-1)
    rv = np.asarray(shard.rv, dtype=float).reshape(-1)
    parkinson = np.asarray(shard.parkinson, dtype=float).reshape(-1)
    if shard.name != shard_name or shard.config.get("data_label") != "SYNTHETIC":
        raise ValueError(f"shard {shard_name!r} must match its name and SYNTHETIC label")
    for label, series in (("returns", returns), ("rv", rv), ("parkinson", parkinson)):
        if series.shape != (n_bars,):
            raise ValueError(
                f"shard {shard_name!r} produced {series.size} {label} rows; needs exactly {n_bars}"
            )
        if not np.isfinite(series).all():
            raise ValueError(f"shard {shard_name!r} produced non-finite {label}")
    if (rv <= 0.0).any() or (parkinson <= 0.0).any():
        raise ValueError(f"shard {shard_name!r} produced non-positive rv/parkinson")
    from quant_fund.utils.numeric import stable_decimal_array

    # Hashing uses tobytes(); collapse libm ulps before the seal and the fit.
    return VolShard(
        shard.name,
        stable_decimal_array(returns),
        stable_decimal_array(rv),
        stable_decimal_array(parkinson),
        dict(shard.config),
    )


def _eval_shard_model(
    shard: VolShard,
    forecaster: VolForecaster,
    horizon: int,
    origins: Array,
    shard_seed: int,
) -> tuple[Array, Array] | str:
    """Return (targets, forecasts) over origins, or an error string."""
    targets = np.empty(origins.size, dtype=float)
    forecasts = np.empty(origins.size, dtype=float)
    try:
        for i, origin in enumerate(origins):
            o = int(origin)
            forecasts[i] = forecaster(
                shard.returns[:o],
                shard.rv[:o],
                shard.parkinson[:o],
                horizon,
                shard_seed + o,
            )
            if not np.isfinite(forecasts[i]) or forecasts[i] <= 0.0:
                raise ValueError(f"forecast {forecasts[i]!r} at origin {o} is not finite/positive")
            targets[i] = float(np.sum(shard.rv[o : o + horizon]))
        return targets, forecasts
    except Exception as exc:  # recorded as an error row, never silent
        return str(exc)


def _resolve_forecasters(
    models: Mapping[str, VolForecaster] | Iterable[str] | None,
) -> dict[str, VolForecaster]:
    """Normalize the model selector to a ``name -> forecaster`` map."""
    if isinstance(models, Mapping):
        return dict(models)
    return resolve_vol_models(models)


def run_vol_bench(
    models: Mapping[str, VolForecaster] | Iterable[str] | None = None,
    shards: Iterable[str] | Mapping[str, VolShardGenerator] | None = None,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    min_history: int = 300,
    n_origins: int = 24,
    stride: int | None = None,
    n_bars: int | None = None,
    seed: int = 0,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Walk-forward vol bench: QLIKE + MSE on h-step cumulative realized variance.

    Scores are proper rules only; per-origin loss differentials vs the
    ``har`` reference use a Newey-West ``vol_loss_diff`` (negative
    ``dm_qlike_mean`` = model beats the reference). A model that fails
    at an origin records an ``error`` row (visible, never silent); the
    harness itself fails closed on degenerate arguments or shard output.
    Returns the results frame plus the unsealed receipt payload.
    """
    forecasters = _resolve_forecasters(models)
    if not forecasters:
        raise ValueError("vol bench requires at least one model")
    for name in forecasters:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("model names must be nonempty strings")

    if isinstance(shards, Mapping):
        resolved_shards = shards
    else:
        resolved_shards = resolve_vol_shard_generators(shards)
    if not resolved_shards:
        raise ValueError("vol bench requires at least one shard")
    for name in resolved_shards:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("shard names must be nonempty strings")

    horizon_set = _check_horizons(horizons)
    min_history = _check_int(min_history, "min_history", 30)
    n_origins = _check_int(n_origins, "n_origins", 10)
    stride_eff = max(horizon_set) if stride is None else _check_int(stride, "stride", 1)
    required_bars = min_history + (n_origins - 1) * stride_eff + max(horizon_set)
    n_bars = required_bars if n_bars is None else _check_int(n_bars, "n_bars", 1)
    if n_bars < required_bars:
        raise ValueError(f"n_bars={n_bars} cannot host the schedule; needs >= {required_bars}")

    columns = [
        "shard",
        "model",
        "horizon",
        "status",
        "error",
        "seed",
        "n_origins",
        "qlike",
        "mse",
        "dm_qlike_mean",
        "dm_qlike_se",
        "dm_qlike_t",
        "epromotion_evalue",
        "epromotion_origin",
        "epromotion_anytime_p",
    ]
    rows: list[dict[str, Any]] = []
    shard_meta: dict[str, Any] = {}
    for shard_index, (shard_name, generator) in enumerate(resolved_shards.items()):
        shard_seed = int(seed) + shard_index
        shard = _validate_shard(generator(n_bars, shard_seed), shard_name, n_bars)
        shard_meta[shard_name] = {
            "n": int(n_bars),
            "seed": shard_seed,
            "returns_sha256": hash_bytes(shard.returns.tobytes()),
            "rv_sha256": hash_bytes(shard.rv.tobytes()),
            "parkinson_sha256": hash_bytes(shard.parkinson.tobytes()),
            "config": shard.config,
        }
        for horizon in horizon_set:
            origins = build_origins(
                n_dates=n_bars,
                h=horizon,
                min_history=min_history,
                stride=stride_eff,
                n_origins=n_origins,
            )
            losses: dict[str, tuple[Array, Array]] = {}
            row_refs: dict[str, dict[str, Any]] = {}
            for model_name, forecaster in forecasters.items():
                row: dict[str, Any] = {
                    "shard": shard_name,
                    "model": model_name,
                    "horizon": horizon,
                    "status": "ok",
                    "error": None,
                    "seed": shard_seed,
                    "n_origins": int(origins.size),
                    "qlike": None,
                    "mse": None,
                    "dm_qlike_mean": None,
                    "dm_qlike_se": None,
                    "dm_qlike_t": None,
                    "epromotion_evalue": None,
                    "epromotion_origin": None,
                    "epromotion_anytime_p": None,
                }
                scored = _eval_shard_model(shard, forecaster, horizon, origins, shard_seed)
                if isinstance(scored, str):
                    row["status"] = "error"
                    row["error"] = scored
                else:
                    targets, forecasts = scored
                    losses[model_name] = (targets, forecasts)
                    row["qlike"] = float(np.mean(qlike_loss(targets, forecasts)))
                    row["mse"] = float(np.mean(mse_loss(targets, forecasts)))
                rows.append(row)
                row_refs[model_name] = row
            if DM_REFERENCE in losses:
                ref_targets, ref_forecasts = losses[DM_REFERENCE]
                for model_name, (targets, forecasts) in losses.items():
                    if model_name == DM_REFERENCE:
                        continue
                    diff = vol_loss_diff(targets, forecasts, ref_forecasts, loss="qlike")
                    row_refs[model_name]["dm_qlike_mean"] = float(diff["mean_diff"])
                    row_refs[model_name]["dm_qlike_se"] = float(diff["se"])
                    row_refs[model_name]["dm_qlike_t"] = float(diff["t"])
                    # Anytime-valid promotion: per-origin e-process on the
                    # QLIKE stream vs the HAR incumbent (challenger=model).
                    proc = LossEProcess(alpha=0.05, init_scale=_EPROMOTION_INIT_SCALE)
                    for c_loss, b_loss in zip(
                        qlike_loss(targets, forecasts).tolist(),
                        qlike_loss(ref_targets, ref_forecasts).tolist(),
                        strict=True,
                    ):
                        proc.update(c_loss, b_loss)
                    final_state = proc.states[-1]
                    row_refs[model_name]["epromotion_evalue"] = float(final_state.evalue)
                    row_refs[model_name]["epromotion_origin"] = proc.promotion_origin
                    row_refs[model_name]["epromotion_anytime_p"] = float(final_state.anytime_p)

    frame = pl.DataFrame(
        rows,
        schema={
            "shard": pl.String,
            "model": pl.String,
            "horizon": pl.Int64,
            "status": pl.String,
            "error": pl.String,
            "seed": pl.Int64,
            "n_origins": pl.Int64,
            "qlike": pl.Float64,
            "mse": pl.Float64,
            "dm_qlike_mean": pl.Float64,
            "dm_qlike_se": pl.Float64,
            "dm_qlike_t": pl.Float64,
            "epromotion_evalue": pl.Float64,
            "epromotion_origin": pl.Int64,
            "epromotion_anytime_p": pl.Float64,
        },
        orient="row",
    ).select(columns)

    shard_digests = {
        name: {
            "returns_sha256": meta["returns_sha256"],
            "rv_sha256": meta["rv_sha256"],
            "parkinson_sha256": meta["parkinson_sha256"],
        }
        for name, meta in shard_meta.items()
    }
    inputs_sha256 = hash_bytes(
        canonical_json_bytes(
            {
                "shards": shard_digests,
                "models": sorted(str(k) for k in forecasters),
                "horizons": [int(h) for h in horizon_set],
                "min_history": min_history,
                "n_origins": n_origins,
                "stride": stride_eff,
                "n_bars": n_bars,
                "seed": int(seed),
            }
        )
    )
    # Corpus-level fingerprint: digest over the evaluated stream content only —
    # receipts across lanes that evaluated the same shard set agree on it,
    # which is what the cross-receipt lattice edges on.
    dataset_sha256 = hash_bytes(canonical_json_bytes({"shards": shard_digests}))
    receipt: dict[str, Any] = {
        "schema": VOL_BENCH_SCHEMA,
        "kind": "vol_bench",
        "claim": "research_only",
        "data_label": "SYNTHETIC",
        "live_pnl_claim": False,
        "meta": {
            "generated_at": datetime.now(UTC).isoformat(),
            "git_revision": git_revision(),
        },
        "seed": int(seed),
        "horizons": [int(h) for h in horizon_set],
        "min_history": min_history,
        "n_origins": n_origins,
        "stride": stride_eff,
        "n_bars": n_bars,
        "target": "cumulative_realized_variance",
        "dm_reference": DM_REFERENCE,
        "models": sorted(str(k) for k in forecasters),
        "shards": shard_meta,
        "inputs_sha256": inputs_sha256,
        "dataset_sha256": dataset_sha256,
        "n_rows": len(rows),
        "n_error_rows": sum(1 for row in rows if row["status"] != "ok"),
        # The frame keeps full precision for the closed-form QLIKE check.
        # The sealed results are decimal-stable so the replay hash does not
        # follow libm.
        "results": _sealed_result_rows(rows),
    }
    return frame, receipt


def _atomic_write_text(path: Path, content: str) -> None:
    """Publish a complete immutable text artifact without replacing an existing one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise FileExistsError(f"receipt path is a symlink: {path}")
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"receipt already exists with different content: {path}")
        return
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if path.is_symlink() or path.read_text(encoding="utf-8") != content:
                raise FileExistsError(
                    f"receipt already exists with different content: {path}"
                ) from None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def vol_bench_contract_errors(receipt: Mapping[str, Any]) -> list[str]:
    """Fail-closed contract for a ``vol_bench.v1`` payload (writer + verifier)."""
    research_blob = {key: value for key, value in receipt.items() if key != "live_pnl_claim"}
    errors: list[str] = []
    if receipt.get("schema") != VOL_BENCH_SCHEMA:
        errors.append("schema_not_vol_bench_v1")
    if receipt.get("kind") != "vol_bench":
        errors.append("kind_not_vol_bench")
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


def vol_bench_dataset_identity(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Shard content digests bound by a v2 ``dataset_hash``."""
    shards = receipt.get("shards")
    if not isinstance(shards, Mapping):
        raise ValueError("vol-bench receipt has no shards block")
    dataset: dict[str, Any] = {}
    for name, meta in shards.items():
        if not isinstance(meta, Mapping):
            raise ValueError(f"vol-bench shard {name!r} metadata is not an object")
        dataset[str(name)] = {
            "returns_sha256": meta.get("returns_sha256"),
            "rv_sha256": meta.get("rv_sha256"),
            "parkinson_sha256": meta.get("parkinson_sha256"),
        }
    return dataset


def vol_bench_params(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """The run parameters bound by a v2 ``params_hash``."""
    return {
        "seed": receipt.get("seed"),
        "horizons": receipt.get("horizons"),
        "min_history": receipt.get("min_history"),
        "n_origins": receipt.get("n_origins"),
        "stride": receipt.get("stride"),
        "n_bars": receipt.get("n_bars"),
        "models": receipt.get("models"),
        "dm_reference": receipt.get("dm_reference"),
    }


def vol_bench_verdict(receipt: Mapping[str, Any]) -> str:
    """pass iff every shard/model/horizon cell scored without error."""
    n_error_rows = receipt.get("n_error_rows")
    return "pass" if n_error_rows == 0 else "fail"


def vol_bench_receipt_v2(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap a ``vol_bench.v1`` payload in the unified ``receipt.v2`` envelope.

    The v1 payload is embedded verbatim under ``payload``; the envelope binds
    the shard content digests, run params, this module's source hash, and the
    loaded numeric stack. Validates the v1 contract first — a malformed v1
    receipt is never wrapped.
    """
    from quant_fund.research.receipt_v2 import build_receipt_v2

    body = {key: value for key, value in receipt.items() if key != "meta"}
    if vol_bench_contract_errors(body):
        raise ValueError("vol-bench receipt violates its synthetic research contract")
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
        dataset=vol_bench_dataset_identity(body),
        params=vol_bench_params(body),
        code_files=(Path(__file__),),
        verdict=vol_bench_verdict(body),
        payload=dict(body),
        generated_at=None if generated_at is None else str(generated_at),
        revision=None if revision is None else str(revision),
    )


def vol_bench_v2_consistency_errors(envelope: Mapping[str, Any]) -> list[str]:
    """Re-derive a vol-bench receipt.v2 envelope's bound digests from its payload."""
    errors: list[str] = []
    payload = envelope.get("payload")
    if not isinstance(payload, Mapping):
        return ["payload_not_object"]
    contract_errors = vol_bench_contract_errors(payload)
    errors.extend(f"payload_{name}" for name in contract_errors)
    if contract_errors:
        return errors
    try:
        dataset = vol_bench_dataset_identity(payload)
    except ValueError as exc:
        return [*errors, f"payload_{exc}"]
    if hash_bytes(canonical_json_bytes(dataset)) != envelope.get("dataset_hash"):
        errors.append("dataset_hash_mismatch")
    if hash_bytes(canonical_json_bytes(vol_bench_params(payload))) != envelope.get("params_hash"):
        errors.append("params_hash_mismatch")
    if vol_bench_verdict(payload) != envelope.get("verdict"):
        errors.append("verdict_mismatch")
    return errors


def write_vol_bench_receipt(
    receipt: Mapping[str, Any],
    receipts_dir: Path | str = Path("receipts"),
    *,
    receipt_version: int = 1,
) -> Path:
    """Seal a vol-bench receipt and write ``receipts/vol_bench_<hash>.json``.

    The filename hash is the sha256 of the canonical receipt payload; the
    same digest is embedded as ``receipt_sha256`` (the fleet_eval seal
    convention). The write is atomic and fail-closed on tampering.
    ``receipt_version=2`` wraps the v1 payload in the unified ``receipt.v2``
    envelope before sealing.
    """
    from quant_fund.research.receipt_v2 import seal_receipt

    if receipt_version == 1:
        body = {key: value for key, value in receipt.items() if key != "meta"}
        if vol_bench_contract_errors(body):
            raise ValueError("vol-bench receipt violates its synthetic research contract")
    elif receipt_version == 2:
        body = vol_bench_receipt_v2(receipt)
    else:
        raise ValueError(f"receipt_version must be 1 or 2, got {receipt_version!r}")
    payload = seal_receipt(body)
    path = Path(receipts_dir) / f"vol_bench_{payload['receipt_sha256'][:16]}.json"
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


__all__ = [
    "DM_REFERENCE",
    "VOL_BENCH_SCHEMA",
    "VOL_MODEL_REGISTRY",
    "VOL_SHARD_GENERATORS",
    "VolForecaster",
    "VolShard",
    "VolShardGenerator",
    "break_vol",
    "garch_vol",
    "resolve_vol_models",
    "resolve_vol_shard_generators",
    "rough_vol",
    "run_vol_bench",
    "vol_bench_contract_errors",
    "vol_bench_dataset_identity",
    "vol_bench_params",
    "vol_bench_receipt_v2",
    "vol_bench_v2_consistency_errors",
    "vol_bench_verdict",
    "write_vol_bench_receipt",
]
