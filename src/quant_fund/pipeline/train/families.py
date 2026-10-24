"""Remaining forecast-family trainers or scientific benches.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.config.models import AppConfig
from quant_fund.metrics.risk import losses_from_returns
from quant_fund.models.alpha import HistoricalMeanAlpha
from quant_fund.models.base import artifact_identity, load_joblib_artifact
from quant_fund.models.deep_rl import PolicyGradientRanker, run_policy_gradient_panel
from quant_fund.models.quantile_bandit import QuantileThompson
from quant_fund.models.regime import GaussianHMMRegime, SingleStateRegime, VolThresholdRegime
from quant_fund.models.rl import (
    LinearThompsonRanker,
    LinUCBRanker,
    run_linucb_panel,
    run_thompson_panel,
)
from quant_fund.models.tail import DrawdownClassifier, GaussianTail, HistoricalTail
from quant_fund.pipeline.artifact_manifest import (
    UNSUPERVISED_LABEL,
    ArtifactManifestError,
    identity_for_training,
    identity_from_artifact,
    save_training_artifact,
    verify_artifact_manifest,
)
from quant_fund.pipeline.dataset import design_matrix, panel
from quant_fund.registry.mlflow_store import attach_artifact_identity, configure_tracking, log_run
from quant_fund.reporting.report import write_evidence_report
from quant_fund.utils.hashing import canonical_frame_fingerprint, canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision, git_worktree_sha256
from quant_fund.utils.seeds import set_global_seed

from .distribution import train_distribution, train_distribution_auto
from .ranking import RANKING_MODEL_NAMES, train_calibration, train_ranking
from .splits import _aligned_label_end_times, _label_horizon, _require_model, _walk_forward_splits
from .volatility import train_volatility, train_volatility_auto


def train_alpha(config: AppConfig, model_name: str = "ridge") -> dict[str, Any]:
    _require_model(
        model_name,
        {"mean", *RANKING_MODEL_NAMES},
        "alpha",
    )
    if model_name == "mean":
        # reuse ranking path with historical mean
        label = config.train.ranking_target
        df = panel(config, label=label)
        x, y, _, feats, _ = design_matrix(df, label)
        m = HistoricalMeanAlpha().fit(x, y)
        path = Path(config.data.root) / "metadata" / "alpha_mean.joblib"
        save_training_artifact(
            m,
            path,
            identity=identity_for_training(
                df,
                config=config,
                label=label,
                features=feats,
                label_horizon_bars=_label_horizon(label),
            ),
        )
        return {"metrics": {"mean": float(m.mean_)}, "path": str(path)}
    return train_ranking(config, "ridge" if model_name == "ridge" else model_name)


def train_regime(config: AppConfig, model_name: str = "hmm") -> dict[str, Any]:
    _require_model(model_name, {"hmm", "threshold", "single", "single_state"}, "regime")
    df = panel(config)
    cols = [c for c in ["mkt_ret_1", "mkt_vol_20", "cs_dispersion", "breadth"] if c in df.columns]
    sub = df.select(["event_time", *cols]).unique("event_time").drop_nulls().sort("event_time")
    dates = sub["event_time"].to_numpy()
    x = sub.select(cols).to_numpy().astype(float)

    def make_model() -> Any:
        if model_name == "hmm":
            return GaussianHMMRegime(config.train.n_hmm_states, config.train.random_seed)
        if model_name == "threshold":
            return VolThresholdRegime()
        if model_name in {"single", "single_state"}:
            return SingleStateRegime()
        raise ValueError(f"unknown regime model {model_name!r}")

    heldout_ll: list[float] = []
    for train_mask, test_mask in _walk_forward_splits(dates, config, horizon_bars=1):
        if not train_mask.any() or not test_mask.any():
            continue
        fold_model = make_model().fit(x[train_mask])
        if model_name == "hmm":
            heldout_ll.append(float(fold_model.aic_bic(x[test_mask])["avg_ll"]))
    if not heldout_ll and model_name == "hmm":
        raise ValueError("walk-forward training produced no trainable/evaluable fold")

    # Persist a production model fit on all labeled history; all reported HMM
    # likelihood is from untouched fold test blocks, never this refit.
    m = make_model().fit(x)
    extra = {"oos_avg_ll": float(np.mean(heldout_ll))} if heldout_ll else {}
    path = Path(config.data.root) / "metadata" / f"regime_{model_name}.joblib"
    save_training_artifact(
        m,
        path,
        identity=identity_for_training(
            df,
            config=config,
            label=UNSUPERVISED_LABEL,
            features=cols,
            label_horizon_bars=1,
        ),
    )
    return {"metrics": extra, "path": str(path), "labels": getattr(m, "labels", {})}


def train_tail(config: AppConfig, model_name: str = "historical") -> dict[str, Any]:
    _require_model(model_name, {"historical", "gaussian", "drawdown"}, "tail")
    df = panel(config)
    return_labels = [c for c in df.columns if c.startswith("future_return")]
    if not return_labels:
        raise ValueError("gold panel has no future_return* label column for tail training")
    label = return_labels[0]
    if model_name == "drawdown":
        event_labels = [c for c in df.columns if c.startswith("future_tail_event")]
        label = event_labels[0] if event_labels else label
    x, y, dates, tail_features, _ = design_matrix(df, label)
    label_end_times = _aligned_label_end_times(df, label, tail_features)

    def make_model() -> Any:
        if model_name == "gaussian":
            return GaussianTail()
        if model_name == "drawdown":
            return DrawdownClassifier()
        if model_name == "historical":
            return HistoricalTail()
        raise ValueError(f"unknown tail model {model_name!r}")

    breach_rates: list[float] = []
    es_excessions: list[float] = []
    brier_scores: list[float] = []
    for train_mask, test_mask in _walk_forward_splits(
        dates,
        config,
        horizon_bars=_label_horizon(label),
        label_end_times=label_end_times,
    ):
        if not train_mask.any() or not test_mask.any():
            continue
        fold_model = make_model().fit(x[train_mask], y[train_mask])
        if model_name == "drawdown":
            pred = np.asarray(fold_model.predict(x[test_mask]), dtype=float)
            brier_scores.append(float(np.mean((pred - (y[test_mask] > 0.5)) ** 2)))
            continue
        var, es = fold_model.predict_var_es()
        losses = losses_from_returns(y[test_mask])
        breaches = losses >= var
        breach_rates.append(float(np.mean(breaches)))
        es_losses = losses[breaches]
        if es_losses.size:
            es_excessions.append(float(np.mean(es_losses) - es))
    if not (breach_rates or brier_scores):
        raise ValueError("walk-forward training produced no trainable/evaluable fold")

    # Persist a final estimator fit on all available history; metrics remain OOS.
    m = make_model().fit(x, y)
    path = Path(config.data.root) / "metadata" / f"tail_{model_name}.joblib"
    save_training_artifact(
        m,
        path,
        identity=identity_for_training(
            df,
            config=config,
            label=label,
            features=tail_features,
            label_horizon_bars=_label_horizon(label),
        ),
    )
    metrics: dict[str, float] = {}
    if breach_rates:
        var, es = m.predict_var_es()
        metrics = {
            "oos_var_breach_rate": float(np.mean(breach_rates)),
            "oos_es_excess": float(np.mean(es_excessions)) if es_excessions else float("nan"),
            "var": var,
            "es": es,
        }
    else:
        metrics = {"oos_brier": float(np.mean(brier_scores))}
    return {"metrics": metrics, "path": str(path)}


def train_reinforcement(config: AppConfig, model_name: str = "linucb") -> dict[str, Any]:
    """Train the paper-only contextual ranking policy on a causal gold panel.

    This is an online contextual-bandit evaluation, not a backtest: rewards are
    forecast targets and the trace never represents executable P&L.  The policy
    sees each decision date once, selects top-k rows, then updates only from the
    realized labels for that date.  The persisted artifact is the final policy
    state and all reported metrics are descriptive research diagnostics.
    """
    if model_name == "auto":
        return train_reinforcement_auto(config)
    if model_name == "policy_gradient":
        return train_policy_gradient(config)
    if model_name == "quantile_thompson":
        return train_quantile_bandit(config)
    if model_name not in {"linucb", "thompson"}:
        raise ValueError(f"unknown reinforcement model {model_name!r}")
    set_global_seed(config.train.random_seed)
    label = config.train.ranking_target
    df = panel(config, label=label)
    x, y, dates, features, _ = design_matrix(df, label)
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("reinforcement training requires a non-empty feature panel")
    # Oracle is diagnostic-only: it is the realized same-date reward ordering,
    # never an input to the policy.
    requested_top_k = max(1, int(config.constraints.max_positions or 3))
    # LinUCB's panel evaluator requires at least 2*k arms per date so that the
    # selected set is not the entire cross-section. Cap k for narrow universes.
    n_arms = int(
        np.max([np.sum(np.asarray(dates) == date) for date in set(np.asarray(dates).tolist())])
    )
    top_k = min(requested_top_k, max(1, n_arms // 2))
    runner = run_linucb_panel if model_name == "linucb" else run_thompson_panel
    trace = runner(
        x,
        y,
        dates,
        oracle=y,
        top_k=top_k,
        alpha=1.0,
        seed=config.train.random_seed,
    )
    if trace.policy_reward.size == 0:
        raise ValueError("reinforcement training produced no evaluable decision dates")
    policy_mean = float(np.mean(trace.policy_reward))
    uniform_mean = float(np.mean(trace.uniform_reward))
    oracle_mean = float(np.mean(trace.oracle_reward))
    metrics = {
        "mean_policy_reward": policy_mean,
        "mean_uniform_reward": uniform_mean,
        "mean_oracle_reward": oracle_mean,
        "mean_advantage_vs_uniform": policy_mean - uniform_mean,
        "mean_regret_vs_oracle": float(np.mean(trace.cumulative_regret)),
        "n_dates": float(len(trace.dates)),
        "n_features": float(x.shape[1]),
    }
    policy = (
        LinUCBRanker(x.shape[1], alpha=1.0)
        if model_name == "linucb"
        else LinearThompsonRanker(x.shape[1], alpha=1.0, seed=config.train.random_seed)
    )
    # Refit the final state causally in the same order used for evaluation.
    for date in sorted(set(np.asarray(dates).tolist())):
        mask = np.asarray(dates) == date
        finite = np.isfinite(y[mask]) & np.isfinite(x[mask]).all(axis=1)
        if not finite.any():
            continue
        scores = policy.scores(x[mask][finite])
        for row in np.argsort(scores)[-top_k:]:
            policy.update(x[mask][finite][row], float(y[mask][finite][row]))
    configure_tracking()
    run_id = log_run(
        family="reinforcement",
        name=model_name,
        params={"features": features, "label": label, "model": model_name},
        metrics=metrics,
        tags={"data": config.data.source, "claim": "research_only"},
    )
    path = Path(config.data.root) / "metadata" / f"rl_{model_name}.joblib"
    save_training_artifact(
        {"policy": policy, "policy_name": model_name, "features": features, "label": label},
        path,
        identity=identity_for_training(
            df,
            config=config,
            label=label,
            features=features,
            label_horizon_bars=_label_horizon(label),
        ),
    )
    return {
        "metrics": metrics,
        "run_id": run_id,
        "path": str(path),
        "research_only": True,
        "live_pnl_claim": False,
        "data_source": "SYNTHETIC" if config.data.source == "synthetic" else config.data.source,
    }


def train_reinforcement_auto(config: AppConfig) -> dict[str, Any]:
    """Select a contextual policy by chronological research diagnostics."""
    candidates = ("linucb", "thompson", "quantile_thompson", "policy_gradient")
    results: list[dict[str, Any]] = []
    unavailable: dict[str, str] = {}
    for candidate in candidates:
        try:
            results.append(train_reinforcement(config, candidate))
        except ImportError as exc:
            if candidate != "policy_gradient":
                raise
            unavailable[candidate] = str(exc)
    viable = [
        result
        for result in results
        if np.isfinite(float(result["metrics"].get("mean_advantage_vs_uniform", np.nan)))
    ]
    if not viable:
        raise ValueError("automatic RL selection produced no finite candidate metric")
    selected = max(
        viable,
        key=lambda result: float(result["metrics"]["mean_advantage_vs_uniform"]),
    )
    selected_payload = load_joblib_artifact(Path(str(selected["path"])))
    if not isinstance(selected_payload, dict) or "policy" not in selected_payload:
        raise ValueError("selected RL artifact is malformed")
    selected_name = Path(str(selected["path"])).stem.removeprefix("rl_")
    selected_payload["policy_name"] = selected_name
    auto_path = Path(config.data.root) / "metadata" / "rl_auto.joblib"
    save_training_artifact(
        selected_payload,
        auto_path,
        identity=identity_from_artifact(Path(str(selected["path"]))),
    )
    candidate_diagnostics = {
        Path(str(result["path"])).stem.removeprefix("rl_"): result["metrics"] for result in results
    }
    candidate_diagnostics.update(
        {name: {"unavailable": reason} for name, reason in unavailable.items()}
    )
    return {
        **selected,
        "path": str(auto_path),
        "model": "auto",
        "selected_model": selected_name,
        "selection_metric": "mean_advantage_vs_uniform",
        "candidates": candidate_diagnostics,
    }


def train_quantile_bandit(config: AppConfig) -> dict[str, Any]:
    """Train and persist the quantile-Thompson contextual bandit."""
    set_global_seed(config.train.random_seed)
    label = config.train.ranking_target
    df = panel(config, label=label)
    x, y, dates, features, _ = design_matrix(df, label)
    top_k = max(1, int(config.constraints.max_positions or 3))
    n_dates = max(len(set(dates.tolist())), 1)
    arms_per_date = max(1, int(x.shape[0] // n_dates))
    top_k = min(top_k, max(1, arms_per_date // 2))
    bandit = QuantileThompson(seed=config.train.random_seed)
    trace = bandit.run_panel(x, y, dates, k=top_k)
    if not trace.policy_reward.size:
        raise ValueError("quantile bandit training produced no evaluable dates")
    metrics = {
        "mean_policy_reward": float(np.mean(trace.policy_reward)),
        "mean_uniform_reward": float(np.mean(trace.uniform_reward)),
        "mean_oracle_reward": float(np.mean(trace.oracle_reward)),
        "mean_advantage_vs_uniform": float(
            np.mean(trace.policy_reward) - np.mean(trace.uniform_reward)
        ),
        "mean_regret_vs_oracle": float(np.mean(trace.cumulative_regret)),
        "n_dates": float(len(trace.dates)),
    }
    configure_tracking()
    run_id = log_run(
        family="reinforcement",
        name="quantile_thompson",
        params={"features": features, "label": label, "model": "quantile_thompson"},
        metrics=metrics,
        tags={"data": config.data.source, "claim": "research_only"},
    )
    path = Path(config.data.root) / "metadata" / "rl_quantile_thompson.joblib"
    save_training_artifact(
        {
            "policy": bandit,
            "policy_name": "quantile_thompson",
            "features": features,
            "label": label,
        },
        path,
        identity=identity_for_training(
            df,
            config=config,
            label=label,
            features=features,
            label_horizon_bars=_label_horizon(label),
        ),
    )
    return {
        "metrics": metrics,
        "run_id": run_id,
        "path": str(path),
        "features": features,
        "research_only": True,
        "live_pnl_claim": False,
        "data_source": "SYNTHETIC" if config.data.source == "synthetic" else config.data.source,
    }


def train_policy_gradient(config: AppConfig) -> dict[str, Any]:
    """Train and persist the optional PyTorch contextual policy."""
    set_global_seed(config.train.random_seed)
    label = config.train.ranking_target
    df = panel(config, label=label)
    x, y, dates, features, _ = design_matrix(df, label)
    top_k = max(1, int(config.constraints.max_positions or 3))
    n_dates = max(len(set(dates.tolist())), 1)
    arms_per_date = max(1, int(x.shape[0] // n_dates))
    effective_top_k = min(top_k, max(1, arms_per_date // 2))
    trace = run_policy_gradient_panel(
        x,
        y,
        dates,
        top_k=effective_top_k,
        seed=config.train.random_seed,
    )
    if not trace.policy_reward.size:
        raise ValueError("policy-gradient training produced no evaluable dates")
    metrics = {
        "mean_policy_reward": float(np.mean(trace.policy_reward)),
        "mean_uniform_reward": float(np.mean(trace.uniform_reward)),
        "mean_advantage_vs_uniform": float(
            np.mean(trace.policy_reward) - np.mean(trace.uniform_reward)
        ),
        "mean_regret_vs_uniform": float(np.mean(trace.cumulative_regret)),
        "n_dates": float(len(trace.dates)),
        "n_features": float(x.shape[1]),
    }
    model = PolicyGradientRanker(x.shape[1], seed=config.train.random_seed).fit(
        x, y, dates, top_k=effective_top_k
    )
    configure_tracking()
    run_id = log_run(
        family="reinforcement",
        name="policy_gradient",
        params={"features": features, "label": label, "model": "policy_gradient"},
        metrics=metrics,
        tags={"data": config.data.source, "claim": "research_only"},
    )
    path = Path(config.data.root) / "metadata" / "rl_policy_gradient.joblib"
    save_training_artifact(
        {
            "policy": model,
            "policy_name": "policy_gradient",
            "features": features,
            "label": label,
        },
        path,
        identity=identity_for_training(
            df,
            config=config,
            label=label,
            features=features,
            label_horizon_bars=_label_horizon(label),
        ),
    )
    return {
        "metrics": metrics,
        "run_id": run_id,
        "path": str(path),
        "features": features,
        "research_only": True,
        "live_pnl_claim": False,
        "data_source": "SYNTHETIC" if config.data.source == "synthetic" else config.data.source,
    }


def train_robinhood_plus(
    config: AppConfig, model_name: str = "hierarchical_markov"
) -> dict[str, Any]:
    """Persist the robinhood+ engine card. The tokenizer is unsupervised."""
    _require_model(model_name, {"hierarchical_markov", "transformer"}, "robinhood_plus")
    from quant_fund.models.robinhood_plus.bench import bench_robinhood_plus
    from quant_fund.models.robinhood_plus.constants import KLINE_FEATURE_NAMES
    from quant_fund.models.robinhood_plus.engine import RobinhoodPlusEngine

    engine = RobinhoodPlusEngine(
        lookback=config.robinhood_plus.lookback,
        pred_len=config.robinhood_plus.pred_len,
        sample_count=config.robinhood_plus.sample_count,
        s1_bits=config.robinhood_plus.s1_bits,
        s2_bits=config.robinhood_plus.s2_bits,
        seed=config.train.random_seed,
        decoder=model_name,
    )
    engine.fit(np.zeros((2, 1)), np.zeros(2))
    path = Path(config.data.root) / "metadata" / f"robinhood_plus_{model_name}.joblib"
    df = panel(config)
    # The card is an unsupervised K-line tokenizer, not a tabular ranker.
    # Ranking-zoo columns are often absent on a price panel and must not be
    # what the artifact claims it was fit on.
    engine_features = [str(name) for name in KLINE_FEATURE_NAMES]
    save_training_artifact(
        engine,
        path,
        identity=identity_for_training(
            df,
            config=config,
            label=UNSUPERVISED_LABEL,
            features=engine_features,
            label_horizon_bars=1,
        ),
    )
    receipt = bench_robinhood_plus(df, config)
    return {
        "metrics": {
            "mean_ic": receipt.get("mean_ic"),
            "n_ok": receipt.get("n_ok"),
            "ic_n_dates": receipt.get("ic_n_dates"),
        },
        "path": str(path),
        "family": "robinhood_plus",
        "research_only": True,
        "execution_claim": "research_only",
    }


def train_family(config: AppConfig, family: str, model_name: str | None = None) -> dict[str, Any]:
    dispatch: dict[str, Callable[[], dict[str, Any]]] = {
        "ranking": lambda: train_ranking(config, model_name or "ridge"),
        "calibration": lambda: train_calibration(config, model_name or "isotonic"),
        "alpha": lambda: train_alpha(config, model_name or "ridge"),
        "distribution": lambda: (
            train_distribution_auto(config)
            if model_name == "auto"
            else train_distribution(config, model_name or "gaussian")
        ),
        "volatility": lambda: (
            train_volatility_auto(config)
            if model_name == "auto"
            else train_volatility(config, model_name or "ewma")
        ),
        "regime": lambda: train_regime(config, model_name or "hmm"),
        "tail": lambda: train_tail(config, model_name or "historical"),
        "reinforcement": lambda: train_reinforcement(config, model_name or "linucb"),
        "covariance": lambda: {"metrics": {}, "note": "covariance is estimated at forecast time"},
        "liquidity": lambda: {"metrics": {}, "note": "liquidity uses parameterized cost model"},
        "robinhood_plus": lambda: train_robinhood_plus(config, model_name or "hierarchical_markov"),
    }
    if family not in dispatch:
        raise ValueError(f"unknown family {family}")
    result = dispatch[family]()
    if config.data.source == "synthetic":
        result["data_source"] = "SYNTHETIC"
    artifact_path = result.get("path")
    if isinstance(artifact_path, str):
        try:
            result.update(artifact_identity(Path(artifact_path)))
        except ValueError as exc:
            result["manifest_valid"] = False
            result["artifact_identity_error"] = str(exc)
        # Dataset identity is verified, never assumed: an artifact whose
        # manifest carries no (or an unverified) identity is recorded loudly
        # here and blocks promotion downstream (see proof.promotion_receipt).
        try:
            identity = verify_artifact_manifest(Path(artifact_path))
            result["dataset_identity"] = identity.model_dump(mode="json")
            result["dataset_identity_valid"] = True
        except ArtifactManifestError as exc:
            result["dataset_identity_valid"] = False
            result["dataset_identity_error"] = str(exc)
        try:
            result["dataset_content_sha256"] = canonical_frame_fingerprint(panel(config))
        except (OSError, ValueError, RuntimeError) as exc:
            result["dataset_fingerprint_error"] = str(exc)
    if (
        isinstance(result.get("run_id"), str)
        and result.get("manifest_valid") is True
        and isinstance(result.get("artifact_sha256"), str)
    ):
        attach_artifact_identity(
            str(result["run_id"]),
            {
                "artifact_sha256": result["artifact_sha256"],
                "manifest_valid": result["manifest_valid"],
                "artifact_class": result.get("artifact_class"),
                "manifest_schema": result.get("manifest_schema"),
            },
        )
    report_paths = write_evidence_report(
        Path(config.data.root),
        candidates={family: result.get("metrics", {})},
        provenance={
            "data_source": result.get("data_source", config.data.source),
            "artifact_path": result.get("path"),
            "artifact_sha256": result.get("artifact_sha256"),
            "manifest_valid": result.get("manifest_valid"),
            "config_sha256": hash_bytes(canonical_json_bytes(config.model_dump(mode="json"))),
            "dataset_content_sha256": result.get("dataset_content_sha256"),
            "git_revision": git_revision(),
            "git_worktree_sha256": git_worktree_sha256(),
        },
    )
    result["evidence_report"] = {key: path.as_posix() for key, path in report_paths.items()}
    return result


__all__ = [
    "train_alpha",
    "train_family",
    "train_policy_gradient",
    "train_quantile_bandit",
    "train_regime",
    "train_reinforcement",
    "train_reinforcement_auto",
    "train_robinhood_plus",
    "train_tail",
]
