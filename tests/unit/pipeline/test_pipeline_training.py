"""Training pipeline helpers + train_family fail-closed edges."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_fund.config import load_config
from quant_fund.config.models import ValidationConfig
from quant_fund.pipeline.train import (
    _chronological_split,
    _label_horizon,
    _make_ranker,
    _walk_forward_splits,
    train_calibration,
    train_calibration_auto,
    train_distribution,
    train_distribution_auto,
    train_family,
    train_ranking,
    train_ranking_auto,
    train_regime,
    train_reinforcement,
    train_reinforcement_auto,
    train_volatility,
    train_volatility_auto,
)
from tests.support.panel_sources import materialize_panel_sources, save_bound_stub


def test_train_distribution_auto_selects_lowest_finite_pinball(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quant_fund.pipeline import train as train_module

    scores = {
        "empirical": 0.4,
        "gaussian": 0.2,
        "linear_qr": float("nan"),
        "xgboost": 0.3,
        "lightgbm": 0.25,
    }

    def fake_train(config, model_name):
        path = Path(config.data.root) / "metadata" / f"dist_{model_name}.joblib"
        save_bound_stub(config, {"features": ["f0"], "model": model_name}, path)
        return {
            "path": str(path),
            "metrics": {"mean_pinball": scores[model_name], "n_oos_rows": 20},
        }

    monkeypatch.setattr(train_module, "train_distribution", fake_train)
    cfg = load_config("configs/research.yaml")
    cfg.data.root = materialize_panel_sources(tmp_path)
    result = train_distribution_auto(cfg)
    assert result["selected_model"] == "gaussian"
    assert Path(result["path"]).name == "dist_auto.joblib"


def test_train_volatility_auto_selects_lowest_finite_qlike(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quant_fund.pipeline import train as train_module

    scores = {"rolling": 0.4, "ewma": 0.1, "har": 0.2, "xgboost": float("nan"), "lightgbm": 0.3}

    def fake_train(config, model_name):
        path = Path(config.data.root) / "metadata" / f"vol_{model_name}.joblib"
        save_bound_stub(config, {"features": ["f0"], "model": model_name}, path)
        return {"path": str(path), "metrics": {"qlike": scores[model_name], "n_oos_rows": 20}}

    monkeypatch.setattr(train_module, "train_volatility", fake_train)
    cfg = load_config("configs/research.yaml")
    cfg.data.root = materialize_panel_sources(tmp_path)
    result = train_volatility_auto(cfg)
    assert result["selected_model"] == "ewma"
    assert Path(result["path"]).name == "vol_auto.joblib"


def test_train_distribution_auto_rejects_trivially_small_oos_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quant_fund.models.base import save_joblib_artifact
    from quant_fund.pipeline import train as train_module

    def fake_train(config, model_name):
        path = Path(config.data.root) / "metadata" / f"dist_{model_name}.joblib"
        save_joblib_artifact({"model": model_name}, path)
        return {"path": str(path), "metrics": {"mean_pinball": 0.01, "n_oos_rows": 1}}

    monkeypatch.setattr(train_module, "train_distribution", fake_train)
    cfg = load_config("configs/research.yaml")
    cfg.data.root = materialize_panel_sources(tmp_path)
    with pytest.raises(ValueError, match="no finite candidate metric"):
        train_distribution_auto(cfg)


def test_train_reinforcement_auto_selects_best_finite_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quant_fund.models.rl import LinUCBRanker
    from quant_fund.pipeline import train as train_module

    advantages = {
        "linucb": 0.01,
        "thompson": 0.05,
        "quantile_thompson": float("nan"),
        "policy_gradient": 0.02,
    }

    def fake_train(config, model_name):
        path = Path(config.data.root) / "metadata" / f"rl_{model_name}.joblib"
        save_bound_stub(
            config,
            {"policy": LinUCBRanker(1), "policy_name": model_name, "features": ["f0"]},
            path,
        )
        return {
            "path": str(path),
            "metrics": {"mean_advantage_vs_uniform": advantages[model_name]},
        }

    monkeypatch.setattr(train_module, "train_reinforcement", fake_train)
    cfg = load_config("configs/research.yaml")
    cfg.data.root = materialize_panel_sources(tmp_path)
    result = train_reinforcement_auto(cfg)
    assert result["selected_model"] == "thompson"
    assert result["selection_metric"] == "mean_advantage_vs_uniform"
    assert Path(result["path"]).name == "rl_auto.joblib"
    assert Path(result["path"]).is_file()


def test_train_ranking_auto_selects_best_finite_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quant_fund.models.ranking import RidgeRanker
    from quant_fund.pipeline import train as train_module

    scores = {"ridge": 0.01, "elasticnet": 0.04, "neural": float("nan"), "ensemble": 0.02}

    def fake_train(config, model_name):
        path = Path(config.data.root) / "metadata" / f"ranker_{model_name}.joblib"
        model = RidgeRanker().fit(np.ones((4, 2)), np.arange(4, dtype=float))
        model.features = ["f0", "f1"]
        save_bound_stub(config, model, path)
        return {"path": str(path), "metrics": {"mean_ic": scores[model_name]}}

    monkeypatch.setattr(train_module, "train_ranking", fake_train)
    cfg = load_config("configs/research.yaml")
    cfg.data.root = materialize_panel_sources(tmp_path)
    result = train_ranking_auto(cfg)
    assert result["selected_model"] == "elasticnet"
    assert result["selection_metric"] == "mean_ic"
    assert Path(result["path"]).name == "ranker_auto.joblib"
    assert Path(result["path"]).is_file()


def test_label_horizon_uses_configured_forward_target() -> None:
    assert _label_horizon("future_excess_return_20") == 20
    assert _label_horizon("custom_target") == 5


def test_chronological_split_is_date_level_and_purges_forward_labels() -> None:
    unique_dates = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(10)]
    dates = np.repeat(np.array(unique_dates, dtype=object), 2)

    train_mask, test_mask = _chronological_split(
        dates,
        train_fraction=0.7,
        horizon_bars=2,
        embargo_bars=1,
    )

    train_dates = set(dates[train_mask])
    test_dates = set(dates[test_mask])
    assert train_dates.isdisjoint(test_dates)
    assert train_dates == set(unique_dates[:4])
    assert test_dates == set(unique_dates[7:])


def test_walk_forward_splits_are_date_level_purged_and_cover_all_test_blocks() -> None:
    unique_dates = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(20)]
    dates = np.repeat(np.array(unique_dates, dtype=object), 3)
    config = ValidationConfig(train_bars=6, val_bars=2, test_bars=2, scheme="expanding")

    splits = _walk_forward_splits(dates, config, horizon_bars=2)

    assert len(splits) == 3
    test_blocks = [set(dates[test]) for _, test in splits]
    assert len(set().union(*test_blocks)) == 6
    assert all(
        left.isdisjoint(right)
        for i, left in enumerate(test_blocks)
        for right in test_blocks[i + 1 :]
    )
    for train, test in splits:
        assert set(dates[train]).isdisjoint(set(dates[test]))
        train_indices = [unique_dates.index(value) for value in set(dates[train])]
        test_indices = [unique_dates.index(value) for value in set(dates[test])]
        assert max(train_indices) + 2 < min(test_indices)


def test_short_walk_forward_uses_explicit_sparse_label_endpoints() -> None:
    unique_dates = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(8)]
    dates = np.repeat(np.array(unique_dates, dtype=object), 2)
    endpoints = np.repeat(
        np.array(
            [
                unique_dates[4],
                unique_dates[2],
                unique_dates[3],
                unique_dates[5],
                unique_dates[5],
                unique_dates[6],
                unique_dates[7],
                unique_dates[7],
            ],
            dtype=object,
        ),
        2,
    )
    config = ValidationConfig(
        train_bars=6,
        val_bars=2,
        test_bars=2,
        scheme="expanding",
        embargo_bars=0,
    )

    splits = _walk_forward_splits(
        dates,
        config,
        horizon_bars=2,
        label_end_times=endpoints,
    )

    assert len(splits) == 1
    train, test = splits[0]
    assert set(dates[train]) == {unique_dates[1], unique_dates[2]}
    assert set(dates[test]) == set(unique_dates[4:])


def test_chronological_split_empty_or_single_date_fail_closed() -> None:
    with pytest.raises(ValueError, match="at least 2 unique dates"):
        _chronological_split(np.array([], dtype=object))
    with pytest.raises(ValueError, match="at least 2 unique dates"):
        _chronological_split(np.array([datetime(2020, 1, 1, tzinfo=UTC)], dtype=object))


def test_chronological_split_bad_fraction_and_horizon() -> None:
    dates = np.array(
        [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(5)],
        dtype=object,
    )
    with pytest.raises(ValueError, match="train_fraction"):
        _chronological_split(dates, train_fraction=0.0)
    with pytest.raises(ValueError, match="train_fraction"):
        _chronological_split(dates, train_fraction=1.0)
    with pytest.raises(ValueError, match="horizon_bars"):
        _chronological_split(dates, horizon_bars=-1)


def test_train_family_unknown_family_fail_closed() -> None:
    cfg = load_config("configs/research.yaml")
    with pytest.raises(ValueError, match="unknown family"):
        train_family(cfg, "not_a_real_family")


def test_make_ranker_unknown_model_fail_closed() -> None:
    cfg = load_config("configs/research.yaml")
    with pytest.raises(ValueError, match="unknown ranking model"):
        _make_ranker("not_a_ranker", cfg)


def test_train_family_unknown_models_fail_closed() -> None:
    cfg = load_config("configs/research.yaml")
    with pytest.raises(ValueError, match="unknown ranking model"):
        train_family(cfg, "ranking", "bogus_ranker")
    with pytest.raises(ValueError, match="unknown distribution model"):
        train_family(cfg, "distribution", "bogus_dist")
    with pytest.raises(ValueError, match="unknown volatility model"):
        train_family(cfg, "volatility", "bogus_vol")
    with pytest.raises(ValueError, match="unknown regime model"):
        train_family(cfg, "regime", "bogus_regime")
    with pytest.raises(ValueError, match="unknown tail model"):
        train_family(cfg, "tail", "bogus_tail")


def test_train_family_dispatches_calibration(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config("configs/research.yaml")
    monkeypatch.setattr(
        "quant_fund.pipeline.train.train_calibration",
        lambda config, model: {"model": model, "research_only": True},
    )
    result = train_family(cfg, "calibration", "platt")
    assert result == {
        "model": "platt",
        "research_only": True,
        "data_source": "SYNTHETIC",
        "evidence_report": {
            "json": "data/metadata/reports/evidence_report.json",
            "markdown": "data/metadata/reports/evidence_report.md",
        },
    }


def test_train_ranking_empty_panel_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config("configs/research.yaml")
    empty = pl.DataFrame(
        schema={
            "event_time": pl.Datetime("us", "UTC"),
            "security_id": pl.Utf8,
            cfg.train.ranking_target: pl.Float64,
            "ret_1": pl.Float64,
        }
    )
    monkeypatch.setattr("quant_fund.pipeline.train.panel", lambda *a, **k: empty)
    with pytest.raises(ValueError, match="non-empty frame"):
        train_ranking(cfg, "ridge")


def test_train_ranking_no_evaluable_fold_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config("configs/research.yaml")
    label = cfg.train.ranking_target
    # Non-empty panel so design_matrix succeeds, but zero walk-forward folds.
    frame = pl.DataFrame(
        {
            "event_time": [datetime(2020, 1, 1, tzinfo=UTC)] * 4,
            "security_id": [f"A{i}" for i in range(4)],
            label: [0.1, -0.2, 0.05, 0.0],
            "ret_1": [0.01, -0.01, 0.0, 0.02],
            "mom_20": [0.1, 0.2, 0.0, -0.1],
            "vol_20": [0.2, 0.2, 0.2, 0.2],
            "reversal_1": [0.0, 0.1, -0.1, 0.0],
            "amihud": [1e-6] * 4,
        }
    )
    monkeypatch.setattr("quant_fund.pipeline.train.panel", lambda *a, **k: frame)
    monkeypatch.setattr(
        "quant_fund.pipeline.train._walk_forward_splits",
        lambda *a, **k: [],
    )
    with pytest.raises(ValueError, match="no trainable/evaluable fold"):
        train_ranking(cfg, "ridge")


def test_train_distribution_empty_panel_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config("configs/research.yaml")
    empty = pl.DataFrame(
        schema={
            "event_time": pl.Datetime("us", "UTC"),
            "security_id": pl.Utf8,
            cfg.train.distribution_target: pl.Float64,
            "ret_1": pl.Float64,
        }
    )
    monkeypatch.setattr("quant_fund.pipeline.train.panel", lambda *a, **k: empty)
    with pytest.raises(ValueError, match="non-empty frame"):
        train_distribution(cfg, "gaussian")


def test_train_calibration_missing_score_column_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = load_config("configs/research.yaml")
    empty = pl.DataFrame({"event_time": [], cfg.train.ranking_target: []})
    monkeypatch.setattr("quant_fund.pipeline.train.panel", lambda *a, **k: empty)
    with pytest.raises(ValueError, match="calibration requires columns"):
        train_calibration(cfg, "isotonic")


def test_train_calibration_persists_causal_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = materialize_panel_sources(tmp_path)
    label = cfg.train.ranking_target
    frame = pl.DataFrame(
        {
            "event_time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(20)],
            "cs_pct_mom_20": np.linspace(0.02, 0.98, 20),
            label: np.where(np.arange(20) % 2 == 0, 0.02, -0.01),
        }
    )
    monkeypatch.setattr("quant_fund.pipeline.train.panel", lambda *a, **k: frame)
    train_mask = np.zeros(20, dtype=bool)
    train_mask[:10] = True
    test_mask = ~train_mask
    monkeypatch.setattr(
        "quant_fund.pipeline.train._walk_forward_splits",
        lambda *a, **k: [(train_mask, test_mask)],
    )
    result = train_calibration(cfg, "isotonic")
    assert result["research_only"] is True
    assert np.isfinite(result["metrics"]["oos_brier"])
    assert Path(result["path"]).is_file()
    assert Path(f"{result['path']}.sha256").is_file()
    from quant_fund.models.calibration import ProbabilityCalibrator

    loaded = ProbabilityCalibrator.load(Path(result["path"]))
    assert loaded.score_feature == "cs_pct_mom_20"
    assert loaded.label == label


def test_train_calibration_auto_selects_lowest_finite_brier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quant_fund.models.calibration import ProbabilityCalibrator
    from quant_fund.pipeline import train as train_module

    briers = {"isotonic": 0.12, "platt": 0.08}

    def fake_train(config, model_name):
        path = Path(config.data.root) / "metadata" / f"calibrator_{model_name}.joblib"
        model = ProbabilityCalibrator(model_name).fit(
            np.linspace(0.1, 0.9, 12), np.asarray([0, 1] * 6, dtype=float)
        )
        save_bound_stub(config, model, path)
        return {"path": str(path), "metrics": {"oos_brier": briers[model_name]}}

    monkeypatch.setattr(train_module, "train_calibration", fake_train)
    cfg = load_config("configs/research.yaml")
    cfg.data.root = materialize_panel_sources(tmp_path)
    result = train_calibration_auto(cfg)
    assert result["selected_model"] == "platt"
    assert result["selection_metric"] == "oos_brier"
    assert Path(result["path"]).name == "calibrator_auto.joblib"


def test_train_volatility_empty_panel_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config("configs/research.yaml")
    empty = pl.DataFrame(
        schema={
            "event_time": pl.Datetime("us", "UTC"),
            "security_id": pl.Utf8,
            cfg.train.volatility_target: pl.Float64,
            "vol_20": pl.Float64,
        }
    )
    monkeypatch.setattr("quant_fund.pipeline.train.panel", lambda *a, **k: empty)
    with pytest.raises(ValueError, match="non-empty frame"):
        train_volatility(cfg, "ewma")


def test_train_regime_empty_panel_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config("configs/research.yaml")
    empty = pl.DataFrame(
        schema={
            "event_time": pl.Datetime("us", "UTC"),
            "mkt_ret_1": pl.Float64,
            "mkt_vol_20": pl.Float64,
            "cs_dispersion": pl.Float64,
            "breadth": pl.Float64,
        }
    )
    monkeypatch.setattr("quant_fund.pipeline.train.panel", lambda *a, **k: empty)
    # unique/drop_nulls on empty → empty x/dates; walk-forward then fails closed for hmm.
    with pytest.raises(
        ValueError, match="no trainable/evaluable fold|at least|empty|chronological"
    ):
        train_regime(cfg, "hmm")


def test_train_family_passthrough_families_do_not_train() -> None:
    cfg = load_config("configs/research.yaml")
    cov = train_family(cfg, "covariance")
    liq = train_family(cfg, "liquidity")
    assert "note" in cov and "metrics" in cov
    assert "note" in liq and "metrics" in liq
    # No live claim keys.
    assert cov.get("live_pnl_claim") is not True
    assert liq.get("live_pnl_claim") is not True


def test_train_reinforcement_persists_research_only_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = materialize_panel_sources(tmp_path)
    label = cfg.train.ranking_target
    dates = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(8)]
    rows = []
    for date in dates:
        for name, signal in [("A", 1.0), ("B", 0.2), ("C", -0.5), ("D", -1.0)]:
            rows.append(
                {
                    "event_time": date,
                    "security_id": name,
                    label: signal,
                    "ret_1": signal,
                    "mom_20": signal,
                    "vol_20": 0.2,
                    "reversal_1": signal,
                    "amihud": 1e-6,
                }
            )
    frame = pl.DataFrame(rows)
    monkeypatch.setattr("quant_fund.pipeline.train.panel", lambda *a, **k: frame)
    monkeypatch.setattr("quant_fund.pipeline.train.configure_tracking", lambda: None)
    monkeypatch.setattr("quant_fund.pipeline.train.log_run", lambda **kwargs: "run-test")
    result = train_reinforcement(cfg)
    assert result["research_only"] is True
    assert result["live_pnl_claim"] is False
    assert Path(result["path"]).is_file()
    assert result["metrics"]["n_dates"] > 0


def test_train_policy_gradient_persists_research_only_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = materialize_panel_sources(tmp_path)
    label = cfg.train.ranking_target
    dates = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(8)]
    rows = []
    for date in dates:
        for name, signal in [("A", 1.0), ("B", 0.2), ("C", -0.5), ("D", -1.0)]:
            rows.append(
                {
                    "event_time": date,
                    "security_id": name,
                    label: signal,
                    "ret_1": signal,
                    "mom_20": signal,
                    "vol_20": 0.2,
                    "reversal_1": signal,
                    "amihud": 1e-6,
                }
            )
    monkeypatch.setattr("quant_fund.pipeline.train.panel", lambda *a, **k: pl.DataFrame(rows))
    monkeypatch.setattr("quant_fund.pipeline.train.configure_tracking", lambda: None)
    monkeypatch.setattr("quant_fund.pipeline.train.log_run", lambda **kwargs: "run-pg-test")
    result = train_reinforcement(cfg, "policy_gradient")
    assert result["research_only"] is True
    assert result["live_pnl_claim"] is False
    assert Path(result["path"]).is_file()


def test_train_quantile_thompson_persists_research_only_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = materialize_panel_sources(tmp_path)
    label = cfg.train.ranking_target
    dates = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(8)]
    rows = []
    for date in dates:
        for name, signal in [("A", 1.0), ("B", 0.2), ("C", -0.5), ("D", -1.0)]:
            rows.append(
                {
                    "event_time": date,
                    "security_id": name,
                    label: signal,
                    "ret_1": signal,
                    "mom_20": signal,
                    "vol_20": 0.2,
                    "reversal_1": signal,
                    "amihud": 1e-6,
                }
            )
    monkeypatch.setattr("quant_fund.pipeline.train.panel", lambda *a, **k: pl.DataFrame(rows))
    monkeypatch.setattr("quant_fund.pipeline.train.configure_tracking", lambda: None)
    monkeypatch.setattr("quant_fund.pipeline.train.log_run", lambda **kwargs: "run-qt-test")
    result = train_reinforcement(cfg, "quantile_thompson")
    assert result["research_only"] is True
    assert result["live_pnl_claim"] is False
    assert Path(result["path"]).is_file()
