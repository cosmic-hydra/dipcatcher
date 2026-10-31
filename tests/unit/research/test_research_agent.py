import math
from pathlib import Path

import numpy as np
import polars as pl

from quant_fund.config import load_config
from quant_fund.research.agent import (
    _benchmark_scorecard,
    _git_worktree_sha256,
    _provenance,
    run_research,
)
from quant_fund.research.catalog import (
    FORBIDDEN_RESEARCH_METRIC_KEYS,
    REQUIRED_BENCHMARK_FAMILIES,
    RESEARCH_RECEIPT_SCHEMA_VERSION,
    family_blob_forbidden_metrics_absent,
)
from quant_fund.utils.hashing import canonical_frame_fingerprint


def test_provenance_changes_when_materialized_data_changes() -> None:
    cfg = load_config("configs/research.yaml")
    left = pl.DataFrame({"event_time": [1], "security_id": ["A"], "x": [1.0]})
    right = left.with_columns(pl.lit(2.0).alias("x"))
    assert (
        _provenance(cfg, left, "x")["dataset_content_sha256"]
        != _provenance(cfg, right, "x")["dataset_content_sha256"]
    )


def test_canonical_frame_fingerprint_is_order_invariant_and_counts_duplicates() -> None:
    left = pl.DataFrame(
        {
            "event_time": [2, 1, 1],
            "security_id": ["B", "A", "A"],
            "x": [0.2, 0.1, 0.1],
        }
    )
    right = pl.DataFrame(
        {
            "x": [0.1, 0.2, 0.1],
            "security_id": ["A", "B", "A"],
            "event_time": [1, 2, 1],
        }
    )
    changed = right.with_columns(
        pl.when(pl.col("x") == 0.2).then(0.3).otherwise(pl.col("x")).alias("x")
    )
    assert canonical_frame_fingerprint(left) == canonical_frame_fingerprint(right)
    assert canonical_frame_fingerprint(left) != canonical_frame_fingerprint(changed)


def test_provenance_hashes_external_northset_book_contents(tmp_path: Path) -> None:
    cfg = load_config("configs/research.yaml")
    book = tmp_path / "book.parquet"
    book.write_bytes(b"first")
    cfg.northset.book_panel_path = str(book)
    frame = pl.DataFrame({"event_time": [1], "security_id": ["A"], "x": [1.0]})
    first = _provenance(cfg, frame, "x", northset_frame=frame)
    book.write_bytes(b"other")
    second = _provenance(cfg, frame, "x", northset_frame=frame)
    assert (
        first["northset_inputs"]["book_panel_sha256"]
        != second["northset_inputs"]["book_panel_sha256"]
    )
    assert first["run_id"] != second["run_id"]


def test_worktree_fingerprint_is_a_real_sha256() -> None:
    fingerprint = _git_worktree_sha256()
    assert len(fingerprint) == 64
    assert all(char in "0123456789abcdef" for char in fingerprint)


def test_research_recovers_synthetic_oracle(tmp_path: Path) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.data.synthetic_n_assets = 36
    cfg.data.synthetic_n_days = 220
    cfg.validation.train_bars = 80
    cfg.validation.val_bars = 20
    cfg.validation.test_bars = 20
    nb = run_research(cfg, include_live_optional=False)
    assert nb.synthetic is True
    assert nb.data_source == "SYNTHETIC"
    assert len(nb.provenance["run_id"]) == 64
    assert len(nb.provenance["config_sha256"]) == 64
    assert len(nb.provenance["dataset_content_sha256"]) == 64
    assert len(nb.provenance["northset_inputs_sha256"]) == 64
    assert nb.provenance["point_in_time"] is True
    assert nb.provenance["execution_claim"] == "research_only"
    assert nb.provenance["benchmark_catalog_version"] == 2
    receipt_root = tmp_path / "metadata" / "research"
    assert (receipt_root / nb.artifacts["immutable_json"]).is_file()
    assert (receipt_root / nb.artifacts["immutable_markdown"]).is_file()
    assert nb.firm == "Artificial Hedge"
    assert nb.schema_version == RESEARCH_RECEIPT_SCHEMA_VERSION
    overfit = nb.backtest_overfitting
    assert overfit["claim"] == "research_diagnostic_only"
    assert overfit["research_only"] is True
    assert overfit["n_trials"] >= 2
    assert overfit["n_trials_effective"] <= overfit["n_trials"]
    assert 0.0 <= float(overfit["pbo"]) <= 1.0
    assert float(overfit["dsr"]) <= float(overfit["psr"]) + 1e-8
    assert float(overfit["dsr_counted_trials"]) <= float(overfit["psr"]) + 1e-8
    assert family_blob_forbidden_metrics_absent(overfit)
    assert "PBO=" in (tmp_path / "metadata" / "research" / "latest.md").read_text()
    assert "Sharpe" not in nb.disclaimer
    by = {r["name"]: r for r in nb.rankers}
    assert "oracle_raw" in by
    oracle = by["oracle_raw"]
    assert oracle["mean_ic"] > 0.15
    assert oracle["p_ic"] < 0.05
    assert "sharpe" not in oracle
    assert "volatility" in nb.families

    assert "reinforcement" in nb.families
    assert "drawdown" in nb.families
    assert "conformal" in nb.families
    assert nb.families["cpcv"]["purge_embargo_valid"] is True
    assert nb.families["cpcv"]["claim"] == "validation_integrity_only"
    # The catalog requires these families. Optional diagnostic benches may
    # honestly emit an empty result for this synthetic panel or Python build.
    assert all(nb.scorecard[name]["executed"] for name in REQUIRED_BENCHMARK_FAMILIES)
    assert all(nb.scorecard[name]["finite_observation"] for name in REQUIRED_BENCHMARK_FAMILIES)
    assert all(item["forbidden_metrics_absent"] for item in nb.scorecard.values())
    conf = nb.families["conformal"]
    assert "sharpe" not in str(conf).lower()
    assert conf.get("aci", {}).get("coverage", 0.0) >= 0.85
    raw_cov = conf.get("gaussian_raw", {}).get("coverage", 1.0)
    assert raw_cov < 0.90
    scaled = conf.get("scaled_gaussian", {}).get("coverage")
    if scaled is not None:
        assert float(scaled) >= 0.80
    rl_feats = [str(f).lower() for f in nb.families.get("reinforcement", {}).get("features", [])]
    qb_feats = [str(f).lower() for f in nb.families.get("quantile_bandit", {}).get("features", [])]
    assert not any("planted" in f for f in rl_feats)
    assert not any("planted" in f for f in qb_feats)
    jp = nb.families.get("jackknife_plus", {})
    assert float(jp.get("coverage_floor", 0.8)) == 0.8
    assert float(jp.get("coverage", 0.0)) >= 0.78
    caps = nb.families.get("interval_risk", {})
    assert "bind_wide" in caps and "bind_tight" in caps
    wide_b = float(caps.get("bind_wide", float("nan")))
    tight_b = float(caps.get("bind_tight", float("nan")))
    assert str(caps.get("weight_rule")) == "equal_weight_per_date"
    if math.isfinite(wide_b) and math.isfinite(tight_b):
        assert wide_b >= tight_b
        if float(caps.get("frac_binding", 1.0)) < 1.0 - 1e-12:
            assert wide_b > tight_b
    sota = (
        "evalues",
        "jackknife_plus",
        "crc",
        "weighted_conformal",
        "interval_risk",
        "quantile_bandit",
    )
    metric = {
        "evalues": "e_final",
        "jackknife_plus": "coverage",
        "crc": "risk",
        "weighted_conformal": "coverage",
        "interval_risk": "frac_binding",
        "quantile_bandit": "mean_regret_vs_oracle",
    }
    for fam in sota:
        assert fam in nb.families
        blob = nb.families[fam]
        assert blob
        assert "sharpe" not in str(blob).lower()
        val = blob.get(metric[fam])
        assert val is not None
        assert math.isfinite(float(val))
    dumped = (tmp_path / "metadata" / "research" / "latest.md").read_text()
    assert "SYNTHETIC" in dumped
    assert "proprietary research lab" in dumped.lower()
    assert "sharpe" not in dumped.lower()
    h1 = next(h for h in nb.hypotheses if h.id == "H1_ranking_oracle")
    assert h1.reject_raw is True
    h7 = next(h for h in nb.hypotheses if h.id == "H7_aci_coverage")
    assert "nominal" in h7.statement.lower() or "α" in h7.statement
    mond = conf.get("mondrian_aci", {})
    assert mond.get("coverage", 0.0) >= 0.85
    assert mond.get("worst_x_coverage", 0.0) >= 0.75
    h8 = next(h for h in nb.hypotheses if h.id == "H8_mondrian_high_vol")
    assert "high-vol" in h8.statement.lower() or "X" in h8.statement
    h10 = next(h for h in nb.hypotheses if h.id == "H10_jackknife_coverage")
    assert "2α" in h10.statement or "2a" in h10.statement.lower()
    assert h10.family == "bound"
    assert h10.reject_fdr is False
    assert not math.isfinite(h10.p_value)
    assert h10.meets_floor == (
        float(jp.get("coverage", 0.0)) + 1e-12 >= float(jp.get("coverage_floor", 0.8))
    )
    h5 = next(h for h in nb.hypotheses if h.id == "H5_drawdown_brier")
    assert h5.p_value not in (0.0, 1.0) or not math.isfinite(h5.p_value)
    ns = nb.families["northset"]
    assert ns["product"] == "Northset"
    assert "live_pnl_claim" not in ns
    assert ns["ohlc_identity_rate"] >= 1.0 - 1e-12
    assert ns["book_uncrossed_rate"] >= 1.0 - 1e-12
    assert ns["session_reconstructs_daily_rate"] >= 1.0 - 1e-12
    assert ns["session_volume_conservation_rate"] >= 1.0 - 1e-12
    assert ns["session_chain_rate"] >= 1.0 - 1e-12
    assert np.isfinite(float(ns["garman_klass_qlike_vs_cc"]))
    assert np.isfinite(float(ns["rogers_satchell_qlike_vs_cc"]))
    assert "ofi_mean_ic" in ns
    assert "wick_skew_p_ic" in ns
    assert "clv_p_ic" in ns
    assert "vpin_mean" in ns
    assert "sweep_any_rate" in ns
    assert "sweep_reject_signed_p_ic" in ns
    assert "sweep_follow_signed_p_ic" in ns
    assert "sweep_evidence" in ns
    assert ns["sweep_evidence"]["timing_contract"] == "event_close_then_next_open"
    assert len(ns["sweep_evidence"]["event_studies"]) == 6
    assert "sharpe" not in str(ns).lower()
    ids = [h.id for h in nb.hypotheses]
    assert len(ids) == len(set(ids)), "research hypothesis IDs must be globally unique"
    h20 = next(h for h in nb.hypotheses if h.id == "H20_northset_ohlc")
    assert h20.family == "bound"
    assert h20.meets_floor is True
    h21 = next(h for h in nb.hypotheses if h.id == "H21_northset_book")
    assert h21.family == "bound"
    assert h21.meets_floor is True
    h23 = next(h for h in nb.hypotheses if h.id == "H23_northset_session")
    assert h23.meets_floor is True
    h24 = next(h for h in nb.hypotheses if h.id == "H24_northset_volume")
    assert h24.meets_floor is True
    h29 = next(h for h in nb.hypotheses if h.id == "H29_northset_chain")
    assert h29.meets_floor is True
    for hid in (
        "H35_northset_reject_event",
        "H36_northset_follow_event",
        "H37_northset_reject_placebo",
        "H38_northset_follow_placebo",
        "H39_northset_reject_cost",
        "H40_northset_follow_cost",
        "H41_northset_reject_stability",
        "H42_northset_follow_stability",
        "H43_northset_session_book_vpin",
        "H44_northset_reject_control",
        "H45_northset_follow_control",
        "H46_northset_reject_liq_control",
        "H47_northset_follow_liq_control",
        "H48_northset_follow_oot",
        "H49_northset_follow_name_cluster",
        "H50_northset_follow_two_way_cluster",
        "H51_northset_follow_overnight_gap",
    ):
        next(h for h in nb.hypotheses if h.id == hid)
    h13 = next(h for h in nb.hypotheses if h.id == "H13_interval_caps")
    assert h13.p_value != 0.0
    assert "H5-style" not in h13.test
    if str(caps.get("weight_rule")) == "equal_weight_per_date":
        assert "equal-weight" in h13.statement.lower()
    cal = {h.id for h in nb.hypotheses if h.family == "calibration"}
    disc = {h.id for h in nb.hypotheses if h.family == "discovery"}
    assert cal.isdisjoint(disc)
    assert "H10_jackknife_coverage" not in cal and "H10_jackknife_coverage" not in disc
    for hid in (
        "H9_eprocess_aci",
        "H10_jackknife_coverage",
        "H11_crc_var",
        "H12_weighted_cqr",
        "H13_interval_caps",
        "H14_quantile_thompson",
    ):
        next(h for h in nb.hypotheses if h.id == hid)


def test_research_receipt_records_runtime_fingerprint(tmp_path: Path) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.data.synthetic_n_assets = 6
    cfg.data.synthetic_n_days = 50
    cfg.universe.min_history_bars = 10
    nb = run_research(cfg)

    runtime = nb.provenance["runtime"]
    assert runtime["python"]
    assert runtime["implementation"]
    assert runtime["platform"]
    assert runtime["packages"]["numpy"]


def test_forbidden_research_metric_keys_frozen() -> None:
    assert (
        frozenset({"sharpe", "sortino", "calmar", "pnl", "nav"}) == FORBIDDEN_RESEARCH_METRIC_KEYS
    )


def test_family_blob_rejects_forbidden_metric_keys() -> None:
    """Scorecard / family blobs fail closed on Sharpe/Sortino/Calmar/pnl/nav keys."""
    clean = {"coverage": 0.9, "mean_width": 0.1, "claim": "research_metric_only"}
    assert family_blob_forbidden_metrics_absent(clean) is True
    assert family_blob_forbidden_metrics_absent({"nested": clean}) is True
    # exact keys
    for key in ("sharpe", "sortino", "calmar", "pnl", "nav"):
        assert family_blob_forbidden_metrics_absent({key: 1.0}) is False
        assert family_blob_forbidden_metrics_absent({key.upper(): 1.0}) is False
    # compound / nested keys (underscore tokens)
    assert family_blob_forbidden_metrics_absent({"flag_high_sharpe": True}) is False
    assert family_blob_forbidden_metrics_absent({"shock_down_pnl": -1.0}) is False
    assert family_blob_forbidden_metrics_absent({"calmar_diagnostic": 0.5}) is False
    assert family_blob_forbidden_metrics_absent({"mean_nav": 1e6}) is False
    assert family_blob_forbidden_metrics_absent({"outer": {"inner_sortino": 0.2}}) is False
    # value text alone must not trip key scanner (keys only)
    assert family_blob_forbidden_metrics_absent({"note": "sharpe is forbidden"}) is True


def test_benchmark_scorecard_flags_forbidden_metrics() -> None:
    families = {
        "clean": {"coverage": 0.91, "risk": 0.05},
        "poison_sharpe": {"coverage": 0.9, "sharpe": 9.9},
        "poison_sortino": {"sortino": 1.2},
        "poison_calmar": {"calmar": 0.4},
        "poison_pnl": {"pnl": 100.0},
        "poison_nav": {"nav": 1_000_000.0},
        "poison_nested": {"diag": {"flag_high_sharpe": True}},
    }
    card = _benchmark_scorecard(families)
    assert card["clean"]["forbidden_metrics_absent"] is True
    for name in (
        "poison_sharpe",
        "poison_sortino",
        "poison_calmar",
        "poison_pnl",
        "poison_nav",
        "poison_nested",
    ):
        assert card[name]["forbidden_metrics_absent"] is False, name
        assert card[name]["executed"] is True
