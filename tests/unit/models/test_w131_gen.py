"""Wave-131 generative-sequence module tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.consistency_ts import bench_consistency_ts
from quant_fund.models.consistency_ts import synth_bimodal as cs_bi
from quant_fund.models.energy_ts import bench_energy_ts, synth_normal_anom
from quant_fund.models.flow_matching_ts import (
    bench_flow_matching_ts,
    synth_regime_windows,
)
from quant_fund.models.perceiver_ts import bench_perceiver_ts, synth_long_window
from quant_fund.models.score_sde_ts import bench_score_sde_ts, synth_bimodal
from quant_fund.models.vq_vae_ts import bench_vq_vae_ts, synth_prototype_mix


class TestVqVae:
    def test_mix(self) -> None:
        x, lab = synth_prototype_mix(40, 32, np.random.default_rng(0))
        assert x.shape == (40, 32)

    def test_bench(self) -> None:
        out = bench_vq_vae_ts(seed=3, n=120, iters=200)
        assert out["synthetic_vqvae_active_codes"] >= 2


class TestFlowMatching:
    def test_windows(self) -> None:
        x, r = synth_regime_windows(40, 24, np.random.default_rng(0))
        assert x.shape == (40, 24)

    def test_bench(self) -> None:
        out = bench_flow_matching_ts(seed=5, n=120, iters=200)
        assert out["synthetic_flow_margin_vs_gauss"] > -0.5


class TestScoreSde:
    def test_bimodal(self) -> None:
        x = synth_bimodal(40, 24, np.random.default_rng(0))
        assert x.shape == (40, 24)

    def test_bench(self) -> None:
        out = bench_score_sde_ts(seed=7, n=120, iters=200, gen_steps=10)
        assert np.isfinite(out["synthetic_scoresde_mmd"])


class TestConsistency:
    def test_bimodal(self) -> None:
        x = cs_bi(40, 24, np.random.default_rng(0))
        assert x.shape == (40, 24)

    def test_bench(self) -> None:
        out = bench_consistency_ts(seed=11, n=120, iters=300)
        assert out["synthetic_consistency_steps"] == 2.0


class TestEnergy:
    def test_anom(self) -> None:
        x, y = synth_normal_anom(40, 24, np.random.default_rng(0))
        assert y.sum() == 20

    def test_bench(self) -> None:
        out = bench_energy_ts(seed=13, n=120, iters=150)
        assert 0 <= out["synthetic_energy_auc"] <= 1


class TestPerceiver:
    def test_window(self) -> None:
        x, y = synth_long_window(40, 256, np.random.default_rng(0))
        assert x.shape == (40, 256)

    def test_bench(self) -> None:
        out = bench_perceiver_ts(seed=17, n=80, iters=200)
        assert out["synthetic_perceiver_dot_ratio"] < 1.0
