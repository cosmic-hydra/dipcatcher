"""Wave-159 optimizer canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models._opt_synth import opt_task
from quant_fund.models.adafactor_opt import bench_adafactor_opt
from quant_fund.models.lamb_opt import bench_lamb_opt
from quant_fund.models.lion_opt import bench_lion_opt
from quant_fund.models.lookahead_opt import bench_lookahead_opt
from quant_fund.models.muon_opt import _ns, bench_muon_opt
from quant_fund.models.sophia_opt import bench_sophia_opt


class TestOptSynth:
    def test_task(self) -> None:
        x, y = opt_task(seed=3, n=32, d=8)
        assert x.shape == (32, 8) and y.shape == (32,)


class TestMuon:
    def test_ns_orthogonal(self) -> None:
        import torch

        # Five Newton-Schulz steps stay within atol on this draw. A local
        # generator keeps the global torch RNG alone for later tests.
        generator = torch.Generator().manual_seed(16)
        g = torch.randn(8, 8, generator=generator)
        x = _ns(g)
        gram = x @ x.T
        assert torch.allclose(gram, torch.eye(8), atol=0.35)

    def test_bench(self) -> None:
        out = bench_muon_opt(seed=5, iters=15)
        assert np.isfinite(out["synthetic_muon_loss"])


class TestLion:
    def test_bench(self) -> None:
        out = bench_lion_opt(seed=7, iters=15)
        assert np.isfinite(out["synthetic_lion_loss"])


class TestSophia:
    def test_bench(self) -> None:
        out = bench_sophia_opt(seed=9, iters=8, hess_every=4)
        assert np.isfinite(out["synthetic_sophia_loss"])


class TestLookahead:
    def test_bench(self) -> None:
        out = bench_lookahead_opt(seed=11, iters=12, k=4)
        assert np.isfinite(out["synthetic_look_loss"])


class TestLamb:
    def test_bench(self) -> None:
        out = bench_lamb_opt(seed=13, iters=15)
        assert np.isfinite(out["synthetic_lamb_loss"])


class TestAdafactor:
    def test_bench(self) -> None:
        out = bench_adafactor_opt(seed=15, iters=15)
        assert np.isfinite(out["synthetic_af_loss"])
        assert out["synthetic_af_mem_ratio"] < 1
