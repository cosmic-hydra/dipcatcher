"""Equivalence of the research-compute kernels against the previous formulas.

SYNTHETIC inputs only. These checks are correctness tests, not market evidence.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from numpy.typing import NDArray

from quant_fund.compute.cache import FitCache
from quant_fund.compute.parallel import derive_seed, process_map
from quant_fund.metrics.cross_section import date_ic_series, decile_portfolios
from quant_fund.metrics.scoring import pearson_ic, rank_ic
from quant_fund.models.asset_pricing import date_groups
from quant_fund.models.covariance import dcc_trailing_complete_window, ewma_cov
from quant_fund.models.cs_papers import _mean_date_ics
from quant_fund.models.ranking import _demean_by_date
from quant_fund.models.volatility import (
    _GARCH_FIT_CACHE,
    GARCHVol,
    HARVol,
    RollingVol,
    ewma_variance,
)

_ULP = np.finfo(np.float64).eps


def _within_ulps(actual: NDArray[np.float64], expected: NDArray[np.float64], ulps: int) -> None:
    """LLVM and NumPy reductions can disagree by a few ulps on the same formula."""
    scale = np.maximum(1.0, np.maximum(np.abs(actual), np.abs(expected)))
    gap = np.abs(np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64))
    finite = np.isfinite(gap) & np.isfinite(scale)
    assert np.array_equal(np.isfinite(actual), np.isfinite(expected))
    assert float(np.max(gap[finite] / (_ULP * scale[finite]), initial=0.0)) <= ulps


def _ewma_reference(log_returns: NDArray[np.float64], lam: float) -> NDArray[np.float64]:
    var = np.empty_like(log_returns)
    var[0] = log_returns[0] ** 2
    for t in range(1, log_returns.size):
        var[t] = lam * var[t - 1] + (1.0 - lam) * log_returns[t - 1] ** 2
    return var


def _ewma_cov_reference(window: NDArray[np.float64], lam: float) -> NDArray[np.float64]:
    cov = np.outer(window[0], window[0])
    one = 1.0 - lam
    for t in range(1, window.shape[0]):
        cov = lam * cov + one * np.outer(window[t], window[t])
    return 0.5 * (cov + cov.T)


def _date_groups_reference(dates: NDArray[np.generic]) -> list[NDArray[np.intp]]:
    buckets: dict[object, list[int]] = {}
    order: list[object] = []
    for i, stamp in enumerate(dates):
        key = stamp.item() if isinstance(stamp, np.generic) else stamp
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(i)
    return [np.asarray(buckets[key], dtype=np.intp) for key in order]


def _demean_reference(y: NDArray[np.float64], dates: NDArray[np.generic]) -> NDArray[np.float64]:
    out = np.asarray(y, dtype=float).copy()
    keys = np.asarray(dates)
    for key in np.unique(keys):
        sl = keys == key
        block = out[sl]
        finite = np.isfinite(block)
        if finite.any():
            out[sl] = block - float(np.mean(block[finite]))
    return out


def _mean_date_ic_reference(
    column: NDArray[np.float64], y: NDArray[np.float64], dates: NDArray[np.generic]
) -> float:
    ics: list[float] = []
    for idx in _date_groups_reference(dates):
        a = column[idx]
        b = y[idx]
        finite = np.isfinite(a) & np.isfinite(b)
        if int(finite.sum()) < 5:
            continue
        if float(np.std(a[finite])) < 1e-12 or float(np.std(b[finite])) < 1e-12:
            continue
        ics.append(float(np.corrcoef(a[finite], b[finite])[0, 1]))
    return float(np.mean(ics)) if ics else 0.0


def test_rolling_vol_matches_per_window_std() -> None:
    rng = np.random.default_rng(0)
    series = rng.normal(size=4000)
    window = 20
    got = RollingVol(window).predict_from_returns(series)
    ref = np.full_like(series, np.nan)
    for i in range(window, series.size + 1):
        ref[i - 1] = np.std(series[i - window : i], ddof=1)
    assert np.array_equal(got, ref, equal_nan=True)


def test_ewma_variance_matches_python_recurrence_within_ulps() -> None:
    """Numba lowers the same recurrence. Measured gap is at most a few ulps.

    NaN / Inf propagation matches the Python loop exactly, including ``lam``
    of 0 and 1, where a filter rewrite would not.
    """
    rng = np.random.default_rng(1)
    finite = np.ascontiguousarray(rng.normal(size=5000))
    dirty = np.array([0.1, np.nan, 0.2, np.inf, -0.1], dtype=np.float64)
    for lam in (0.0, 0.94, 1.0):
        for series in (finite, dirty, np.array([0.3], dtype=np.float64)):
            got = ewma_variance(series, lam)
            ref = _ewma_reference(np.asarray(series, dtype=np.float64), lam)
            if series.size < 10 or lam == 1.0:
                assert np.array_equal(got, ref, equal_nan=True)
            else:
                _within_ulps(got, ref, ulps=4)
                assert np.array_equal(np.isfinite(got), np.isfinite(ref))


def test_har_trailing_means_match_python_slices() -> None:
    rng = np.random.default_rng(2)
    rv = np.abs(rng.normal(size=3000))
    rv[10] = np.nan
    design = HARVol.har_design(rv)
    lagged = np.full(rv.shape, np.nan)
    lagged[1:] = rv[:-1]
    assert np.array_equal(design[:, 1], lagged, equal_nan=True)
    for column, window in ((2, 5), (3, 22)):
        ref = np.full(rv.shape, np.nan)
        for t in range(window, rv.size):
            ref[t] = np.mean(lagged[t - window + 1 : t + 1])
        assert np.array_equal(design[:, column], ref, equal_nan=True)


def test_ewma_cov_matches_outer_product_within_an_ulp() -> None:
    """In-place updates match ``lam * cov + (1-lam) * r r'`` to about one ulp.

    The outer-product form and the element form are the same arithmetic with
    a different evaluation order inside BLAS. ``lam`` of 0 and 1 stay exact
    on this draw.
    """
    rng = np.random.default_rng(3)
    raw = rng.normal(size=(40, 6))
    raw[3, 1] = np.nan
    for lam in (0.0, 0.94, 1.0):
        got = ewma_cov(raw, lam)
        window = np.ascontiguousarray(
            dcc_trailing_complete_window(raw, min_rows=2), dtype=np.float64
        )
        ref = _ewma_cov_reference(window, lam)
        if lam in (0.0, 1.0):
            assert np.array_equal(got, ref)
        else:
            _within_ulps(got, ref, ulps=4)


@pytest.mark.parametrize("kind", ["datetime64[us]", "datetime64[ns]", "int64"])
def test_date_groups_match_first_seen_dict(kind: str) -> None:
    rng = np.random.default_rng(4)
    if kind.startswith("datetime64"):
        base = np.arange(12).astype(kind)
        dates = rng.choice(base, size=200)
    else:
        dates = rng.integers(0, 12, size=200)
    got = date_groups(dates)
    ref = _date_groups_reference(dates)
    assert len(got) == len(ref)
    for left, right in zip(got, ref, strict=True):
        assert np.array_equal(left, right)


def test_demean_matches_unique_loop_on_datetime_and_int() -> None:
    rng = np.random.default_rng(5)
    y = rng.normal(size=300)
    y[4] = np.nan
    dates_i = rng.integers(0, 9, size=300)
    dates_d = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-01-10"))[dates_i].astype(
        "datetime64[us]"
    )
    for dates in (dates_i, dates_d):
        got = _demean_by_date(y, dates)
        ref = _demean_reference(y, dates)
        assert np.array_equal(got, ref, equal_nan=True)


def test_date_ic_and_deciles_match_corrcoef_reference() -> None:
    """Centered dot product versus ``np.corrcoef`` / ``rank_ic``.

    The two Pearson reductions differ by about one ulp (two-pass sum of
    squares versus NumPy's covariance kernel). Date order follows the
    previous ``group_by(..., maintain_order=True)`` on stringified stamps.
    """
    rng = np.random.default_rng(6)
    n_names = 8
    dates = np.repeat(np.arange(np.datetime64("2021-01-01"), np.datetime64("2021-01-31")), n_names)
    order = rng.permutation(dates.size)
    dates = dates[order]
    scores = rng.normal(size=dates.size)
    y = 0.2 * scores + rng.normal(size=dates.size)
    scores[0] = np.nan
    got = date_ic_series(scores, y, dates, min_names=5)
    # Match ``_date_keys``: datetime64 uses str().
    frame = pl.DataFrame({"score": scores, "y": y, "date": [str(d) for d in dates]})
    pearson: list[float] = []
    spearman: list[float] = []
    kept: list[object] = []
    for date, grp in frame.group_by("date", maintain_order=True):
        if grp.height < 5:
            continue
        s = grp["score"].to_numpy().astype(float)
        t = grp["y"].to_numpy().astype(float)
        mask = np.isfinite(s) & np.isfinite(t)
        if int(mask.sum()) < 5:
            continue
        pearson.append(pearson_ic(s[mask], t[mask]))
        spearman.append(rank_ic(s[mask], t[mask]))
        kept.append(date[0] if isinstance(date, tuple) else date)
    assert got.dates == kept
    np.testing.assert_allclose(got.pearson, pearson, rtol=1e-12, atol=1e-15, equal_nan=True)
    np.testing.assert_allclose(got.spearman, spearman, rtol=1e-12, atol=1e-15, equal_nan=True)
    dec = decile_portfolios(scores, y, dates, n_buckets=5, min_names=5)
    assert dec.n_buckets == 5
    assert len(dec.dates) == len(dec.long_short)


def test_balanced_date_ic_matches_corrcoef_with_ties() -> None:
    """Equal-sized finite dates take the matrix path, including average-rank ties."""
    rng = np.random.default_rng(11)
    n_names = 8
    dates = np.repeat(np.arange(np.datetime64("2022-01-01"), np.datetime64("2022-01-13")), n_names)
    scores = rng.normal(size=dates.size)
    scores[0] = scores[1]
    y = 0.15 * scores + rng.normal(size=dates.size)
    got = date_ic_series(scores, y, dates, min_names=5)
    frame = pl.DataFrame({"score": scores, "y": y, "date": [str(d) for d in dates]})
    pearson: list[float] = []
    spearman: list[float] = []
    kept: list[object] = []
    for date, grp in frame.group_by("date", maintain_order=True):
        s = grp["score"].to_numpy().astype(float)
        t = grp["y"].to_numpy().astype(float)
        pearson.append(pearson_ic(s, t))
        spearman.append(rank_ic(s, t))
        kept.append(date[0] if isinstance(date, tuple) else date)
    assert got.dates == kept
    np.testing.assert_allclose(got.pearson, pearson, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(got.spearman, spearman, rtol=1e-12, atol=1e-15)


def test_mean_date_ics_match_corrcoef_column_loop() -> None:
    rng = np.random.default_rng(7)
    n = 400
    dates = np.repeat(np.arange(20), 20)
    x = rng.normal(size=(n, 3))
    y = x[:, 0] * 0.4 + rng.normal(size=n)
    got = _mean_date_ics(x, y, dates)
    ref = np.array([_mean_date_ic_reference(x[:, j], y, dates) for j in range(3)])
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


def test_fit_cache_round_trip_and_miss() -> None:
    cache = FitCache(max_entries=2)
    left = np.arange(4, dtype=np.float64)
    right = left + 1.0
    key = cache.hash_key(("garch", 1), (left,))
    other = cache.hash_key(("garch", 1), (right,))
    assert key != other
    assert cache.get(key) is None
    cache.put(key, {"sigma": 0.2})
    assert cache.get(key) == {"sigma": 0.2}
    cache.put(other, {"sigma": 0.3})
    third = cache.hash_key(("garch", 2), (left,))
    cache.put(third, {"sigma": 0.4})
    assert cache.get(key) is None  # evicted
    assert cache.get(third) == {"sigma": 0.4}


def test_garch_content_hash_skips_refit() -> None:
    rng = np.random.default_rng(8)
    series = rng.normal(scale=0.01, size=80)
    _GARCH_FIT_CACHE.clear()
    try:
        first = GARCHVol(min_obs=40).fit_returns(series)
        forecast = first.forecast(horizon=1)
        second = GARCHVol(min_obs=40).fit_returns(series)
        again = second.forecast(horizon=1)
        assert second.fit_status == first.fit_status
        assert again["variance"][0] == pytest.approx(forecast["variance"][0])
        assert again["cumulative_variance"][0] == pytest.approx(forecast["cumulative_variance"][0])
        other = GARCHVol(p=1, q=1, min_obs=40, vol="gjr").fit_returns(series)
        assert other is not second
        _GARCH_FIT_CACHE.clear()
        third = GARCHVol(min_obs=40).fit_returns(series)
        assert third.forecast(horizon=1)["variance"][0] == pytest.approx(forecast["variance"][0])
    finally:
        _GARCH_FIT_CACHE.clear()


def test_derive_seed_is_stable_and_index_dependent() -> None:
    assert derive_seed(0, 0) == 465150766
    assert derive_seed(1, 0) == 776153793
    assert derive_seed(7, 3) == 1672421989
    assert derive_seed(7, 4) == 944952506
    assert derive_seed(7, 3) != derive_seed(7, 4)


def _seeded_dot(item: tuple[int, NDArray[np.float64]], seed: int) -> float:
    scale, vector = item
    draw = np.random.default_rng(seed).normal(size=vector.shape[0])
    return float(scale) + float(np.dot(vector, draw))


def test_process_map_matches_serial_seeds_and_order() -> None:
    rng = np.random.default_rng(9)
    items = [(i, rng.normal(size=16)) for i in range(6)]
    serial = [_seeded_dot(item, derive_seed(11, i)) for i, item in enumerate(items)]
    parallel = process_map(_seeded_dot, items, base_seed=11, min_items=4)
    assert parallel == serial


def test_process_start_method_is_spawn_off_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    import quant_fund.compute.parallel as parallel

    monkeypatch.setattr(parallel.sys, "platform", "darwin")
    assert parallel._process_start_method() == "spawn"
    monkeypatch.setattr(parallel.sys, "platform", "win32")
    assert parallel._process_start_method() == "spawn"
    monkeypatch.setattr(parallel.sys, "platform", "linux")
    assert parallel._process_start_method() == "fork"


def test_process_map_falls_back_when_fork_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unsupported_fork(_method: str) -> None:
        raise ValueError("cannot find context for fork")

    monkeypatch.setattr("quant_fund.compute.parallel.multiprocessing.get_context", unsupported_fork)
    items = [(i, np.ones(4)) for i in range(4)]
    expected = [_seeded_dot(item, derive_seed(3, i)) for i, item in enumerate(items)]
    assert process_map(_seeded_dot, items, base_seed=3, max_workers=2) == expected


def test_process_map_falls_back_when_task_cannot_be_pickled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pickle

    import quant_fund.compute.parallel as parallel

    class FakePool:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def map(self, fn: object, payloads: object) -> list[object]:
            raise pickle.PicklingError("unpicklable task")

        def shutdown(self, wait: bool = True) -> None:
            return None

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", FakePool)
    items = [(i, np.ones(4)) for i in range(4)]
    expected = [_seeded_dot(item, derive_seed(3, i)) for i, item in enumerate(items)]
    assert process_map(_seeded_dot, items, base_seed=3, max_workers=2) == expected


def test_process_map_short_input_stays_inline() -> None:
    out = process_map(_seeded_dot, [(1, np.ones(4))], base_seed=3, min_items=4)
    assert out == [_seeded_dot((1, np.ones(4)), derive_seed(3, 0))]


def _boom_task(item: int, _seed: int) -> int:
    if item == 2:
        raise ValueError("task failed")
    return item


def test_process_map_propagates_task_errors() -> None:
    with pytest.raises(ValueError, match="task failed"):
        process_map(_boom_task, [0, 1, 2, 3], base_seed=1)


def test_garch_origin_tasks_match_across_processes() -> None:
    from quant_fund.pipeline.train.volatility import _garch_origin_task

    rng = np.random.default_rng(10)
    spec = {
        "p": 1,
        "q": 1,
        "dist": "normal",
        "vol": "garch",
        "min_obs": 40,
        "mean": "Constant",
        "power": 2.0,
        "series_scope": "security_level_ret_1",
    }
    items = [
        (spec, rng.normal(scale=0.01, size=90), 1, float(rng.normal(scale=0.01))) for _ in range(4)
    ]
    _GARCH_FIT_CACHE.clear()
    try:
        serial = [_garch_origin_task(item, derive_seed(0, i)) for i, item in enumerate(items)]
        parallel = process_map(_garch_origin_task, items, base_seed=0, min_items=4)
    finally:
        _GARCH_FIT_CACHE.clear()
    assert len(parallel) == len(serial)
    for left, right in zip(parallel, serial, strict=True):
        assert left[0] == pytest.approx(right[0], rel=1e-12, abs=1e-15)
        assert left[1] == right[1]
        assert left[2]["log_score"] == pytest.approx(right[2]["log_score"], rel=1e-12, abs=1e-12)
        assert left[2]["pit"] == pytest.approx(right[2]["pit"], rel=1e-12, abs=1e-12)
