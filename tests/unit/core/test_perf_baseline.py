"""Contract tests for the perf-baseline regression harness (scripts/perf_baseline.py).

Timing-harness correctness only — these tests assert the JSON contract and the
comparison logic with tiny sizes; they assert nothing about wall-clock speed.
"""

from __future__ import annotations

import json
import math

import pytest
from scripts import perf_baseline as pb

TINY_SIZES: dict[str, int] = {
    "crps_quantiles": 200,
    "stationary_bootstrap": 40,
    "reality_check_battery": 40,
    "conformal_quantile": 500,
    "e_bh": 20,
    "energy_score": 40,
}

ALL_BENCHES = set(pb.BENCHES)


@pytest.fixture
def tiny_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pb, "BENCH_SIZES", dict(TINY_SIZES))
    # The calibration workload is sized for CI stability, not test speed.
    monkeypatch.setattr(pb, "_CALIBRATION_N", 2_000)
    monkeypatch.setattr(pb, "_CALIBRATION_REPS", 2)


def _record(tmp_path, name: str = "base.json") -> dict:
    path = tmp_path / name
    pb.record(path)
    assert path.exists()
    return json.loads(path.read_text(encoding="utf-8"))


def test_record_produces_valid_baseline(tiny_sizes, tmp_path, monkeypatch) -> None:
    payload = _record(tmp_path)
    for key in ("created_utc", "git_revision", "python_version", "numpy_version", "platform"):
        assert key in payload, f"missing key: {key}"
    assert isinstance(payload["git_revision"], str) and payload["git_revision"]
    benches = payload["benches"]
    assert set(benches) == ALL_BENCHES
    for name, entry in benches.items():
        assert entry["repeats"] == pb.REPEATS
        assert entry["normalized"] > 0.0 and math.isfinite(entry["normalized"]), name
        best = entry["min_seconds"]
        assert math.isfinite(best) and best > 0.0, name


def test_record_sizes_hook_is_used(tiny_sizes, tmp_path, monkeypatch) -> None:
    payload = _record(tmp_path)
    assert payload["git_revision"] == pb.git_revision()


def test_compare_identical_files_passes(tiny_sizes, tmp_path, monkeypatch) -> None:
    base = tmp_path / "base.json"
    pb.record(base)
    cur = tmp_path / "cur.json"
    cur.write_bytes(base.read_bytes())  # exact copy: no timing noise
    result = pb.compare(base, cur)
    assert result.passed
    assert result.regressions == ()
    assert result.missing == ()


def test_compare_flags_two_x_slowdown_at_default_threshold(
    tiny_sizes, tmp_path, monkeypatch
) -> None:
    base = tmp_path / "base.json"
    pb.record(base)
    current = json.loads(base.read_text(encoding="utf-8"))
    name = "crps_quantiles"
    current["benches"][name]["normalized"] *= 2.0
    cur = tmp_path / "cur.json"
    cur.write_text(json.dumps(current), encoding="utf-8")
    result = pb.compare(base, cur)
    assert not result.passed
    assert [r[0] for r in result.regressions] == [name]
    assert result.regressions[0][3] == pytest.approx(2.0)
    loose = pb.compare(base, cur, threshold=3.0)
    assert loose.passed
    assert loose.regressions == ()


def test_compare_reports_missing_benches(tiny_sizes, tmp_path, monkeypatch) -> None:
    base = tmp_path / "base.json"
    pb.record(base)
    current = json.loads(base.read_text(encoding="utf-8"))
    dropped = "energy_score"
    del current["benches"][dropped]
    cur = tmp_path / "cur.json"
    cur.write_text(json.dumps(current), encoding="utf-8")
    result = pb.compare(base, cur)
    assert dropped in result.missing
    assert not result.passed


def test_compare_flags_speedups_as_informational(tiny_sizes, tmp_path, monkeypatch) -> None:
    base = tmp_path / "base.json"
    pb.record(base)
    current = json.loads(base.read_text(encoding="utf-8"))
    name = "conformal_quantile"
    current["benches"][name]["normalized"] *= 0.25
    cur = tmp_path / "cur.json"
    cur.write_text(json.dumps(current), encoding="utf-8")
    result = pb.compare(base, cur)
    assert [s[0] for s in result.speedups] == [name]
    assert result.passed


def test_compare_fail_closed_unknown_bench_name(tiny_sizes, tmp_path, monkeypatch) -> None:
    base = tmp_path / "base.json"
    pb.record(base)
    current = json.loads(base.read_text(encoding="utf-8"))
    current["benches"]["not_a_real_bench"] = {"normalized": 1e-3, "repeats": 5}
    cur = tmp_path / "cur.json"
    cur.write_text(json.dumps(current), encoding="utf-8")
    with pytest.raises(pb.BaselineError, match="unknown bench"):
        pb.compare(base, cur)


def test_compare_fail_closed_malformed_json(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"benches": {}}), encoding="utf-8")
    with pytest.raises(pb.BaselineError, match="malformed JSON"):
        pb.compare(bad, good)
    with pytest.raises(pb.BaselineError, match="malformed JSON"):
        pb.compare(good, bad)


def test_compare_fail_closed_bad_shape_and_bad_min(tmp_path) -> None:
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"benches": {"e_bh": {"normalized": 0.0}}}), encoding="utf-8")
    cur = tmp_path / "cur.json"
    cur.write_text(json.dumps({"benches": {"e_bh": {"normalized": 1e-3}}}), encoding="utf-8")
    with pytest.raises(pb.BaselineError, match="finite and > 0"):
        pb.compare(base, cur)
    shape = tmp_path / "shape.json"
    shape.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    with pytest.raises(pb.BaselineError, match="'benches' mapping"):
        pb.compare(shape, cur)


def test_compare_fail_closed_negative_or_nan_threshold(tiny_sizes, tmp_path, monkeypatch) -> None:
    base = tmp_path / "base.json"
    pb.record(base)
    with pytest.raises(ValueError, match="threshold"):
        pb.compare(base, base, threshold=-1.0)
    with pytest.raises(ValueError, match="threshold"):
        pb.compare(base, base, threshold=0.0)
    with pytest.raises(ValueError, match="threshold"):
        pb.compare(base, base, threshold=float("nan"))


def test_cli_record_compare_roundtrip(tiny_sizes, tmp_path, monkeypatch, capsys) -> None:
    base = tmp_path / "base.json"
    cur = tmp_path / "cur.json"
    assert pb.main(["record", "--output", str(base)]) == 0
    cur.write_bytes(base.read_bytes())  # identical copy: timing noise must not flip the result
    assert pb.main(["compare", "--baseline", str(base), "--current", str(cur)]) == 0
    captured = capsys.readouterr()
    assert "PASS" in captured.out
    assert pb.main(["compare", "--baseline", str(base), "--current", str(cur), "--json"]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["passed"] is True
    assert set(parsed) == {"regressions", "speedups", "missing", "passed"}


def test_cli_compare_failure_exit_code(tiny_sizes, tmp_path, monkeypatch, capsys) -> None:
    base = tmp_path / "base.json"
    pb.record(base)
    current = json.loads(base.read_text(encoding="utf-8"))
    current["benches"]["stationary_bootstrap"]["normalized"] *= 2.0
    cur = tmp_path / "cur.json"
    cur.write_text(json.dumps(current), encoding="utf-8")
    assert pb.main(["compare", "--baseline", str(base), "--current", str(cur)]) == 1
    assert "REGRESSION" in capsys.readouterr().out


def test_cli_fail_closed_paths_exit_nonzero(tiny_sizes, tmp_path, monkeypatch, capsys) -> None:
    base = tmp_path / "base.json"
    pb.record(base)
    bad = tmp_path / "bad.json"
    bad.write_text("{nope", encoding="utf-8")
    assert pb.main(["compare", "--baseline", str(bad), "--current", str(base)]) == 2
    assert "error:" in capsys.readouterr().err
    assert (
        pb.main(["compare", "--baseline", str(base), "--current", str(base), "--threshold", "-1"])
        == 2
    )
    assert "error:" in capsys.readouterr().err
    assert (
        pb.main(["compare", "--baseline", str(tmp_path / "missing.json"), "--current", str(base)])
        == 2
    )
    assert "error:" in capsys.readouterr().err


def test_time_bench_caps_wide_numeric_pools(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class _Ctx:
        def __enter__(self) -> _Ctx:
            seen["entered"] = True
            return self

        def __exit__(self, *args: object) -> bool:
            seen["exited"] = True
            return False

    def fake_limits(**kwargs: object) -> _Ctx:
        seen["limits"] = kwargs.get("limits")
        return _Ctx()

    import threadpoolctl

    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: [{"num_threads": 8}])
    monkeypatch.setattr(threadpoolctl, "threadpool_limits", fake_limits)
    monkeypatch.setattr(pb, "REPEATS", 1)
    monkeypatch.setattr(pb, "_CALIBRATION_N", 8)
    monkeypatch.setattr(pb, "_CALIBRATION_REPS", 1)
    pb._time_bench(lambda rng, n: 0, 1, 4)
    assert seen["limits"] == 1
    assert seen["entered"] is True
    assert seen["exited"] is True


def test_time_bench_leaves_single_thread_pools_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    import threadpoolctl

    def fail_limits(**kwargs: object) -> None:
        raise AssertionError(kwargs)

    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: [{"num_threads": 1}])
    monkeypatch.setattr(threadpoolctl, "threadpool_limits", fail_limits)
    monkeypatch.setattr(pb, "REPEATS", 1)
    monkeypatch.setattr(pb, "_CALIBRATION_N", 8)
    monkeypatch.setattr(pb, "_CALIBRATION_REPS", 1)
    best, normalized, cal = pb._time_bench(lambda rng, n: 0, 1, 4)
    assert best > 0.0 and normalized > 0.0 and cal > 0.0


def test_registry_and_size_hooks_stay_in_sync() -> None:
    assert set(pb.BENCHES) == set(pb.BENCH_SIZES) == set(pb.BENCH_SEEDS)
    for name, size in pb.BENCH_SIZES.items():
        assert size >= 1, name
