"""``run_research(jobs=...)`` schedules the live-optional benches; it must not change a blob.

The pool uses spawn, so the children cannot see monkeypatches: these tests run
real benches, picked from the cheapest measured ones (< 0.5 s each).
"""

from __future__ import annotations

import json

import pytest

from quant_fund.research.agent import _live_optional_benches
from quant_fund.research.live_optional_pool import _run_live_optional

_CHEAP = ("rough_paths", "distributional_ml", "leakage_redteam")


def _canon(blobs: dict[str, object]) -> str:
    return json.dumps(blobs, sort_keys=True, default=repr)


def test_pool_matches_sequential_blob_for_blob() -> None:
    assert set(_CHEAP) <= set(_live_optional_benches())
    sequential = _run_live_optional(_CHEAP, jobs=1, seed=7)
    pooled = _run_live_optional(_CHEAP, jobs=2, seed=7)
    assert list(pooled) == list(_CHEAP)
    assert _canon(pooled) == _canon(sequential)
    assert all(sequential.values())


def test_pool_reraises_a_failing_bench() -> None:
    with pytest.raises(KeyError, match="no_such_bench"):
        _run_live_optional(("rough_paths", "no_such_bench"), jobs=2, seed=7)
