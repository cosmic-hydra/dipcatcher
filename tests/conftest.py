"""Lab suite hooks: session caches, lake locks, and the slow-test list."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# The default test lane is explicitly offline.  MLflow otherwise starts a
# background configuration fetch on first use, even for file-backed tracking.
# Disable that optional telemetry before test modules import MLflow so the
# ``not network`` gate cannot disclose environment metadata or depend on egress.
os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
# GitHub Actions sets GITHUB_ACTIONS. Typer's option highlighter then splits
# "--flag" into "\x1b[1m-\x1b[0m\x1b[1m-flag", so substring asserts fail.
# NO_COLOR does not clear that when GITHUB_ACTIONS is set; Typer reads
# _TYPER_FORCE_DISABLE_TERMINAL at import. If rich_utils is already loaded,
# also clear the cached force bit.
os.environ["NO_COLOR"] = "1"
os.environ["_TYPER_FORCE_DISABLE_TERMINAL"] = "1"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# Do NOT pin OPENBLAS_CORETYPE. Forcing HASWELL segfaults numpy.linalg.lstsq
# on AMD EPYC. Leave core detection to OpenBLAS.
# Mirror Makefile TEST_ENV / CI thread pinning before any native extension
# can load a second OpenMP runtime. setdefault: make/CI env wins when set.
for _k, _v in (
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("VECLIB_MAXIMUM_THREADS", "1"),
    ("KMP_INIT_AT_FORK", "FALSE"),
    ("PYTHONFAULTHANDLER", "1"),
    ("TOKENIZERS_PARALLELISM", "false"),
):
    os.environ.setdefault(_k, _v)
if os.environ.get("OPENBLAS_CORETYPE", "").upper() == "HASWELL":
    os.environ.pop("OPENBLAS_CORETYPE", None)
if "typer.rich_utils" in sys.modules:
    sys.modules["typer.rich_utils"].FORCE_TERMINAL = False  # type: ignore[attr-defined]

import pytest  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.support.session_cache import bench_stats, data_stats, install  # noqa: E402

install()

_FIXTURE_TIMES: list[tuple[float, str, str]] = []
_SLOW_PATH = Path(__file__).with_name("slow_nodeids.txt")


def _slow_nodeids() -> set[str]:
    if not _SLOW_PATH.is_file():
        return set()
    wanted: set[str] = set()
    for line in _SLOW_PATH.read_text(encoding="utf-8").splitlines():
        nodeid = line.split("#", 1)[0].strip()
        if nodeid:
            wanted.add(nodeid)
    return wanted


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark node ids listed in ``tests/slow_nodeids.txt``.

    The PR gate runs ``-m 'not slow'``. The scheduled / workflow_dispatch full
    suite does not. Paths are node ids, not filenames, so moving a module
    without updating the list keeps the test in the fast gate.
    """
    wanted = _slow_nodeids()
    if not wanted:
        return
    slow = pytest.mark.slow
    for item in items:
        if item.nodeid in wanted:
            item.add_marker(slow)


@pytest.hookimpl(hookwrapper=True)
def pytest_fixture_setup(fixturedef: pytest.FixtureDef[object], request: pytest.FixtureRequest):
    started = time.perf_counter()
    outcome = yield
    elapsed = time.perf_counter() - started
    outcome.get_result()
    if elapsed >= 0.25:
        _FIXTURE_TIMES.append((elapsed, fixturedef.argname, request.node.nodeid))


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    log_path = os.environ.get("DIP_FIXTURE_LOG")
    if not log_path:
        return
    path = Path(log_path)
    worker = os.environ.get("PYTEST_XDIST_WORKER", "controller")
    # The controller does not run tests. Each worker writes its own file so
    # hit/miss counts are not dropped on the floor.
    if worker != "controller":
        path = path.with_name(f"{path.stem}.{worker}{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"worker {worker}",
        f"bench_cache {bench_stats()}",
        f"data_cache {data_stats()}",
        f"exitstatus {exitstatus}",
        "slowest fixtures (setup >= 0.25s):",
    ]
    ordered = sorted(_FIXTURE_TIMES, reverse=True)[:40]
    for elapsed, name, nodeid in ordered:
        lines.append(f"{elapsed:8.2f}s  {name}  {nodeid}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


collect_ignore_glob = ["**/__pycache__/**"]


# Hypothesis CI profile (A3 #5 supporting infra, DESIGN.md §9.5): CI sets
# HYPOTHESIS_PROFILE=ci for derandomized, fixed-budget property runs. Explicit
# per-test @settings still win over the profile; local runs are unaffected.
if os.environ.get("HYPOTHESIS_PROFILE") == "ci":
    from hypothesis import settings

    settings.register_profile("ci", derandomize=True, max_examples=100)
    settings.load_profile("ci")
