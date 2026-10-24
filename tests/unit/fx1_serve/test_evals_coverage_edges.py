"""Branches in fx1.serve.evals the all-green audit never takes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum

import numpy as np
import pytest
from pydantic import ValidationError

from fx1.serve import evals as evals_mod
from fx1.serve.evals import (
    EvalRecord,
    EvalSpec,
    EvalSpecItemSchema,
    EvalSpecStore,
    EvalStore,
    _jsonable,
    _report_tasks,
    report_task_items,
    run_eval_record,
)
from fx1.serve.journal import ReplayResult


class _Color(Enum):
    RED = "red"


class _Journal:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self._payloads = payloads
        self.appended: list[dict[str, object]] = []

    def replay(self) -> ReplayResult:
        return ReplayResult(payloads=list(self._payloads), warnings=["replayed"])

    def append(self, payload: dict[str, object]) -> None:
        self.appended.append(payload)

    def compact(self, payloads: list[dict[str, object]]) -> None:
        self.appended.append({"compact": len(payloads)})


def _record(eval_id: str, status: str = "queued") -> EvalRecord:
    return EvalRecord(
        eval_id=eval_id,
        suite="tooluse",
        backend="byok",
        seed=0,
        status=status,  # type: ignore[arg-type]
        created_at=1.0,
    )


def _spec(spec_id: str) -> EvalSpec:
    return EvalSpec(
        spec_id=spec_id,
        name=spec_id,
        data_source_config={"type": "custom"},
        testing_criteria=[],
        metadata={},
        created_at=1.0,
    )


def test_schema_links_and_jsonable_leaves() -> None:
    with pytest.raises(ValidationError, match="checkpoint_dir"):
        EvalSpecItemSchema(suite="tooluse", backend="byok", checkpoint_dir="/ckpt")
    with pytest.raises(ValidationError, match="judge_backend"):
        EvalSpecItemSchema(suite="tooluse", judge_backend="byok")
    assert _jsonable(np.float64(1.5)) == 1.5
    assert _jsonable(_Color.RED) == "red"
    assert _jsonable(date(2026, 1, 2)) == "2026-01-02"
    assert _jsonable({"when": date(2026, 1, 2)}) == {"when": "2026-01-02"}


def test_eval_store_replay_delete_and_eviction() -> None:
    kept = _record("kept", "succeeded")
    journal = _Journal(
        [
            {"evicted": ["stale"], "deleted": "old"},
            {"note": "no record"},
            {"record": kept.model_dump(mode="json"), "key": "idem", "fp": "fp1"},
        ]
    )
    restored = EvalStore(4, journal)  # type: ignore[arg-type]
    assert restored.get("kept") is not None
    assert restored.get("old") is None
    assert restored.recover_warnings == ["replayed"]

    bare = EvalStore(1)
    assert bare.delete("missing") is None
    bare._compact_locked()
    bare.mark(_record("unjournaled"))
    bare.put(_record("a"), None, None)
    bare.put(_record("b"), "kb", "fb")
    assert bare.get("a") is None and bare.get("b") is not None

    queued = EvalStore(2, _Journal([]))  # type: ignore[arg-type]
    queued.put(_record("q"), None, None)
    cancelled = queued.cancel_pending()
    assert [rec.eval_id for rec in cancelled] == ["q"]
    assert cancelled[0].status == "cancelled"
    plain = EvalStore(2)
    plain.put(_record("q2"), None, None)
    assert plain.cancel_pending()[0].status == "cancelled"


def test_spec_store_replay_evicts_and_skips_non_records() -> None:
    live = _spec("live")
    journal = _Journal(
        [
            {"evicted": ["old-spec"], "deleted": "gone"},
            {"note": "skip"},
            {"record": live.model_dump(mode="json")},
        ]
    )
    restored = EvalSpecStore(4, journal)  # type: ignore[arg-type]
    assert restored.get("live") is not None
    assert restored.get("gone") is None

    bare = EvalSpecStore(1)
    bare._compact_locked()
    bare.put(_spec("s1"))
    bare.put(_spec("s2"))
    assert bare.get("s1") is None and bare.get("s2") is not None
    assert bare.delete("s2") is not None
    assert bare.delete("s2") is None


@dataclass
class _Report:
    ok: bool


def test_run_eval_record_seals_success_and_faults(monkeypatch: pytest.MonkeyPatch) -> None:
    def runner(model: object, seed: int, judge: object = None) -> _Report:
        assert model == "model" and seed == 0 and judge == "judge"
        return _Report(ok=True)

    monkeypatch.setattr(evals_mod, "eval_runner", lambda _suite: runner)
    record = _record("run")
    record.suite = "capability"
    run_eval_record(record, model="model", judge="judge")  # type: ignore[arg-type]
    assert record.status == "succeeded" and record.report == {"ok": True}

    def boom(model: object, seed: int, judge: object = None) -> _Report:
        raise RuntimeError("suite down")

    monkeypatch.setattr(evals_mod, "eval_runner", lambda _suite: boom)
    failed = _record("bad")
    failed.suite = "tooluse"
    run_eval_record(failed, model="model")  # type: ignore[arg-type]
    assert failed.status == "failed" and "RuntimeError" in (failed.error or "")


def test_report_task_rows_skip_non_objects() -> None:
    report = {"results": ["nope", {"task": "alpha", "passed": True}, {"task": 1, "passed": True}]}
    assert report_task_items(report)[0]["name"] == "alpha"
    assert _report_tasks(report) == {"alpha": True}
    assert report_task_items({"results": "nope"}) == []
