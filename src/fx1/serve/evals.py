"""fx-1 eval submissions — the seeded suite battery over the wire.

Every eval the harness ships takes a ``ModelFn``; a resolved serving
backend *is* one. This module is the bridge between them:

- ``EVAL_SUITES`` — lazily-loaded registry (suite name -> runner) so the
  eval import chain never loads for a server that only serves
  completions. ``accepts_judge`` marks the suites that take a grader
  ``ModelFn`` (capability and the external-bench adapter).
- ``EvalRecord`` — the bounded-store record a submission produces:
  suite/seed/backend + the serialized report, sealed on export as
  ``fx1_eval_record.v1``.
- ``EvalStore`` — LRU of records with an Idempotency-Key index (the
  same contract ``_JobStore`` gives run jobs).
- ``metered_model`` — wraps the resolved backend so every model call
  inside an eval lands on the metrics ledger under
  ``eval:{suite}:{backend}`` — never silent.
- ``run_eval_record`` — the worker: drive the runner, serialize the
  report fail-closed, stamp finish. Called from the jobs executor under
  the shared inflight/drain contract.

Evals always run under the decode pin ``{"temperature": 0.0}`` —
recorded on the record so a sealed eval receipt states the exact decode
config the evidence was produced under.
"""

from __future__ import annotations

import importlib
import json
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from math import comb
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from fx1.eval.suite import ModelFn
from fx1.serve.backends import SamplingParams
from fx1.serve.journal import JobJournal, _ClaimLocks

__all__ = [
    "EVAL_SUITES",
    "EvalDiff",
    "EvalDiffDelta",
    "EvalDiffSignificance",
    "EvalRecord",
    "EvalSpec",
    "EvalSpecItemSchema",
    "EvalSpecStore",
    "EvalStore",
    "EvalTaskTransition",
    "diff_eval_records",
    "eval_record_receipt",
    "eval_runner",
    "metered_model",
    "report_task_items",
    "run_eval_record",
    "run_wire",
    "spec_wire",
]

# suite -> (module, function, accepts_judge). Lazy: importing the eval
# package chain (bank seeding, adapters) costs real time at import, so
# the registry stores dotted paths and resolves on first run.
_EVAL_RUNNERS: dict[str, tuple[str, str, bool]] = {
    "capability": ("fx1.eval.capability", "run_capability_eval", True),
    "calibration": ("fx1.eval.calibration_eval", "run_calibration_eval", False),
    "tooluse": ("fx1.eval.tooluse_eval", "run_tooluse_eval", False),
    "retrieval": ("fx1.eval.retrieval_eval", "run_retrieval_eval", False),
    "ts_reasoning": ("fx1.eval.ts_reasoning", "run_ts_reasoning_eval", False),
    "ext_bench": ("fx1.eval.ext_bench", "run_ext_bench_eval", True),
    "options_reasoning": (
        "fx1.eval.options_reasoning_eval",
        "run_options_reasoning_eval",
        False,
    ),
}

EVAL_SUITES: tuple[str, ...] = tuple(_EVAL_RUNNERS)

EvalSuiteName = Literal[
    "capability",
    "calibration",
    "tooluse",
    "retrieval",
    "ts_reasoning",
    "ext_bench",
    "options_reasoning",
]

# Decode pin every eval runs under — recorded verbatim on the record so
# the sealed receipt states the config the evidence came from.
EVAL_SAMPLING = SamplingParams()


def eval_runner(suite: str) -> Callable[..., Any]:
    """Resolve a suite name to its ``run_*_eval`` callable.

    Fail closed: unknown suite names are a KeyError at submit time, never
    a worker fault. Import happens here (not module top) so the eval
    chain loads only when an eval actually runs.
    """
    module_name, fn_name, _accepts_judge = _EVAL_RUNNERS[suite]
    fn: Callable[..., Any] = getattr(importlib.import_module(module_name), fn_name)
    return fn


def suite_accepts_judge(suite: str) -> bool:
    return _EVAL_RUNNERS[suite][2]


class EvalSpecItemSchema(BaseModel):
    """The suite knobs an eval spec pins — same fields as the
    ``/harness/evals`` submission minus credentials (a spec is stored,
    export-safe state: BYOK keys attach to the run body only)."""

    model_config = ConfigDict(extra="forbid")

    suite: EvalSuiteName
    seed: int = Field(default=0, ge=0)
    backend: Literal["hosted_k3", "local_fx1", "byok"] | None = None
    fallbacks: list[Literal["hosted_k3", "local_fx1", "byok"]] = Field(
        default_factory=list, max_length=2
    )
    checkpoint_dir: str | None = None
    judge_backend: Literal["hosted_k3", "local_fx1", "byok"] | None = None
    timeout_s: float | None = Field(default=None, gt=0, le=3600)

    @model_validator(mode="after")
    def _schema_valid(self) -> EvalSpecItemSchema:
        if len(set(self.fallbacks)) != len(self.fallbacks) or self.backend in self.fallbacks:
            raise ValueError("fallbacks must be distinct and not repeat the backend")
        chain = {*self.fallbacks} | ({self.backend} if self.backend is not None else set())
        if self.checkpoint_dir is not None and "local_fx1" not in chain:
            raise ValueError("checkpoint_dir applies only to a 'local_fx1' link")
        if self.judge_backend is not None and not suite_accepts_judge(self.suite):
            raise ValueError("judge_backend is supported on judge suites only")
        return self


def _jsonable(obj: Any) -> Any:
    """Deep-normalize a report subtree to JSON-safe leaves — numpy
    arrays/scalars, tuples/sets, enums, and date-likes convert;
    anything still unserializable raises at the caller's check."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Enum):
        return _jsonable(obj.value)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in obj]
    return obj


def report_dump(report: Any) -> dict[str, Any]:
    """Serialize a suite report fail-closed — pydantic or dataclass,
    nothing else seals into a record. The dict must be JSON-safe: it is
    journaled and served verbatim, so numpy leaves/dataclasses are
    normalized away rather than left to crash the serializer later."""
    if isinstance(report, BaseModel):
        out = _jsonable(report.model_dump(mode="python"))
    elif is_dataclass(report) and not isinstance(report, type):
        out = _jsonable(asdict(report))
    else:
        raise TypeError(f"eval report is not serializable: {type(report).__name__}")
    # model_dump and asdict both return a dict, so this guard never fires.
    if not isinstance(out, dict):  # pragma: no cover
        raise TypeError("eval report did not serialize to a dict")  # pragma: no cover
    json.dumps(out)  # fail-closed proof the dump is wire-safe
    return out


def metered_model(
    backend: Any,
    *,
    metric_key: str,
    record: Callable[[str, bool, float, dict[str, int] | None], None],
) -> ModelFn:
    """Wrap ``backend`` as a ModelFn that meters every call.

    Each suite call lands on the complete ledger under ``metric_key``
    (``eval:{suite}:{backend}``) — eval traffic is metered model work,
    visible but never conflated with user completions.
    """

    def _fn(messages: list[dict[str, str]]) -> str:
        started = time.monotonic()
        try:
            out = backend.complete(messages, sampling=EVAL_SAMPLING)
        except Exception:
            record(metric_key, False, (time.monotonic() - started) * 1000.0, None)
            raise
        record(
            metric_key,
            True,
            (time.monotonic() - started) * 1000.0,
            getattr(backend, "last_usage", None),
        )
        out_str: str = out
        return out_str

    return _fn


class EvalRecord(BaseModel):
    """One submitted eval's durable record — sealed on receipt export.

    Never carries request-side credentials (``byok``/keys): the record is
    evidence, and evidence must be safe to export.
    """

    model_config = ConfigDict(extra="forbid")

    eval_id: str
    suite: str
    # The serving link actually used (attempts carries the full chain).
    backend: str
    seed: int
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    created_at: float
    finished_at: float | None = None
    report: dict[str, Any] | None = None
    error: str | None = None
    # Ordered chain trace — {backend, ok, error_class} per link tried.
    attempts: list[dict[str, Any]] | None = None
    # The decode pin the eval ran under (always the 0.0 default today).
    sampling: dict[str, Any] | None = None
    # Terminal-state webhook (the job contract): on every terminal
    # transition the record is POSTed to ``callback_url`` — never the
    # secret, which lives only as a signing key.
    callback_url: str | None = None
    callback_status: Literal["delivered", "failed"] | None = None
    callback_error: str | None = None
    callback_attempts: int = 0
    # /v1/evals binding: the spec this run was created under and the
    # model string the run requested (``backend`` records the resolved
    # link — a 'ft:…' model maps to 'local_fx1' + the card's checkpoint,
    # so the requested name would otherwise be unrecoverable). Both are
    # absent on direct ``/harness/evals`` submissions.
    eval_spec: str | None = None
    eval_model: str | None = None
    _callback_secret: str | None = PrivateAttr(default=None)
    _callback_fired: bool = PrivateAttr(default=False)
    _callback_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)


class EvalStore:
    """Bounded LRU of eval records + an Idempotency-Key index — the
    ``_JobStore`` contract for eval submissions. Eviction drops the
    idempotency mapping with the record.

    With a ``JobJournal`` bound (``--state-dir`` on serve), every
    transition is journaled before the store mutates, and boot replays
    the chain: terminal records come back as-was; evals still queued or
    running at the crash are restored as ``failed`` with an honest
    restart error (never re-run — the request payload isn't journaled),
    and their idempotency keys still resolve so a retried submission
    returns the lost record instead of duplicating work.
    """

    def __init__(self, max_entries: int, journal: JobJournal | None = None) -> None:
        self._lock = threading.Lock()
        self._max = max_entries
        self._claims = _ClaimLocks(max_entries)
        self._records: OrderedDict[str, EvalRecord] = OrderedDict()
        self._keys: OrderedDict[str, tuple[str, str]] = OrderedDict()
        self._record_key: dict[str, str] = {}
        self._record_fp: dict[str, str] = {}
        self._journal = journal
        self.recover_warnings: list[str] = []
        if journal is not None:
            res = journal.replay()
            self.recover_warnings = list(res.warnings)
            now = time.time()
            for payload in res.payloads:
                for evict in payload.get("evicted") or ():
                    self._drop(str(evict))
                deleted = payload.get("deleted")
                if deleted is not None:
                    self._drop(str(deleted))
                    continue
                if "record" not in payload:
                    continue
                rec = EvalRecord.model_validate(payload["record"])
                # Signing secrets are not journaled; recovered records never re-deliver.
                rec._callback_fired = True
                self._records[rec.eval_id] = rec
                self._records.move_to_end(rec.eval_id)
                key, fp = payload.get("key"), payload.get("fp")
                if key is not None and fp is not None:
                    self._keys[key] = (fp, rec.eval_id)
                    self._record_key[rec.eval_id] = key
                    self._record_fp[rec.eval_id] = fp
            for rec in self._records.values():
                if rec.status in ("queued", "running"):
                    rec.status = "failed"
                    rec.error = "process restarted before the eval reached a terminal state"
                    rec.finished_at = now
            self._compact_locked()

    def _drop(self, eval_id: str) -> None:
        self._records.pop(eval_id, None)
        key = self._record_key.pop(eval_id, None)
        self._record_fp.pop(eval_id, None)
        if key is not None:
            self._keys.pop(key, None)

    def _record(self, rec: EvalRecord) -> dict[str, Any]:
        return {
            "record": rec.model_dump(mode="json"),
            "key": self._record_key.get(rec.eval_id),
            "fp": self._record_fp.get(rec.eval_id),
        }

    def _compact_locked(self) -> None:
        """Rewrite the journal with only the live state — called on boot
        post-replay so dead history and torn tails don't accumulate."""
        if self._journal is not None:
            self._journal.compact([self._record(r) for r in self._records.values()])

    def mark(self, rec: EvalRecord) -> None:
        """Journal a status transition made outside the store (the worker
        mutates ``rec`` in place; this makes each hop durable)."""
        if self._journal is not None:
            with self._lock:
                self._journal.append(self._record(rec))

    def start(self, eval_id: str) -> EvalRecord | None:
        """Atomic queued→running claim — the worker's handshake against
        the cancel path. Takes the same lock ``cancel`` does, so a
        cancel that lands first can never be overwritten back to
        running. Returns the record only when it was still queued;
        ``None`` tells the worker to drop it (cancelled, evicted, or
        unknown)."""
        with self._lock:
            rec = self._records.get(eval_id)
            if rec is None or rec.status != "queued":
                return None
            rec.status = "running"
            if self._journal is not None:
                self._journal.append(self._record(rec))
            return rec

    @property
    def capacity(self) -> int:
        return self._max

    def get(self, eval_id: str) -> EvalRecord | None:
        with self._lock:
            return self._records.get(eval_id)

    def list_records(
        self,
        status: str | None = None,
        suite: str | None = None,
        limit: int | None = None,
        spec: str | None = None,
    ) -> tuple[list[EvalRecord], int]:
        """Newest-first snapshot, optionally filtered; returns
        (page, total-before-paging) like the jobs list. ``spec`` filters
        to the /v1/evals runs bound to one eval spec."""
        with self._lock:
            records = list(self._records.values())
        records.reverse()
        if status is not None:
            records = [r for r in records if r.status == status]
        if suite is not None:
            records = [r for r in records if r.suite == suite]
        if spec is not None:
            records = [r for r in records if r.eval_spec == spec]
        total = len(records)
        if limit is not None:
            records = records[:limit]
        return records, total

    def delete(self, eval_id: str) -> EvalRecord | None:
        """Remove a terminal record + its idempotency mapping — the run
        delete for ``/v1/evals``. Journaled as a tombstone so replay
        can't resurrect it."""
        with self._lock:
            rec = self._records.get(eval_id)
            if rec is None:
                return None
            self._drop(eval_id)
            if self._journal is not None:
                self._journal.append({"deleted": eval_id})
            return rec

    def cancel(self, eval_id: str) -> tuple[EvalRecord | None, str]:
        """Cooperative cancel: a 'queued' record flips to 'cancelled' (the
        worker frees its slot on dequeue without running). Running and
        terminal records report their status so the route can 409."""
        with self._lock:
            rec = self._records.get(eval_id)
            if rec is None:
                return None, "missing"
            if rec.status == "queued":
                rec.status = "cancelled"
                rec.finished_at = time.time()
                if self._journal is not None:
                    self._journal.append(self._record(rec))
                return rec, "cancelled"
            return rec, rec.status

    def cancel_pending(self) -> list[EvalRecord]:
        """Shutdown path: queued records flip to cancelled so their futures
        never start."""
        with self._lock:
            out = [r for r in self._records.values() if r.status == "queued"]
            for rec in out:
                rec.status = "cancelled"
                rec.finished_at = time.time()
                if self._journal is not None:
                    self._journal.append(self._record(rec))
            return out

    def claim_lock(self, key: str | None) -> AbstractContextManager[None]:
        """Serialize a key's lookup -> create -> record insert: a keyed
        retry waits for an in-flight twin instead of double-executing
        next to it (the job/eval submit contract post-claim-locks)."""
        return self._claims.hold(key)

    def async_claim_lock(self, key: str | None) -> AbstractAsyncContextManager[None]:
        return self._claims.ahold(key)

    def get_key(self, key: str) -> tuple[str, str] | None:
        with self._lock:
            hit = self._keys.get(key)
            if hit is not None:
                self._keys.move_to_end(key)
            return hit

    def put(
        self,
        record: EvalRecord,
        key: str | None,
        fingerprint: str | None,
    ) -> None:
        with self._lock:
            self._records[record.eval_id] = record
            self._records.move_to_end(record.eval_id)
            if key is not None and fingerprint is not None:
                self._keys[key] = (fingerprint, record.eval_id)
                self._record_key[record.eval_id] = key
                self._record_fp[record.eval_id] = fingerprint
            evicted: list[str] = []
            while len(self._records) > self._max:
                evicted_id, _ = self._records.popitem(last=False)
                old_key = self._record_key.pop(evicted_id, None)
                self._record_fp.pop(evicted_id, None)
                if old_key is not None:
                    self._keys.pop(old_key, None)
                evicted.append(evicted_id)
            if self._journal is not None:
                payload = self._record(record)
                if evicted:
                    payload["evicted"] = evicted
                self._journal.append(payload)


def run_eval_record(
    record: EvalRecord,
    *,
    model: ModelFn,
    judge: ModelFn | None = None,
) -> None:
    """Worker: drive the suite runner, serialize the report, stamp finish.

    Called under the shared inflight slot by the jobs executor; the
    caller owns status transitions ('running' in, terminal out). A runner
    fault lands on the record as ``failed`` + error — never propagates
    into the worker loop.
    """
    try:
        fn = eval_runner(record.suite)
        kwargs: dict[str, Any] = {}
        if suite_accepts_judge(record.suite):
            kwargs["judge"] = judge
        report = fn(model, seed=record.seed, **kwargs)
        record.report = report_dump(report)
        record.status = "succeeded"
    except Exception as exc:  # noqa: BLE001 — worker faults land in the record
        record.error = f"{type(exc).__name__}: {exc}"
        record.status = "failed"
    finally:
        record.finished_at = time.time()


def eval_record_receipt(record: dict[str, Any]) -> dict[str, Any]:
    """Seal an eval record as ``fx1_eval_record.v1`` — the same claim as
    the completion/job receipts: these bytes were the recorded run."""
    from fx1.serve.ops_receipt import _ops_receipt

    return _ops_receipt("fx1_eval_record", "fx1_eval_record.v1", record)


# ---- eval diffs: the promotion-gate primitive -------------------------------
#
# ``GET /harness/evals/{base}/diff/{candidate}`` and the SDK twin diff two
# terminal eval records: which seeded tasks flipped, which direction the
# honesty/ship gate moved, and the by_kind counter deltas. A diff is
# *comparable* only when both records ran the same suite over the same
# bank (``eval_bank_sha256``) at the same seed — a cross-bank diff is still
# served but reads ``comparable: false`` / ``verdict: "unknown"`` rather
# than pretending the numbers mean anything.


class EvalTaskTransition(BaseModel):
    """One seeded task whose pass/fail flipped between base and candidate."""

    model_config = ConfigDict(extra="forbid")

    task: str
    base: bool
    candidate: bool
    direction: Literal["fixed", "regressed"]


class EvalDiffDelta(BaseModel):
    """A numeric leaf that moved between the two reports' ``by_kind``."""

    model_config = ConfigDict(extra="forbid")

    path: str
    base: float
    candidate: float
    delta: float


class EvalDiffSignificance(BaseModel):
    """McNemar exact sign test on the discordant task pairs.

    ``n_fixed`` tasks went fail→pass and ``n_regressed`` went pass→fail;
    under the null the split is a fair coin, so ``p_value`` is the exact
    two-sided sign test. ``verdict`` stays the observed direction — the
    significance block is the separate evidence of whether the move is
    distinguishable from noise at this bank size.
    """

    model_config = ConfigDict(extra="forbid")

    n_fixed: int
    n_regressed: int
    p_value: float
    significant_p05: bool


class EvalDiff(BaseModel):
    """Deterministic comparison of two terminal eval records."""

    model_config = ConfigDict(extra="forbid")

    object: Literal["eval_diff"] = "eval_diff"
    base_eval_id: str
    candidate_eval_id: str
    base_backend: str
    candidate_backend: str
    same_suite: bool
    same_seed: bool
    same_bank: bool
    comparable: bool
    gate_base: bool | None
    gate_candidate: bool | None
    gate_transition: Literal["opened", "closed", "unchanged", "unknown"]
    tasks_fixed: list[str]
    tasks_regressed: list[str]
    tasks_only_base: list[str]
    tasks_only_candidate: list[str]
    deltas: list[EvalDiffDelta]
    significance: EvalDiffSignificance | None
    verdict: Literal["improved", "regressed", "unchanged", "unknown"]


def report_task_items(report: dict[str, Any]) -> list[dict[str, Any]]:
    """The suite's per-task verdict rows — ``[{name, passed, row}]`` over
    the declared verdict containers. The ``/v1/evals`` output_items
    surface serves these verbatim; an eval with no per-task array
    returns an empty list (never fabricated rows)."""
    for container, name_key, flag_key in _TASK_VERDICT_SHAPES:
        items = report.get(container)
        if not isinstance(items, list):
            continue
        out: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            flag = item.get(flag_key)
            name = item.get(name_key)
            if isinstance(flag, bool) and isinstance(name, str):
                out.append({"name": name, "passed": flag, "row": item})
        if out:
            return out
    return []


# ---- /v1/evals: the OpenAI Evals-shaped spec/run surface -------------------
#
# OpenAI's evals API separates the *eval* (a named container declaring the
# datasource shape + grading criteria) from its *runs* (executions against
# a model). The harness's ``/harness/evals`` record IS a run; the spec is
# the new container: ``data_source_config.item_schema`` pins the suite
# knobs (suite/seed/fallbacks/judge — never credentials), and each run
# binds ``EvalRecord.eval_spec`` back to it.
#
# ``EvalSpecStore`` is the same bounded-LRU + hash-chained-journal
# contract as EvalStore, minus idempotency (specs are synchronous creates
# — the dedupe contract lives on run creation, which rides _submit_eval).


class EvalSpec(BaseModel):
    """A declared eval: name + datasource config + grading criteria."""

    model_config = ConfigDict(extra="forbid")

    spec_id: str
    name: str
    # OpenAI's shape: {"type": "custom", "item_schema": {...}} — the
    # item_schema carries the harness submission knobs (suite, seed,
    # fallbacks, judge_backend, checkpoint_dir, timeout_s). Validated
    # against EvalSpecItemSchema at create — never request-side creds.
    data_source_config: dict[str, Any]
    testing_criteria: list[dict[str, Any]]
    metadata: dict[str, str]
    created_at: float


class EvalSpecStore:
    """Bounded LRU of eval specs + tombstone-journaled deletes.

    Payload kinds on ``eval_specs.jsonl``: ``{"record": spec, "evicted"?
    : [...]}`` on write, ``{"deleted": spec_id}`` on delete. Boot replay
    applies them in order — a deleted spec never resurrects.
    """

    def __init__(self, max_entries: int, journal: JobJournal | None = None) -> None:
        self._lock = threading.Lock()
        self._max = max_entries
        self._specs: OrderedDict[str, EvalSpec] = OrderedDict()
        self._journal = journal
        self.recover_warnings: list[str] = []
        if journal is not None:
            res = journal.replay()
            self.recover_warnings = list(res.warnings)
            for payload in res.payloads:
                for evict in payload.get("evicted") or ():
                    self._specs.pop(str(evict), None)
                deleted = payload.get("deleted")
                if deleted is not None:
                    self._specs.pop(str(deleted), None)
                    continue
                if "record" not in payload:
                    continue
                spec = EvalSpec.model_validate(payload["record"])
                self._specs[spec.spec_id] = spec
                self._specs.move_to_end(spec.spec_id)
            self._compact_locked()

    def _drop(self, spec_id: str) -> None:
        self._specs.pop(spec_id, None)

    def _write(self, payload: dict[str, Any]) -> None:
        if self._journal is not None:
            self._journal.append(payload)

    def _compact_locked(self) -> None:
        if self._journal is not None:
            self._journal.compact(
                [{"record": s.model_dump(mode="json")} for s in self._specs.values()]
            )

    def put(self, spec: EvalSpec) -> None:
        with self._lock:
            self._specs[spec.spec_id] = spec
            self._specs.move_to_end(spec.spec_id)
            evicted: list[str] = []
            while len(self._specs) > self._max:
                evicted_id, _ = self._specs.popitem(last=False)
                evicted.append(evicted_id)
            payload: dict[str, Any] = {"record": spec.model_dump(mode="json")}
            if evicted:
                payload["evicted"] = evicted
            self._write(payload)

    def get(self, spec_id: str) -> EvalSpec | None:
        with self._lock:
            return self._specs.get(spec_id)

    def update(self, spec: EvalSpec) -> None:
        """Journal a spec mutation — name/metadata always, the declared
        shape on the unbound hot-reload path (the route/SDK freeze it once
        a run binds)."""
        with self._lock:
            self._write({"record": spec.model_dump(mode="json")})

    def delete(self, spec_id: str) -> EvalSpec | None:
        with self._lock:
            spec = self._specs.pop(spec_id, None)
            if spec is not None:
                self._write({"deleted": spec_id})
            return spec

    def list_specs(self, *, limit: int, after: str | None) -> tuple[list[EvalSpec], bool]:
        """Newest-first page — ``after`` is a spec-id cursor like the jobs
        list; returns (page, has_more). An unknown cursor raises
        ``ValueError`` — a mistyped cursor must fail closed, never
        masquerade as end-of-list."""
        with self._lock:
            specs = list(self._specs.values())
        specs.reverse()
        if after is not None:
            idx = next((i for i, s in enumerate(specs) if s.spec_id == after), None)
            if idx is None:
                raise ValueError(f"cursor {after!r} is not a spec id")
            specs = specs[idx + 1 :]
        page = specs[:limit]
        return page, len(specs) > limit


_RUN_STATUS: dict[str, str] = {
    "queued": "queued",
    "running": "in_progress",
    "succeeded": "completed",
    "failed": "failed",
    "cancelled": "canceled",
}


def spec_wire(spec: EvalSpec) -> dict[str, Any]:
    """The OpenAI eval object for a stored spec."""
    return {
        "id": spec.spec_id,
        "object": "eval",
        "name": spec.name,
        "data_source_config": spec.data_source_config,
        "testing_criteria": spec.testing_criteria,
        "metadata": spec.metadata,
        "created_at": int(spec.created_at),
    }


def run_wire(record: EvalRecord) -> dict[str, Any]:
    """The OpenAI eval.run object over an EvalRecord.

    ``result_counts`` appears only on completed runs — a failed/canceled
    run carries ``error``, never fabricated task counts. The suites emit
    one verdict stream per task (not a per-criterion breakdown), so
    ``per_testing_criteria_results`` stays empty rather than pretending a
    per-criterion measurement exists.
    """
    out: dict[str, Any] = {
        "id": f"evalrun_{record.eval_id}",
        "object": "eval.run",
        "eval_id": record.eval_spec,
        "model": record.eval_model or record.backend,
        "status": _RUN_STATUS.get(record.status, record.status),
        "created_at": int(record.created_at),
        "suite": record.suite,
        "seed": record.seed,
        "backend": record.backend,
        "per_testing_criteria_results": [],
        # the run's sealed evidence twin — the /harness/evals receipt
        "receipt_url": f"/harness/evals/{record.eval_id}/receipt",
    }
    if record.status == "succeeded" and isinstance(record.report, dict):
        items = report_task_items(record.report)
        passed = sum(1 for t in items if t["passed"])
        out["result_counts"] = {
            "total": len(items),
            "passed": passed,
            "failed": len(items) - passed,
            "errored": 0,
        }
    if record.status in ("failed", "cancelled"):
        out["error"] = {
            "code": "eval_run_failed" if record.status == "failed" else "eval_run_canceled",
            "message": record.error or record.status,
        }
    return out


# Per-task verdict shapes by suite: run_suite-style ``results`` carry
# ``task``/``passed``; tooluse's ``outcomes`` carry ``task_id``/``completed``;
# retrieval's ``results`` carry ``question_id``/``correct``. Each pair is
# the suite's declared per-task verdict — the diffs read all three.
_TASK_VERDICT_SHAPES: tuple[tuple[str, str, str], ...] = (
    ("results", "task", "passed"),
    ("outcomes", "task_id", "completed"),
    ("results", "question_id", "correct"),
)


def _report_tasks(report: dict[str, Any]) -> dict[str, bool]:
    """task name -> passed, over the report's per-task arrays; absent → empty."""
    out: dict[str, bool] = {}
    for container, name_key, flag_key in _TASK_VERDICT_SHAPES:
        items = report.get(container)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            flag = item.get(flag_key)
            name = item.get(name_key)
            if isinstance(flag, bool) and isinstance(name, str):
                out.setdefault(name, flag)
    return out


def _numeric_leaves(node: Any, prefix: str = "") -> dict[str, float]:
    """Flatten numeric leaves of a report subtree; bools are not numbers."""
    out: dict[str, float] = {}
    if isinstance(node, dict):
        for key in node:
            out.update(_numeric_leaves(node[key], f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        out[prefix] = float(node)
    return out


def _sign_test_pvalue(k: int, n: int) -> float:
    """Exact two-sided sign-test p on Binomial(n, 0.5), integer-exact.

    The extreme set under the symmetric null is both tails at
    ``|X - n/2| >= |k - n/2|`` — the tails are disjoint unless k sits on
    the center (p = 1). Deterministic ``math.comb`` arithmetic; a float
    binomial coefficient or a normal approximation would drift across
    platforms.
    """
    if n <= 0 or 2 * k == n:
        return 1.0
    d = min(k, n - k)
    tail: int = sum(comb(n, j) for j in range(d + 1))
    p: float = min(1.0, 2.0 * tail / 2**n)
    return p


def diff_eval_records(base: EvalRecord, candidate: EvalRecord) -> EvalDiff:
    """Diff two eval records into a promotion-gate verdict.

    Fails closed on the caller's side: both records must be terminal with
    a serialized report — the route 409s otherwise; this function assumes
    the contract and never invents a diff over a missing report.
    """
    base_report = base.report or {}
    cand_report = candidate.report or {}

    same_suite = base.suite == candidate.suite
    same_seed = base.seed == candidate.seed
    base_bank = base_report.get("eval_bank_sha256")
    cand_bank = cand_report.get("eval_bank_sha256")
    same_bank = isinstance(base_bank, str) and isinstance(cand_bank, str) and base_bank == cand_bank
    # Banks are seeded by construction: same suite + same seed pins the
    # bank. A stamped mismatch overrides that; absent stamps don't —
    # suites that don't emit ``eval_bank_sha256`` stay diffable while
    # ``same_bank`` still reports the stamp evidence honestly.
    bank_mismatch = (
        isinstance(base_bank, str) and isinstance(cand_bank, str) and base_bank != cand_bank
    )
    comparable = same_suite and same_seed and not bank_mismatch

    base_tasks = _report_tasks(base_report)
    cand_tasks = _report_tasks(cand_report)
    if not comparable:
        # A mismatched bank makes task-name pairing meaningless — the
        # same name in a different bank is a different task.
        base_tasks = {}
        cand_tasks = {}

    transitions: list[EvalTaskTransition] = []
    for name in sorted(base_tasks.keys() & cand_tasks.keys()):
        if base_tasks[name] != cand_tasks[name]:
            transitions.append(
                EvalTaskTransition(
                    task=name,
                    base=base_tasks[name],
                    candidate=cand_tasks[name],
                    direction="fixed" if cand_tasks[name] else "regressed",
                )
            )
    tasks_regressed = [t.task for t in transitions if t.direction == "regressed"]
    tasks_fixed = [t.task for t in transitions if t.direction == "fixed"]

    gate_base = base_report.get("honesty_gate_passed")
    gate_cand = cand_report.get("honesty_gate_passed")
    gate_b = gate_base if isinstance(gate_base, bool) else None
    gate_c = gate_cand if isinstance(gate_cand, bool) else None
    gate_transition: Literal["opened", "closed", "unchanged", "unknown"]
    if gate_b is None or gate_c is None:
        gate_transition = "unknown"
    elif gate_b == gate_c:
        gate_transition = "unchanged"
    else:
        gate_transition = "opened" if gate_c else "closed"

    base_leaves = _numeric_leaves(base_report.get("by_kind"))
    cand_leaves = _numeric_leaves(cand_report.get("by_kind"))
    deltas = [
        EvalDiffDelta(
            path=p,
            base=base_leaves[p],
            candidate=cand_leaves[p],
            delta=cand_leaves[p] - base_leaves[p],
        )
        for p in sorted(base_leaves.keys() & cand_leaves.keys())
        if base_leaves[p] != cand_leaves[p]
    ]

    if not comparable:
        verdict: Literal["improved", "regressed", "unchanged", "unknown"] = "unknown"
        significance = None
    else:
        if tasks_regressed or gate_transition == "closed":
            verdict = "regressed"
        elif tasks_fixed or gate_transition == "opened":
            verdict = "improved"
        else:
            verdict = "unchanged"
        n_fixed = len(tasks_fixed)
        n_regressed = len(tasks_regressed)
        p = _sign_test_pvalue(n_regressed, n_fixed + n_regressed)
        significance = EvalDiffSignificance(
            n_fixed=n_fixed,
            n_regressed=n_regressed,
            p_value=p,
            significant_p05=p < 0.05,
        )

    return EvalDiff(
        base_eval_id=base.eval_id,
        candidate_eval_id=candidate.eval_id,
        base_backend=base.backend,
        candidate_backend=candidate.backend,
        same_suite=same_suite,
        same_seed=same_seed,
        same_bank=same_bank,
        comparable=comparable,
        gate_base=gate_b,
        gate_candidate=gate_c,
        gate_transition=gate_transition,
        tasks_fixed=tasks_fixed,
        tasks_regressed=tasks_regressed,
        tasks_only_base=sorted(base_tasks.keys() - cand_tasks.keys()),
        tasks_only_candidate=sorted(cand_tasks.keys() - base_tasks.keys()),
        deltas=deltas,
        significance=significance,
        verdict=verdict,
    )
