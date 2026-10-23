"""jobs_audit — ``/harness/jobs`` + ``/harness/runs`` deep audit.

Probe battery over the async job orchestration surface: the full
``queued→running→terminal`` lifecycle (transition ordering, monotonic
timestamps, no status regression, terminal records immutable), the
submission validator's refusal envelopes, the executor contract
(exception → ``failed`` + error, nonzero exit → ``succeeded`` with
``result.ok=False`` — ``status`` tracks execution, ``ok`` the
command's verdict), Idempotency-Key dedup (body-``replayed`` marker /
409 conflict / parallel single-execution / credential scoping — no
replay header on this surface), the gate (drain latch, ``max_inflight``
admission, auth-before-gate ordering), signed terminal webhooks
(fire-once, retry policy, HMAC — unsigned sends no webhook headers;
failure never resurrects; ``callback_*`` bookkeeping lands after the
terminal flip), list/filter/paging honesty, ``--state-dir`` journaled
durability (terminal restores as-was, in-flight fails closed honestly,
idempotency keys survive), ``/harness/jobs/batch`` per-item failure
isolation, the ``/harness/runs`` synchronous twin (same
gate/idempotency/envelope rules), per-key metering (``uses`` bills
every authorized call including gate-refused; rpm refusal code is
``rate_limited``), the receipt's digested ``record`` projection, and
the store-level claim contract.

Found while building this lane (fixed in the same commit):

* ``_JobStore`` had no atomic start claim: the executor worker did a
  lock-free ``if job.status == "cancelled"`` check-then-set, so a
  cancel landing between the check and the ``running`` write
  resurrected a cancelled job — it ran to ``succeeded`` and journaled
  ``cancelled→running→succeeded`` (and replayed the same resurrection
  on restart). The eval lane fixed the same defect class with
  ``EvalStore.start``; ``_JobStore`` now gets the same primitive and
  ``_exec`` runs the transition through it under the store lock.
* ``job_store.put`` ran *after* ``jobs_executor.submit``: the worker's
  first ``mark`` could journal a transition for a record the journal
  had never seen, the GET surface 404'd a job whose submission had
  already 202'd, and a refused ``submit`` left a journaled ghost
  record. The store now registers the job before hand-off, and a
  refused ``submit`` leaves a ``{"deleted": job_id}`` tombstone so
  replay drops it (the EvalStore convention).
* ``/harness/runs`` let a runner fault (e.g. ``TimeoutExpired`` on an
  over-time command) escape as a bare plain-text 500 while the jobs
  twin captured the same fault honestly into ``job.error`` — the route
  now folds unhandled executor faults into the ``{"detail","code"}``
  envelope (500 ``internal``).

Sealed ``jobs_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

from quant_fund.schemas.receipt import verify_receipt_payload
from quant_fund.utils.reproducibility import git_revision

__all__ = ["jobs_audit", "jobs_audit_bench"]

_API_KEY_ENV = "FX1_API_KEY"
_ROOT = "k3y-material"
_WAIT_S = 15.0
_JOBS = "/harness/jobs"
_RUNS = "/harness/runs"
_DRAIN = "/harness/drain"
_IDEM = "Idempotency-Key"
_TERMINAL = {"succeeded", "failed", "cancelled"}
_RANK = {"queued": 0, "running": 1, "succeeded": 2, "failed": 2, "cancelled": 2}
_JOB_FIELDS = {
    "job_id",
    "status",
    "created_at",
    "finished_at",
    "result",
    "error",
    "callback_url",
    "callback_status",
    "callback_error",
    "callback_attempts",
}
_RESULT_FIELDS = {
    "command",
    "exit_code",
    "stdout",
    "stderr",
    "ok",
    "timeout_s",
    "replayed",
    "stdout_truncated",
    "stderr_truncated",
}


# ---------------------------------------------------------------------------
# Stub plumbing — runners, app/client factories, poll helpers
# ---------------------------------------------------------------------------


def _fast_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
    del argv, timeout_s
    return 0, "ok", ""


class _Runner:
    """``Harness`` runner stub — records argv, optional release gate,
    optional fault injection; everything the executor probes need to
    know about what actually ran."""

    def __init__(
        self,
        gate: threading.Event | None = None,
        *,
        raise_exc: BaseException | None = None,
        exit_code: int = 0,
        stdout: str = "ok",
        stderr: str = "",
    ) -> None:
        self.calls: list[list[str]] = []
        self.gate = gate
        self.entered = threading.Event()
        self.raise_exc = raise_exc
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr

    def __call__(self, argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        del timeout_s
        self.calls.append(list(argv))
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(30)
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.exit_code, self.stdout, self.stderr


def _make_app(
    *,
    runner: Callable[[list[str], int], tuple[int, str, str]] | None = None,
    state_dir: Path | None = None,
    **create_kw: Any,
) -> FastAPI:
    """``create_app`` under the ambient env (``_audit_context`` already
    swept ``FX1_*``); the backend resolver is never exercised by the
    harness surface."""
    import fx1.serve.api as api_mod  # noqa: PLC0415
    from fx1.harness import Harness  # noqa: PLC0415
    from fx1.serve.conv_audit import _StubBackend  # noqa: PLC0415

    return api_mod.create_app(
        harness=Harness(runner=runner or _fast_runner),
        backend_resolver=lambda *a, **k: _StubBackend(),
        state_dir=state_dir,
        **create_kw,
    )


def _client(
    runner: Callable[[list[str], int], tuple[int, str, str]] | None = None,
    api_key: str | None = None,
    **create_kw: Any,
) -> TestClient:
    """TestClient for a fresh app; ``api_key`` installs the root key."""
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from fx1.serve.conv_audit import _RESOURCES  # noqa: PLC0415

    if api_key is None:
        os.environ.pop(_API_KEY_ENV, None)
    else:
        os.environ[_API_KEY_ENV] = api_key
    app = _make_app(runner=runner, **create_kw)
    client = TestClient(app, raise_server_exceptions=False)
    # The app keeps a jobs executor pool; register its shutdown so probe
    # sections don't leak threads across the battery.
    stack = _RESOURCES.get(None)
    if stack is not None:
        stack.callback(app.state.jobs_executor.shutdown, False, cancel_futures=True)
    return client


def _h(auth: str | None = _ROOT, **extra: str) -> dict[str, str]:
    out = {"X-API-Key": auth} if auth else {}
    out.update(extra)
    return out


def _ih(key: str, auth: str | None = _ROOT, **extra: str) -> dict[str, str]:
    return _h(auth, **{_IDEM: key, **extra})


def _err_code(r: Any) -> str | None:
    if not r.headers.get("content-type", "").startswith("application/json"):
        return None
    code = r.json().get("code")
    return code if isinstance(code, str) else None


def _replay_hdr(r: Any) -> bool:
    return bool(r.headers.get("x-fx1-idempotent-replay") == "true")


def _submit(
    client: TestClient,
    command: str = "doctor",
    *,
    key: str | None = None,
    auth: str | None = None,
    **fields: Any,
) -> Any:
    body: dict[str, Any] = {"command": command, **fields}
    headers = _h(auth) if auth else {}
    if key is not None:
        headers[_IDEM] = key
    return client.post(_JOBS, json=body, headers=headers)


def _record(client: TestClient, job_id: str, auth: str | None = None) -> Any:
    return client.get(f"{_JOBS}/{job_id}", headers=_h(auth) if auth else {})


def _wait(
    client: TestClient, job_id: str, timeout: float = _WAIT_S, auth: str | None = None
) -> dict[str, Any]:
    """Poll the record until terminal; returns the last seen body."""
    end = time.monotonic() + timeout
    st: dict[str, Any] = {}
    while time.monotonic() < end:
        st = _record(client, job_id, auth).json()
        if st.get("status") in _TERMINAL:
            return st
        time.sleep(0.05)
    return st


def _wait_hits(sink: Any, n: int, timeout: float = _WAIT_S) -> None:
    end = time.monotonic() + timeout
    while len(sink.hits) < n and time.monotonic() < end:
        time.sleep(0.05)


def _busy_executor(client: TestClient, sleep_s: float = 1.5) -> None:
    """Park every worker thread on a raw sleeper — outside the inflight
    accounting — so a submitted job acquires its slot yet stays
    ``queued`` behind the future queue. The deterministic way to reach
    the queued-cancel path (the executor is sized ``max_inflight``)."""
    from typing import cast as _cast

    from fastapi import FastAPI  # noqa: PLC0415

    executor = _cast(FastAPI, client.app).state.jobs_executor
    release = threading.Event()
    hold = release.wait
    for _ in range(int(getattr(executor, "_max_workers", 4))):  # noqa: SLF001
        executor.submit(lambda: hold(timeout=sleep_s))
    time.sleep(0.1)


def _key_card(client: TestClient, root: str, key_id: str) -> dict[str, Any]:
    return dict(client.get(f"/harness/keys/{key_id}", headers=_h(root)).json())


def _mint(client: TestClient, root: str, **fields: Any) -> dict[str, Any]:
    r = client.post("/harness/keys", json=fields, headers=_h(root))
    assert r.status_code == 201, f"key mint refused: {r.status_code} {r.text}"
    return dict(r.json())


# ---------------------------------------------------------------------------
# Probes: submit + lifecycle
# ---------------------------------------------------------------------------


def _submit_lifecycle_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    runner = _Runner()
    client = _client(runner)

    r = _submit(client)
    out["submit_202"] = r.status_code == 202
    body = r.json()
    job_id = body["job_id"]
    out["submit_shape"] = set(body) == {"job_id", "status", "replayed"} and (
        body["replayed"] is False
    )
    out["submit_location_header"] = r.headers.get("location") == f"{_JOBS}/{job_id}"
    # The submit response renders the record live — status is a
    # monotone-forward member of the lifecycle set, never a lie about
    # "queued" when the worker already ran it.
    out["submit_status_live_member"] = body["status"] in _RANK

    rec = _wait(client, job_id)
    out["lifecycle_queued_to_succeeded"] = rec["status"] == "succeeded"
    out["record_shape_pinned"] = set(rec) == _JOB_FIELDS
    out["timestamps_monotonic"] = (
        isinstance(rec["created_at"], (int, float))
        and isinstance(rec["finished_at"], (int, float))
        and rec["finished_at"] >= rec["created_at"]
    )
    res = rec["result"]
    out["result_shape_pinned"] = set(res) == _RESULT_FIELDS
    out["result_honest"] = (
        res["command"] == "doctor"
        and res["exit_code"] == 0
        and res["ok"] is True
        and res["stdout"] == "ok"
        and res["stderr"] == ""
        and res["stdout_truncated"] is False
        and res["stderr_truncated"] is False
        and res["replayed"] is False
    )
    out["runner_called_once"] = len(runner.calls) == 1

    # nonzero exit: the job lifecycle still succeeded — ``status`` tracks
    # execution, ``result.ok`` carries the command's own verdict
    runner2 = _Runner(exit_code=3, stdout="partial", stderr="bad ticks")
    client2 = _client(runner2)
    rec2 = _wait(client2, _submit(client2).json()["job_id"])
    out["exit_nonzero_succeeded"] = rec2["status"] == "succeeded"
    out["exit_nonzero_result_ok_false"] = (
        rec2["result"]["ok"] is False and rec2["result"]["exit_code"] == 3
    )
    out["failed_result_mirrored"] = (
        rec2["result"]["exit_code"] == 3
        and rec2["result"]["ok"] is False
        and rec2["result"]["stdout"] == "partial"
        and rec2["result"]["stderr"] == "bad ticks"
    )
    out["failed_result_not_truncated"] = (
        rec2["result"]["stdout_truncated"] is False and rec2["result"]["stderr_truncated"] is False
    )

    # runner fault → failed + error names the exception type
    runner3 = _Runner(raise_exc=RuntimeError("kaboom"))
    client3 = _client(runner3)
    rec3 = _wait(client3, _submit(client3).json()["job_id"])
    out["worker_fault_failed"] = rec3["status"] == "failed"
    out["worker_fault_error_field"] = (
        rec3["error"] is not None
        and "RuntimeError" in rec3["error"]
        and "kaboom" in rec3["error"]
        and rec3["result"] is None
    )

    # status never regresses: poll through the transition
    gate = threading.Event()
    runner4 = _Runner(gate=gate)
    client4 = _client(runner4)
    jid4 = _submit(client4).json()["job_id"]
    seen: list[str] = []
    deadline = time.monotonic() + _WAIT_S
    while not runner4.entered.is_set() and time.monotonic() < deadline:
        seen.append(_record(client4, jid4).json()["status"])
        time.sleep(0.02)
    gate.set()
    seen.append(_wait(client4, jid4)["status"])
    out["status_never_regresses"] = (
        all(_RANK[seen[i]] <= _RANK[seen[i + 1]] for i in range(len(seen) - 1))
        and seen[-1] == "succeeded"
    )

    # a terminal record never flips again
    rec4 = _record(client4, jid4).json()
    again = _record(client4, jid4).json()
    out["terminal_record_stable"] = (
        again["status"] == rec4["status"]
        and again["finished_at"] == rec4["finished_at"]
        and again["result"] == rec4["result"]
    )
    return out


# ---------------------------------------------------------------------------
# Probes: submission validation envelopes
# ---------------------------------------------------------------------------


def _validation_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client = _client()

    cases = {
        "missing_command_422": {},
        "null_command_422": {"command": None},
        "command_wrong_type_422": {"command": 7},
        "spec_wrong_type_422": {"command": "doctor", "extra_args": "--verbose"},
        "extra_field_forbidden_422": {"command": "doctor", "bogus": 1},
        "callback_scheme_422": {"command": "doctor", "callback_url": "ftp://x/hook"},
        "callback_missing_host_422": {"command": "doctor", "callback_url": "http:///x"},
        "callback_userinfo_422": {
            "command": "doctor",
            "callback_url": "http://u:p@127.0.0.1/hook",
        },
        "secret_without_url_422": {"command": "doctor", "callback_secret": "s"},
        "extra_args_over_cap_422": {
            "command": "doctor",
            "extra_args": [f"a{i}" for i in range(65)],
        },
    }
    for name, body in cases.items():
        r = client.post(_JOBS, json=body)
        out[name] = r.status_code == 422 and _err_code(r) == "validation"

    r = client.post(_JOBS, content=b"{not json", headers={"Content-Type": "application/json"})
    out["malformed_json_422"] = r.status_code in (400, 422) and isinstance(_err_code(r), str)

    r = client.post(_JOBS, json={"command": "definitely-not-a-command"})
    out["unknown_command_404"] = r.status_code == 404 and _err_code(r) == "not_found"
    out["unknown_command_detail_names_it"] = "definitely-not-a-command" in str(
        r.json().get("detail", "")
    )

    out["get_unknown_404"] = _record(client, "never-submitted").status_code == 404
    out["receipt_unknown_404"] = client.get(f"{_JOBS}/never-submitted/receipt").status_code == 404
    out["events_unknown_404"] = client.get(f"{_JOBS}/never-submitted/events").status_code == 404
    out["cancel_unknown_404"] = client.delete(f"{_JOBS}/never-submitted").status_code == 404

    # config containment: on /harness/jobs the escape runs inside the
    # worker → honest failed record (the validator is Harness.run, not
    # the request model); the /harness/runs twin 422s the request.
    client2 = _client()
    r2 = client2.post(_JOBS, json={"command": "doctor", "config": "/etc/hostname"})
    out["config_escape_submit_accepted"] = r2.status_code == 202
    if r2.status_code == 202:
        rec2 = _wait(client2, r2.json()["job_id"])
        out["config_escape_fails_closed"] = rec2["status"] == "failed" and "allowlist" in str(
            rec2.get("error") or ""
        )
        out["config_escape_no_result"] = rec2.get("result") is None
    return out


# ---------------------------------------------------------------------------
# Probes: transitions — the cancel matrix + the dequeue race
# ---------------------------------------------------------------------------


def _transition_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    runner = _Runner()
    client = _client(runner, max_inflight=1)

    # queued cancel: the lone worker thread is parked on a raw sleeper —
    # the submit acquires its inflight slot yet sits queued behind the
    # future queue, so DELETE lands while it is genuinely "queued"
    _busy_executor(client, sleep_s=1.5)
    jid_b = _submit(client).json()["job_id"]
    d = client.delete(f"{_JOBS}/{jid_b}")
    out["cancel_queued_200_cancelled"] = d.status_code == 200 and d.json()["status"] == "cancelled"
    out["cancel_queued_finished_at_set"] = d.json().get("finished_at") is not None
    # past the sleeper window the future dequeues — start() must lose
    time.sleep(2.0)
    rec_b = _record(client, jid_b).json()
    out["cancelled_never_runs"] = rec_b["status"] == "cancelled" and runner.calls == []

    # cancel a cancelled job → idempotent 200
    d2 = client.delete(f"{_JOBS}/{jid_b}")
    out["cancel_cancelled_idempotent_200"] = (
        d2.status_code == 200 and d2.json()["status"] == "cancelled"
    )

    # running cancel → 409, job completes
    gate2 = threading.Event()
    runner2 = _Runner(gate=gate2)
    client2 = _client(runner2, max_inflight=1)
    jid_c = _submit(client2).json()["job_id"]
    deadline = time.monotonic() + _WAIT_S
    while not runner2.entered.is_set() and time.monotonic() < deadline:
        time.sleep(0.02)
    d3 = client2.delete(f"{_JOBS}/{jid_c}")
    out["cancel_running_409"] = d3.status_code == 409 and _err_code(d3) == "conflict"
    out["cancel_running_detail_states_status"] = "running" in str(d3.json().get("detail", ""))
    gate2.set()
    rec_c = _wait(client2, jid_c)
    out["running_completes_after_409"] = rec_c["status"] == "succeeded"

    # terminal cancel → 409, record keeps terminal status
    d4 = client2.delete(f"{_JOBS}/{jid_c}")
    out["cancel_terminal_409"] = d4.status_code == 409
    out["cancel_terminal_keeps_record"] = _record(client2, jid_c).json()["status"] == "succeeded"

    # cancel racing dequeue, N rounds on fresh apps: each round ends
    # either cancelled-and-never-ran or running-then-succeeded — never
    # cancelled-then-ran (the resurrect the start() claim forbids)
    rounds = 0
    honest = 0
    for i in range(10):
        rn = _Runner()
        cn = _client(rn, max_inflight=1)
        # worker parked; the job's cancel races the sleeper expiry
        _busy_executor(cn, sleep_s=0.02 + (i % 5) * 0.04)
        rb = cn.post(_JOBS, json={"command": "doctor"})
        jb = rb.json()["job_id"]
        dr = cn.delete(f"{_JOBS}/{jb}")
        rec = _wait(cn, jb)
        rounds += 1
        if dr.status_code == 200:
            honest += int(rec["status"] == "cancelled" and rec["result"] is None)
        elif dr.status_code == 409:
            honest += int(rec["status"] == "succeeded" and rec["result"] is not None)
    out["cancel_race_outcome_honest"] = rounds == 10 and honest == rounds
    return out


# ---------------------------------------------------------------------------
# Probes: idempotency
# ---------------------------------------------------------------------------


def _idempotency_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    runner = _Runner()
    client = _client(runner)

    r1 = _submit(client, key="dup-1")
    r2 = _submit(client, key="dup-1")
    out["idem_replay_same_job"] = (
        r1.status_code == 202
        and r2.status_code == 202
        and r2.json()["job_id"] == r1.json()["job_id"]
        and r2.json()["replayed"] is True
    )
    # the jobs surface marks replays in the body — no replay header
    out["idem_replay_marker_body"] = (
        r2.json()["replayed"] is True and r1.json()["replayed"] is False
    )
    out["idem_replay_no_marker_header"] = not _replay_hdr(r2) and not _replay_hdr(r1)
    _wait(client, r1.json()["job_id"])
    out["idem_replay_never_reruns"] = len(runner.calls) == 1

    r4 = _submit(client, key="dup-1", extra_args=["--verbose"])
    out["idem_conflict_409"] = r4.status_code == 409 and _err_code(r4) == "idempotency_conflict"
    out["idem_conflict_executes_nothing"] = len(runner.calls) == 1

    ra = _submit(client, key="k-a")
    rb = _submit(client, key="k-b")
    out["idem_distinct_keys_independent"] = ra.json()["job_id"] != rb.json()["job_id"]
    rn1, rn2 = _submit(client), _submit(client)
    out["idem_no_key_always_new"] = rn1.json()["job_id"] != rn2.json()["job_id"]

    r_over = _submit(client, key="k" * 300)
    out["idem_key_overlong_400"] = r_over.status_code == 400 and isinstance(_err_code(r_over), str)

    # header wins over the in-body field
    rh = _submit(client, key="hdr-k", idempotency_key="body-k")
    rb2 = _submit(client, key="hdr-k")
    rbody = _submit(client, idempotency_key="body-k")
    out["idem_header_wins_over_body"] = (
        rh.json()["job_id"] == rb2.json()["job_id"]
        and rbody.json()["job_id"] != rh.json()["job_id"]
    )

    # replay returns the live record status, not a frozen snapshot
    rs = _submit(client, key="live-st")
    _wait(client, rs.json()["job_id"])
    rr = _submit(client, key="live-st")
    out["idem_replay_live_status"] = rr.json()["status"] == "succeeded"

    # credential scoping: same key string under different creds → different work
    client2 = _client(_Runner(), api_key=_ROOT)
    minted = _mint(client2, _ROOT)
    r_x = client2.post(_JOBS, json={"command": "doctor"}, headers=_ih("shared-key", minted["key"]))
    r_y = client2.post(_JOBS, json={"command": "doctor"}, headers=_ih("shared-key", _ROOT))
    out["idem_scoped_per_credential"] = r_x.json()["job_id"] != r_y.json()["job_id"]

    # parallel same-key submits → exactly one job + one execution
    gate = threading.Event()
    prunner = _Runner(gate=gate)
    os.environ.pop(_API_KEY_ENV, None)  # leaked root key would 401 the raw POSTs
    papp = _make_app(runner=prunner, max_inflight=4)

    async def fan_out() -> list[Any]:
        import httpx  # noqa: PLC0415

        async with (
            papp.router.lifespan_context(papp),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(papp), base_url="http://localhost"
            ) as ac,
        ):
            first = asyncio.create_task(
                ac.post(_JOBS, json={"command": "doctor"}, headers={_IDEM: "conc.j"})
            )
            while not prunner.entered.is_set():
                await asyncio.sleep(0.005)
            rest = [
                asyncio.create_task(
                    ac.post(_JOBS, json={"command": "doctor"}, headers={_IDEM: "conc.j"})
                )
                for _ in range(3)
            ]
            gate.set()
            return [await first] + list(await asyncio.gather(*rest))

    rs = asyncio.run(fan_out())
    oks = [r for r in rs if r.status_code == 202]
    jids = {r.json().get("job_id") for r in oks}
    out["idem_parallel_single_job"] = len(oks) == 4 and len(jids) == 1
    out["idem_parallel_single_execution"] = len(prunner.calls) == 1

    # parallel distinct keys → distinct ids
    prunner2 = _Runner()
    papp2 = _make_app(runner=prunner2, max_inflight=8)

    async def fan_distinct() -> list[Any]:
        import httpx  # noqa: PLC0415

        async with (
            papp2.router.lifespan_context(papp2),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(papp2), base_url="http://localhost"
            ) as ac,
        ):
            responses = list(
                await asyncio.gather(
                    *[
                        ac.post(
                            _JOBS,
                            json={"command": "doctor"},
                            headers={_IDEM: f"dk-{i}"},
                        )
                        for i in range(6)
                    ]
                )
            )
            # Do not leave the lifespan until every accepted worker has
            # entered the runner. Shutdown cancels queued futures, which made
            # this probe measure scheduler timing instead of the API contract.
            deadline = time.monotonic() + _WAIT_S
            while len(prunner2.calls) < 6 and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            return responses

    ds = asyncio.run(fan_distinct())
    djids = {r.json().get("job_id") for r in ds if r.status_code == 202}
    out["parallel_submit_distinct_ids"] = len(ds) == 6 and len(djids) == 6
    out["parallel_all_ran"] = len(prunner2.calls) == 6
    return out


# ---------------------------------------------------------------------------
# Probes: the gate — drain latch, admission cap, ordering
# ---------------------------------------------------------------------------


def _gate_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    gate = threading.Event()
    runner = _Runner(gate=gate)
    client = _client(runner, api_key=_ROOT, max_inflight=1)

    _mint(client, _ROOT)
    # minted pre-drain: the key lifecycle refuses under the latch, so the
    # read-scope credential for the scope-beats-gate probe must exist first
    ro = client.post("/harness/keys", json={"scopes": ["read"]}, headers=_h(_ROOT))

    # in-flight job survives a mid-run drain; new work refused
    jid_a = _submit(client, auth=_ROOT).json()["job_id"]
    deadline = time.monotonic() + _WAIT_S
    while not runner.entered.is_set() and time.monotonic() < deadline:
        time.sleep(0.02)

    d = client.post(_DRAIN, headers=_h(_ROOT))
    out["drain_200_latches"] = d.status_code == 200 and d.json()["draining"] is True

    r_drained = _submit(client, auth=_ROOT)
    out["drain_submit_503"] = r_drained.status_code == 503 and _err_code(r_drained) == "draining"
    out["drain_detail_declares"] = "draining" in str(r_drained.json().get("detail", ""))
    r_unk_drained = _submit(client, "no-such-cmd", auth=_ROOT)
    out["drain_jobs_404_beats_503"] = (
        r_unk_drained.status_code == 404 and _err_code(r_unk_drained) == "not_found"
    )
    r_runs_drained = client.post(_RUNS, json={"command": "no-such-cmd"}, headers=_h(_ROOT))
    out["drain_runs_503_beats_404"] = (
        r_runs_drained.status_code == 503 and _err_code(r_runs_drained) == "draining"
    )

    # reads + cancel stay open under drain
    out["drain_get_list_200"] = client.get(_JOBS, headers=_h(_ROOT)).status_code == 200
    out["drain_get_record_200"] = (
        client.get(f"{_JOBS}/{jid_a}", headers=_h(_ROOT)).status_code == 200
    )
    out["drain_health_reflects"] = (
        client.get("/health", headers=_h(_ROOT)).json().get("draining") is True
    )
    out["drain_idempotent_200"] = client.post(_DRAIN, headers=_h(_ROOT)).status_code == 200
    out["no_unlatch_route"] = client.post("/harness/undrain", headers=_h(_ROOT)).status_code in (
        404,
        405,
    )

    # batch under drain: every item refused, honestly counted
    rb = client.post(
        f"{_JOBS}/batch",
        json={"jobs": [{"command": "doctor"}, {"command": "doctor"}]},
        headers=_h(_ROOT),
    )
    body = rb.json()
    out["drain_batch_refuses_itemwise"] = (
        rb.status_code == 202
        and body["submitted"] == 0
        and body["failed"] == 2
        and all(item.get("code") == "draining" for item in body["jobs"])
    )

    # auth beats the gate; scope beats the gate
    r_bad = client.post(_JOBS, json={"command": "doctor"}, headers=_h("not-the-key"))
    out["auth_before_gate_401"] = r_bad.status_code == 401
    r_scope = client.post(_JOBS, json={"command": "doctor"}, headers=_h(ro.json()["key"]))
    out["scope_before_gate_403"] = r_scope.status_code == 403

    # in-flight completes past the drain; wait_s unblocks on empty
    gate.set()
    rec_a = _wait(client, jid_a, auth=_ROOT)
    out["drain_inflight_completes"] = rec_a["status"] == "succeeded"
    d2 = client.post(f"{_DRAIN}?wait_s=5", headers=_h(_ROOT))
    out["drain_wait_returns_drained"] = d2.status_code == 200 and d2.json().get("drained") is True

    # capacity: max_inflight=1 — second submit while first runs → 503
    gate2 = threading.Event()
    runner2 = _Runner(gate=gate2)
    client2 = _client(runner2, max_inflight=1)
    _submit(client2)
    deadline = time.monotonic() + _WAIT_S
    while not runner2.entered.is_set() and time.monotonic() < deadline:
        time.sleep(0.02)
    r_cap = _submit(client2)
    out["over_capacity_503"] = r_cap.status_code == 503 and _err_code(r_cap) == "over_capacity"
    out["over_capacity_retry_after"] = int(r_cap.headers.get("Retry-After", "0")) >= 1
    out["over_capacity_no_record"] = client2.get(_JOBS, params={"limit": 500}).json()["total"] == 1
    gate2.set()
    out["capacity_recovers_after_release"] = _submit(client2).status_code == 202
    return out


# ---------------------------------------------------------------------------
# Probes: webhooks — fire-once, HMAC, retry policy, no resurrection
# ---------------------------------------------------------------------------


def _webhook_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    from fx1.serve.webhook_audit import _Sink  # noqa: PLC0415
    from fx1.serve.webhooks import (  # noqa: PLC0415
        WEBHOOK_MAX_ATTEMPTS,
        verify_webhook,
    )

    sink = _Sink()
    try:
        runner = _Runner()
        client = _client(runner)

        secret = "whsec-test"
        job_id = _submit(client, callback_url=sink.url("/hook"), callback_secret=secret).json()[
            "job_id"
        ]
        rec = _wait(client, job_id)
        _wait_hits(sink, 1)
        out["callback_fires_once"] = len(sink.hits) == 1
        hit = sink.hits[0]
        out["callback_hmac_verifies"] = verify_webhook(
            secret,
            hit.headers.get("X-Fx1-Webhook-Timestamp", ""),
            hit.headers.get("X-Fx1-Webhook-Signature", ""),
            hit.body,
        )
        payload = json.loads(hit.body)
        out["callback_payload_terminal"] = (
            payload.get("job_id") == job_id and payload.get("status") == "succeeded"
        )
        # bookkeeping lands after the terminal flip — re-read post-hit
        rec2 = rec
        deadline = time.monotonic() + _WAIT_S
        while rec2.get("callback_status") != "delivered" and time.monotonic() < deadline:
            time.sleep(0.05)
            rec2 = _record(client, job_id).json()
        out["callback_record_fields"] = (
            rec2["callback_url"] == sink.url("/hook")
            and rec2["callback_status"] == "delivered"
            and rec2["callback_attempts"] >= 1
            and rec2["callback_error"] is None
        )
        out["secret_never_on_record"] = "callback_secret" not in rec and (
            secret not in json.dumps(rec)
        )

        # re-reads never re-fire
        _record(client, job_id)
        _record(client, job_id)
        time.sleep(0.15)
        out["callback_never_repeats"] = len(sink.hits) == 1

        # unsigned callback: no webhook headers at all — the signature
        # AND timestamp headers only exist when a secret is configured
        r2 = _submit(client, callback_url=sink.url("/hook2"))
        _wait(client, r2.json()["job_id"])
        _wait_hits(sink, 2)
        hit2 = sink.hits[1]
        out["callback_unsigned_no_headers"] = (
            hit2.headers.get("X-Fx1-Webhook-Signature") is None
            and hit2.headers.get("X-Fx1-Webhook-Timestamp") is None
        )

        # endpoint 5xx → retry to the cap, then failed on the record —
        # the job itself never resurrects
        r3 = _submit(client, callback_url=sink.url("/fail"), callback_secret=secret)
        jid3 = r3.json()["job_id"]
        _wait(client, jid3)
        rec3: dict[str, Any] = {}
        deadline = time.monotonic() + _WAIT_S
        while time.monotonic() < deadline:
            rec3 = _record(client, jid3).json()
            if rec3.get("callback_status") == "failed":
                break
            time.sleep(0.05)
        out["callback_failure_marks_failed"] = rec3.get("callback_status") == "failed"
        out["callback_failure_attempts_capped"] = (
            rec3.get("callback_attempts") == WEBHOOK_MAX_ATTEMPTS
        )
        out["callback_failure_error_set"] = bool(rec3.get("callback_error"))
        out["callback_failure_never_resurrects"] = rec3.get("status") == "succeeded"

        # endpoint 4xx → definitive, a single attempt
        r4 = _submit(client, callback_url=sink.url("/reject"))
        jid4 = r4.json()["job_id"]
        _wait(client, jid4)
        rec4: dict[str, Any] = {}
        deadline = time.monotonic() + _WAIT_S
        while time.monotonic() < deadline:
            rec4 = _record(client, jid4).json()
            if rec4.get("callback_status"):
                break
            time.sleep(0.05)
        out["callback_4xx_single_attempt"] = (
            rec4.get("callback_attempts") == 1 and rec4.get("callback_status") == "failed"
        )

        # cancel fires the webhook once (a terminal transition)
        runner_q = _Runner()
        client_q = _client(runner_q, max_inflight=1)
        _busy_executor(client_q, sleep_s=2.5)
        rb = _submit(client_q, callback_url=sink.url("/can"), callback_secret=secret)
        jb = rb.json()["job_id"]
        hits_before = len(sink.hits)
        client_q.delete(f"{_JOBS}/{jb}")
        _wait_hits(sink, hits_before + 1)
        hit_c = sink.hits[-1]
        body_c = json.loads(hit_c.body)
        out["cancel_fires_callback_once"] = hit_c.path == "/can"
        out["cancel_callback_payload_cancelled"] = (
            body_c.get("status") == "cancelled" and body_c.get("job_id") == jb
        )
        out["cancel_callback_signed"] = verify_webhook(
            secret,
            hit_c.headers.get("X-Fx1-Webhook-Timestamp", ""),
            hit_c.headers.get("X-Fx1-Webhook-Signature", ""),
            hit_c.body,
        )
        _wait(client_q, jb)
    finally:
        sink.close()
    return out


# ---------------------------------------------------------------------------
# Probes: waiting — wait_s param, client poll, SSE stream
# ---------------------------------------------------------------------------


def _wait_stream_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    gate = threading.Event()
    runner = _Runner(gate=gate)
    client = _client(runner)

    job_id = _submit(client).json()["job_id"]
    deadline = time.monotonic() + _WAIT_S
    while not runner.entered.is_set() and time.monotonic() < deadline:
        time.sleep(0.02)

    # ?wait_s is not a long-poll knob on the job GET — the param is
    # ignored and the live status returns immediately (SSE + drain's
    # own wait_s are the wait channels; the client polls)
    t0 = time.monotonic()
    r_wait = client.get(f"{_JOBS}/{job_id}?wait_s=5")
    elapsed = time.monotonic() - t0
    out["wait_s_param_inert"] = r_wait.status_code == 200 and elapsed < 2.0
    out["wait_s_returns_live_status"] = r_wait.json()["status"] == "running"

    # SSE: frames stream transitions and the terminal frame ends it
    events: list[dict[str, Any]] = []
    done = threading.Event()

    def consume() -> None:
        try:
            with client.stream("GET", f"{_JOBS}/{job_id}/events?timeout_s=10") as resp:
                for line in resp.iter_lines():
                    if line.startswith("data:"):
                        events.append(json.loads(line[5:].strip()))
                        if events[-1].get("status") in _TERMINAL:
                            done.set()
                            return
        except Exception:  # noqa: BLE001 — stream close is expected
            pass
        done.set()

    th = threading.Thread(target=consume, daemon=True)
    th.start()
    gate.set()
    done.wait(timeout=_WAIT_S)
    th.join(timeout=_WAIT_S)
    statuses = [str(e.get("status")) for e in events]
    out["sse_emits_terminal_frame"] = bool(statuses) and statuses[-1] == "succeeded"
    out["sse_frames_monotone"] = all(
        _RANK.get(statuses[i], -1) <= _RANK.get(statuses[i + 1], 9)
        for i in range(len(statuses) - 1)
    )

    # a finished job's stream returns instantly
    t0 = time.monotonic()
    with client.stream("GET", f"{_JOBS}/{job_id}/events?timeout_s=30") as resp:
        lines = list(resp.iter_lines())
    out["sse_terminal_instant"] = (time.monotonic() - t0) < 10.0 and any(
        '"succeeded"' in ln for ln in lines
    )

    # HarnessClient.wait_run polls to terminal
    from fx1.serve.client import (  # noqa: PLC0415
        HarnessClient,
        HarnessJobError,
        HarnessTransportError,
    )
    from fx1.serve.eval_lifecycle_audit import _tc_transport  # noqa: PLC0415

    gate2 = threading.Event()
    runner2 = _Runner(gate=gate2)
    client2 = _client(runner2)
    hc2 = HarnessClient("http://testserver", transport=_tc_transport(client2))
    jid2 = _submit(client2).json()["job_id"]
    gate2.set()
    res2 = hc2.wait_run(jid2, timeout_s=15.0)
    out["client_wait_run_succeeds"] = res2.ok and res2.stdout == "ok"

    runner3 = _Runner(raise_exc=RuntimeError("boom"))
    client3 = _client(runner3)
    hc3 = HarnessClient("http://testserver", transport=_tc_transport(client3))
    jid3 = _submit(client3).json()["job_id"]
    try:
        hc3.wait_run(jid3, timeout_s=15.0)
        out["client_wait_failed_raises"] = False
    except HarnessJobError as exc:
        out["client_wait_failed_raises"] = "failed" in str(exc)
    except Exception:  # noqa: BLE001
        out["client_wait_failed_raises"] = False

    gate4 = threading.Event()
    runner4 = _Runner(gate=gate4)
    client4 = _client(runner4)
    hc4 = HarnessClient("http://testserver", transport=_tc_transport(client4))
    jid4 = _submit(client4).json()["job_id"]
    try:
        hc4.wait_run(jid4, timeout_s=0.05)
        out["client_wait_timeout_raises"] = False
    except HarnessTransportError:
        out["client_wait_timeout_raises"] = True
    except Exception:  # noqa: BLE001
        out["client_wait_timeout_raises"] = False
    out["wait_timeout_job_still_running"] = _record(client4, jid4).json()["status"] in {
        "queued",
        "running",
    }
    gate4.set()
    _wait(client4, jid4)
    return out


# ---------------------------------------------------------------------------
# Probes: list / filter / paging
# ---------------------------------------------------------------------------


def _list_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client = _client(_Runner())

    ids = [_submit(client).json()["job_id"] for _ in range(5)]
    for jid in ids:
        _wait(client, jid)

    r = client.get(_JOBS)
    body = r.json()
    out["list_200_shape"] = r.status_code == 200 and set(body) == {"jobs", "total"}
    got_ids = [j["job_id"] for j in body["jobs"]]
    out["list_newest_first"] = got_ids == list(reversed(ids))
    out["list_total_counts_all"] = body["total"] == 5

    r = client.get(_JOBS, params={"status": "succeeded"})
    out["list_status_filter"] = {j["job_id"] for j in r.json()["jobs"]} == set(ids)
    r = client.get(_JOBS, params={"status": "failed"})
    out["list_filter_empty_honest"] = r.json()["total"] == 0 and r.json()["jobs"] == []
    out["list_status_bad_422"] = client.get(_JOBS, params={"status": "bogus"}).status_code == 422

    page = client.get(_JOBS, params={"limit": 2, "offset": 1}).json()
    out["list_page_window"] = [j["job_id"] for j in page["jobs"]] == got_ids[1:3]
    out["list_total_ignores_page"] = page["total"] == body["total"]
    far = client.get(_JOBS, params={"offset": 999}).json()
    out["list_offset_beyond_empty"] = far["jobs"] == [] and far["total"] == 5
    out["list_limit_cap_422"] = client.get(_JOBS, params={"limit": 501}).status_code == 422
    out["list_offset_negative_422"] = client.get(_JOBS, params={"offset": -1}).status_code == 422
    out["list_limit_zero_422"] = client.get(_JOBS, params={"limit": 0}).status_code == 422
    return out


# ---------------------------------------------------------------------------
# Probes: batch — per-item failure isolation
# ---------------------------------------------------------------------------


def _batch_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    runner = _Runner()
    client = _client(runner)

    r = client.post(
        f"{_JOBS}/batch",
        json={
            "jobs": [
                {"command": "doctor"},
                {"command": "no-such-cmd"},
                {"command": "doctor"},
            ]
        },
    )
    body = r.json()
    out["batch_202_always"] = r.status_code == 202
    out["batch_shape"] = set(body) == {"jobs", "submitted", "failed"}
    out["batch_per_item_isolation"] = (
        body["submitted"] == 2
        and body["failed"] == 1
        and bool(body["jobs"][0].get("job_id"))
        and body["jobs"][1].get("code") == "not_found"
        and body["jobs"][1].get("index") == 1
        and bool(body["jobs"][2].get("job_id"))
        and body["jobs"][1].get("job_id") is None
    )
    ok_ids = [j["job_id"] for j in body["jobs"] if j.get("job_id")]
    for jid in ok_ids:
        _wait(client, jid)
    out["batch_items_reach_terminal"] = all(
        _record(client, jid).json()["status"] in _TERMINAL for jid in ok_ids
    )
    out["batch_failed_item_left_no_record"] = _record(client, "no-such-cmd").status_code == 404

    # whole-batch body validation precedes per-item isolation
    r2 = client.post(f"{_JOBS}/batch", json={"jobs": [{"command": "doctor"}, {"command": 5}]})
    out["batch_malformed_item_422_all"] = r2.status_code == 422
    out["batch_empty_422"] = client.post(f"{_JOBS}/batch", json={"jobs": []}).status_code == 422

    # per-item idempotency keys dedupe within and across batches
    r4 = client.post(
        f"{_JOBS}/batch",
        json={
            "jobs": [
                {"command": "doctor", "idempotency_key": "b-k1"},
                {"command": "doctor", "idempotency_key": "b-k2"},
            ]
        },
    )
    # The executor is async. Count calls only after the first batch has
    # actually run, or a slow worker lands those calls inside the replay
    # window and a faithful replay looks like new work.
    for item in r4.json().get("jobs", []):
        if item.get("job_id"):
            _wait(client, item["job_id"])
    calls_before = len(runner.calls)
    r5 = client.post(
        f"{_JOBS}/batch",
        json={
            "jobs": [
                {"command": "doctor", "idempotency_key": "b-k1"},
                {"command": "doctor", "idempotency_key": "b-k3"},
            ]
        },
    )
    b4, b5 = r4.json(), r5.json()
    if b5["jobs"][1].get("job_id"):
        _wait(client, b5["jobs"][1]["job_id"])
    out["batch_idem_replay_per_item"] = (
        b5["jobs"][0].get("replayed") is True
        and b5["jobs"][0].get("job_id") == b4["jobs"][0].get("job_id")
        and b5["jobs"][1].get("replayed") is False
    )
    out["batch_replay_ran_nothing_new"] = len(runner.calls) == calls_before + 1
    return out


# ---------------------------------------------------------------------------
# Probes: durability — --state-dir journal replay
# ---------------------------------------------------------------------------


def _durability_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    from fx1.serve.conv_audit import _temporary_directory  # noqa: PLC0415

    state_dir = _temporary_directory()

    # app 1: one clean-terminal job + one worker-failure job + a key
    runner = _Runner()
    c1 = _client(runner, state_dir=state_dir)
    rec1 = _wait(c1, _submit(c1, key="durable-1").json()["job_id"])
    r2 = c1.post(_JOBS, json={"command": "doctor", "config": "/etc/hostname"})
    rec2 = _wait(c1, r2.json()["job_id"])
    snap1 = {j["job_id"]: j for j in c1.get(_JOBS).json()["jobs"]}
    c1.close()

    # app 2: fresh process on the same dir — replay restores as-was
    runner_c2 = _Runner()
    c2 = _client(runner_c2, state_dir=state_dir)
    snap2 = {j["job_id"]: j for j in c2.get(_JOBS).json()["jobs"]}
    out["restart_terminal_restored"] = (
        snap2.get(rec1["job_id"], {}).get("status") == "succeeded"
        and snap2.get(rec2["job_id"], {}).get("status") == "failed"
    )
    out["restart_result_preserved"] = snap2.get(rec1["job_id"], {}).get("result") == rec1.get(
        "result"
    )
    out["restart_error_preserved"] = snap2.get(rec2["job_id"], {}).get("error") == rec2.get("error")
    out["restart_records_equal_to_snapshot"] = all(
        snap2.get(jid, {}) == rec for jid, rec in snap1.items()
    )

    # idempotency survives restart: same key replays to the same job,
    # and the replay executes nothing
    r3 = c2.post(_JOBS, json={"command": "doctor"}, headers={_IDEM: "durable-1"})
    out["restart_idem_replays_old_job"] = (
        r3.json().get("replayed") is True and r3.json()["job_id"] == rec1["job_id"]
    )
    _wait(c2, rec1["job_id"])
    out["restart_replay_executes_nothing"] = runner_c2.calls == []

    # running job at crash → fail-closed honest restart error
    state_dir2 = _temporary_directory()
    gate = threading.Event()
    runner_q = _Runner(gate=gate)
    c3 = _client(runner_q, state_dir=state_dir2)
    jid4 = _submit(c3).json()["job_id"]
    deadline = time.monotonic() + _WAIT_S
    while not runner_q.entered.is_set() and time.monotonic() < deadline:
        time.sleep(0.02)
    # no lifespan exit — the journal tail still says "running"; a fresh
    # store replays it as a crash
    c4 = _client(_Runner(), state_dir=state_dir2)
    rec4 = c4.get(f"{_JOBS}/{jid4}").json()
    out["restart_running_fails_closed"] = rec4.get("status") == "failed"
    out["restart_failed_reason_honest"] = "restarted" in str(rec4.get("error") or "")
    out["restart_finished_at_stamped"] = rec4.get("finished_at") is not None

    # journal replays with no warnings — the chain verified clean
    # (checked while c3's worker is still parked, before its stale-chain
    # tail append can land)
    from fx1.serve.journal import JobJournal  # noqa: PLC0415

    res = JobJournal(state_dir2 / "jobs.jsonl").replay()
    out["journal_replay_clean"] = res.warnings == []
    gate.set()
    _wait(c3, jid4)

    # queued job at crash → same fail-closed recovery
    state_dir3 = _temporary_directory()
    runner_q2 = _Runner()
    c5 = _client(runner_q2, state_dir=state_dir3, max_inflight=1)
    _busy_executor(c5, sleep_s=2.0)
    jb = _submit(c5).json()["job_id"]
    c6 = _client(_Runner(), state_dir=state_dir3)
    rec_b = c6.get(f"{_JOBS}/{jb}").json()
    out["restart_queued_fails_closed"] = rec_b.get("status") == "failed"

    # drain latch is process-local: a restarted app accepts work again
    c7 = _client(_Runner(), state_dir=state_dir)
    c7.post(_DRAIN)
    drained = c7.post(_JOBS, json={"command": "doctor"})
    c8 = _client(_Runner(), state_dir=state_dir)
    out["drain_latch_process_local"] = (
        drained.status_code == 503 and c8.post(_JOBS, json={"command": "doctor"}).status_code == 202
    )
    return out


# ---------------------------------------------------------------------------
# Probes: /harness/runs — the synchronous twin
# ---------------------------------------------------------------------------


def _runs_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    runner = _Runner()
    client = _client(runner)

    r = client.post(_RUNS, json={"command": "doctor"})
    out["runs_200_shape"] = r.status_code == 200 and set(r.json()) == _RESULT_FIELDS
    body = r.json()
    out["runs_result_honest"] = (
        body["command"] == "doctor"
        and body["exit_code"] == 0
        and body["ok"] is True
        and body["stdout"] == "ok"
        and body["replayed"] is False
    )
    out["runs_no_replay_header"] = not _replay_hdr(r)

    r3 = client.post(_RUNS, json={"command": "no-such-cmd"})
    out["runs_unknown_404"] = r3.status_code == 404 and _err_code(r3) == "not_found"

    r4 = client.post(_RUNS, json={"command": 3})
    out["runs_bad_command_422"] = r4.status_code == 422
    r5 = client.post(_RUNS, json={"command": "doctor", "callback_url": "ftp://x"})
    out["runs_bad_callback_422"] = r5.status_code == 422
    r6 = client.post(_RUNS, json={"command": "doctor", "config": "/etc/passwd"})
    out["runs_config_escape_422"] = r6.status_code == 422 and _err_code(r6) == "validation"

    # idempotency on the sync route: replay never re-runs
    k = "runs-dup"
    r7 = client.post(_RUNS, json={"command": "doctor"}, headers={_IDEM: k})
    calls_mid = len(runner.calls)
    r8 = client.post(_RUNS, json={"command": "doctor"}, headers={_IDEM: k})
    out["runs_idem_replay"] = (
        r8.status_code == 200
        and r8.json()["replayed"] is True
        and {k: v for k, v in r8.json().items() if k != "replayed"}
        == {k: v for k, v in r7.json().items() if k != "replayed"}
        and len(runner.calls) == calls_mid
    )
    r9 = client.post(_RUNS, json={"command": "doctor", "extra_args": ["-x"]}, headers={_IDEM: k})
    out["runs_idem_conflict_409"] = (
        r9.status_code == 409 and _err_code(r9) == "idempotency_conflict"
    )

    # nonzero exit is a 200 honest result, not an error envelope
    runner_bad = _Runner(exit_code=7, stderr="nope")
    client_bad = _client(runner_bad)
    r10 = client_bad.post(_RUNS, json={"command": "doctor"})
    out["runs_nonzero_200_honest"] = (
        r10.status_code == 200
        and r10.json()["ok"] is False
        and r10.json()["exit_code"] == 7
        and r10.json()["stderr"] == "nope"
    )

    # runner fault → enveloped 500, never bare text/plain (the fix)
    runner_exc = _Runner(raise_exc=TimeoutError("command timed out"))
    client_exc = _client(runner_exc)
    r11 = client_exc.post(_RUNS, json={"command": "doctor"})
    out["runs_worker_fault_enveloped"] = (
        r11.status_code == 500
        and r11.headers.get("content-type", "").startswith("application/json")
        and isinstance(_err_code(r11), str)
    )
    out["runs_worker_fault_detail_names_type"] = "TimeoutError" in str(r11.json().get("detail", ""))

    runner_exc2 = _Runner(raise_exc=RuntimeError("disk gone"))
    client_exc2 = _client(runner_exc2)
    r12 = client_exc2.post(_RUNS, json={"command": "doctor"})
    out["runs_any_fault_enveloped"] = r12.status_code == 500 and isinstance(_err_code(r12), str)
    return out


# ---------------------------------------------------------------------------
# Probes: metering — uses billed on admit, never on auth refusal
# ---------------------------------------------------------------------------


def _metering_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    runner = _Runner()
    client = _client(runner, api_key=_ROOT)

    minted = _mint(client, _ROOT)
    raw, kid = minted["key"], minted["id"]
    # minted pre-drain: the key lifecycle refuses under the latch
    ro = client.post("/harness/keys", json={"scopes": ["read"]}, headers=_h(_ROOT))
    ro_raw, ro_id = ro.json()["key"], ro.json()["id"]
    rpm_key = _mint(client, _ROOT, rpm=3)
    rr, rid = rpm_key["key"], rpm_key["id"]
    self_key = _mint(client, _ROOT)

    uses0 = _key_card(client, _ROOT, kid)["uses"]
    _submit(client, auth=raw)
    _submit(client, auth=raw)
    uses1 = _key_card(client, _ROOT, kid)["uses"]
    out["submit_bills_use"] = uses1 - uses0 == 2

    # gate-refused submit still bills (admission happened at authenticate)
    client.post(_DRAIN, headers=_h(_ROOT))
    _submit(client, auth=raw)
    uses2 = _key_card(client, _ROOT, kid)["uses"]
    out["gate_refused_still_bills"] = uses2 - uses1 == 1

    # auth/scope refusals bill nothing
    ro0 = _key_card(client, _ROOT, ro_id)["uses"]
    client.post(_JOBS, json={"command": "doctor"}, headers=_h(ro_raw))
    ro1 = _key_card(client, _ROOT, ro_id)["uses"]
    out["scope_refusal_no_use"] = ro1 == ro0

    # tokens stay zero — harness jobs aren't model calls
    out["tokens_never_billed"] = _key_card(client, _ROOT, kid)["tokens_used"] == 0

    # rpm cap: 4th authorized call on a 3/rpm key 429s and bills no use.
    # The drain-refused submits above still authenticated, so they spend
    # the window — the cap counts authentications, not admissions.
    for _ in range(3):
        client.post(_JOBS, json={"command": "doctor"}, headers=_h(rr))
    r_over = client.post(_JOBS, json={"command": "doctor"}, headers=_h(rr))
    out["rpm_refusal_429"] = r_over.status_code == 429
    out["rpm_code_rate_limited"] = _err_code(r_over) == "rate_limited"
    out["rpm_refusal_no_bill"] = _key_card(client, _ROOT, rid)["uses"] == 3
    out["rpm_retry_after"] = int(r_over.headers.get("Retry-After", "0")) >= 1
    out["rpm_limit_headers"] = r_over.headers.get("x-ratelimit-limit-requests") == "3"

    self_r = client.get("/harness/self", headers=_h(self_key["key"]))
    out["self_meter_shape"] = (
        self_r.status_code == 200
        and self_r.json().get("object") == "self_usage"
        and self_r.json().get("metered") is True
        and self_r.json()["key"]["id"] == self_key["id"]
    )
    return out


# ---------------------------------------------------------------------------
# Probes: store-level claim contract (the resurrect fix primitive)
# ---------------------------------------------------------------------------


def _store_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    import fx1.serve.api as api_mod  # noqa: PLC0415

    def _job(status: str = "queued") -> Any:
        return api_mod.JobStatusResponse.model_validate(
            {
                "job_id": f"job-{status}-{time.time_ns()}",
                "status": status,
                "created_at": time.time(),
                "finished_at": None,
                "result": None,
                "error": None,
            }
        )

    store = api_mod._JobStore(max_entries=8)  # noqa: SLF001 — pins internals
    j = _job()
    store.put(j, None, None)
    claimed = store.start(j.job_id)
    out["start_queued_wins"] = claimed is j and j is not None
    out["start_claim_flips_running"] = j.status == "running"
    out["start_running_loses"] = store.start(j.job_id) is None
    out["start_unknown_loses"] = store.start("never-submitted") is None

    j2 = _job()
    store.put(j2, "key-1", "fp-1")
    cancelled, outcome = store.cancel(j2.job_id)
    out["cancel_then_start_loses"] = (
        outcome == "cancelled"
        and cancelled is not None
        and cancelled.status == "cancelled"
        and store.start(j2.job_id) is None
    )
    out["cancel_running_reports_status"] = store.cancel(j.job_id)[1] == "running"

    j3 = _job("succeeded")
    store.put(j3, None, None)
    out["start_succeeded_loses"] = store.start(j3.job_id) is None
    out["cancel_succeeded_reports"] = store.cancel(j3.job_id)[1] == "succeeded"

    tiny = api_mod._JobStore(max_entries=1)  # noqa: SLF001
    j4, j5 = _job(), _job()
    tiny.put(j4, "k1", "fp1")
    tiny.put(j5, "k2", "fp2")
    out["eviction_drops_idem_mapping"] = (
        tiny.get_key("k1") is None
        and tiny.get(j4.job_id) is None
        and tiny.get_key("k2") is not None
    )

    tiny2 = api_mod._JobStore(max_entries=4)  # noqa: SLF001
    j6 = _job()
    tiny2.put(j6, "dk", "dfp")
    tiny2.delete(j6.job_id)
    out["delete_removes_record_and_key"] = (
        tiny2.get(j6.job_id) is None and tiny2.get_key("dk") is None
    )
    return out


# ---------------------------------------------------------------------------
# Probes: receipt sealing — fx1_job_record.v1
# ---------------------------------------------------------------------------


def _receipt_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client = _client(_Runner())

    job_id = _submit(client).json()["job_id"]
    rec = _wait(client, job_id)

    doc_r = client.get(f"{_JOBS}/{job_id}/receipt")
    out["receipt_200_terminal"] = doc_r.status_code == 200
    doc = doc_r.json()
    out["receipt_schema_pinned"] = (
        doc["kind"] == "fx1_job_record" and doc["schema"] == "fx1_job_record.v1"
    )
    # the receipt's record is a digested projection of the served one:
    # same identity fields, stdout/stderr hashed, callback detail dropped
    proj = doc["record"]
    out["receipt_record_projects_served"] = all(
        proj[k] == rec[k]
        for k in ("job_id", "status", "created_at", "finished_at", "error", "callback_attempts")
    )
    out["receipt_result_digested"] = (
        "stdout" not in proj["result"]
        and proj["result"].get("stdout_sha256")
        == hashlib.sha256(rec["result"]["stdout"].encode()).hexdigest()
        and proj["result"].get("stderr_sha256")
        == hashlib.sha256(rec["result"]["stderr"].encode()).hexdigest()
    )
    out["receipt_verifies"] = verify_receipt_payload(doc)["valid"] is True

    tampered = json.loads(json.dumps(doc))
    tampered["record"]["status"] = "cancelled"
    out["receipt_tamper_fails_closed"] = verify_receipt_payload(tampered)["valid"] is False

    doc2 = client.get(f"{_JOBS}/{job_id}/receipt").json()
    out["receipt_seal_deterministic"] = doc2["receipt_sha256"] == doc["receipt_sha256"]

    # non-terminal records seal too — the receipt pins "as served"
    runner_q = _Runner()
    client_q = _client(runner_q, max_inflight=1)
    _busy_executor(client_q, sleep_s=2.0)
    jq = _submit(client_q).json()["job_id"]
    doc_q = client_q.get(f"{_JOBS}/{jq}/receipt")
    out["receipt_queued_also_seals"] = (
        doc_q.status_code == 200 and verify_receipt_payload(doc_q.json())["valid"] is True
    )
    out["receipt_queued_status_honest"] = doc_q.json()["record"]["status"] == "queued"
    _wait(client_q, jq)
    return out


# ---------------------------------------------------------------------------
# Probes: every refusal arrives enveloped
# ---------------------------------------------------------------------------


def _envelope_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client = _client(_Runner(), api_key=_ROOT)

    ro = client.post("/harness/keys", json={"scopes": ["read"]}, headers=_h(_ROOT))
    refusals = {
        "env_401": client.post(_JOBS, json={"command": "doctor"}, headers=_h("bogus")),
        "env_403": client.post(_JOBS, json={"command": "doctor"}, headers=_h(ro.json()["key"])),
        "env_404_cmd": client.post(_JOBS, json={"command": "nope"}, headers=_h(_ROOT)),
        "env_404_job": client.get(f"{_JOBS}/ghost", headers=_h(_ROOT)),
        "env_422": client.post(_JOBS, json={"command": 5}, headers=_h(_ROOT)),
    }
    for name, r in refusals.items():
        body = r.json()
        out[name] = (
            r.status_code >= 400
            and "detail" in body
            and isinstance(body.get("code"), str)
            and len(body.get("code", "")) > 0
        )
    return out


# ---------------------------------------------------------------------------
# Probes: HarnessClient wire legs
# ---------------------------------------------------------------------------


def _client_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    from fx1.serve.client import (  # noqa: PLC0415
        BackendNotConfiguredError,
        HarnessAuthError,
        HarnessClient,
    )
    from fx1.serve.eval_lifecycle_audit import _tc_transport  # noqa: PLC0415

    runner = _Runner()
    client = _client(runner, api_key=_ROOT)
    hc = HarnessClient("http://testserver", api_key=_ROOT, transport=_tc_transport(client))

    sub = hc.submit_run("doctor")
    job_id = sub
    res_h = hc.wait_run(job_id, timeout_s=15.0)
    out["client_wait_run_honest"] = res_h.ok and res_h.stdout == "ok"
    rec = hc.job_status(job_id)
    out["client_leg_same_shape"] = set(rec) == _JOB_FIELDS and rec["status"] == "succeeded"
    out["client_result_honest"] = rec["result"]["stdout"] == "ok"
    out["client_job_status_roundtrip"] = hc.job_status(job_id)["job_id"] == job_id

    listed = hc.list_jobs(limit=10)
    items = listed.get("jobs", []) if isinstance(listed, dict) else listed
    out["client_list_jobs"] = any(j["job_id"] == job_id for j in items)

    res = hc.run("doctor")
    out["client_run_ok"] = res.ok is True and res.stdout == "ok"

    hc_bad = HarnessClient("http://testserver", api_key="bogus", transport=_tc_transport(client))
    try:
        hc_bad.submit_run("doctor")
        out["client_auth_maps"] = False
    except HarnessAuthError:
        out["client_auth_maps"] = True
    except Exception:  # noqa: BLE001
        out["client_auth_maps"] = False

    try:
        hc.job_status("never-existed")
        out["client_404_maps_keyerror"] = False
    except KeyError:
        out["client_404_maps_keyerror"] = True
    except Exception:  # noqa: BLE001
        out["client_404_maps_keyerror"] = False

    client.post(_DRAIN, headers=_h(_ROOT))
    try:
        hc.submit_run("doctor")
        out["client_503_maps_backend_not_configured"] = False
    except BackendNotConfiguredError:
        out["client_503_maps_backend_not_configured"] = True
    except Exception:  # noqa: BLE001
        out["client_503_maps_backend_not_configured"] = False

    # cancel a queued job through the client
    runner_q = _Runner()
    client_q = _client(runner_q, max_inflight=2)
    _busy_executor(client_q, sleep_s=2.5)
    hc_q = HarnessClient("http://testserver", transport=_tc_transport(client_q))
    ja = _submit(client_q).json()["job_id"]
    rb = hc_q.submit_run("doctor")
    jb = rb
    cancelled = hc_q.cancel_job(jb)
    out["client_cancel_job"] = cancelled["status"] == "cancelled"
    _wait(client_q, ja)
    return out


# ---------------------------------------------------------------------------
# Aggregator + sealed bench
# ---------------------------------------------------------------------------


def jobs_audit() -> dict[str, Any]:
    """Run the jobs/runs battery; returns literal bools."""
    from fx1.serve.conv_audit import _audit_context  # noqa: PLC0415

    with _audit_context():
        # The webhook probes post to a real loopback sink; production
        # callback delivery remains public-network-only unless opted in.
        os.environ["FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS"] = "1"
        out: dict[str, Any] = {}
        out.update(_submit_lifecycle_probes())
        out.update(_validation_probes())
        out.update(_transition_probes())
        out.update(_idempotency_probes())
        out.update(_gate_probes())
        out.update(_webhook_probes())
        out.update(_wait_stream_probes())
        out.update(_list_probes())
        out.update(_batch_probes())
        out.update(_durability_probes())
        out.update(_runs_probes())
        out.update(_metering_probes())
        out.update(_store_probes())
        out.update(_receipt_probes())
        out.update(_envelope_probes())
        out.update(_client_probes())
        return out


def jobs_audit_bench() -> dict[str, Any]:
    """Sealed receipt: every probe True under jobs_audit.v1."""
    r = jobs_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "jobs_audit",
        "schema": "jobs_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": [
                "harness.jobs.submit",
                "harness.jobs.batch",
                "harness.jobs.get",
                "harness.jobs.list",
                "harness.jobs.cancel",
                "harness.jobs.receipt",
                "harness.jobs.events",
                "harness.runs",
                "harness.drain",
                "harness.keys.mint",
                "harness.self",
            ],
            "not_verified": [
                "real multi-process state-dir sharing",
                "webhook delivery under real network partitions",
                "executor soak beyond max_inflight admission refusals",
            ],
            "not_executed": [
                "runner subprocess teardown",
                "signal-driven drain",
            ],
        },
        "interpretation": (
            "Every probe True means: on this checkout the /harness/jobs + "
            "/harness/runs surface submits, dedupes, gates, executes, "
            "cancels, webhooks, lists, meters, and durably replays exactly "
            "as the pinned contract states — transitions order "
            "queued→running→terminal with finished_at ≥ created_at, a "
            "terminal record never flips, a cancelled job never runs (the "
            "store-level start() claim closes the dequeue race), every "
            "refusal arrives in the {detail, code} envelope, idempotency "
            "replays across restarts, webhooks fire once with verifiable "
            "HMAC and a capped retry policy that never resurrects the "
            "job, --state-dir restart fails in-flight jobs closed with an "
            "honest restart error, drain is a one-way latch that lets "
            "in-flight work complete, /harness/runs folds executor "
            "faults into the envelope rather than a bare 500, batch "
            "items fail independently with honest counts, and admission "
            "meters every authorized request while auth/scope refusals "
            "bill nothing. SYNTHETIC stub runners only — no research "
            "claim."
        ),
    }
    canon = json.dumps(out, sort_keys=True, separators=(",", ":"))
    out["receipt_sha256"] = hashlib.sha256(canon.encode()).hexdigest()
    return out


if __name__ == "__main__":
    print(json.dumps(jobs_audit_bench(), indent=1))
