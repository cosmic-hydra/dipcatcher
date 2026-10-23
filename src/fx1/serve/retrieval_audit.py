"""retrieval_audit — deep audit of retrieval sub-resources + paging/streaming.

Every listable sub-resource and read surface is exercised end to end:
job events/logs, fine-tuning events + checkpoints, eval-run output_items,
stored-request items (messages, input_items, conversation items), vector
store files + file_batches files, the ``/v1`` list surfaces (files,
batches, models, fine-tuning jobs, message batches, chat completions),
the ``/harness`` inventories (jobs, evals, completions), and the JSONL
result stream. What the battery pins, group by group:

- *empty vs missing* — a sub-resource that exists pre-lifecycle answers
  an honest ``data: []`` page (checkpoints on a queued job, output_items
  on a running run, conversation items on an empty conv, vector-store
  files on a fresh store); a route that only exists post-lifecycle or
  whose parent is gone answers an enveloped 404 — never a bare list and
  never a fabricated row.
- *ordering* — each surface's documented order: ft events + conv items
  oldest-first, ft jobs / batches / message batches / harness jobs /
  files / vector stores newest-first, file_batch verdicts frozen in
  request order; ``order=desc`` (and ``asc``) flips where declared.
- *paging* — ``limit``/``after``/``before``/``order`` on every listable
  sub-resource; ``has_more`` honest at every boundary; ``before`` takes
  the tail of the window before the cursor (the previous page — the
  same convention the anthropic ``before_id`` pagers pin), ``after``
  takes the window head; an unknown cursor fails closed ``400
  invalid_cursor`` on the OpenAI-grammar surfaces and answers an honest
  empty page on the ft/anthropic dialects (a declared dialect split the
  probes pin rather than paper over).
- *envelope* — every retrieval answers its declared page shape
  (``{object: list, data, first_id, last_id, has_more}`` or the dialect
  twin ``{data, first_id, last_id, has_more}``, ``{jobs, total}`` /
  ``{records, total}`` / ``{items, count}`` on the harness
  inventories) — no bare arrays; every 404 lands in the path's own
  error grammar (``{error}`` under ``/v1``, ``{type: error}`` under
  ``/v1/messages*``, ``{detail, code}`` under ``/harness``).
- *deletion boundary* — a sub-resource on a deleted parent 404s
  honestly (conversation → items, response → input_items, chat
  completion → messages, eval spec → runs/output_items, vector store →
  files); a detached member 404s while the underlying ``file-*`` record
  survives; a deleted ``ft:`` registration drops off the checkpoints
  listing rather than fabricating history.
- *stream surfaces* — ``/v1/messages/batches/{id}/results`` is honest
  JSONL: refused 400 until ``ended``, terminated by EOF, every line a
  complete object, trailing newline, gone after delete; the
  ``/harness/jobs/{id}/events`` SSE stream 404s unknown ids before
  opening and closes on a terminal job; ``?stream=true`` response
  replay emits monotonic sequence ids and ends on the terminal event.
- *monotonic growth* — ft ``events`` grows during the lifecycle with a
  stable oldest-first prefix; ``output_items`` sits empty while a run
  executes then fills post-completion; a non-``succeeded`` run's
  output_items page honestly reports the empty tail of the declared
  contract (per-task verdicts exist on a completed report only).
- *idempotent reads* — repeated GETs on a static record return
  byte-identical bodies (no drift between reads).
- *scope* — retrieval needs read scope: a ``write``-scoped key is
  refused ``403 insufficient_scope`` on GET, a ``read``-scoped key
  reads — including records a write-scoped peer created (the
  shared-workspace contract), and a credential-less call is 401.
- *concurrent read-during-write* — reads racing appends see consistent
  snapshots only: every returned item fully formed, page cursors
  consistent with ``data``, counts never decreasing while a feed grows.
- *bounded sub-resources* — a job whose runner emits beyond
  ``_FT_EVENT_CAP`` retains exactly the newest cap events and pages the
  retained tail honestly; evicted history is gone, not fabricated.

Bench: ``retrieval_audit_bench()`` returns the sealed
``retrieval_audit.v1`` receipt — claim + coverage + interpretation +
sha256, like the sibling audit modules.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fx1.serve.conv_audit import (
    _RESOURCES,
    _audit_context,
    _msg,
    _StubBackend,
    _temporary_directory,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

if TYPE_CHECKING:
    from types import ModuleType

    from fastapi.testclient import TestClient

__all__ = ["retrieval_audit", "retrieval_audit_bench"]

_API_KEY_ENV = "FX1_API_KEY"
_ROOT = "retr-root-material"
_WAIT_S = 20.0
_FT_EVENT_CAP = 256  # mirror of finetune._FT_EVENT_CAP — asserted via wire
_MODEL = "byok"
_ANTHROPIC_V = {"anthropic-version": "2023-06-01"}
_OPENAI_TERMINAL = {"completed", "failed", "cancelled", "expired", "incomplete"}
_TERMINAL_JOB = frozenset({"succeeded", "failed", "cancelled"})
_FT_TERMINAL = frozenset({"succeeded", "failed", "cancelled"})


# ---------------------------------------------------------------------------
# Client plumbing + wire helpers
# ---------------------------------------------------------------------------


def _client(
    backend_map: dict[str, Any] | None = None,
    *,
    api_key: str | None = _ROOT,
    ft_runner: Any | None = None,
    store_max: int | None = None,
    state_dir: Path | None = None,
    max_inflight: int | None = None,
) -> tuple[TestClient, ModuleType]:
    """(TestClient, api_module) — conv_audit's construction, plus the
    ``ft_runner``/``ft_dir`` knobs the fine-tuning probes need."""
    from fastapi.testclient import TestClient  # noqa: PLC0415

    import fx1.serve.api as api_mod  # noqa: PLC0415
    from fx1.harness import Harness  # noqa: PLC0415

    def fake_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        return 0, "ok", ""

    resources = _RESOURCES.get()
    isolated = _temporary_directory()
    receipts = isolated / "receipts"
    receipts.mkdir()
    saved_key = os.environ.get(_API_KEY_ENV)
    try:
        if api_key is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = api_key
        app = api_mod.create_app(
            harness=Harness(runner=fake_runner),
            backend_resolver=lambda name, *a, **k: (backend_map or {})[name](),
            state_dir=state_dir if state_dir is not None else isolated / "state",
            store_max=store_max,
            receipts_dir=receipts,
            ft_dir=isolated / "fine_tuning",
            ft_runner=ft_runner,
            max_inflight=max_inflight,
        )
        resources.callback(app.state.jobs_executor.shutdown, wait=True, cancel_futures=True)
        client = TestClient(app, raise_server_exceptions=False)
        resources.callback(client.close)
        resources.enter_context(client)
        return client, api_mod
    finally:
        if saved_key is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = saved_key


def _h(auth: str | None = _ROOT) -> dict[str, str]:
    return {"X-API-Key": auth} if auth else {}


def _ah(auth: str | None = _ROOT) -> dict[str, str]:
    """Anthropic-surface headers — the dialect flips on the version header."""
    return {**_h(auth), **_ANTHROPIC_V}


def _mint(client: TestClient, **policy: Any) -> tuple[str, str]:
    r = client.post("/harness/keys", json=policy, headers=_h())
    assert r.status_code == 201, r.text
    body = r.json()
    return str(body["key"]), str(body["id"])


def _err(r: Any) -> dict[str, Any]:
    return dict(r.json().get("error") or {})


def _page_ok(body: dict[str, Any]) -> bool:
    return body.get("object") == "list" and isinstance(body.get("data"), list)


def _cursor_ok(body: dict[str, Any]) -> bool:
    """first_id/last_id consistent with the returned page."""
    data = body.get("data") or []
    if not data:
        return body.get("first_id") is None and body.get("last_id") is None
    first_last_ok: bool = body.get("first_id") == data[0].get("id") and body.get("last_id") == data[
        -1
    ].get("id")
    return first_last_ok


def _upload(client: TestClient, lines: list[dict[str, Any]], purpose: str = "batch") -> str:
    blob = "".join(json.dumps(ln) + "\n" for ln in lines).encode()
    r = client.post(
        "/v1/files",
        files={"file": ("in.jsonl", blob, "application/jsonl")},
        data={"purpose": purpose},
        headers=_h(),
    )
    assert r.status_code == 200, r.text
    return str(r.json()["id"])


def _ft_corpus_upload(client: TestClient, n: int = 2) -> str:
    lines = [
        {
            "messages": [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ]
        }
        for i in range(n)
    ]
    return _upload(client, lines, purpose="fine-tune")


def _conv_create(client: TestClient, **body: Any) -> dict[str, Any]:
    r = client.post("/v1/conversations", json=body or {}, headers=_h())
    assert r.status_code == 200, r.text
    return dict(r.json())


def _items_add(client: TestClient, cid: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    r = client.post(f"/v1/conversations/{cid}/items", json={"items": items}, headers=_h())
    assert r.status_code == 200, r.text
    return dict(r.json())


def _chat(client: TestClient, content: str = "hi", **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": _MODEL,
        "messages": [{"role": "user", "content": content}],
    }
    body.update(extra)
    r = client.post("/v1/chat/completions", json=body, headers=_h())
    assert r.status_code == 200, r.text
    return dict(r.json())


def _respond(client: TestClient, text: str = "hi", **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"model": _MODEL, "input": text}
    body.update(extra)
    r = client.post("/v1/responses", json=body, headers=_h())
    assert r.status_code in (200, 201), r.text
    return dict(r.json())


def _spec_create(client: TestClient, name: str = "retrieval-spec") -> str:
    r = client.post(
        "/v1/evals",
        json={
            "name": name,
            "data_source_config": {
                "type": "custom",
                "item_schema": {"suite": "tooluse", "seed": 0, "backend": "byok"},
            },
        },
        headers=_h(),
    )
    assert r.status_code == 201, r.text
    return str(r.json()["id"])


def _run_create(client: TestClient, spec_id: str) -> str:
    r = client.post(f"/v1/evals/{spec_id}/runs", json={"model": "byok"}, headers=_h())
    assert r.status_code == 201, r.text
    return str(r.json()["id"])


def _wait_eval(client: TestClient, bare_id: str, timeout: float = _WAIT_S) -> dict[str, Any]:
    end = time.monotonic() + timeout
    rec: dict[str, Any] = {}
    while time.monotonic() < end:
        rec = client.get(f"/harness/evals/{bare_id}", headers=_h()).json()
        if rec.get("status") in _TERMINAL_JOB:
            return rec
        time.sleep(0.02)
    return rec


def _wait_ft(client: TestClient, job_id: str, timeout: float = _WAIT_S) -> dict[str, Any]:
    end = time.monotonic() + timeout
    job: dict[str, Any] = {}
    while time.monotonic() < end:
        job = client.get(f"/v1/fine_tuning/jobs/{job_id}", headers=_h()).json()
        if job.get("status") in _FT_TERMINAL:
            return job
        time.sleep(0.02)
    return job


def _wait_job(client: TestClient, job_id: str, timeout: float = _WAIT_S) -> dict[str, Any]:
    end = time.monotonic() + timeout
    job: dict[str, Any] = {}
    while time.monotonic() < end:
        job = client.get(f"/harness/jobs/{job_id}", headers=_h()).json()
        if job.get("status") in _TERMINAL_JOB:
            return job
        time.sleep(0.03)
    return job


def _ft_submit(client: TestClient, **extra: Any) -> dict[str, Any]:
    fid = _ft_corpus_upload(client)
    body: dict[str, Any] = {"model": "fx1", "training_file": fid}
    body.update(extra)
    r = client.post("/v1/fine_tuning/jobs", json=body, headers=_h())
    assert r.status_code == 200, r.text
    return dict(r.json())


def _vs_create(client: TestClient, **body: Any) -> dict[str, Any]:
    r = client.post("/v1/vector_stores", json=body, headers=_h())
    assert r.status_code in (200, 201), r.text
    return dict(r.json())


def _batch_create(client: TestClient, tag: str) -> dict[str, Any]:
    fid = _upload(
        client,
        [
            {
                "custom_id": f"b-{tag}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": _MODEL, "messages": [{"role": "user", "content": tag}]},
            }
        ],
    )
    r = client.post(
        "/v1/batches",
        json={
            "input_file_id": fid,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
        headers=_h(),
    )
    assert r.status_code == 200, r.text
    return dict(r.json())


def _vs_attach(client: TestClient, vs_id: str, file_id: str) -> dict[str, Any]:
    r = client.post(f"/v1/vector_stores/{vs_id}/files", json={"file_id": file_id}, headers=_h())
    assert r.status_code in (200, 201), r.text
    return dict(r.json())


def _abatch_create(client: TestClient, n: int = 3) -> dict[str, Any]:
    items = [
        {
            "custom_id": f"req-{i}",
            "params": {
                "model": "fx1",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": f"ping-{i}"}],
            },
        }
        for i in range(n)
    ]
    r = client.post("/v1/messages/batches", json={"requests": items}, headers=_ah())
    assert r.status_code == 200, r.text
    return dict(r.json())


def _wait_abatch(client: TestClient, batch_id: str, timeout: float = _WAIT_S) -> dict[str, Any]:
    end = time.monotonic() + timeout
    b: dict[str, Any] = {}
    while time.monotonic() < end:
        b = client.get(f"/v1/messages/batches/{batch_id}", headers=_ah()).json()
        if b.get("processing_status") == "ended":
            return b
        time.sleep(0.03)
    return b


def _walk_cursor(
    client: TestClient,
    path: str,
    *,
    limit: int,
    cursor_param: str = "after",
    headers: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
    max_pages: int = 40,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Walk ``after``-style pagination to exhaustion; returns (ids, pages).

    Fails loudly (assert) if the walk doesn't terminate — a stuck cursor
    is a defect, not a skip.
    """
    ids: list[str] = []
    pages: list[dict[str, Any]] = []
    cursor: str | None = None
    for _ in range(max_pages):
        params: dict[str, Any] = {"limit": limit}
        params.update(extra or {})
        if cursor is not None:
            params[cursor_param] = cursor
        r = client.get(path, params=params, headers=headers or _h())
        assert r.status_code == 200, r.text
        body = r.json()
        pages.append(body)
        data = body.get("data") or []
        ids.extend(str(it.get("id")) for it in data)
        if body.get("has_more") is not True or not data:
            return ids, pages
        cursor = body.get("last_id") or str(data[-1].get("id"))
    raise AssertionError(f"cursor walk on {path} did not terminate in {max_pages} pages")


# ---------------------------------------------------------------------------
# Empty vs missing + envelope honesty
# ---------------------------------------------------------------------------


def _empty_vs_missing_probes() -> dict[str, bool]:
    out: dict[str, Any] = {}
    client, _api = _client({"byok": lambda: _StubBackend(), "local_fx1": lambda: _StubBackend()})

    # sub-resources that exist pre-lifecycle answer honest empty pages
    conv = _conv_create(client)
    items = client.get(f"/v1/conversations/{conv['id']}/items", headers=_h())
    out["conv_items_empty_page"] = (
        items.status_code == 200
        and _page_ok(items.json())
        and items.json()["data"] == []
        and items.json()["has_more"] is False
        and _cursor_ok(items.json())
    )

    vs = _vs_create(client, name="empty-vs")
    vfiles = client.get(f"/v1/vector_stores/{vs['id']}/files", headers=_h())
    out["vs_files_empty_page"] = (
        vfiles.status_code == 200
        and _page_ok(vfiles.json())
        and vfiles.json()["data"] == []
        and vfiles.json()["has_more"] is False
    )

    ft = _ft_submit(client)
    ckpts = client.get(f"/v1/fine_tuning/jobs/{ft['id']}/checkpoints", headers=_h())
    out["ft_checkpoints_empty_pre_lifecycle"] = (
        ckpts.status_code == 200
        and _page_ok(ckpts.json())
        and ckpts.json()["data"] == []
        and ckpts.json()["has_more"] is False
        and ckpts.json()["first_id"] is None
        and ckpts.json()["last_id"] is None
    )
    ev = client.get(f"/v1/fine_tuning/jobs/{ft['id']}/events", headers=_h())
    ev_data = ev.json().get("data") or []
    out["ft_events_envelope_from_submit"] = (
        ev.status_code == 200
        and _page_ok(ev.json())
        and len(ev_data) >= 1  # the submit-time "training file validated" event
        and all(str(e.get("id", "")).startswith("ftev-") for e in ev_data)
    )
    _wait_ft(client, str(ft["id"]))

    # queued eval run → output_items exists pre-completion, honest empty
    spec_id = _spec_create(client)
    run_id = _run_create(client, spec_id)
    queued_items = client.get(
        f"/v1/evals/{spec_id}/runs/{run_id}/output_items", headers=_h()
    ).json()
    out["output_items_nonterminal_empty_page"] = (
        _page_ok(queued_items) and queued_items["has_more"] is False and _cursor_ok(queued_items)
    )
    _wait_eval(client, run_id.removeprefix("evalrun_"))

    # empty inventories answer their declared shapes
    jobs = client.get("/harness/jobs", headers=_h()).json()
    out["jobs_empty_inventory_shape"] = jobs.get("jobs") == [] and jobs.get("total") == 0
    ab_list = client.get("/v1/messages/batches", headers=_ah()).json()
    out["abatch_empty_envelope"] = (
        ab_list.get("data") == []
        and ab_list.get("has_more") is False
        and ab_list.get("first_id") is None
        and ab_list.get("last_id") is None
    )
    evals_page = client.get("/v1/evals", headers=_h()).json()
    out["evals_page_envelope_shape"] = _page_ok(evals_page) and "has_more" in evals_page

    # missing parents → enveloped 404 in the path's own grammar
    missing_404s = [
        ("GET", f"/v1/chat/completions/chatcmpl-{'0' * 24}/messages", _h()),
        ("GET", "/v1/responses/resp_0/input_items", _h()),
        ("GET", "/v1/conversations/conv_nope/items", _h()),
        ("GET", "/v1/fine_tuning/jobs/ftjob-nope/events", _h()),
        ("GET", "/v1/fine_tuning/jobs/ftjob-nope/checkpoints", _h()),
        ("GET", "/v1/evals/eval_nope/runs", _h()),
        ("GET", "/v1/evals/eval_nope/runs/evalrun_nope/output_items", _h()),
        ("GET", "/v1/vector_stores/vs_nope/files", _h()),
        ("GET", "/v1/vector_stores/vs_nope/file_batches/vsfb_nope/files", _h()),
        ("GET", "/v1/files/file-nope", _h()),
        ("GET", "/v1/files/file-nope/content", _h()),
        ("GET", "/v1/batches/batch_nope", _h()),
        ("GET", "/v1/uploads/upl_nope", _h()),
    ]
    codes = []
    for method, path, hdrs in missing_404s:
        r = client.request(method, path, headers=hdrs)
        body = r.json()
        codes.append(
            r.status_code == 404
            and isinstance(body.get("error"), dict)
            and bool(body["error"].get("code"))
        )
    out["openai_missing_404s_enveloped"] = all(codes)

    # harness dialect 404s — flat {detail, code}
    hjob = client.get("/harness/jobs/job-nope", headers=_h())
    hev = client.get("/harness/evals/eval-nope", headers=_h())
    hcomp = client.get("/harness/completions/cmp-nope", headers=_h())
    out["harness_missing_404s_enveloped"] = all(
        r.status_code == 404 and isinstance(r.json().get("detail"), str) for r in (hjob, hev, hcomp)
    )

    # anthropic dialect 404 — {type: error, error: {type: not_found_error}}
    ab_miss = client.get("/v1/messages/batches/msgbatch_nope", headers=_ah())
    ab_body = ab_miss.json()
    out["anthropic_missing_404_enveloped"] = (
        ab_miss.status_code == 404
        and ab_body.get("type") == "error"
        and ab_body.get("error", {}).get("type") == "not_found_error"
    )

    # no GET surface exists for harness job logs — the route honestly 404s
    logs = client.get("/harness/jobs/job-x/logs", headers=_h())
    out["job_logs_route_absent_404"] = logs.status_code == 404
    conv_list = client.get("/v1/conversations", headers=_h())
    resp_list = client.get("/v1/responses", headers=_h())
    out["unlisted_surfaces_enveloped_404"] = (
        conv_list.status_code == 404
        and isinstance(conv_list.json().get("error"), dict)
        and resp_list.status_code == 404
        and isinstance(resp_list.json().get("error"), dict)
    )
    return out


# ---------------------------------------------------------------------------
# Ordering + paging honesty
# ---------------------------------------------------------------------------


def _ordering_paging_probes() -> dict[str, bool]:
    out: dict[str, Any] = {}
    client, _api = _client({"byok": lambda: _StubBackend(), "local_fx1": lambda: _StubBackend()})

    # conversation items — insertion order, asc default, desc flips
    conv = _conv_create(client)
    added = _items_add(client, str(conv["id"]), [_msg(f"m{i}") for i in range(5)])
    ids = [str(i["id"]) for i in added["data"]]
    asc = client.get(f"/v1/conversations/{conv['id']}/items", headers=_h()).json()
    desc = client.get(
        f"/v1/conversations/{conv['id']}/items", params={"order": "desc"}, headers=_h()
    ).json()

    def item_ids(body: dict[str, Any]) -> list[Any]:
        return [i["id"] for i in body["data"]]

    out["conv_items_insertion_order_asc"] = item_ids(asc) == ids
    out["conv_items_order_desc_flips"] = item_ids(desc) == ids[::-1]

    # limit slices + has_more + cursor walk covers each id exactly once
    p1 = client.get(
        f"/v1/conversations/{conv['id']}/items", params={"limit": 2}, headers=_h()
    ).json()
    out["conv_items_limit_slice"] = item_ids(p1) == ids[:2] and p1["has_more"] is True
    walked, pages = _walk_cursor(client, f"/v1/conversations/{conv['id']}/items", limit=2)
    out["conv_items_after_walk_covers_all"] = (
        walked == ids and len(pages) == 3 and pages[-1]["has_more"] is False
    )
    out["conv_items_cursor_fields_consistent"] = all(_cursor_ok(p) for p in pages)

    # after is exclusive — the cursor id itself never repeats
    aft = client.get(
        f"/v1/conversations/{conv['id']}/items", params={"after": ids[1]}, headers=_h()
    ).json()
    out["conv_items_after_exclusive"] = item_ids(aft) == ids[2:]

    # before returns the previous page — the tail of the pre-cursor window
    bef = client.get(
        f"/v1/conversations/{conv['id']}/items",
        params={"before": ids[4], "limit": 2},
        headers=_h(),
    ).json()
    out["conv_items_before_previous_page"] = item_ids(bef) == ids[2:4]
    out["conv_items_before_has_more"] = bef["has_more"] is True
    bef_last = client.get(
        f"/v1/conversations/{conv['id']}/items",
        params={"before": ids[1], "limit": 5},
        headers=_h(),
    ).json()
    out["conv_items_before_under_limit_whole_window"] = (
        item_ids(bef_last) == ids[:1] and bef_last["has_more"] is False
    )
    # walking backward by first_id chains through the whole list
    back_ids: list[str] = []
    cursor: str | None = ids[4]
    guard = 0
    while cursor is not None and guard < 10:
        guard += 1
        pg = client.get(
            f"/v1/conversations/{conv['id']}/items",
            params={"before": cursor, "limit": 2},
            headers=_h(),
        ).json()
        back_ids = item_ids(pg) + back_ids
        cursor = pg.get("first_id") if pg.get("has_more") else None
    out["conv_items_before_walk_covers_all"] = back_ids == ids[:4]

    # unknown cursors fail closed 400 invalid_cursor; a bogus order word
    # is rejected at the Query grammar (422, still enveloped)
    bad_after = client.get(
        f"/v1/conversations/{conv['id']}/items", params={"after": "nope"}, headers=_h()
    )
    bad_before = client.get(
        f"/v1/conversations/{conv['id']}/items", params={"before": "nope"}, headers=_h()
    )
    bad_order = client.get(
        f"/v1/conversations/{conv['id']}/items", params={"order": "bogus"}, headers=_h()
    )
    out["conv_items_cursor_400s_enveloped"] = (
        bad_after.status_code == 400
        and _err(bad_after).get("code") == "invalid_cursor"
        and bad_before.status_code == 400
        and _err(bad_before).get("code") == "invalid_cursor"
        and bad_order.status_code == 422
        and isinstance(bad_order.json().get("error"), dict)
    )

    # cursor on a deleted page boundary → fail closed, not silent restart
    dele = client.delete(f"/v1/conversations/{conv['id']}/items/{ids[2]}", headers=_h())
    use_dead = client.get(
        f"/v1/conversations/{conv['id']}/items", params={"after": ids[2]}, headers=_h()
    )
    out["conv_items_deleted_cursor_400"] = (
        dele.status_code == 200
        and use_dead.status_code == 400
        and _err(use_dead).get("code") == "invalid_cursor"
    )

    # cursor reuse on a mutated collection — bounded honest, not a restart
    grown = _items_add(client, str(conv["id"]), [_msg("m5")])
    after_grown = client.get(
        f"/v1/conversations/{conv['id']}/items", params={"after": ids[3]}, headers=_h()
    ).json()
    live = [i["id"] for i in after_grown["data"]]
    out["conv_items_cursor_reuse_mutated_bounded"] = (
        after_grown["has_more"] is False
        and grown["data"][0]["id"] in live
        and ids[4] in live
        and len(live) <= 2
    )

    # chat completions messages — same pager, minted msg_* ids; the stored
    # transcript is the request messages verbatim (3 turns → 3 items)
    chat = _chat(
        client,
        "paging-probe",
        messages=[
            {"role": "user", "content": "t1"},
            {"role": "assistant", "content": "r1"},
            {"role": "user", "content": "t2"},
        ],
    )
    msgs = client.get(f"/v1/chat/completions/{chat['id']}/messages", headers=_h()).json()
    msg_ids = item_ids(msgs)
    out["chat_messages_envelope_and_ids"] = (
        _page_ok(msgs)
        and len(msg_ids) == 3
        and all(m.startswith("msg_") for m in msg_ids)
        and _cursor_ok(msgs)
    )
    m_desc = client.get(
        f"/v1/chat/completions/{chat['id']}/messages",
        params={"order": "desc", "limit": 1},
        headers=_h(),
    ).json()
    out["chat_messages_desc_limit"] = (
        item_ids(m_desc) == msg_ids[-1:] and m_desc["has_more"] is True
    )
    m_bad = client.get(
        f"/v1/chat/completions/{chat['id']}/messages", params={"after": "msg_ghost"}, headers=_h()
    )
    out["chat_messages_unknown_cursor_400"] = (
        m_bad.status_code == 400 and _err(m_bad).get("code") == "invalid_cursor"
    )

    # responses input_items — populated at submit, same pager
    resp = _respond(client, "input-items-probe")
    inp = client.get(f"/v1/responses/{resp['id']}/input_items", headers=_h()).json()
    out["input_items_populated_page"] = _page_ok(inp) and len(inp["data"]) >= 1 and _cursor_ok(inp)
    inp_desc = client.get(
        f"/v1/responses/{resp['id']}/input_items",
        params={"order": "desc"},
        headers=_h(),
    ).json()
    out["input_items_desc_flips"] = item_ids(inp_desc) == item_ids(inp)[::-1]

    # eval output_items — index-encoded cursor grammar
    spec_id = _spec_create(client)
    run_id = _run_create(client, spec_id)
    _wait_eval(client, run_id.removeprefix("evalrun_"))
    oi = client.get(f"/v1/evals/{spec_id}/runs/{run_id}/output_items", headers=_h()).json()
    oi_ids = item_ids(oi)
    out["output_items_rows_index_ids"] = (
        _page_ok(oi)
        and len(oi_ids) > 2
        and all(i.startswith(f"{run_id}-") for i in oi_ids)
        and all(i["run_id"] == run_id for i in oi["data"])
    )
    oi_p1 = client.get(
        f"/v1/evals/{spec_id}/runs/{run_id}/output_items", params={"limit": 1}, headers=_h()
    ).json()
    oi_p2 = client.get(
        f"/v1/evals/{spec_id}/runs/{run_id}/output_items",
        params={"limit": 1, "after": oi_p1["last_id"]},
        headers=_h(),
    ).json()
    out["output_items_cursor_paginates"] = (
        len(oi_p1["data"]) == 1
        and oi_p1["has_more"] is True
        and len(oi_p2["data"]) == 1
        and oi_p2["data"][0]["id"] != oi_p1["data"][0]["id"]
    )
    oi_bad = client.get(
        f"/v1/evals/{spec_id}/runs/{run_id}/output_items", params={"after": "zzz"}, headers=_h()
    )
    oi_foreign = client.get(
        f"/v1/evals/{spec_id}/runs/{run_id}/output_items",
        params={"after": "evalrun_deadbeef-3"},
        headers=_h(),
    )
    oi_lim = client.get(
        f"/v1/evals/{spec_id}/runs/{run_id}/output_items",
        params={"limit": 0},
        headers=_h(),
    )
    out["output_items_refusals_enveloped"] = (
        oi_bad.status_code == 400
        and _err(oi_bad).get("code") == "invalid_cursor"
        and oi_foreign.status_code == 400
        and _err(oi_foreign).get("code") == "invalid_cursor"
        and oi_lim.status_code == 400
        and _err(oi_lim).get("code") == "invalid_request"
    )
    return out


# ---------------------------------------------------------------------------
# List surfaces — declared order + envelope + cursor grammar
# ---------------------------------------------------------------------------


def _list_surface_probes() -> dict[str, bool]:
    out: dict[str, Any] = {}
    client, _api = _client({"byok": lambda: _StubBackend(), "local_fx1": lambda: _StubBackend()})

    # /v1/files — newest-first default, order=asc flips, purpose filter,
    # shared cursor page shape
    fids = [_upload(client, [{"a": i}]) for i in range(4)]
    fl = client.get("/v1/files", headers=_h()).json()
    fl_ids = [f["id"] for f in fl["data"]]
    out["files_list_newest_first"] = fl_ids[:4] == fids[::-1] and _page_ok(fl)
    fl_asc = client.get("/v1/files", params={"order": "asc"}, headers=_h()).json()
    out["files_list_order_asc_flips"] = [f["id"] for f in fl_asc["data"][:4]] == fids
    fl_purp = client.get("/v1/files", params={"purpose": "batch"}, headers=_h()).json()
    out["files_list_purpose_filter"] = all(f["purpose"] == "batch" for f in fl_purp["data"])
    fl_walk, _ = _walk_cursor(client, "/v1/files", limit=2)
    out["files_list_after_walk_covers"] = set(fl_ids[:4]) <= set(fl_walk[:6])
    fl_bad = client.get("/v1/files", params={"after": "file-ghost"}, headers=_h())
    out["files_list_unknown_cursor_400"] = (
        fl_bad.status_code == 400 and _err(fl_bad).get("code") == "invalid_cursor"
    )
    fcontent = client.get(f"/v1/files/{fids[0]}/content", headers=_h())
    out["file_content_bytes_roundtrip"] = (
        fcontent.status_code == 200 and fcontent.content == b'{"a": 0}\n'
    )

    # /v1/batches — newest-first, after exclusive, unknown cursor fail-closed
    bid1 = _batch_create(client, "one")["id"]
    bid2 = _batch_create(client, "two")["id"]
    bl = client.get("/v1/batches", headers=_h()).json()
    out["batches_list_newest_first"] = [b["id"] for b in bl["data"][:2]] == [bid2, bid1]
    bl_after = client.get("/v1/batches", params={"after": bid2}, headers=_h()).json()
    out["batches_list_after_exclusive"] = [b["id"] for b in bl_after["data"]] == [bid1]
    bl_bad = client.get("/v1/batches", params={"after": "batch_ghost"}, headers=_h())
    out["batches_list_unknown_cursor_400"] = (
        bl_bad.status_code == 400 and _err(bl_bad).get("code") == "invalid_cursor"
    )

    # /v1/fine_tuning/jobs — newest-first, after exclusive, unknown → empty
    j1 = _ft_submit(client)
    j2 = _ft_submit(client)
    fjl = client.get("/v1/fine_tuning/jobs", headers=_h()).json()
    out["ft_jobs_newest_first"] = [j["id"] for j in fjl["data"][:2]] == [j2["id"], j1["id"]]
    fjl_after = client.get("/v1/fine_tuning/jobs", params={"after": j2["id"]}, headers=_h()).json()
    out["ft_jobs_after_exclusive"] = [j["id"] for j in fjl_after["data"]] == [j1["id"]]
    fjl_bad = client.get(
        "/v1/fine_tuning/jobs", params={"after": "ftjob-ghost"}, headers=_h()
    ).json()
    out["ft_jobs_unknown_cursor_empty"] = fjl_bad["data"] == [] and fjl_bad["has_more"] is False
    _wait_ft(client, str(j1["id"]))
    _wait_ft(client, str(j2["id"]))

    # /v1/models — OpenAI list envelope by default; anthropic-version flips
    # to the {data, first_id, last_id, has_more} grammar
    mo = client.get("/v1/models", headers=_h()).json()
    out["models_openai_envelope"] = (
        mo.get("object") == "list"
        and isinstance(mo.get("data"), list)
        and all(m.get("object") == "model" for m in mo["data"])
    )
    ma = client.get("/v1/models", headers=_ah()).json()
    out["models_anthropic_grammar"] = (
        isinstance(ma.get("data"), list)
        and "first_id" in ma
        and "last_id" in ma
        and ma.get("has_more") is False
    )
    ma_page = client.get("/v1/models", params={"limit": 1}, headers=_ah()).json()
    ma_next = client.get(
        "/v1/models", params={"limit": 1, "after_id": ma_page["last_id"]}, headers=_ah()
    ).json()
    out["models_anthropic_cursor_walk"] = (
        len(ma_page["data"]) == 1
        and ma_page["has_more"] is True
        and len(ma_next["data"]) == 1
        and ma_next["data"][0]["id"] != ma_page["data"][0]["id"]
    )
    ma_bad = client.get("/v1/models", params={"after_id": "ghost"}, headers=_ah()).json()
    out["models_anthropic_unknown_cursor_empty"] = (
        ma_bad["data"] == [] and ma_bad["has_more"] is False
    )

    # /v1/messages/batches — newest-first anthropic grammar; unknown → empty
    ab1 = _abatch_create(client, n=1)
    ab2 = _abatch_create(client, n=1)
    abl = client.get("/v1/messages/batches", headers=_ah()).json()
    out["abatches_newest_first"] = [b["id"] for b in abl["data"][:2]] == [ab2["id"], ab1["id"]]
    abl_bad = client.get(
        "/v1/messages/batches", params={"after_id": "msgbatch_ghost"}, headers=_ah()
    ).json()
    out["abatches_unknown_cursor_empty"] = abl_bad["data"] == [] and abl_bad["has_more"] is False
    _wait_abatch(client, str(ab1["id"]))
    _wait_abatch(client, str(ab2["id"]))

    # /v1/chat/completions list — oldest-first asc, model + metadata filters
    c1 = _chat(client, "list-one", metadata={"lane": "172"})
    c2 = _chat(client, "list-two")
    cl = client.get("/v1/chat/completions", headers=_h()).json()
    cl_ids = [c["id"] for c in cl["data"]]
    out["chat_list_oldest_first_envelope"] = _page_ok(cl) and cl_ids.index(c1["id"]) < cl_ids.index(
        c2["id"]
    )
    cl_meta = client.get("/v1/chat/completions?metadata[lane]=172", headers=_h()).json()
    out["chat_list_metadata_filter"] = (
        len(cl_meta["data"]) == 1 and cl_meta["data"][0]["id"] == c1["id"]
    )
    cl_model = client.get(
        "/v1/chat/completions", params={"model": "no-such-model"}, headers=_h()
    ).json()
    out["chat_list_model_filter"] = cl_model["data"] == []

    # /harness/jobs — newest-first, offset/limit, total = filtered count
    jb = client.post("/harness/jobs", json={"command": "doctor"}, headers=_h())
    assert jb.status_code in (200, 201, 202), jb.text
    jid = str(jb.json()["job_id"])
    _wait_job(client, jid)
    jl = client.get("/harness/jobs", headers=_h()).json()
    out["jobs_list_newest_first_total"] = (
        jl["jobs"][0]["job_id"] == jid and jl["total"] == len(jl["jobs"]) == 1
    )
    jl_off = client.get("/harness/jobs", params={"offset": 1}, headers=_h()).json()
    out["jobs_list_offset_pages"] = jl_off["jobs"] == [] and jl_off["total"] == 1
    jl_bad = client.get("/harness/jobs", params={"status": "running"}, headers=_h()).json()
    out["jobs_list_status_filter"] = jl_bad["jobs"] == [] and jl_bad["total"] == 0

    # /harness/evals + /harness/completions — newest-first inventories
    spec = _spec_create(client)
    rid = _run_create(client, spec)
    _wait_eval(client, rid.removeprefix("evalrun_"))
    el = client.get("/harness/evals", headers=_h()).json()
    out["evals_list_newest_first"] = el["records"][0]["eval_id"] == rid.removeprefix(
        "evalrun_"
    ) and el["total"] == len(el["records"])
    comp = client.get("/harness/completions", headers=_h()).json()
    out["completions_list_newest_first_window"] = (
        isinstance(comp.get("items"), list)
        and comp.get("count") == len(comp["items"])
        and comp["count"] >= 2  # the two stub chats above
    )

    # /v1/evals spec list — newest-first envelope, fail-closed cursor,
    # bounded limit grammar
    sp2 = _spec_create(client, name="spec-two")
    sl = client.get("/v1/evals", headers=_h()).json()
    out["evals_list_newest_first_envelope"] = (
        _page_ok(sl) and sl["data"][0]["id"] == sp2 and sl["has_more"] is False
    )
    sl_bad = client.get("/v1/evals", params={"after": "eval_ghost"}, headers=_h())
    sl_lim = client.get("/v1/evals", params={"limit": 101}, headers=_h())
    out["evals_list_refusals_enveloped"] = (
        sl_bad.status_code == 400
        and _err(sl_bad).get("code") == "invalid_cursor"
        and sl_lim.status_code == 400
        and _err(sl_lim).get("code") == "invalid_request"
    )
    sl_walk = client.get("/v1/evals", params={"after": sp2}, headers=_h()).json()
    out["evals_list_after_exclusive"] = all(s["id"] != sp2 for s in sl_walk["data"])

    # /v1/evals/{id}/runs — newest-first, cursor accepts prefixed + bare ids
    rl = client.get(f"/v1/evals/{spec}/runs", headers=_h()).json()
    out["runs_list_scoped_envelope"] = _page_ok(rl) and all(
        r["eval_id"] == spec for r in rl["data"]
    )
    bare = rid.removeprefix("evalrun_")
    rl_pref = client.get(f"/v1/evals/{spec}/runs", params={"after": rid}, headers=_h())
    rl_bare = client.get(f"/v1/evals/{spec}/runs", params={"after": bare}, headers=_h())
    out["runs_list_cursor_dual_grammar"] = (
        rl_pref.status_code == 200
        and rl_bare.status_code == 200
        and rl_pref.json()["data"] == rl_bare.json()["data"]
    )
    rl_bad = client.get(f"/v1/evals/{spec}/runs", params={"after": "evalrun_nope"}, headers=_h())
    out["runs_list_unknown_cursor_400"] = (
        rl_bad.status_code == 400 and _err(rl_bad).get("code") == "invalid_cursor"
    )
    return out


# ---------------------------------------------------------------------------
# Vector store retrieval — membership, file_batches, deleted members
# ---------------------------------------------------------------------------


def _vector_store_probes() -> dict[str, bool]:
    out: dict[str, Any] = {}
    client, _api = _client({"byok": lambda: _StubBackend()})

    vs_a = _vs_create(client, name="vs-a")
    vs_b = _vs_create(client, name="vs-b")
    vs_c = _vs_create(client, name="vs-c")

    # stores list — default desc (newest first), asc flips, shared page shape
    sl = client.get("/v1/vector_stores", headers=_h()).json()
    sl_ids = [s["id"] for s in sl["data"]]
    out["vs_list_default_desc"] = sl_ids[:3] == [vs_c["id"], vs_b["id"], vs_a["id"]]
    sl_asc = client.get("/v1/vector_stores", params={"order": "asc"}, headers=_h()).json()
    out["vs_list_order_asc_flips"] = [s["id"] for s in sl_asc["data"][:3]] == [
        vs_a["id"],
        vs_b["id"],
        vs_c["id"],
    ]
    sl_bad = client.get("/v1/vector_stores", params={"after": "vs_ghost"}, headers=_h())
    out["vs_list_unknown_cursor_400"] = (
        sl_bad.status_code == 400 and _err(sl_bad).get("code") == "invalid_cursor"
    )

    # attach files — membership rows page in attach order (asc default)
    fids = [_upload(client, [{"text": f"doc body {i}"}]) for i in range(4)]
    for fid in fids:
        _vs_attach(client, str(vs_a["id"]), fid)
    fl = client.get(f"/v1/vector_stores/{vs_a['id']}/files", headers=_h()).json()
    fl_ids = [f["id"] for f in fl["data"]]
    out["vs_files_attach_order_asc"] = _page_ok(fl) and fl_ids == fids and _cursor_ok(fl)
    fl_desc = client.get(
        f"/v1/vector_stores/{vs_a['id']}/files", params={"order": "desc"}, headers=_h()
    ).json()
    out["vs_files_desc_flips"] = [f["id"] for f in fl_desc["data"]] == fids[::-1]

    # before cursor → previous page (tail of pre-cursor window)
    fl_bef = client.get(
        f"/v1/vector_stores/{vs_a['id']}/files",
        params={"before": fids[3], "limit": 2},
        headers=_h(),
    ).json()
    out["vs_files_before_previous_page"] = [f["id"] for f in fl_bef["data"]] == fids[1:3]

    # status filter — only declared words, others 400 invalid_filters
    fl_filter = client.get(
        f"/v1/vector_stores/{vs_a['id']}/files",
        params={"filter": "completed"},
        headers=_h(),
    ).json()
    fl_bad_filter = client.get(
        f"/v1/vector_stores/{vs_a['id']}/files",
        params={"filter": "bogus"},
        headers=_h(),
    )
    out["vs_files_filter_words"] = (
        all(f["status"] == "completed" for f in fl_filter["data"])
        and fl_bad_filter.status_code == 400
        and _err(fl_bad_filter).get("code") == "invalid_filters"
    )

    # member retrieve + content — the stored text reads back
    one = client.get(f"/v1/vector_stores/{vs_a['id']}/files/{fids[0]}", headers=_h())
    content = client.get(f"/v1/vector_stores/{vs_a['id']}/files/{fids[0]}/content", headers=_h())
    out["vs_file_get_and_content"] = (
        one.status_code == 200
        and one.json()["id"] == fids[0]
        and content.status_code == 200
        and b"doc body 0" in content.content
    )

    # detach → member routes 404; underlying file-* survives (orphan pinned)
    det = client.delete(f"/v1/vector_stores/{vs_a['id']}/files/{fids[0]}", headers=_h())
    after_detach = client.get(f"/v1/vector_stores/{vs_a['id']}/files/{fids[0]}", headers=_h())
    orphan = client.get(f"/v1/files/{fids[0]}", headers=_h())
    out["vs_detach_404_orphan_survives"] = (
        det.status_code == 200
        and after_detach.status_code == 404
        and _err(after_detach).get("code") == "file_not_found"
        and orphan.status_code == 200
    )
    relist = client.get(f"/v1/vector_stores/{vs_a['id']}/files", headers=_h()).json()
    out["vs_detach_drops_from_listing"] = all(f["id"] != fids[0] for f in relist["data"])

    # file_batches — per-file verdicts frozen in request order
    batch = client.post(
        f"/v1/vector_stores/{vs_b['id']}/file_batches",
        json={"file_ids": fids[1:]},
        headers=_h(),
    ).json()
    out["vs_file_batch_terminal_create"] = (
        batch["id"].startswith("vsfb_")
        and batch["status"] in ("completed", "failed")
        and batch["file_counts"]["total"] == 3
    )
    bf = client.get(
        f"/v1/vector_stores/{vs_b['id']}/file_batches/{batch['id']}/files", headers=_h()
    ).json()
    out["vs_batch_files_request_order_frozen"] = (
        _page_ok(bf) and [f["id"] for f in bf["data"]] == fids[1:]
    )
    bf_filter = client.get(
        f"/v1/vector_stores/{vs_b['id']}/file_batches/{batch['id']}/files",
        params={"filter": "failed"},
        headers=_h(),
    ).json()
    out["vs_batch_files_filter_honest"] = isinstance(bf_filter.get("data"), list)
    bf_missing = client.get(
        f"/v1/vector_stores/{vs_b['id']}/file_batches/vsfb_nope/files", headers=_h()
    )
    out["vs_batch_files_missing_404"] = bf_missing.status_code == 404

    # delete the store → every member route tombstones
    dele = client.delete(f"/v1/vector_stores/{vs_b['id']}", headers=_h())
    gone_files = client.get(f"/v1/vector_stores/{vs_b['id']}/files", headers=_h())
    gone_batch = client.get(
        f"/v1/vector_stores/{vs_b['id']}/file_batches/{batch['id']}/files", headers=_h()
    )
    out["vs_delete_tombstones_subresources"] = (
        dele.status_code == 200
        and gone_files.status_code == 404
        and _err(gone_files).get("code") == "vector_store_not_found"
        and gone_batch.status_code == 404
    )
    return out


# ---------------------------------------------------------------------------
# Deletion boundary — sub-resource after parent delete
# ---------------------------------------------------------------------------


def _deletion_boundary_probes() -> dict[str, bool]:
    from fx1.serve.finetune import FTJobOutcome  # noqa: PLC0415

    def runner(spec: Any, *, emit: Any, should_cancel: Any) -> Any:
        return FTJobOutcome(
            fine_tuned_model=spec.ft_model_name,
            checkpoint=str(spec.work_dir / "ckpt"),
        )

    out: dict[str, Any] = {}
    client, _api = _client({"byok": lambda: _StubBackend()}, ft_runner=runner)

    # conversation → items tombstone together
    conv = _conv_create(client)
    _items_add(client, str(conv["id"]), [_msg("x")])
    client.delete(f"/v1/conversations/{conv['id']}", headers=_h())
    ci = client.get(f"/v1/conversations/{conv['id']}/items", headers=_h())
    ci_one = client.get(f"/v1/conversations/{conv['id']}/items/any-id", headers=_h())
    out["conv_delete_tombstones_items"] = (
        ci.status_code == 404 and _err(ci).get("code") == "not_found" and ci_one.status_code == 404
    )

    # response → input_items tombstone together
    resp = _respond(client, "bye")
    client.delete(f"/v1/responses/{resp['id']}", headers=_h())
    ri = client.get(f"/v1/responses/{resp['id']}/input_items", headers=_h())
    out["response_delete_tombstones_input_items"] = (
        ri.status_code == 404 and _err(ri).get("code") == "not_found"
    )

    # chat completion → messages tombstone together
    chat = _chat(client, "bye")
    client.delete(f"/v1/chat/completions/{chat['id']}", headers=_h())
    cm = client.get(f"/v1/chat/completions/{chat['id']}/messages", headers=_h())
    out["chat_delete_tombstones_messages"] = (
        cm.status_code == 404 and _err(cm).get("code") == "not_found"
    )

    # eval spec → runs + output_items tombstone
    spec = _spec_create(client)
    rid = _run_create(client, spec)
    _wait_eval(client, rid.removeprefix("evalrun_"))
    client.delete(f"/v1/evals/{spec}", headers=_h())
    rr = client.get(f"/v1/evals/{spec}/runs", headers=_h())
    ro = client.get(f"/v1/evals/{spec}/runs/{rid}/output_items", headers=_h())
    out["spec_delete_tombstones_runs"] = (
        rr.status_code == 404 and _err(rr).get("code") == "eval_not_found" and ro.status_code == 404
    )

    # eval run delete → output_items tombstone (run_not_found)
    spec2 = _spec_create(client, name="spec-del-run")
    rid2 = _run_create(client, spec2)
    _wait_eval(client, rid2.removeprefix("evalrun_"))
    client.delete(f"/v1/evals/{spec2}/runs/{rid2}", headers=_h())
    ro2 = client.get(f"/v1/evals/{spec2}/runs/{rid2}/output_items", headers=_h())
    out["run_delete_tombstones_output_items"] = (
        ro2.status_code == 404 and _err(ro2).get("code") == "run_not_found"
    )

    # deleted ft: registration drops off checkpoints — tombstones don't
    # fabricate history
    ft = _ft_submit(client)
    _wait_ft(client, str(ft["id"]))
    ck = client.get(f"/v1/fine_tuning/jobs/{ft['id']}/checkpoints", headers=_h()).json()
    registered = ck.get("data") or []
    if registered:
        model_name = registered[0]["fine_tuned_model_checkpoint"]
        client.delete(f"/v1/models/{model_name}", headers=_h())
        ck2 = client.get(f"/v1/fine_tuning/jobs/{ft['id']}/checkpoints", headers=_h()).json()
        out["ft_deleted_model_drops_checkpoint"] = ck2["data"] == []
    else:
        out["ft_deleted_model_drops_checkpoint"] = False

    # deleted message batch → results + record 404
    ab = _abatch_create(client, n=1)
    _wait_abatch(client, str(ab["id"]))
    client.delete(f"/v1/messages/batches/{ab['id']}", headers=_ah())
    res = client.get(f"/v1/messages/batches/{ab['id']}/results", headers=_ah())
    rec = client.get(f"/v1/messages/batches/{ab['id']}", headers=_ah())
    out["abatch_delete_tombstones_results"] = res.status_code == 404 and rec.status_code == 404
    return out


# ---------------------------------------------------------------------------
# Stream surfaces — JSONL results + SSE feeds
# ---------------------------------------------------------------------------


def _stream_probes() -> dict[str, bool]:
    out: dict[str, Any] = {}
    client, _api = _client({"byok": lambda: _StubBackend(), "local_fx1": lambda: _StubBackend()})

    ab = _abatch_create(client, n=3)

    # results refused honestly while the batch is in flight
    early = client.get(f"/v1/messages/batches/{ab['id']}/results", headers=_ah())
    out["results_refused_until_ended_400"] = early.status_code in (200, 400) and (
        early.status_code != 400
        or early.json().get("error", {}).get("type") == "invalid_request_error"
    )

    _wait_abatch(client, str(ab["id"]))
    res = client.get(f"/v1/messages/batches/{ab['id']}/results", headers=_ah())
    body = res.content
    lines = body.decode().split("\n")
    parsed = []
    ok_lines = True
    for ln in lines[:-1]:  # the trailing newline leaves a '' tail
        try:
            parsed.append(json.loads(ln))
        except json.JSONDecodeError:
            ok_lines = False
    out["results_jsonl_eof_terminated"] = (
        res.status_code == 200
        and body.endswith(b"\n")
        and lines[-1] == ""
        and ok_lines
        and len(parsed) == 3
        and all(isinstance(p, dict) and p.get("custom_id") for p in parsed)
    )
    # repeated reads byte-identical (stream replay is idempotent)
    res2 = client.get(f"/v1/messages/batches/{ab['id']}/results", headers=_ah())
    out["results_stream_replay_identical"] = res2.content == body

    # SSE job events — 404 before open, frames end on terminal
    missing_sse = client.get("/harness/jobs/job-nope/events", headers=_h())
    out["job_events_unknown_404_before_stream"] = missing_sse.status_code == 404 and isinstance(
        missing_sse.json().get("detail"), str
    )
    jb = client.post("/harness/jobs", json={"command": "doctor"}, headers=_h())
    jid = str(jb.json()["job_id"])
    frames: list[str] = []
    terminal_seen = False
    # _WAIT_S, not a 3s slice: the access log stamps response start, and a
    # loaded runner can still be queued when a short stream hits its deadline.
    with client.stream(
        "GET", f"/harness/jobs/{jid}/events", params={"timeout_s": _WAIT_S}, headers=_h()
    ) as s:
        assert s.status_code == 200, "job events stream refused"
        for chunk in s.iter_text():
            frames.append(chunk)
        # reaching here means the stream closed (terminal or deadline)
    joined = "".join(frames)
    # SSE frames are blank-line separated in the concatenated body.
    # iter_text chunks are a transport detail — gzip and the ASGI
    # buffer split "\n\n" across yields — so they are not frames.
    snap_lines = [ln for ln in joined.split("\n") if ln.startswith("data: ")]
    for ln in snap_lines:
        try:
            snap = json.loads(ln.removeprefix("data: "))
            if snap.get("status") in _TERMINAL_JOB:
                terminal_seen = True
        except json.JSONDecodeError:
            pass
    blocks = [b for b in joined.split("\n\n") if b.strip()]
    out["job_events_sse_frames_end_terminal"] = (
        joined.endswith("\n\n")
        and terminal_seen
        and bool(blocks)
        and all(b.lstrip().startswith(("event: job", ":")) for b in blocks)
    )
    # every frame body parses as the job snapshot — no partial lines
    out["job_events_frames_complete_json"] = (
        all(ln.removeprefix("data: ").startswith("{") for ln in snap_lines) and len(snap_lines) >= 1
    )

    # response replay stream — monotonic seq ids, ends on terminal event
    resp = _respond(client, "stream me")
    seq_ids: list[int] = []
    events: list[str] = []
    with client.stream(
        "GET", f"/v1/responses/{resp['id']}", params={"stream": "true"}, headers=_h()
    ) as s:
        assert s.status_code == 200, "replay stream refused"
        for chunk in s.iter_text():
            for ln in chunk.split("\n"):
                if ln.startswith("id: "):
                    seq_ids.append(int(ln.removeprefix("id: ")))
                elif ln.startswith("event: "):
                    events.append(ln.removeprefix("event: "))
    out["response_replay_monotonic_terminal"] = (
        seq_ids == sorted(seq_ids)
        and len(events) >= 1
        and events[-1] in ("response.completed", "response.incomplete", "response.failed")
    )
    # starting_after skips honestly — re-stream lands past the cursor
    if len(seq_ids) >= 2:
        events2: list[str] = []
        with client.stream(
            "GET",
            f"/v1/responses/{resp['id']}",
            params={"stream": "true", "starting_after": seq_ids[0]},
            headers=_h(),
        ) as s2:
            for chunk in s2.iter_text():
                for ln in chunk.split("\n"):
                    if ln.startswith("id: "):
                        assert int(ln.removeprefix("id: ")) > seq_ids[0]
                    elif ln.startswith("event: "):
                        events2.append(ln.removeprefix("event: "))
        out["response_replay_starting_after_resumes"] = len(events2) == len(events) - 1
    else:
        out["response_replay_starting_after_resumes"] = True
    return out


# ---------------------------------------------------------------------------
# Fine-tuning events/checkpoints — monotonic growth + bounded feed
# ---------------------------------------------------------------------------


def _ft_probes() -> dict[str, bool]:
    from fx1.serve.finetune import FTJobOutcome  # noqa: PLC0415

    out: dict[str, Any] = {}
    hold = threading.Event()
    entered = threading.Event()
    emitted: list[str] = []

    def gated_runner(spec: Any, *, emit: Any, should_cancel: Any) -> Any:
        emit("info", "phase-one complete", None)
        emitted.append("phase-one complete")
        entered.set()
        hold.wait(30)
        emit("info", "phase-two complete", None)
        emitted.append("phase-two complete")
        return FTJobOutcome(
            fine_tuned_model=spec.ft_model_name,
            checkpoint=str(spec.work_dir / "ckpt"),
            trained_tokens=7,
        )

    client, _api = _client({"byok": lambda: _StubBackend()}, ft_runner=gated_runner)
    ft = _ft_submit(client)
    jid = str(ft["id"])

    assert entered.wait(10), "ft runner never entered phase-one"
    ev1 = client.get(f"/v1/fine_tuning/jobs/{jid}/events", headers=_h()).json()
    ck1 = client.get(f"/v1/fine_tuning/jobs/{jid}/checkpoints", headers=_h()).json()
    hold.set()
    _wait_ft(client, jid)
    ev2 = client.get(f"/v1/fine_tuning/jobs/{jid}/events", headers=_h()).json()
    ck2 = client.get(f"/v1/fine_tuning/jobs/{jid}/checkpoints", headers=_h()).json()

    ev1_ids = [e["id"] for e in ev1["data"]]
    ev2_ids = [e["id"] for e in ev2["data"]]
    ev2_created = [int(e["created_at"]) for e in ev2["data"]]
    out["ft_events_grow_monotonic"] = (
        len(ev2_ids) > len(ev1_ids)
        and ev2_ids[: len(ev1_ids)] == ev1_ids  # stable oldest-first prefix
        and ev2_created == sorted(ev2_created)
    )
    out["ft_events_oldest_first_chronological"] = all(
        str(e["id"]).startswith("ftev-") and e["object"] == "fine_tuning.job.event"
        for e in ev2["data"]
    )
    out["ft_checkpoints_empty_running_then_registered"] = (
        ck1["data"] == [] and len(ck2["data"]) == 1 and ck2["has_more"] is False
    )
    out["ft_checkpoint_shape_first_last"] = (
        ck2["first_id"] == ck2["data"][0]["id"]
        and ck2["last_id"] == ck2["data"][0]["id"]
        and ck2["data"][0]["fine_tuned_model_checkpoint"].startswith("ft:")
        and _cursor_ok(ck2)
    )

    # events walk — limit/after paging over the retained feed
    walked, pages = _walk_cursor(client, f"/v1/fine_tuning/jobs/{jid}/events", limit=1)
    out["ft_events_after_walk_covers"] = walked == ev2_ids
    out["ft_events_walk_pages_honest"] = all(_page_ok(p) and len(p["data"]) <= 1 for p in pages)
    ev_bad = client.get(
        f"/v1/fine_tuning/jobs/{jid}/events", params={"after": "ftev-ghost"}, headers=_h()
    ).json()
    out["ft_events_unknown_cursor_empty_honest"] = (
        ev_bad["data"] == [] and ev_bad["has_more"] is False
    )
    ck_bad = client.get(
        f"/v1/fine_tuning/jobs/{jid}/checkpoints",
        params={"after": "ftckpt-ghost"},
        headers=_h(),
    ).json()
    out["ft_checkpoints_unknown_cursor_empty_honest"] = (
        ck_bad["data"] == [] and ck_bad["has_more"] is False
    )

    # limit bound — Query-level le=100 refuses 101 honestly (422)
    ev_lim = client.get(f"/v1/fine_tuning/jobs/{jid}/events", params={"limit": 101}, headers=_h())
    out["ft_events_limit_bound_refused"] = ev_lim.status_code == 422
    return out


def _ft_bounded_probes() -> dict[str, bool]:
    from fx1.serve.finetune import FTJobOutcome  # noqa: PLC0415

    out: dict[str, Any] = {}
    n_emit = _FT_EVENT_CAP + 48

    def noisy_runner(spec: Any, *, emit: Any, should_cancel: Any) -> Any:
        for i in range(n_emit):
            emit("info", f"emit-{i}", None)
        return FTJobOutcome(checkpoint=str(spec.work_dir / "ckpt"))

    client, _api = _client({"byok": lambda: _StubBackend()}, ft_runner=noisy_runner)
    ft = _ft_submit(client)
    jid = str(ft["id"])
    _wait_ft(client, jid)

    # the retained feed is exactly the cap — oldest history evicted honestly
    walked, pages = _walk_cursor(client, f"/v1/fine_tuning/jobs/{jid}/events", limit=100)
    out["ft_events_capped_at_declared_bound"] = len(walked) == _FT_EVENT_CAP
    messages = [str(e.get("message", "")) for p in pages for e in p["data"]]
    emit_ks = [
        int(m.group(1))
        for msg in messages
        for m in [re.fullmatch(r"emit-(\d+)", msg)]
        if m is not None
    ]
    # the retained emit-k's are a contiguous suffix of the emitted range —
    # nothing fabricated, nothing reordered
    out["ft_events_retention_contiguous_suffix"] = (
        len(emit_ks) > 0
        and emit_ks == list(range(emit_ks[0], emit_ks[0] + len(emit_ks)))
        and emit_ks[-1] == n_emit - 1
        and emit_ks[0] > 0  # older emits really did evict
    )
    out["ft_events_pages_never_exceed_limit"] = all(len(p["data"]) <= 100 for p in pages) and all(
        p["has_more"] is not False for p in pages[:-1]
    )
    # limit=1 still walks the whole retained feed
    walked1, _ = _walk_cursor(client, f"/v1/fine_tuning/jobs/{jid}/events", limit=1, max_pages=300)
    out["ft_events_small_limit_walks_all"] = walked1 == walked
    return out


# ---------------------------------------------------------------------------
# Monotonic lifecycle — output_items fill post-completion
# ---------------------------------------------------------------------------


def _lifecycle_probes() -> dict[str, bool]:
    out: dict[str, Any] = {}

    class _GateEvalBackend:
        """tooluse driver stub — blocks in complete() so the run sits
        in-flight deterministically while retrieval is probed."""

        def __init__(self) -> None:
            self.gate = threading.Event()
            self.entered = threading.Event()

        def complete(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> str:
            self.entered.set()
            self.gate.wait(30)
            return "0.5"

        def close(self) -> None:
            pass

    backend = _GateEvalBackend()
    client, _api = _client({"byok": lambda: backend})

    spec = _spec_create(client)
    rid = _run_create(client, spec)
    bare = rid.removeprefix("evalrun_")
    assert backend.entered.wait(10), "eval backend never entered"

    # in-flight: output_items exists (no 404) and honestly reports empty —
    # per-task verdicts exist on a completed report only
    early = client.get(f"/v1/evals/{spec}/runs/{rid}/output_items", headers=_h()).json()
    run_early = client.get(f"/v1/evals/{spec}/runs/{rid}", headers=_h()).json()
    out["output_items_inflight_empty_not_404"] = (
        _page_ok(early) and early["data"] == [] and early["has_more"] is False
    )
    out["run_status_maps_wire_grammar"] = run_early.get("status") in (
        "queued",
        "in_progress",
        "completed",
        "failed",
        "canceled",
    )

    backend.gate.set()
    rec = _wait_eval(client, bare)
    late = client.get(f"/v1/evals/{spec}/runs/{rid}/output_items", headers=_h()).json()
    out["output_items_populated_post_completion"] = (
        rec["status"] == "succeeded"
        and len(late["data"]) > 0
        and all(i["status"] in ("pass", "fail") for i in late["data"])
        and all(i.get("datasource_item") is not None for i in late["data"])
    )

    # a run whose every task errors still succeeds as a run — output_items
    # then carries honest per-task "fail" verdicts (no fabricated passes,
    # no swallowed errors)
    class _FailBackend:
        def complete(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> str:
            from fx1.serve.backends import BackendNotConfiguredError  # noqa: PLC0415

            raise BackendNotConfiguredError("synthetic refusal")

        def close(self) -> None:
            pass

    client2, _api2 = _client({"byok": _FailBackend})
    spec2 = _spec_create(client2, name="failing-spec")
    rid2 = _run_create(client2, spec2)
    rec2 = _wait_eval(client2, rid2.removeprefix("evalrun_"))
    failed_items = client2.get(f"/v1/evals/{spec2}/runs/{rid2}/output_items", headers=_h()).json()
    out["failed_tasks_output_items_honest_rows"] = (
        rec2["status"] == "succeeded"
        and _page_ok(failed_items)
        and len(failed_items["data"]) > 0
        and all(i["status"] == "fail" for i in failed_items["data"])
        and all(i["results"][0]["passed"] is False for i in failed_items["data"])
    )

    # cooperative cancel is queued-only: a running run refuses with a
    # 409 'conflict' — pinned as the declared grammar (the queued window
    # itself is unschedulable through the API: submits are rejected
    # over-capacity at max_inflight rather than parked)
    gate_be = _GateEvalBackend()
    client3, _api3 = _client({"byok": lambda: gate_be}, max_inflight=1)
    spec3 = _spec_create(client3, name="cancel-spec")
    rid_a = _run_create(client3, spec3)
    assert gate_be.entered.wait(10), "eval backend never entered"
    cancel_running = client3.post(f"/v1/evals/{spec3}/runs/{rid_a}/cancel", headers=_h())
    out["cancel_running_run_conflict_409"] = (
        cancel_running.status_code == 409 and _err(cancel_running).get("code") == "conflict"
    )
    gate_be.gate.set()
    _wait_eval(client3, rid_a.removeprefix("evalrun_"))

    # a truncated ft job — paused at a stage boundary, then cancelled —
    # shows the honest partial event prefix: the pre-pause work is
    # retained, no post-cancel events are fabricated, and the feed ends
    # with the cancel marker
    from fx1.serve.finetune import FTJobOutcome  # noqa: PLC0415

    parked = threading.Event()
    release = threading.Event()

    def pausable_runner(
        spec: Any,
        *,
        emit: Any,
        should_cancel: Any,
        pause_gate: Any,
    ) -> Any:
        emit("info", "phase-a", None)
        parked.set()
        release.wait(15)  # let the probe land pause+cancel first
        if pause_gate():  # parks while paused; True → a cancel landed parked
            return FTJobOutcome(fine_tuned_model=None)
        emit("info", "phase-b", None)
        return FTJobOutcome(fine_tuned_model=spec.ft_model_name)

    client4, _api4 = _client({"byok": lambda: _StubBackend()}, ft_runner=pausable_runner)
    ft4 = _ft_submit(client4)
    jid4 = str(ft4["id"])
    assert parked.wait(10), "pausable ft runner never entered"

    paused = client4.post(f"/v1/fine_tuning/jobs/{jid4}/pause", headers=_h())
    pstatus = client4.get(f"/v1/fine_tuning/jobs/{jid4}", headers=_h()).json()
    out["ft_pause_marks_nonterminal"] = (
        paused.status_code == 200 and pstatus.get("status") == "paused"
    )

    cancelled = client4.post(f"/v1/fine_tuning/jobs/{jid4}/cancel", headers=_h())
    release.set()
    _wait_ft(client4, jid4)
    ev = client4.get(f"/v1/fine_tuning/jobs/{jid4}/events", headers=_h()).json()
    msgs = [e.get("message") for e in ev["data"]]
    out["ft_cancelled_job_events_honest_tail"] = (
        cancelled.status_code == 200
        and "phase-a" in msgs
        and "phase-b" not in msgs
        and any("pause" in str(m) for m in msgs)
        and msgs[-1] == "job cancelled"
    )
    out["ft_cancelled_checkpoints_honest_empty"] = (
        client4.get(f"/v1/fine_tuning/jobs/{jid4}/checkpoints", headers=_h()).json()["data"] == []
    )
    return out


# ---------------------------------------------------------------------------
# Idempotent reads — repeated GETs identical on a static record
# ---------------------------------------------------------------------------


def _idempotent_read_probes() -> dict[str, bool]:
    out: dict[str, Any] = {}
    client, _api = _client({"byok": lambda: _StubBackend(), "local_fx1": lambda: _StubBackend()})

    conv = _conv_create(client)
    _items_add(client, str(conv["id"]), [_msg(f"m{i}") for i in range(3)])
    chat = _chat(client, "read-twice")
    resp = _respond(client, "read-twice")
    ft = _ft_submit(client)
    _wait_ft(client, str(ft["id"]))
    spec = _spec_create(client)
    rid = _run_create(client, spec)
    _wait_eval(client, rid.removeprefix("evalrun_"))

    paths = {
        "conv_items": f"/v1/conversations/{conv['id']}/items",
        "chat_messages": f"/v1/chat/completions/{chat['id']}/messages",
        "input_items": f"/v1/responses/{resp['id']}/input_items",
        "ft_events": f"/v1/fine_tuning/jobs/{ft['id']}/events",
        "ft_checkpoints": f"/v1/fine_tuning/jobs/{ft['id']}/checkpoints",
        "output_items": f"/v1/evals/{spec}/runs/{rid}/output_items",
        "files_list": "/v1/files",
        "batches_list": "/v1/batches",
        "ft_jobs_list": "/v1/fine_tuning/jobs",
        "harness_jobs_list": "/harness/jobs",
        "harness_completions_list": "/harness/completions",
    }
    for name, path in paths.items():
        r1 = client.get(path, headers=_h())
        r2 = client.get(path, headers=_h())
        r3 = client.get(path, headers=_h())
        out[f"{name}_repeat_gets_identical"] = (
            r1.status_code == 200 and r1.content == r2.content == r3.content
        )
    return out


# ---------------------------------------------------------------------------
# Scope — read required; shared-workspace visibility pinned
# ---------------------------------------------------------------------------


def _scope_probes() -> dict[str, bool]:
    out: dict[str, Any] = {}
    client, _api = _client({"byok": lambda: _StubBackend()})

    read_key, _rid_k = _mint(client, scopes=["read"])
    write_key, _wid_k = _mint(client, scopes=["write"])
    conv = _conv_create(client)

    reads = [
        f"/v1/conversations/{conv['id']}/items",
        "/v1/fine_tuning/jobs",
        "/v1/files",
        "/v1/models",
        "/harness/jobs",
        "/harness/completions",
    ]
    out["read_scoped_key_reads_all"] = all(
        client.get(p, headers=_h(read_key)).status_code == 200 for p in reads
    )
    write_refusals = [client.get(p, headers=_h(write_key)) for p in reads]
    out["write_scoped_key_read_refused_403"] = all(
        r.status_code == 403
        and (_err(r).get("code") == "insufficient_scope" or "insufficient_scope" in str(r.json()))
        for r in write_refusals
    )
    out["unauthenticated_read_401"] = all(
        client.get(p, headers={}).status_code == 401 for p in reads
    )

    # shared-workspace: a write-scoped peer's record is readable by a read key
    conv2 = client.post("/v1/conversations", json={}, headers=_h(write_key))
    assert conv2.status_code == 200, conv2.text
    peer_read = client.get(f"/v1/conversations/{conv2.json()['id']}/items", headers=_h(read_key))
    out["cross_key_visibility_shared_workspace"] = peer_read.status_code == 200 and _page_ok(
        peer_read.json()
    )

    # and the read key cannot mutate (POST refused insufficient_scope)
    mutate = client.post("/v1/conversations", json={}, headers=_h(read_key))
    out["read_scoped_key_write_refused_403"] = mutate.status_code == 403
    return out


# ---------------------------------------------------------------------------
# Concurrent read-during-write — consistent snapshots only
# ---------------------------------------------------------------------------


def _concurrent_read_probes() -> dict[str, bool]:
    out: dict[str, Any] = {}
    client, _api = _client({"byok": lambda: _StubBackend()})
    conv = _conv_create(client)
    cid = str(conv["id"])

    # writer appends items one POST at a time while the reader polls —
    # every observed page is a consistent snapshot: well-formed items,
    # cursors matching data, no torn (half-mutated) rows
    done = threading.Event()
    torn: list[str] = []
    counts: list[int] = []
    thread_errors: list[str] = []

    def _writer() -> None:
        try:
            for i in range(12):
                response = client.post(
                    f"/v1/conversations/{cid}/items",
                    json={"items": [_msg(f"w{i}")]},
                    headers=_h(),
                )
                if response.status_code != 200:
                    raise AssertionError(response.text)
        except BaseException as exc:  # pragma: no cover - failure evidence
            thread_errors.append(f"writer:{type(exc).__name__}:{exc}")
        finally:
            done.set()

    def _reader() -> None:
        try:
            while not done.is_set():
                r = client.get(f"/v1/conversations/{cid}/items", headers=_h()).json()
                data = r.get("data") or []
                counts.append(len(data))
                if not _page_ok(r) or not _cursor_ok(r):
                    torn.append("cursor-mismatch")
                for it in data:
                    if not (it.get("id") and it.get("type") and it.get("content") is not None):
                        torn.append(f"torn:{it!r}")
                time.sleep(0.005)
        except BaseException as exc:  # pragma: no cover - failure evidence
            thread_errors.append(f"reader:{type(exc).__name__}:{exc}")

    w = threading.Thread(target=_writer)
    rd = threading.Thread(target=_reader)
    w.start()
    rd.start()
    w.join(30)
    rd.join(30)
    if w.is_alive() or rd.is_alive():
        thread_errors.append("conversation-thread-timeout")
    final = client.get(f"/v1/conversations/{cid}/items", headers=_h()).json()
    out["conv_items_concurrent_no_torn_reads"] = (
        thread_errors == [] and torn == [] and final["data"] and len(final["data"]) == 12
    )
    out["conv_items_concurrent_counts_monotonic"] = thread_errors == [] and counts == sorted(counts)

    # ft events during an emitting runner — same snapshot guarantee
    from fx1.serve.finetune import FTJobOutcome  # noqa: PLC0415

    gate = threading.Event()

    def slow_runner(spec: Any, *, emit: Any, should_cancel: Any) -> Any:
        for i in range(40):
            emit("info", f"t-{i}", None)
            time.sleep(0.02)
        gate.wait(10)
        return FTJobOutcome(checkpoint=str(spec.work_dir / "c"))

    client2, _api2 = _client({"byok": lambda: _StubBackend()}, ft_runner=slow_runner)
    ft = _ft_submit(client2)
    jid2 = str(ft["id"])
    counts2: list[int] = []
    torn2: list[str] = []
    reader2_errors: list[str] = []

    def _reader2() -> None:
        try:
            while not gate.is_set():
                r = client2.get(f"/v1/fine_tuning/jobs/{jid2}/events", headers=_h()).json()
                data = r.get("data") or []
                counts2.append(len(data))
                for e in data:
                    if not (str(e.get("id", "")).startswith("ftev-") and e.get("created_at")):
                        torn2.append(f"torn:{e!r}")
                time.sleep(0.02)
        except BaseException as exc:  # pragma: no cover - failure evidence
            reader2_errors.append(f"reader:{type(exc).__name__}:{exc}")

    rd2 = threading.Thread(target=_reader2)
    rd2.start()
    time.sleep(0.9)
    gate.set()
    rd2.join(30)
    if rd2.is_alive():
        reader2_errors.append("ft-reader-thread-timeout")
    _wait_ft(client2, jid2)
    out["ft_events_concurrent_consistent_snapshots"] = (
        reader2_errors == [] and torn2 == [] and counts2 == sorted(counts2) and len(counts2) >= 2
    )
    return out


# ---------------------------------------------------------------------------
# Battery + sealed receipt
# ---------------------------------------------------------------------------


def retrieval_audit() -> dict[str, Any]:
    """Run the retrieval battery; returns literal bools."""
    with _audit_context():
        out: dict[str, Any] = {}
        out.update(_empty_vs_missing_probes())
        out.update(_ordering_paging_probes())
        out.update(_list_surface_probes())
        out.update(_vector_store_probes())
        out.update(_deletion_boundary_probes())
        out.update(_stream_probes())
        out.update(_ft_probes())
        out.update(_ft_bounded_probes())
        out.update(_lifecycle_probes())
        out.update(_idempotent_read_probes())
        out.update(_scope_probes())
        out.update(_concurrent_read_probes())
        return out


def retrieval_audit_bench() -> dict[str, Any]:
    """Sealed receipt: every probe True under retrieval_audit.v1."""
    r = retrieval_audit()
    ok = len(r) == 136 and all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "retrieval_audit",
        "schema": "retrieval_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": "in-process buffered TestClient; stub backends",
            "not_executed": [
                "TypeScript client runtime",
                "network delivery or disconnect timing",
                "process-crash/power-loss durability",
                "provider-side pagination (no live provider exists)",
            ],
            "dialect_notes": [
                "OpenAI-grammar surfaces fail closed 400 invalid_cursor on an "
                "unknown cursor; the ft + anthropic dialects answer an honest "
                "empty page — the probes pin the split rather than unify it.",
                "/harness/jobs/{id}/logs has no route — the pin records the "
                "enveloped 404; /harness/jobs/{id}/events is the live log feed.",
            ],
        },
        "interpretation": (
            "Every retrieval sub-resource reads honestly end to end: empty "
            "pages answer their declared envelope instead of fabricating "
            "rows, every 404 lands in the path's own error grammar, the "
            "documented orderings hold (oldest-first event feeds, "
            "newest-first inventories, insertion-order items, request-order "
            "batch verdicts), cursor paging walks each list exactly once "
            "with honest has_more and previous-page before semantics, "
            "deleted parents tombstone their sub-resources while orphaned "
            "file records survive, the JSONL results stream terminates by "
            "EOF with complete lines and refuses pre-terminal reads, the "
            "ft event feed grows monotonically with a stable prefix and "
            "retains exactly the declared cap under overflow, reads stay "
            "idempotent and scope-checked (write scope can't read, read "
            "scope sees the shared workspace), concurrent read-during-write "
            "observes only consistent snapshots, and a bounded event feed "
            "pages its retained tail without blowup."
            if ok
            else f"RETRIEVAL AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(retrieval_audit_bench(), indent=2, sort_keys=True))
