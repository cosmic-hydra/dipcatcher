"""Line and branch edges for fx1.serve.vectorstores left open by the fx1 run."""

from __future__ import annotations

from typing import Any

import pytest

from fx1.serve.journal import ReplayResult
from fx1.serve.vectorstores import (
    VS_FILTER_LEAVES,
    VS_MAX_STORES_ID,
    VectorStoreError,
    VectorStoreStore,
    VSBatchRec,
    VSFileRec,
    VSMeta,
    _filters_match,
    validate_attributes,
    validate_chunking_strategy,
    validate_expires_after,
    validate_filters,
)


def _raises(status: int, code: str, fn: Any, *args: Any, **kwargs: Any) -> VectorStoreError:
    with pytest.raises(VectorStoreError) as exc:
        fn(*args, **kwargs)
    assert exc.value.status == status
    assert exc.value.code == code
    return exc.value


def test_attribute_filter_and_policy_validation() -> None:
    err = _raises(400, "invalid_request", validate_attributes, ["nope"])
    assert "object" in str(err)
    err = _raises(400, "invalid_request", validate_attributes, {"": "x"})
    assert "non-empty" in str(err)
    assert validate_attributes({"ok": True, "n": 1, "s": "t"})["ok"] is True

    err = _raises(400, "invalid_filters", validate_filters, "nope")
    assert "object" in str(err)
    err = _raises(400, "invalid_filters", validate_filters, {"type": "or", "filters": []})
    assert "non-empty" in str(err)
    leaves = [{"type": "eq", "key": "n", "value": i} for i in range(VS_FILTER_LEAVES + 1)]
    err = _raises(400, "invalid_filters", validate_filters, {"type": "and", "filters": leaves})
    assert "16" in str(err)
    err = _raises(400, "invalid_filters", validate_filters, {"type": "eq", "key": "", "value": 1})
    assert "filters.key" in str(err)
    err = _raises(
        400,
        "invalid_filters",
        validate_filters,
        {"type": "in", "key": "n", "value": []},
    )
    assert "non-empty" in str(err)
    err = _raises(
        400,
        "invalid_filters",
        validate_filters,
        {"type": "nin", "key": "n", "value": [1, {"nested": True}]},
    )
    assert "scalars" in str(err)
    err = _raises(
        400,
        "invalid_filters",
        validate_filters,
        {"type": "eq", "key": "n", "value": {"no": 1}},
    )
    assert "scalar" in str(err)

    err = _raises(400, "invalid_request", validate_chunking_strategy, "auto")
    assert "object" in str(err)
    err = _raises(400, "invalid_expires_after", validate_expires_after, "later")
    assert "object" in str(err)

    with pytest.raises(ValueError, match="idem_max"):
        VectorStoreStore(2, idem_max=0)


def test_filter_match_comparisons() -> None:
    attrs = {"n": 5, "tag": "alpha", "flag": True}
    assert _filters_match(
        {
            "type": "or",
            "filters": [
                {"type": "eq", "key": "tag", "value": "nope"},
                {"type": "eq", "key": "tag", "value": "alpha"},
            ],
        },
        attrs,
    )
    assert _filters_match({"type": "nin", "key": "tag", "value": ["zzz"]}, attrs)
    assert _filters_match({"type": "gt", "key": "flag", "value": 0}, attrs) is False
    assert _filters_match({"type": "gt", "key": "n", "value": True}, attrs) is False
    assert _filters_match({"type": "gt", "key": "n", "value": 3}, attrs) is True
    assert _filters_match({"type": "gte", "key": "n", "value": 5}, attrs) is True
    assert _filters_match({"type": "lt", "key": "n", "value": 3}, attrs) is False
    assert _filters_match({"type": "lte", "key": "n", "value": 5}, attrs) is True


class _Journal:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = payloads
        self.compacted: list[dict[str, Any]] | None = None

    def replay(self) -> ReplayResult:
        return ReplayResult(payloads=self.payloads)

    def compact(self, live: list[dict[str, Any]]) -> None:
        self.compacted = live

    def append(self, payload: dict[str, Any]) -> None:
        self.payloads.append(payload)


def _reader(file_id: str) -> tuple[bytes, str] | None:
    blobs = {
        "file-a": b"alpha beta alpha",
        "file-b": b"alpha beta beta",
        "file-empty": b"",
    }
    if file_id not in blobs:
        return None
    return blobs[file_id], f"{file_id}.txt"


def test_replay_orphan_update_and_idem_eviction() -> None:
    known = VSMeta(vs_id="vs_known", name="old", created_at=1).model_dump(
        mode="json", exclude={"files", "usage_bytes"}
    )
    orphan = VSFileRec(
        file_id="file-orphan",
        vector_store_id="vs_missing",
        created_at=1,
        status="completed",
        usage_bytes=1,
        filename="o.txt",
    )
    batch = VSBatchRec(
        batch_id="vsfb_x",
        vector_store_id="vs_missing",
        created_at=1,
        status="completed",
        file_ids=["file-a"],
        files=[],
        counts={"completed": 1, "failed": 0, "total": 1},
    )
    journal = _Journal(
        [
            {"vs": known, "vs_idem": {"key": "keep", "fp": "fp", "resp": {"ok": True}}},
            {"vs_update": {"vs_id": "vs_known", "name": "renamed", "metadata": {"k": "v"}}},
            {"vs_update": {"vs_id": "vs_absent", "name": "nope", "metadata": {}}},
            {"vs_file_delete": {"vs_id": "vs_absent", "file_id": "file-a"}},
            {"vs_batch": batch.model_dump(mode="json")},
            {"vs_file": orphan.model_dump(mode="json")},
            {"vs_touch": {"vs_id": "vs_absent", "last_active_at": 9}},
            {"vs_idem_evicted": ["keep"]},
        ]
    )
    store = VectorStoreStore(4, journal=journal, file_reader=_reader)
    assert store.get("vs_known")["name"] == "renamed"
    assert any("journaled without its store" in warning for warning in store.recover_warnings)
    assert "keep" not in store._idem
    assert store._batches == {}


def test_membership_idempotency_and_missing_content() -> None:
    store = VectorStoreStore(4, max_files=4, file_reader=_reader, idem_max=1)
    created = store.create(name="edges")
    vs_id = created["id"]
    attached = store.attach(vs_id, "file-a", _pin_store=False, attributes={"n": 5, "tag": "alpha"})
    assert attached["id"] == "file-a"
    assert store._inflight.get(vs_id, 0) == 0
    store.attach(vs_id, "file-b", attributes={"n": 1, "tag": "beta", "flag": True})

    with store._condition:
        store._pin_locked(vs_id)
        store._pin_locked(vs_id)
        store._unpin_locked(vs_id)
        assert store._inflight[vs_id] == 1
        store._unpin_locked(vs_id)
        assert vs_id not in store._inflight

    store._detach(vs_id, "file-missing")
    store._detach("vs_nope", "file-a")
    bare = VectorStoreStore(1)
    bare._compact_locked()

    meta = store._stores[vs_id]
    empty = VSFileRec(
        file_id="file-empty",
        vector_store_id=vs_id,
        created_at=1,
        status="completed",
        usage_bytes=0,
        filename="e.txt",
    )
    store._index_file(meta, empty, b"")
    assert empty.status == "failed"
    assert empty.last_error is not None and empty.last_error["code"] == "empty_file"

    meta.expires_at = 1
    err = _raises(
        410,
        "vector_store_expired",
        store._prepare_file_locked,
        meta,
        "file-new",
        (b"hello there", "n.txt"),
        attributes={},
        chunking_strategy={"type": "auto"},
    )
    assert vs_id in str(err)
    meta.expires_at = None

    err = _raises(404, "file_not_found", store.file_content, vs_id, "file-missing")
    assert "not attached" in str(err)
    err = _raises(404, "vector_store_not_found", store._store, "")
    assert "not found" in str(err)
    err = _raises(404, "vector_store_not_found", store._store, "x" * (VS_MAX_STORES_ID + 1))
    assert "not found" in str(err)

    with pytest.raises(ValueError, match="fingerprint"):
        store._idem_transition_locked("idem-1", None, {"ok": True})
    store._idem["idem-old"] = ("fp-old", {"ok": False})
    err = _raises(
        409,
        "idempotency_conflict",
        store._idem_transition_locked,
        "idem-old",
        "fp-new",
        {"ok": True},
    )
    assert "already committed" in str(err)
    payload, nxt = store._idem_transition_locked("idem-new", "fp-new", {"ok": True})
    assert payload["vs_idem_evicted"] == ["idem-old"]
    assert nxt is not None and "idem-new" in nxt

    hits = store.search(
        [vs_id],
        "alpha",
        filters={"type": "or", "filters": [{"type": "eq", "key": "tag", "value": "alpha"}]},
    )
    assert {hit["file_id"] for hit in hits} == {"file-a"}
    numeric = store.search([vs_id], "alpha", filters={"type": "lte", "key": "n", "value": 1})
    assert {hit["file_id"] for hit in numeric} == {"file-b"}
    assert store.search([vs_id], "alpha", filters={"type": "gt", "key": "tag", "value": 1}) == []
