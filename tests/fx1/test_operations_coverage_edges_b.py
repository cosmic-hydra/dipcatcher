"""Coverage for operation IO, registry dispatch, and workspace containment.

Synthetic files only. Failures must be explicit errors; hashes and pages are exact.
"""

from __future__ import annotations

import ast
import hashlib
import io
import os
import stat
import sys
import types
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from fx1.operations import base, read_csv, read_jsonl, read_toml
from fx1.operations import inspect_numpy_array as npy
from fx1.operations import inspect_parquet as parquet
from fx1.operations import inspect_zip as zipmod
from fx1.operations.base import (
    InputModel,
    Operation,
    OperationContext,
    OutputModel,
    WorkspaceReader,
    canonical_json,
)
from fx1.operations.registry import (
    execute_operation,
    get_operation,
    invoke_operation_tool,
    list_operations,
    operation_tool_specs,
)


@pytest.fixture
def context(tmp_path: Path) -> OperationContext:
    return OperationContext(workspace_root=tmp_path)


def _write(root: Path, name: str, payload: bytes) -> None:
    (root / name).write_bytes(payload)


def _npy(header: bytes, payload: bytes = b"", version: tuple[int, int] = (1, 0)) -> bytes:
    width = 2 if version == (1, 0) else 4
    return b"\x93NUMPY" + bytes(version) + len(header).to_bytes(width, "little") + header + payload


class _ProbeIn(InputModel):
    text: str


class _ProbeOut(OutputModel):
    echoed: str


def _probe(request: _ProbeIn, context: OperationContext) -> _ProbeOut:
    return _ProbeOut(echoed=request.text)


def _operation(**overrides: object) -> Operation[Any, Any]:
    kwargs: dict[str, object] = {
        "id": "features.probe_op",
        "kind": "feature",
        "description": "synthetic probe",
        "input_model": _ProbeIn,
        "output_model": _ProbeOut,
        "handler": _probe,
    }
    kwargs.update(overrides)
    return Operation(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# base.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spelling",
    ["", "   ", "/abs/a.csv", "../a.csv", "a/../../b.csv", "dir/../x.csv"],
)
def test_relative_paths_reject_escapes(spelling: str) -> None:
    with pytest.raises(ValueError, match="relative|traverse"):
        base._relative_data_path(spelling, (".csv",))


def test_relative_path_suffix_and_windows_names(monkeypatch: pytest.MonkeyPatch) -> None:
    assert base._relative_data_path("DATA.CSV", (".csv",)) == Path("DATA.CSV")
    with pytest.raises(ValueError, match="extension"):
        base._relative_data_path("a.parquet", (".csv",))
    monkeypatch.setattr(base.os, "name", "nt")
    assert base._relative_data_path("ok.csv", (".csv",)) == Path("ok.csv")
    for spelling in ("CON.csv", "foo.", "foo ", "a<b.csv", "a\x01b.csv"):
        with pytest.raises(ValueError, match="Windows"):
            base._relative_data_path(spelling, (".csv", ".txt", ""))


def test_windows_path_normalizes_prefixes() -> None:
    unc = base._windows_path("\\\\?\\UNC\\server\\share\\a.csv")
    extended = base._windows_path("\\\\?\\C:\\data\\a.csv")
    plain = base._windows_path("C:\\data\\a.csv")
    assert unc.parts[0].startswith("\\\\server\\")
    assert extended.drive == "C:"
    assert plain == extended
    with pytest.raises(OSError, match="unavailable"):
        base._windows_opened_path(0)


def test_posix_open_refuses_windows_and_weak_platforms(
    context: OperationContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(base.sys, "platform", "win32")
    with pytest.raises(OSError, match="POSIX descriptor"):
        base._open_posix_data_file(context.workspace_root, Path("a.csv"))
    monkeypatch.setattr(base.sys, "platform", "linux")
    monkeypatch.setattr(base.os, "supports_dir_fd", set())
    with pytest.raises(OSError, match="safely open"):
        base._open_posix_data_file(context.workspace_root, Path("a.csv"))


def test_posix_open_refuses_missing_nofollow(
    context: OperationContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr(base.os, "O_NOFOLLOW")
    with pytest.raises(OSError, match="safely open"):
        base._open_posix_data_file(context.workspace_root, Path("a.csv"))


def test_open_binary_reads_nested_files_and_rejects_links(
    context: OperationContext, tmp_path: Path
) -> None:
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "a.csv").write_bytes(b"col\n1\n")
    with context.open_binary("sub/a.csv", suffixes=(".csv",), max_bytes=100) as reader:
        assert reader.readable() is True
        assert reader.read() == b"col\n1\n"
        digest = reader.source_sha256
    assert digest == hashlib.sha256(b"col\n1\n").hexdigest()
    assert context.read_bytes("sub/a.csv", suffixes=(".csv",)) == b"col\n1\n"
    (tmp_path / "link.csv").symlink_to(nested / "a.csv")
    with pytest.raises(OSError), context.open_binary("link.csv", suffixes=(".csv",)):
        pass
    (tmp_path / "dir.csv").mkdir()
    with (
        pytest.raises(ValueError, match="regular file"),
        context.open_binary("dir.csv", suffixes=(".csv",)),
    ):
        pass
    (tmp_path / "big.csv").write_bytes(b"abcdef")
    with (
        pytest.raises(ValueError, match="exceeds"),
        context.open_binary("big.csv", suffixes=(".csv",), max_bytes=2),
    ):
        pass
    for bad in (True, -1, "3"):
        with (
            pytest.raises(ValueError, match="max_bytes"),
            context.open_binary("sub/a.csv", suffixes=(".csv",), max_bytes=bad),
        ):
            pass
    with pytest.raises(ValueError, match="regular file"):
        context.resolve_file("dir.csv", suffixes=(".csv",))
    (tmp_path / "real.csv").write_bytes(b"ok")
    assert context.resolve_file("real.csv", suffixes=(".csv",)).name == "real.csv"
    with pytest.raises(FileNotFoundError):
        context.resolve_file("gone.csv", suffixes=(".csv",))
    with pytest.raises(ValueError, match="extension"):
        context.resolve_file("real.csv", suffixes=(".txt",))


def test_workspace_reader_bounds_digest_and_closed_reads(tmp_path: Path) -> None:
    path = tmp_path / "blob.csv"
    path.write_bytes(b"abcdef")
    descriptor = os.open(path, os.O_RDONLY)
    reader = WorkspaceReader(descriptor, max_bytes=100)
    assert reader.readinto(bytearray()) == 0
    with pytest.raises(ValueError, match="entire file"):
        _ = reader.source_sha256
    chunk = bytearray(3)
    assert reader.readinto(chunk) == 3
    assert bytes(chunk) == b"abc"
    reader.bytes_read = reader._max_bytes + 5
    with pytest.raises(ValueError, match="exceeds"):
        reader.readinto(bytearray(8))
    reader.close()
    with pytest.raises(ValueError, match="closed"):
        reader.readinto(bytearray(4))
    reader.close()
    grown = os.open(path, os.O_RDONLY)
    bounded = WorkspaceReader(grown, max_bytes=2)
    with pytest.raises(ValueError, match="exceeds"):
        bounded.readinto(bytearray(8))
    bounded.close()


def test_read_bytes_rejects_empty_reader(
    context: OperationContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    @contextmanager
    def _empty(*_args: object, **_kwargs: object) -> Any:
        class _Stream:
            def read(self) -> None:
                return None

        yield _Stream()

    monkeypatch.setattr(OperationContext, "open_binary", _empty)
    with pytest.raises(OSError, match="no bytes"):
        context.read_bytes("a.csv", suffixes=(".csv",))


def test_unsupported_platform_open_fails_closed(
    context: OperationContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.csv").write_bytes(b"x")
    monkeypatch.setattr(base.os, "name", "nt")
    monkeypatch.setattr(base.sys, "platform", "linux")
    with (
        pytest.raises(OSError, match="POSIX or Windows"),
        context.open_binary("a.csv", suffixes=(".csv",)),
    ):
        pass


def test_windows_handle_inspection_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ctypes

    payload = tmp_path / "a.csv"
    payload.write_bytes(b"abc")
    monkeypatch.setattr(base.os, "name", "nt")
    monkeypatch.setattr(base.sys, "platform", "win32")
    monkeypatch.setattr(base.os, "O_BINARY", 0, raising=False)
    monkeypatch.setattr(base.os, "O_NOINHERIT", 0, raising=False)
    responses: list[tuple[int, str | None]] = []

    def _final(_handle: object, buffer: Any, _capacity: int, _flags: int) -> int:
        size, text = responses.pop(0)
        if text is not None:
            buffer.value = text
        return size

    kernel = MagicMock()
    kernel.GetFinalPathNameByHandleW.side_effect = _final
    monkeypatch.setattr(ctypes, "WinDLL", lambda *_args, **_kwargs: kernel, raising=False)
    monkeypatch.setattr(
        ctypes,
        "WinError",
        lambda code=0, descr=None: OSError(code, descr or "winerror"),
        raising=False,
    )
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 1, raising=False)
    monkeypatch.setitem(sys.modules, "msvcrt", MagicMock(get_osfhandle=MagicMock(return_value=1)))

    class _Root:
        def __str__(self) -> str:
            return "C:\\data\\ws"

        def __truediv__(self, _other: object) -> Path:
            return payload

    context = OperationContext(workspace_root=tmp_path)
    object.__setattr__(context, "workspace_root", _Root())

    responses.append((0, None))
    with pytest.raises(OSError), context.open_binary("a.csv", suffixes=(".csv",)):
        pass
    responses.extend([(2048, None), (12, "\\\\?\\C:\\data\\ws\\a.csv")])
    with context.open_binary("a.csv", suffixes=(".csv",), max_bytes=10) as reader:
        assert reader.read() == b"abc"
    responses.extend([(2048, None), (12, "\\\\?\\c:\\data\\ws\\a.csv")])
    with context.open_binary("a.csv", suffixes=(".csv",)) as reader:
        assert reader.read() == b"abc"
    responses.append((70_000, None))
    with (
        pytest.raises(OSError, match="path length"),
        context.open_binary("a.csv", suffixes=(".csv",)),
    ):
        pass
    escapes = ["relative.csv", "D:\\data\\ws\\a.csv", "C:\\data\\ws", "C:\\data\\other\\a.csv"]
    for text in escapes:
        responses.append((12, text))
        with (
            pytest.raises(ValueError, match="escapes"),
            context.open_binary("a.csv", suffixes=(".csv",)),
        ):
            pass
    responses.append((12, "C:\\data\\ws\\a.exe"))
    with (
        pytest.raises(ValueError, match="extension"),
        context.open_binary("a.csv", suffixes=(".csv",)),
    ):
        pass
    responses.append((12, "C:\\data\\ws\\a.csv"))
    with (
        pytest.raises(ValueError, match="exceeds"),
        context.open_binary("a.csv", suffixes=(".csv",), max_bytes=0),
    ):
        pass


def test_canonical_json_and_operation_contract(context: OperationContext) -> None:
    assert canonical_json({"b": 1, "a": 2}) == b'{"a":2,"b":1}'
    with pytest.raises(ValueError, match="exceeds"):
        canonical_json({"k": "x" * 40}, max_bytes=8)
    with pytest.raises(ValueError, match="identifier"):
        _operation(id="features.Bad")
    with pytest.raises(ValueError, match="namespace"):
        _operation(id="skills.probe_op", kind="feature")
    with pytest.raises(ValueError, match="nonempty"):
        _operation(description="  ")
    foreign = type("Foreign", (InputModel,), {})
    foreign.__module__ = "fx1.operations.simple_returns"
    with pytest.raises(ValueError, match="capability file"):
        _operation(input_model=foreign)
    foreign_out = type("ForeignOut", (OutputModel,), {})
    foreign_out.__module__ = "fx1.operations.simple_returns"
    with pytest.raises(ValueError, match="capability file"):
        _operation(output_model=foreign_out)
    operation = _operation()
    described = operation.describe()
    assert described["id"] == "features.probe_op"
    assert described["market_evidence"] is False
    invoked = operation.invoke({"text": "hi"}, context)
    assert invoked["result"] == {"echoed": "hi"}
    assert len(invoked["input_sha256"]) == 64

    def _bad(request: _ProbeIn, _context: OperationContext) -> _ProbeOut:
        return {"echoed": request.text}  # type: ignore[return-value]

    with pytest.raises(TypeError, match="output schema"):
        _operation(handler=_bad).invoke({"text": "hi"}, context)


# ---------------------------------------------------------------------------
# registry.py
# ---------------------------------------------------------------------------


def test_registry_rejects_bad_exports_and_dispatches_tools(
    context: OperationContext, tmp_path: Path
) -> None:
    with pytest.raises(KeyError, match="unregistered"):
        get_operation("features.not_registered")
    module = types.ModuleType("fx1.operations.simple_returns")
    module.OPERATION = object()  # type: ignore[attr-defined]
    get_operation.cache_clear()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("fx1.operations.registry.import_module", lambda _name: module)
        with pytest.raises(TypeError, match="OPERATION"):
            get_operation("features.simple_returns")
        from fx1.operations import rolling_rank

        module.OPERATION = rolling_rank.OPERATION  # type: ignore[attr-defined]
        with pytest.raises(ValueError, match="identity"):
            get_operation("features.simple_returns")

        def _handler(request: _ProbeIn, _context: OperationContext) -> _ProbeOut:
            return _ProbeOut(echoed=request.text)

        _ProbeIn.__module__ = _ProbeOut.__module__ = _handler.__module__ = (
            "fx1.operations.elsewhere"
        )
        module.OPERATION = Operation(  # type: ignore[attr-defined]
            id="features.simple_returns",
            kind="feature",
            description="mismatched module",
            input_model=_ProbeIn,
            output_model=_ProbeOut,
            handler=_handler,
        )
        with pytest.raises(ValueError, match="identity"):
            get_operation("features.simple_returns")
    get_operation.cache_clear()
    assert get_operation("features.rolling_rank").id == "features.rolling_rank"
    with pytest.raises(ValueError, match="offset|limit"):
        list_operations(offset=-1)
    with pytest.raises(ValueError, match="limit"):
        list_operations(limit=0)
    with pytest.raises(ValueError, match="kind"):
        list_operations(kind="other")  # type: ignore[arg-type]
    page = list_operations(kind="plugin", query="Read", limit=1)
    assert page["has_more"] is True
    assert page["results"][0]["kind"] == "plugin"
    assert list_operations(query="not-a-real-capability-token")["matching_count"] == 0
    names = {item["function"]["name"] for item in operation_tool_specs()}
    assert names == {"list_operations", "describe_operation", "execute_operation"}
    listed = invoke_operation_tool(
        "list_operations", {"query": "rank", "kind": "feature", "limit": 5}
    )
    assert listed["matching_count"] >= 1
    described = invoke_operation_tool(
        "describe_operation", {"operation_id": "features.rolling_rank"}
    )
    assert "input_schema" in described
    executed = invoke_operation_tool(
        "execute_operation",
        {
            "operation_id": "features.rolling_rank",
            "arguments": {"values": [1.0, 2.0, 3.0], "window": 3},
        },
        workspace_root=tmp_path,
    )
    assert executed["result"]["percentiles"][-1] == 1.0
    direct = execute_operation(
        "skills.audit_bar_integrity",
        {"bars": [{"open": 1, "high": 1, "low": 1, "close": 1, "volume": 0}]},
        workspace_root=tmp_path,
    )
    assert direct["result"]["passed"] is True
    with pytest.raises(KeyError, match="unknown operation tool"):
        invoke_operation_tool("nope", {})
    with pytest.raises(ValidationError):
        invoke_operation_tool("list_operations", {"limit": 0})
    # restore probe models so later collection in this process stays local
    _ProbeIn.__module__ = _ProbeOut.__module__ = __name__


# ---------------------------------------------------------------------------
# read_csv / read_jsonl / read_toml
# ---------------------------------------------------------------------------


def test_csv_pages_strings_and_rejects_hostile_input(
    context: OperationContext, tmp_path: Path
) -> None:
    _write(tmp_path, "ok.csv", b"a,b\n\n1,2\n\n3,4\n")
    page = read_csv.execute(read_csv.Input(path="ok.csv", offset=1, limit=1), context)
    assert page.columns == ["a", "b"]
    assert page.rows == [{"a": "3", "b": "4"}]
    assert page.total_rows == 2
    assert page.blank_records_skipped == 2
    assert page.has_more is False
    assert page.source_sha256 == hashlib.sha256(b"a,b\n\n1,2\n\n3,4\n").hexdigest()
    _write(tmp_path, "wide.tsv", b"a;b\n1;2\n")
    parsed = read_csv.execute(read_csv.Input(path="wide.tsv", delimiter=";"), context)
    assert parsed.rows == [{"a": "1", "b": "2"}]
    for delimiter in ("\n", "\r", "\0", '"'):
        with pytest.raises(ValidationError, match="delimiter"):
            read_csv.Input(path="ok.csv", delimiter=delimiter)
    _write(tmp_path, "long.csv", b"a,b\n" + b"x" * 1_000_001 + b",2\n")
    with pytest.raises(ValueError, match="physical lines"):
        read_csv.execute(read_csv.Input(path="long.csv"), context)
    _write(tmp_path, "nul.csv", b"a,b\n1,\0\n")
    with pytest.raises(ValueError, match="NUL"):
        read_csv.execute(read_csv.Input(path="nul.csv"), context)
    for name, blob, pattern in (
        ("empty.csv", b"", "nonempty header"),
        ("blank.csv", b"a,,b\n", "blank names"),
        ("dup.csv", b"a,a\n1,2\n", "duplicate"),
        ("longname.csv", b"a," + b"n" * 257 + b"\n1,2\n", "256"),
        ("ragged.csv", b"a,b\n1\n", "cells"),
        ("quote.csv", b'a,b\n"unterminated,2\n', "invalid CSV"),
        ("bad.csv", b"\xff\n", "UTF-8"),
    ):
        _write(tmp_path, name, blob)
        with pytest.raises(ValueError, match=pattern):
            read_csv.execute(read_csv.Input(path=name), context)
    header = ",".join(f"c{index}" for index in range(257)) + "\n"
    _write(tmp_path, "cols.csv", header.encode())
    with pytest.raises(ValueError, match="256 columns"):
        read_csv.execute(read_csv.Input(path="cols.csv"), context)
    _write(tmp_path, "many.csv", b"a\n" + b"1\n" * 100_001)
    with pytest.raises(ValueError, match="100000"):
        read_csv.execute(read_csv.Input(path="many.csv"), context)
    _write(tmp_path, "page.csv", b"a\n" + (b"x" * 100_000 + b"\n") * 11)
    with pytest.raises(ValueError, match="1000000 cell"):
        read_csv.execute(read_csv.Input(path="page.csv", limit=20), context)
    with pytest.raises(ValueError, match="extension"):
        read_csv.execute(read_csv.Input(path="ok.txt"), context)


def test_jsonl_validates_objects_and_pages(context: OperationContext, tmp_path: Path) -> None:
    payload = b'{"a": 1.5, "b": [true, {"c": "x"}]}\n\n{"b": 2}\n{"a": 3}\n'
    _write(tmp_path, "ok.jsonl", payload)
    page = read_jsonl.execute(read_jsonl.Input(path="ok.jsonl", offset=1, limit=1), context)
    assert page.total_rows == 3
    assert page.blank_lines_skipped == 1
    assert page.rows == [{"b": 2}]
    assert page.physical_line_numbers == [3]
    assert page.columns == ["a", "b"]
    assert page.has_more is True
    assert page.source_sha256 == hashlib.sha256(payload).hexdigest()
    cases = {
        "dup.jsonl": (b'{"a": 1, "a": 2}\n', "duplicate"),
        "nan.jsonl": (b'{"a": NaN}\n', "nonfinite"),
        "inf.jsonl": (b'{"a": 1e309}\n', "finite float"),
        "list.jsonl": (b"[1, 2]\n", "JSON object"),
        "deep.jsonl": (('{"a":' * 70 + "1" + "}" * 70 + "\n").encode(), "nesting"),
        "surrogate.jsonl": (b'{"a": "\\ud800"}\n', "invalid JSON"),
        "long.jsonl": (b'{"a":"' + b"x" * 1_000_001 + b'"}\n', "physical lines"),
        "bad.jsonl": (b"\xff\n", "UTF-8"),
    }
    for name, (blob, pattern) in cases.items():
        _write(tmp_path, name, blob)
        with pytest.raises(ValueError, match=pattern):
            read_jsonl.execute(read_jsonl.Input(path=name), context)
    wide = {f"k{index}": 1 for index in range(1001)}
    _write(tmp_path, "wide.jsonl", (str(wide).replace("'", '"') + "\n").encode())
    with pytest.raises(ValueError, match="1000 distinct"):
        read_jsonl.execute(read_jsonl.Input(path="wide.jsonl"), context)
    line = '{"a":"' + ("x" * 100_000) + '"}\n'
    _write(tmp_path, "page.jsonl", (line * 11).encode())
    with pytest.raises(ValueError, match="1000000 source"):
        read_jsonl.execute(read_jsonl.Input(path="page.jsonl", limit=20), context)
    _write(tmp_path, "many.jsonl", b'{"a":1}\n' * 100_001)
    with pytest.raises(ValueError, match="100000 records"):
        read_jsonl.execute(read_jsonl.Input(path="many.jsonl"), context)
    with pytest.raises(ValueError, match="extension"):
        read_jsonl.execute(read_jsonl.Input(path="ok.txt"), context)


def test_toml_types_paths_and_defensive_values(context: OperationContext, tmp_path: Path) -> None:
    _write(
        tmp_path,
        "ok.toml",
        b"""
outside = 1979-05-27
[inner]
offset = 1979-05-27T07:32:00Z
local_dt = 1979-05-27T07:32:00
local_d = 1980-01-02
local_t = 07:32:00
flag = true
name = "a"
number = 1.5
nums = [1, 2]
""",
    )
    selected = read_toml.execute(read_toml.Input(path="ok.toml", table_path=["inner"]), context)
    assert selected.data["number"] == 1.5
    assert selected.data["flag"] is True
    assert selected.data["nums"] == [1, 2]
    kinds = {item.kind for item in selected.temporal_values}
    assert kinds == {"offset_datetime", "local_datetime", "local_date", "local_time"}
    assert all(item.path[:1] == ["inner"] for item in selected.temporal_values)
    assert selected.source_sha256 == hashlib.sha256((tmp_path / "ok.toml").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="does not resolve"):
        read_toml.execute(read_toml.Input(path="ok.toml", table_path=["missing"]), context)
    with pytest.raises(ValueError, match="TOML table"):
        read_toml.execute(read_toml.Input(path="ok.toml", table_path=["inner", "nums"]), context)
    _write(tmp_path, "nf.toml", b"x = inf\n")
    with pytest.raises(ValueError, match="nonfinite"):
        read_toml.execute(read_toml.Input(path="nf.toml"), context)
    _write(tmp_path, "bad.toml", b"x = \n")
    with pytest.raises(ValueError, match="invalid UTF-8 TOML"):
        read_toml.execute(read_toml.Input(path="bad.toml"), context)
    _write(tmp_path, "bin.toml", b"\xff\n")
    with pytest.raises(ValueError, match="invalid UTF-8 TOML"):
        read_toml.execute(read_toml.Input(path="bin.toml"), context)
    _write(tmp_path, "key.toml", ('"' + "k" * 257 + '" = 1\n').encode())
    with pytest.raises(ValueError, match="256"):
        read_toml.execute(read_toml.Input(path="key.toml"), context)
    _write(tmp_path, "deep.toml", ("x = " + "{a=" * 66 + "1" + "}" * 66 + "\n").encode())
    with pytest.raises(ValueError, match="20000|nesting"):
        read_toml.execute(read_toml.Input(path="deep.toml"), context)
    _write(tmp_path, "nodes.toml", b"x = [" + b"1," * 20_001 + b"1]\n")
    with pytest.raises(ValueError, match="20000"):
        read_toml.execute(read_toml.Input(path="nodes.toml"), context)
    times = "[t]\n" + "\n".join(f"k{index} = 1980-01-01" for index in range(201))
    _write(tmp_path, "times.toml", times.encode())
    with pytest.raises(ValueError, match="200 temporal"):
        read_toml.execute(read_toml.Input(path="times.toml", table_path=["t"]), context)

    def _loads(value: object) -> Any:
        return lambda _text: value

    monkey = pytest.MonkeyPatch()
    _write(tmp_path, "odd.toml", b"a = 1\n")
    monkey.setattr(read_toml.tomllib, "loads", _loads({1: "v"}))
    with pytest.raises(ValueError, match="keys must be strings"):
        read_toml.execute(read_toml.Input(path="odd.toml"), context)
    monkey.setattr(read_toml.tomllib, "loads", _loads({"k": object()}))
    with pytest.raises(ValueError, match="unsupported TOML"):
        read_toml.execute(read_toml.Input(path="odd.toml"), context)
    monkey.undo()


# ---------------------------------------------------------------------------
# inspect_zip / inspect_numpy_array / inspect_parquet
# ---------------------------------------------------------------------------


def _zip(members: list[tuple[str, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members:
            info = zipfile.ZipInfo(name, date_time=(2024, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED if data == b"" else zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return buffer.getvalue()


def test_zip_metadata_flags_and_limits(context: OperationContext, tmp_path: Path) -> None:
    payload = _zip(
        [
            ("", b"x"),
            ("../evil.txt", b"x"),
            ("/abs/file.txt", b"x"),
            ("aux.txt", b"x"),
            ("clean.txt", b"xy"),
            ("empty.txt", b""),
        ]
    )
    _write(tmp_path, "h.zip", payload)
    result = zipmod.execute(zipmod.Input(path="h.zip", limit=2), context)
    assert result.entry_count == 6
    assert result.has_more is True
    assert result.members[0].name == ""
    assert "empty_name" in result.path_issue_counts
    assert result.path_issue_counts["parent_traversal"] == 1
    full = zipmod.execute(zipmod.Input(path="h.zip", limit=20), context)
    empty = next(item for item in full.members if item.name == "empty.txt")
    assert empty.expansion_ratio is None
    assert full.source_sha256 == hashlib.sha256(payload).hexdigest()
    link = io.BytesIO()
    with zipfile.ZipFile(link, "w") as archive:
        info = zipfile.ZipInfo("link.txt", date_time=(2024, 1, 1, 0, 0, 0))
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, b"target")
        archive.writestr("dup.txt", b"1")
        archive.writestr("dup.txt", b"2")
        archive.writestr(zipfile.ZipInfo("dir/"), b"")
    _write(tmp_path, "flags.zip", link.getvalue())
    flags = zipmod.execute(zipmod.Input(path="flags.zip", limit=10), context)
    by_name = {item.name: item for item in flags.members}
    assert by_name["link.txt"].symlink is True
    assert by_name["dup.txt"].duplicate_name is True
    assert by_name["dir/"].directory is True
    assert flags.symlink_count == 1
    assert flags.duplicate_name_count == 1
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for index in range(10_001):
            archive.writestr(f"m{index}.txt", b"x")
    _write(tmp_path, "many.zip", buffer.getvalue())
    with pytest.raises(ValueError, match="10000"):
        zipmod.execute(zipmod.Input(path="many.zip"), context)
    _write(tmp_path, "long.zip", _zip([("n" * 4097, b"x")]))
    with pytest.raises(ValueError, match="4096"):
        zipmod.execute(zipmod.Input(path="long.zip"), context)
    _write(tmp_path, "bad.zip", b"not a zip")
    with pytest.raises(ValueError, match="ZIP"):
        zipmod.execute(zipmod.Input(path="bad.zip"), context)
    direct = zipmod._path_issues("")
    assert "empty_name" in direct
    assert "control_character" in zipmod._path_issues("a\x01b")
    assert "windows_ambiguous_name" in zipmod._path_issues("aux.txt")
    assert "absolute_or_drive_path" in zipmod._path_issues("C:/x.txt")


def test_npy_headers_versions_and_caps(context: OperationContext, tmp_path: Path) -> None:
    buffer = io.BytesIO()
    np.save(buffer, np.arange(6, dtype=">f8").reshape(2, 3), allow_pickle=False)
    _write(tmp_path, "ok.npy", buffer.getvalue())
    result = npy.execute(npy.Input(path="ok.npy"), context)
    assert result.shape == [2, 3]
    assert result.element_count == 6
    assert result.dtype == ">f8"
    assert result.source_bytes == len(buffer.getvalue())
    structured = io.BytesIO()
    np.save(structured, np.zeros(2, dtype=[("x", "<i2"), ("y", "<f4")]))
    _write(tmp_path, "struct.npy", structured.getvalue())
    fields = npy.execute(npy.Input(path="struct.npy"), context)
    assert [item.name for item in fields.fields] == ["x", "y"]
    header = b"{'descr': '<f8', 'fortran_order': True, 'shape': (1,)}\n"
    _write(tmp_path, "fort.npy", _npy(header, b"\x00" * 8))
    assert npy.execute(npy.Input(path="fort.npy"), context).fortran_order is True
    version2 = io.BytesIO()
    np.lib.format.write_array(version2, np.arange(2), version=(2, 0))
    _write(tmp_path, "v2.npy", version2.getvalue())
    assert npy.execute(npy.Input(path="v2.npy"), context).format_version == "2.0"
    header_v3 = b"{'descr': '<f8', 'fortran_order': False, 'shape': (1,)}"
    padding = b" " * (16 - ((10 + len(header_v3) + 1) % 16))
    _write(tmp_path, "v3.npy", _npy(header_v3 + padding + b"\n", b"\x00" * 8, version=(3, 0)))
    assert npy.execute(npy.Input(path="v3.npy"), context).format_version == "3.0"
    shape = "(" + ",".join(["0"] * 20_001) + ",)"
    bomb = f"{{'descr': '|u1', 'fortran_order': False, 'shape': {shape}}}\n".encode()
    assert len(bomb) <= 65_536
    assert sum(1 for _ in ast.walk(ast.parse(bomb.decode().strip(), mode="eval"))) > 20_000
    bomb = _npy(bomb)
    wide = np.zeros(1, dtype=[(f"f{index}", "u1") for index in range(257)])
    wide_buf = io.BytesIO()
    np.save(wide_buf, wide)
    hostile = [
        (b"", "not an NPY"),
        (b"\x93NUMPY\x04\x00\x00\x00", "unsupported"),
        (b"\x93NUMPY\x02\x00\x01\x00\x00", "truncated NPY header length"),
        (_npy(b"{'descr': '<f8'", b""), "newline"),
        (
            b"\x93NUMPY\x01\x00"
            + (80).to_bytes(2, "little")
            + b"{'descr': '<f8', 'fortran_order': False, 'shape': (1,)}\n",
            "truncated NPY header",
        ),
        (bomb, "20000"),
        (wide_buf.getvalue(), "256 fields"),
        (_npy(b"{'descr': '|u1', 'fortran_order': False, 'shape': (2000000000,)}\n"), "billion"),
        (_npy(b"{'descr': '<f8', 'fortran_order': False, 'shape': (1,)}\n", b"x" * 16), "match"),
        (_npy(b"{'descr': '|O8', 'fortran_order': False, 'shape': (1,)}\n", b"x" * 8), "pickle"),
        (_npy(b"", version=(2, 0)), "between 1 and 65536"),
        (_npy(b"{'descr': '<f8', 'fortran_order': False}\n"), "exactly"),
        (_npy(b"{'a': 1, 'b': 2, 'c': 3}\n"), "exactly"),
        (_npy(b"[1]\n"), "dictionary"),
        (_npy(b"{'descr': '<f8', 'fortran_order': False, 'shape': [1]}\n"), "tuple"),
        (_npy(b"{'descr': '<f8', 'fortran_order': 1, 'shape': (1,)}\n"), "boolean"),
        (_npy(b"{'descr': 1, 'fortran_order': False, 'shape': (1,)}\n"), "string or"),
        (_npy(b"{'descr': 'nope', 'fortran_order': False, 'shape': (1,)}\n"), "unsupported"),
    ]
    for blob, pattern in hostile:
        _write(tmp_path, "bad.npy", blob)
        with pytest.raises(ValueError, match=pattern):
            npy.execute(npy.Input(path="bad.npy"), context)


def test_parquet_schema_pages_and_caps(context: OperationContext, tmp_path: Path) -> None:
    table = pa.table({"a": [1, 2, 3], "s": ["x", "y", "z"]})
    pq.write_table(table, tmp_path / "ok.parquet", row_group_size=1)
    result = parquet.execute(parquet.Input(path="ok.parquet", row_group_limit=1), context)
    assert result.total_rows == 3
    assert result.row_group_count == 3
    assert result.has_more_row_groups is True
    assert result.row_groups[0].rows == 1
    assert (
        result.source_sha256 == hashlib.sha256((tmp_path / "ok.parquet").read_bytes()).hexdigest()
    )
    past = parquet.execute(parquet.Input(path="ok.parquet", row_group_offset=9), context)
    assert past.row_groups == []
    assert past.has_more_row_groups is False
    wide = pa.table({f"c{index}": pa.array([1], type=pa.int8()) for index in range(257)})
    pq.write_table(wide, tmp_path / "wide.parquet")
    with pytest.raises(ValueError, match="256 top-level"):
        parquet.execute(parquet.Input(path="wide.parquet"), context)
    fields = [pa.field(f"c{index}", pa.int8()) for index in range(1025)]
    nested = pa.table({"s": pa.array([{}], type=pa.struct(fields))})
    pq.write_table(nested, tmp_path / "leaves.parquet")
    with pytest.raises(ValueError, match="1024 leaf"):
        parquet.execute(parquet.Input(path="leaves.parquet"), context)
    groups = io.BytesIO()
    writer = pq.ParquetWriter(groups, pa.schema([("a", pa.int8())]))
    one = pa.table({"a": pa.array([1], type=pa.int8())})
    for _index in range(10_001):
        writer.write_table(one)
    writer.close()
    _write(tmp_path, "groups.parquet", groups.getvalue())
    with pytest.raises(ValueError, match="10000 row groups"):
        parquet.execute(parquet.Input(path="groups.parquet"), context)
    _write(tmp_path, "bad.parquet", b"not parquet")
    with pytest.raises(ValueError, match="invalid or unsupported Parquet"):
        parquet.execute(parquet.Input(path="bad.parquet"), context)
