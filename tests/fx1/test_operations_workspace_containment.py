"""SYNTHETIC adversarial tests for operations workspace containment.

``OperationContext`` is the chokepoint every file-reading capability routes
through: path spelling is checked without touching the filesystem
(``_relative_data_path``), POSIX opens traverse each component by descriptor
with ``O_NOFOLLOW`` (``_open_posix_data_file``), and ``WorkspaceReader``
bounds bytes and hashes them only after EOF. These probes attack each layer
directly — traversal spellings, link substitution at every depth, FIFO and
directory substitution, byte-budget growth, and digest-before-EOF misuse —
instead of relying on the readers' happy paths to reach them.

All results are correctness fixtures over synthetic files, never market
evidence.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from fx1.operations.base import (
    InputModel,
    Operation,
    OperationContext,
    OutputModel,
    WorkspaceReader,
    _open_posix_data_file,
    _relative_data_path,
    _windows_opened_path,
    canonical_json,
)

POSIX = os.name == "posix"


@pytest.fixture
def ctx(tmp_path: Path) -> OperationContext:
    return OperationContext(workspace_root=tmp_path)


class _OkInput(InputModel):
    text: str


class _OkOutput(OutputModel):
    echoed: str


def _ok_handler(request: _OkInput, context: OperationContext) -> _OkOutput:
    return _OkOutput(echoed=request.text)


# ----------------------- path spelling (_relative_data_path) -----------------


@pytest.mark.parametrize(
    "spelling",
    [
        "",
        "   ",
        "/abs/passwd",
        "../outside.csv",
        "a/../../b.csv",
        "a/./../b.csv",
        "dir/../x.csv",
    ],
)
def test_relative_path_rejects_escape_spellings(spelling: str) -> None:
    with pytest.raises(ValueError, match="relative|traverse"):
        _relative_data_path(spelling, (".csv",))


def test_relative_path_rejects_wrong_suffix() -> None:
    with pytest.raises(ValueError, match="extension"):
        _relative_data_path("data.parquet", (".csv", ".tsv"))


def test_relative_path_accepts_case_folded_suffix() -> None:
    path = _relative_data_path("DATA.CSV", (".csv",))
    assert path == Path("DATA.CSV")


# ------------------------------ resolve_file ---------------------------------


def test_resolve_file_accepts_ancestor_symlink(tmp_path: Path) -> None:
    """``/tmp`` and ``/var`` are symlinks on macOS; files under them stay inside."""
    real_parent = tmp_path / "private"
    real_parent.mkdir()
    workspace = real_parent / "work"
    workspace.mkdir()
    (workspace / "inside.csv").write_text("ok", encoding="utf-8")
    link_parent = tmp_path / "var"
    link_parent.symlink_to(real_parent, target_is_directory=True)
    ctx = OperationContext(workspace_root=link_parent / "work")
    found = ctx.resolve_file("inside.csv", suffixes=(".csv",))
    assert found == (workspace / "inside.csv").resolve()
    assert found.read_text(encoding="utf-8") == "ok"


def test_resolve_file_rejects_symlink_escaping_workspace(
    ctx: OperationContext, tmp_path: Path
) -> None:
    outside = tmp_path.parent / "outside_secret.csv"
    outside.write_text("secret", encoding="utf-8")
    try:
        (tmp_path / "link.csv").symlink_to(outside)
        with pytest.raises(ValueError, match="inside the workspace"):
            ctx.resolve_file("link.csv", suffixes=(".csv",))
    finally:
        outside.unlink(missing_ok=True)


def test_resolve_file_rejects_directory_and_missing(ctx: OperationContext, tmp_path: Path) -> None:
    (tmp_path / "sub.csv").mkdir()
    with pytest.raises(ValueError, match="regular file"):
        ctx.resolve_file("sub.csv", suffixes=(".csv",))
    with pytest.raises(OSError):
        ctx.resolve_file("gone.csv", suffixes=(".csv",))


@pytest.mark.skipif(not POSIX, reason="POSIX descriptor traversal only")
def test_resolve_file_rejects_nested_symlink_dir(ctx: OperationContext, tmp_path: Path) -> None:
    outside_dir = tmp_path.parent / "outside_dir"
    outside_dir.mkdir(exist_ok=True)
    (outside_dir / "loot.csv").write_text("x", encoding="utf-8")
    (tmp_path / "inner").symlink_to(outside_dir)
    try:
        with pytest.raises(ValueError):
            ctx.resolve_file("inner/loot.csv", suffixes=(".csv",))
    finally:
        (outside_dir / "loot.csv").unlink(missing_ok=True)
        outside_dir.rmdir()


# --------------------- descriptor traversal (_open_posix_data_file) ----------


@pytest.mark.skipif(not POSIX, reason="POSIX descriptor traversal only")
def test_posix_open_rejects_symlink_leaf(ctx: OperationContext, tmp_path: Path) -> None:
    target = tmp_path.parent / "real.csv"
    target.write_text("data", encoding="utf-8")
    (tmp_path / "alias.csv").symlink_to(target)
    try:
        with pytest.raises(OSError):
            _open_posix_data_file(ctx.workspace_root, Path("alias.csv"))
    finally:
        target.unlink(missing_ok=True)


@pytest.mark.skipif(not POSIX, reason="POSIX descriptor traversal only")
def test_posix_open_rejects_symlinked_directory_component(
    ctx: OperationContext, tmp_path: Path
) -> None:
    outside_dir = tmp_path.parent / "shadow"
    outside_dir.mkdir(exist_ok=True)
    (outside_dir / "loot.csv").write_text("x", encoding="utf-8")
    (tmp_path / "dir").symlink_to(outside_dir)
    try:
        with pytest.raises(OSError):
            _open_posix_data_file(ctx.workspace_root, Path("dir/loot.csv"))
    finally:
        (outside_dir / "loot.csv").unlink(missing_ok=True)
        outside_dir.rmdir()


@pytest.mark.skipif(not POSIX, reason="POSIX descriptor traversal only")
def test_posix_open_hardlink_inside_workspace_opens(ctx: OperationContext, tmp_path: Path) -> None:
    """Name-based containment: an in-workspace hard link is not a symlink, so
    O_NOFOLLOW traversal opens it. Documented — the spell cannot traverse."""
    source = tmp_path / "real.csv"
    source.write_bytes(b"data")
    linked = tmp_path / "hard.csv"
    os.link(source, linked)
    descriptor = _open_posix_data_file(ctx.workspace_root, Path("hard.csv"))
    try:
        assert os.read(descriptor, 4) == b"data"
    finally:
        os.close(descriptor)


# ------------------------------- open_binary ---------------------------------


@pytest.mark.skipif(not POSIX, reason="POSIX descriptor traversal only")
def test_open_binary_reads_bounded_and_hashes(ctx: OperationContext, tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_bytes(b"col\n1\n")
    with ctx.open_binary("a.csv", suffixes=(".csv",), max_bytes=100) as reader:
        content = reader.read()
        assert content == b"col\n1\n"
        assert reader.bytes_read == 6
        assert reader.reached_eof is True
        assert reader.source_sha256 == hashlib.sha256(b"col\n1\n").hexdigest()


@pytest.mark.skipif(not POSIX, reason="POSIX descriptor traversal only")
def test_open_binary_exact_budget_passes_over_by_one_fails(
    ctx: OperationContext, tmp_path: Path
) -> None:
    (tmp_path / "exact.csv").write_bytes(b"x" * 64)
    with ctx.open_binary("exact.csv", suffixes=(".csv",), max_bytes=64) as reader:
        assert reader.read() == b"x" * 64
    (tmp_path / "over.csv").write_bytes(b"x" * 65)
    with (
        pytest.raises(ValueError, match="exceeds"),
        ctx.open_binary("over.csv", suffixes=(".csv",), max_bytes=64),
    ):
        pass


@pytest.mark.skipif(not POSIX, reason="POSIX descriptor traversal only")
@pytest.mark.parametrize("bad", [True, -1, 1.5, "64"])
def test_open_binary_rejects_noninteger_or_negative_budget(
    ctx: OperationContext, tmp_path: Path, bad: object
) -> None:
    (tmp_path / "a.csv").write_bytes(b"1")
    with (
        pytest.raises(ValueError, match="nonnegative integer"),
        ctx.open_binary("a.csv", suffixes=(".csv",), max_bytes=bad),  # type: ignore[arg-type]
    ):
        pass


@pytest.mark.skipif(not POSIX, reason="POSIX descriptor traversal only")
def test_open_binary_rejects_fifo_without_blocking(ctx: OperationContext, tmp_path: Path) -> None:
    """O_NONBLOCK makes a substituted FIFO fail instead of hanging on open."""
    fifo = tmp_path / "pipe.csv"
    os.mkfifo(fifo)
    with pytest.raises((OSError, ValueError)), ctx.open_binary("pipe.csv", suffixes=(".csv",)):
        pass


@pytest.mark.skipif(not POSIX, reason="POSIX descriptor traversal only")
def test_open_binary_rejects_directory_leaf(ctx: OperationContext, tmp_path: Path) -> None:
    (tmp_path / "dir.csv").mkdir()
    with pytest.raises((OSError, ValueError)), ctx.open_binary("dir.csv", suffixes=(".csv",)):
        pass


# ------------------------------- WorkspaceReader -----------------------------


@pytest.mark.skipif(not POSIX, reason="descriptor-based reader")
def test_reader_refuses_digest_before_eof(tmp_path: Path) -> None:
    target = tmp_path / "big.csv"
    target.write_bytes(b"abcd")
    descriptor = os.open(target, os.O_RDONLY)
    try:
        reader = WorkspaceReader(descriptor, max_bytes=16)
        with pytest.raises(ValueError, match="entire file"):
            _ = reader.source_sha256
        reader.read(2)
        with pytest.raises(ValueError, match="entire file"):
            _ = reader.source_sha256
        reader.read()
        assert reader.source_sha256 == hashlib.sha256(b"abcd").hexdigest()
        reader.close()
    finally:
        if not reader.closed:
            os.close(descriptor)


@pytest.mark.skipif(not POSIX, reason="descriptor-based reader")
def test_reader_fails_on_growth_beyond_budget(tmp_path: Path) -> None:
    """A file larger than the declared budget trips readinto, not the fstat gate."""
    target = tmp_path / "grown.csv"
    target.write_bytes(b"x" * 128)
    descriptor = os.open(target, os.O_RDONLY)
    try:
        reader = WorkspaceReader(descriptor, max_bytes=64)
        with pytest.raises(ValueError, match="exceeds"):
            reader.read()
        reader.close()
    finally:
        if not reader.closed:
            os.close(descriptor)


@pytest.mark.skipif(not POSIX, reason="descriptor-based reader")
def test_reader_closed_read_fails_closed(tmp_path: Path) -> None:
    target = tmp_path / "a.csv"
    target.write_bytes(b"1")
    descriptor = os.open(target, os.O_RDONLY)
    reader = WorkspaceReader(descriptor, max_bytes=8)
    reader.close()
    with pytest.raises(ValueError, match="closed"):
        reader.read(1)


def test_read_bytes_collects_and_bounds(ctx: OperationContext, tmp_path: Path) -> None:
    (tmp_path / "e.csv").write_bytes(b"")
    assert ctx.read_bytes("e.csv", suffixes=(".csv",)) == b""
    (tmp_path / "b.csv").write_bytes(b"12")
    assert ctx.read_bytes("b.csv", suffixes=(".csv",), max_bytes=2) == b"12"
    with pytest.raises(ValueError, match="exceeds"):
        ctx.read_bytes("b.csv", suffixes=(".csv",), max_bytes=1)


# --------------------------- canonical_json budget ---------------------------


def test_canonical_json_enforces_byte_budget_during_encode() -> None:
    with pytest.raises(ValueError, match="budget"):
        canonical_json({"pad": "x" * 1000}, max_bytes=64)


def test_canonical_json_rejects_nonfinite() -> None:
    with pytest.raises((TypeError, ValueError)):
        canonical_json({"v": float("nan")})


def test_canonical_json_is_order_independent() -> None:
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


# ------------------------- Operation contract gates --------------------------


def _operation(**overrides: object) -> Operation:
    kwargs: dict[str, object] = {
        "id": "features.test_op",
        "kind": "feature",
        "description": "synthetic probe operation",
        "input_model": _OkInput,
        "output_model": _OkOutput,
        "handler": _ok_handler,
    }
    kwargs.update(overrides)
    return Operation(**kwargs)  # type: ignore[arg-type]


def test_operation_rejects_bad_id_spellings() -> None:
    for bad in ("Features.test_op", "features", "features.test-op", "x.test_op", "skills.test_op"):
        with pytest.raises(ValueError, match="namespace|identifier"):
            _operation(id=bad)


def test_operation_rejects_empty_description() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        _operation(description="   ")


def test_operation_rejects_model_handler_module_split() -> None:
    class _ForeignInput(InputModel):
        pass

    _ForeignInput.__module__ = "fx1.operations.simple_returns"
    with pytest.raises(ValueError, match="same file|capability file"):
        _operation(input_model=_ForeignInput)


def test_invoke_envelope_hashes_and_validates(ctx: OperationContext) -> None:
    op = _operation()
    result = op.invoke({"text": "hi"}, ctx)
    assert result["schema"] == "fx1.operation-result/v1"
    assert result["result"] == {"echoed": "hi"}
    assert result["market_evidence"] is False
    assert len(result["input_sha256"]) == 64 and len(result["output_sha256"]) == 64


def test_invoke_rejects_undeclared_and_nonfinite_input(ctx: OperationContext) -> None:
    op = _operation()
    with pytest.raises(ValidationError):
        op.invoke({"text": "hi", "smuggled": True}, ctx)

    class _NumInput(InputModel):
        v: float

    def _num_handler(request: _NumInput, context: OperationContext) -> _OkOutput:
        return _OkOutput(echoed=str(request.v))

    num_op = _operation(input_model=_NumInput, handler=_num_handler)
    with pytest.raises(ValidationError):
        num_op.invoke({"v": float("inf")}, ctx)


def test_invoke_rejects_output_outside_declared_schema(ctx: OperationContext) -> None:
    def _bad_handler(request: _OkInput, context: OperationContext) -> _OkOutput:
        return {"echoed": request.text}  # type: ignore[return-value]

    op = _operation(handler=_bad_handler)
    with pytest.raises(TypeError, match="outside its declared output schema"):
        op.invoke({"text": "hi"}, ctx)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX box asserts the failure path")
def test_windows_handle_inspection_fails_closed_off_windows() -> None:
    with pytest.raises(OSError, match="unavailable|Windows"):
        _windows_opened_path(0)
