"""Execution contracts shared by independently implemented harness operations."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

OperationKind = Literal["feature", "skill", "plugin"]


class InputModel(BaseModel):
    """Reject undeclared input fields and nonfinite numeric values by default."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class OutputModel(BaseModel):
    """Operation results must have a declared, JSON-compatible schema."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def _relative_data_path(relative_path: str, suffixes: tuple[str, ...]) -> Path:
    """Validate the supplied spelling without following filesystem links."""
    path = Path(relative_path)
    if not relative_path.strip() or not path.parts or path.anchor or ".." in path.parts:
        raise ValueError("path must be relative to the workspace and cannot traverse parents")
    if os.name == "nt" and any(
        PureWindowsPath(part).is_reserved()
        or part.endswith((".", " "))
        or any(character in '<>:"|?*' or ord(character) < 32 for character in part)
        for part in path.parts
    ):
        raise ValueError("path cannot contain Windows devices, streams, or ambiguous names")
    if path.suffix.lower() not in suffixes:
        raise ValueError(f"file extension must be one of {suffixes}")
    return path


def _open_posix_data_file(root: Path, relative: Path) -> int:
    """Traverse directories by descriptor and reject links in every component."""
    if sys.platform == "win32":
        raise OSError("POSIX descriptor traversal is unavailable on Windows")
    if os.open not in os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW"):
        raise OSError("this platform cannot safely open workspace files without following links")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    # Walk the frozen absolute root too: O_NOFOLLOW on root alone would leave
    # its ancestors exposed to replacement between canonicalization and open.
    directory_fd = os.open(root.anchor, directory_flags)
    try:
        for component in (*root.parts[1:], *relative.parts[:-1]):
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        # O_NONBLOCK prevents a substituted FIFO from blocking before fstat can
        # reject it. It does not change ordinary regular-file reads.
        return os.open(
            relative.name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_fd,
        )
    finally:
        os.close(directory_fd)


def _windows_path(value: str) -> PureWindowsPath:
    """Normalize extended DOS/UNC prefixes without consulting the filesystem."""
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return PureWindowsPath(value)


def _windows_opened_path(descriptor: int) -> PureWindowsPath:
    """Obtain the normalized final path of the actual handle, before any read."""
    if sys.platform != "win32":
        raise OSError("Windows file-handle inspection is unavailable on this platform")
    import ctypes
    import msvcrt
    from ctypes import wintypes

    # These imports and APIs are reached only on Windows. A missing API fails
    # closed; no string-only fallback substitutes for inspecting this handle.
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_final_path = kernel32.GetFinalPathNameByHandleW
    get_final_path.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
    get_final_path.restype = wintypes.DWORD
    handle = wintypes.HANDLE(msvcrt.get_osfhandle(descriptor))
    capacity = 1024
    while capacity <= 65_536:
        buffer = ctypes.create_unicode_buffer(capacity)
        size = get_final_path(handle, buffer, capacity, 0)
        if size == 0:
            raise ctypes.WinError(ctypes.get_last_error())
        if size >= capacity:
            capacity = size + 1
            continue
        return _windows_path(buffer.value)
    raise OSError("opened file path exceeds the supported Windows path length")


class WorkspaceReader(io.RawIOBase):
    """Nonseekable reader that bounds and hashes bytes from one checked descriptor."""

    def __init__(self, descriptor: int, max_bytes: int) -> None:
        super().__init__()
        self._descriptor = descriptor
        self._max_bytes = max_bytes
        self._digest = hashlib.sha256()
        self.bytes_read = 0
        self.reached_eof = False

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        """Read no more than one block and fail on growth beyond the byte budget."""
        if self.closed:
            raise ValueError("read of closed workspace file")
        view = memoryview(buffer).cast("B")
        if not view:
            return 0
        size = min(len(view), 65_536, self._max_bytes - self.bytes_read + 1)
        if size <= 0:
            raise ValueError(f"input file exceeds {self._max_bytes} bytes")
        chunk = os.read(self._descriptor, size)
        if not chunk:
            self.reached_eof = True
            return 0
        self.bytes_read += len(chunk)
        if self.bytes_read > self._max_bytes:
            raise ValueError(f"input file exceeds {self._max_bytes} bytes")
        self._digest.update(chunk)
        view[: len(chunk)] = chunk
        return len(chunk)

    @property
    def source_sha256(self) -> str:
        """A whole-read digest is available only after observing end of file."""
        if not self.reached_eof:
            raise ValueError("source digest requires reading the entire file")
        return self._digest.hexdigest()

    def close(self) -> None:
        if not self.closed:
            try:
                os.close(self._descriptor)
            finally:
                super().close()


def _workspace_realpath(root: Path) -> Path:
    """Collapse ancestor symlinks; do not follow the final path component.

    macOS maps ``/tmp`` to ``/private/tmp`` and ``/var`` to ``/private/var``.
    ``Path.resolve`` follows those ancestors, while ``os.path.abspath`` does
    not, so a containment check of a resolved file against an abspath root
    rejects every file in a pytest workspace. Keeping the final component
    lexical still rejects a workspace directory that was swapped for a symlink.
    """
    absolute = Path(os.path.abspath(root))
    parent = os.path.realpath(absolute.parent)
    return Path(parent) / absolute.name


@dataclass(frozen=True)
class OperationContext:
    """Authority supplied by the host, never by model-generated arguments."""

    workspace_root: Path

    def __post_init__(self) -> None:
        # Pin relative host roots without following a newly substituted link.
        # Harness already resolves its host root when it is constructed.
        object.__setattr__(self, "workspace_root", Path(os.path.abspath(self.workspace_root)))

    def resolve_file(self, relative_path: str, *, suffixes: tuple[str, ...]) -> Path:
        """Inspect a file path; use read_bytes for containment enforced at open."""
        path = _relative_data_path(relative_path, suffixes)
        root = self.workspace_root
        resolved = (root / path).resolve(strict=True)
        # ``root`` is abspath, not realpath, so a workspace under /tmp or
        # /var keeps that spelling while the file realpath is /private/tmp
        # or /private/var on macOS. Compare against the parent realpath plus
        # the lexical final component: ancestor symlinks collapse, and
        # replacing the workspace directory itself with a symlink still
        # fails closed.
        contained = _workspace_realpath(root)
        if not resolved.is_relative_to(contained) or not resolved.is_file():
            raise ValueError("path must identify a regular file inside the workspace")
        if resolved.suffix.lower() not in suffixes:
            raise ValueError(f"file extension must be one of {suffixes}")
        return resolved

    @contextmanager
    def open_binary(
        self,
        relative_path: str,
        *,
        suffixes: tuple[str, ...],
        max_bytes: int = 2_000_000,
    ) -> Iterator[WorkspaceReader]:
        """Open a bounded, hash-tracked stream from a contained regular file.

        POSIX rejects all symlinks through descriptor-relative traversal. Windows
        checks the final path of the opened handle, then reads that same handle.
        Unsupported filesystems or failed handle inspection fail closed.
        """
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError("max_bytes must be a nonnegative integer")
        relative = _relative_data_path(relative_path, suffixes)
        if os.name == "posix":
            descriptor = _open_posix_data_file(self.workspace_root, relative)
        elif sys.platform == "win32":
            descriptor = os.open(
                self.workspace_root / relative,
                os.O_RDONLY | os.O_BINARY | os.O_NOINHERIT,
            )
        else:
            raise OSError("contained workspace reads require POSIX or Windows handle support")
        reader: WorkspaceReader | None = None
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("path must identify a regular file inside the workspace")
            if os.name == "nt":
                opened_path = _windows_opened_path(descriptor)
                root = _windows_path(str(self.workspace_root))
                # Directory spelling is deliberately exact: Windows permits
                # case-sensitive directories, so casefold-based Path containment
                # could admit a distinct sibling. Harness supplies a canonical
                # host root; noncanonical spellings fail closed here.
                if (
                    not opened_path.is_absolute()
                    or opened_path.anchor.casefold() != root.anchor.casefold()
                    or len(opened_path.parts) <= len(root.parts)
                    or opened_path.parts[1 : len(root.parts)] != root.parts[1:]
                ):
                    raise ValueError("opened file escapes the host workspace")
                if opened_path.suffix.lower() not in suffixes:
                    raise ValueError(f"file extension must be one of {suffixes}")
            if metadata.st_size > max_bytes:
                raise ValueError(f"input file exceeds {max_bytes} bytes")
            reader = WorkspaceReader(descriptor, max_bytes)
            yield reader
        finally:
            if reader is None:
                os.close(descriptor)
            else:
                reader.close()

    def read_bytes(
        self,
        relative_path: str,
        *,
        suffixes: tuple[str, ...],
        max_bytes: int = 2_000_000,
    ) -> bytes:
        """Collect a bounded small file; streaming callers should use open_binary."""
        with self.open_binary(relative_path, suffixes=suffixes, max_bytes=max_bytes) as stream:
            content = stream.read()
            if content is None:
                raise OSError("workspace file read unexpectedly produced no bytes")
            return content


def canonical_json(value: object, *, max_bytes: int = 2_000_000) -> bytes:
    """Bound canonical JSON as it is encoded; never construct an oversized result."""
    encoder = json.JSONEncoder(
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    chunks: list[bytes] = []
    total = 0
    for fragment in encoder.iterencode(value):
        encoded = fragment.encode("utf-8")
        total += len(encoded)
        if total > max_bytes:
            raise ValueError(f"canonical JSON exceeds the {max_bytes}-byte budget")
        chunks.append(encoded)
    return b"".join(chunks)


@dataclass(frozen=True)
class Operation[InputT: InputModel, OutputT: OutputModel]:
    """One reviewed implementation file and its input/output types."""

    id: str
    kind: OperationKind
    description: str
    input_model: type[InputT]
    output_model: type[OutputT]
    handler: Callable[[InputT, OperationContext], OutputT]
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"(?:features|skills|plugins)\.[a-z][a-z0-9_]*", self.id):
            raise ValueError("operation id must be a namespaced, stable identifier")
        prefix = {"feature": "features", "skill": "skills", "plugin": "plugins"}[self.kind]
        if self.id.split(".", 1)[0] != prefix:
            raise ValueError("operation id namespace must agree with its kind")
        if not self.description.strip():
            raise ValueError("operation description must be nonempty")
        if self.input_model.__module__ != self.handler.__module__:
            raise ValueError("input schema and implementation must live in the capability file")
        if self.output_model.__module__ != self.handler.__module__:
            raise ValueError("output schema and implementation must live in the capability file")

    def describe(self) -> dict[str, Any]:
        """Schemas for a host to validate and present this operation to an AI."""
        return {
            "id": self.id,
            "kind": self.kind,
            "description": self.description,
            "version": self.version,
            "module": self.handler.__module__,
            "input_schema": self.input_model.model_json_schema(),
            "output_schema": self.output_model.model_json_schema(),
            "implementation": "independent",
            "market_evidence": False,
        }

    def invoke(self, arguments: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        """Validate before execution and return finite JSON with input/output hashes."""
        request = self.input_model.model_validate(arguments)
        input_bytes = canonical_json(request.model_dump(mode="json"))
        result = self.handler(request, context)
        if not isinstance(result, self.output_model):
            raise TypeError(f"{self.id} returned a value outside its declared output schema")
        output_value = result.model_dump(mode="json")
        output_bytes = canonical_json(output_value)
        return {
            "schema": "fx1.operation-result/v1",
            "operation_id": self.id,
            "version": self.version,
            "kind": self.kind,
            "result": output_value,
            "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
            "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
            "research_only": True,
            "live_pnl_claim": False,
            "market_evidence": False,
        }
