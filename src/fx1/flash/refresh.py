"""Flash-context refresh: re-check an entry's sources instead of guessing.

Refreshing never rewrites the finding itself — it re-fetches each recorded
source and records what happened (reachable / unreachable / changed status),
stamping ``refresh_attempted_at`` and a plain-language ``refresh_note``.
``refreshed_at`` advances only when every recorded source is reachable. When
a source is gone, the existing evidence clock is preserved so retrieval
demotes stale findings; the text keeps its provenance history. A ``fetch_fn``
injection point keeps this testable offline (the default lazily imports the
web-research fetcher).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from fx1.flash.store import FlashEntry, FlashStore

FetchFn = Callable[[str], object]  # url -> object with .status / .error


@dataclass
class RefreshReport:
    """What re-checking an entry's sources found."""

    entry_id: str
    checked: list[str] = field(default_factory=list)
    reachable: list[str] = field(default_factory=list)
    unreachable: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def all_reachable(self) -> bool:
        return bool(self.checked) and not self.unreachable


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _probe_source(fetch_fn: FetchFn, url: str) -> str | None:
    """Return ``None`` when reachable, else a short unreachable reason."""
    try:
        result = fetch_fn(url)
        status = getattr(result, "status", None)
        error = getattr(result, "error", None)
        if status is not None and 200 <= status < 400 and not error:
            return None
        return str(error or f"HTTP {status}")
    except Exception as exc:  # noqa: BLE001 — network faults must not kill refresh
        return f"{type(exc).__name__}: {exc}"


def _refresh_note(report: RefreshReport) -> str:
    if not report.checked:
        return "no sources recorded; nothing to refresh"
    ratio = f"{len(report.reachable)}/{len(report.checked)} sources reachable"
    if report.all_reachable:
        return f"{ratio}; finding refreshed without change"
    return f"{ratio}; unreachable: {', '.join(report.unreachable)}"


def refresh_entry(
    store: FlashStore,
    entry: FlashEntry,
    *,
    fetch_fn: FetchFn | None = None,
) -> RefreshReport:
    """Re-fetch *entry*'s sources and stamp the refresh outcome.

    Sources without a URL are skipped. A failed fetch (network or policy) is
    reported as unreachable-with-reason, never silently dropped. With no
    sources at all the entry is left untouched and the report says so.
    """
    if fetch_fn is None:
        from fx1.webresearch.fetch import fetch_page

        def fetch_fn(url: str) -> object:
            return fetch_page(url)

    report = RefreshReport(entry_id=entry.id)
    for source in entry.sources:
        url = source.url
        if not url:
            continue
        report.checked.append(url)
        reason = _probe_source(fetch_fn, url)
        if reason is None:
            report.reachable.append(url)
        else:
            report.unreachable.append(f"{url} ({reason})")

    report.note = _refresh_note(report)
    if not report.checked:
        return report
    attempted_at = _now_iso()
    patch = {
        "refresh_attempted_at": attempted_at,
        "refresh_note": report.note,
    }
    if report.all_reachable:
        patch["refreshed_at"] = attempted_at
    store.update(entry.id, patch)
    return report


def refresh(
    store: FlashStore, entry_id: str, *, fetch_fn: FetchFn | None = None
) -> RefreshReport | None:
    """Refresh by id; ``None`` when the id does not exist."""
    entry = store.get(entry_id)
    if entry is None:
        return None
    return refresh_entry(store, entry, fetch_fn=fetch_fn)
