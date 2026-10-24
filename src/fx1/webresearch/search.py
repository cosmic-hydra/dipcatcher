"""Web search — keyless DuckDuckGo (HTML endpoints) with injectable seams.

``DuckDuckGoLiteSearcher`` queries the no-JavaScript DDG endpoints and parses
the result list from HTML with the stdlib parser. There is no API key and no
third-party dependency; failures degrade to an empty result list plus a
reported error, never fabricated hits. ``NullSearcher`` keeps the engine
runnable offline (tests, air-gapped labs).

Searches are polite: a small inter-search pause (budget-controlled) and a
descriptive user agent. Search queries are built by callers from the *user's
goal text only* — never from private flash context (that is an approval-gated
decision enforced one layer up).
"""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

from fx1.webresearch.fetch import DEFAULT_USER_AGENT

DDG_LITE_URL = "https://lite.duckduckgo.com/lite/"
DDG_HTML_URL = "https://html.duckduckgo.com/html/"
_DDG_SEARCH_HOSTS = frozenset({"lite.duckduckgo.com", "html.duckduckgo.com"})
DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MAX_RESULTS = 8


@dataclass
class SearchHit:
    """One search result: a URL, title, snippet, and which engine found it."""

    url: str
    title: str = ""
    snippet: str = ""
    engine: str = ""


@dataclass
class SearchResult:
    query: str
    hits: list[SearchHit] = field(default_factory=list)
    error: str = ""
    elapsed_s: float = 0.0


class _DDGResultParser(HTMLParser):
    """Extracts result-link anchors and their sibling snippets from DDG HTML.

    DDG emits the snippet ``<td class='result-snippet'>`` *after* the result
    anchor closes, so a hit stays pending across its ``</a>`` and is flushed
    when the next result link starts (or at ``close()``).
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hits: list[SearchHit] = []
        self._current: SearchHit | None = None
        self._in_link = False
        self._in_snippet = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        if (
            tag == "a"
            and attr_map.get("rel") == "nofollow"
            and attr_map.get("class") in {"result-link", "result__a"}
        ):
            # A new result link closes the previous one: the snippet cell
            # follows the anchor, so flush only when the next result starts.
            self._flush()
            self._current = SearchHit(url=attr_map.get("href") or "", title="")
            self._in_link = True
        elif (
            tag in {"td", "span"}
            and self._current is not None
            and attr_map.get("class") in {"result-snippet", "result__snippet"}
        ):
            self._in_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_link:
            self._in_link = False  # keep _current: the snippet cell follows
        elif self._in_snippet and tag in {"td", "span"}:
            self._in_snippet = False

    def handle_data(self, data: str) -> None:
        if self._current is None:
            return
        if self._in_link:
            self._current.title += data
        elif self._in_snippet:
            self._current.snippet += data

    def _flush(self) -> None:
        if self._current is not None and self._current.url:
            self.hits.append(self._current)
        self._current = None

    def close(self) -> None:
        super().close()
        self._flush()


def _resolve_ddg_url(href: str, engine: str) -> str:
    """DDG wraps result URLs in its redirector; unwrap the ``uddg`` target."""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlparse(href)
    if "duckduckgo.com" in (parsed.netloc or ""):
        query = urllib.parse.parse_qs(parsed.query)
        target = query.get("uddg", [None])[0]
        if target:
            return urllib.parse.unquote(target)
    return href


def _parse_results(html: str, engine: str) -> list[SearchHit]:
    parser = _DDGResultParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 — parse what we can
        pass
    hits: list[SearchHit] = []
    seen: set[str] = set()
    for raw in parser.hits:
        url = _resolve_ddg_url(raw.url, engine)
        title = " ".join(raw.title.split())
        snippet = " ".join(raw.snippet.split())
        if not url or url in seen:
            continue
        seen.add(url)
        hits.append(SearchHit(url=url, title=title, snippet=snippet, engine=engine))
    return hits


# A direct-answer JSON endpoint that works keyless as a fallback is omitted
# deliberately: keyless coverage is HTML-only, and mixing endpoints would make
# offline tests dishonest about what a live run can do.


class Searcher:
    """Search protocol: ``search(query, *, max_results, timeout_s)``."""

    def search(
        self,
        query: str,
        *,
        max_results: int = DEFAULT_MAX_RESULTS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> SearchResult:
        raise NotImplementedError


class DuckDuckGoLiteSearcher(Searcher):
    """Keyless DDG via the lite endpoint, HTML fallback to the html endpoint."""

    def __init__(
        self,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        open_fn: Callable[[urllib.request.Request, float], Any] | None = None,
        pause_s: float = 0.0,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._user_agent = user_agent
        self._open_fn = open_fn
        self._pause_s = pause_s
        self._now = now_fn
        self._last_search_at = 0.0

    def _get(self, url: str, timeout_s: float) -> bytes:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname not in _DDG_SEARCH_HOSTS:
            raise ValueError("search endpoint must be an approved DuckDuckGo HTTPS host")
        request = urllib.request.Request(url, headers={"User-Agent": self._user_agent})
        if self._open_fn is not None:
            with self._open_fn(request, timeout_s) as response:
                return bytes(response.read())
        # Scheme and host are allowlisted immediately above.
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # nosec B310
            return bytes(response.read())

    def search(
        self,
        query: str,
        *,
        max_results: int = DEFAULT_MAX_RESULTS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> SearchResult:
        started = self._now()
        if self._pause_s:
            wait = self._pause_s - (self._now() - self._last_search_at)
            if wait > 0:
                time.sleep(wait)
        self._last_search_at = self._now()
        result = SearchResult(query=query)
        encoded = urllib.parse.urlencode({"q": query})
        for engine, endpoint in (("ddg-lite", DDG_LITE_URL), ("ddg-html", DDG_HTML_URL)):
            try:
                body = self._get(f"{endpoint}?{encoded}", timeout_s)
            except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
                result.error = f"search transport fault ({type(exc).__name__})"
                continue
            try:
                html = body.decode("utf-8", errors="replace")
            except UnicodeError:  # pragma: no cover — errors="replace" never raises UnicodeError
                result.error = "search response not decodable"  # pragma: no cover
                continue  # pragma: no cover
            hits = _parse_results(html, engine)
            if hits:
                result.hits = hits[:max_results]
                result.error = ""
                break
            result.error = f"{engine} returned no parseable results"
        result.elapsed_s = round(self._now() - started, 3)
        return result


class NullSearcher(Searcher):
    """Offline stand-in: empty results, explicit error — never fabricated hits."""

    def search(
        self,
        query: str,
        *,
        max_results: int = DEFAULT_MAX_RESULTS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> SearchResult:
        del max_results, timeout_s
        return SearchResult(query=query, error="null searcher: offline")


def search_ddg(query: str, **kwargs: Any) -> SearchResult:
    """One-shot convenience: a fresh lite searcher per call (no shared pause state)."""
    return DuckDuckGoLiteSearcher().search(query, **kwargs)
