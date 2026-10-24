"""Edge coverage for fx1.webresearch fetch, search, and the research engine.

Every socket is injected. Clocks and sleeps are stubbed so nothing waits.
"""

from __future__ import annotations

import threading
import urllib.error
import urllib.request
from decimal import Decimal

import pytest

from fx1.webresearch import engine as eng
from fx1.webresearch import search as search_mod
from fx1.webresearch.engine import (
    Claim,
    ClaimGroup,
    ResearchBudget,
    Researcher,
    ResearchReport,
    SourceDoc,
    _note_claim_conflict,
    _numeric_facts,
    _overlap,
    extract_claims,
    group_claims,
)
from fx1.webresearch.fetch import (
    FetchPolicyError,
    FetchResult,
    _LimitedRedirectHandler,
    _robots_disallowed,
    _TextExtractor,
    clear_robots_cache,
    fetch_page,
    host_is_public,
    html_to_text,
    url_policy_error,
)
from fx1.webresearch.search import (
    DDG_HTML_URL,
    DDG_LITE_URL,
    DuckDuckGoLiteSearcher,
    SearchHit,
    SearchResult,
    _parse_results,
    search_ddg,
)


class _CM:
    def __init__(
        self,
        body: bytes,
        *,
        url: str = "https://example.com/a",
        content_type: str = "text/plain",
        status: int = 200,
    ) -> None:
        self._body = body
        self.status = status
        self.headers = {"Content-Type": content_type}
        self._url = url

    def read(self, n: int = -1) -> bytes:
        if n < 0 or n >= len(self._body):
            body, self._body = self._body, b""
            return body
        chunk, self._body = self._body[:n], self._body[n:]
        return chunk

    def geturl(self) -> str:
        return self._url

    def __enter__(self) -> _CM:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _public_resolve(host: str, port: int, *args: object, **kwargs: object) -> list[tuple]:
    return [(2, 1, 6, "", ("1.1.1.1", port))]


def test_host_and_url_policy_edges() -> None:
    assert host_is_public("1.1.1.1") == (True, "")
    assert host_is_public("8.8.8.8")[0] is True

    def weird(host: str, port: int, *args: object, **kwargs: object) -> list[tuple]:
        return [(2, 1, 6, "", ("not-an-ip", port)), (2, 1, 6, "", ("1.1.1.1", port))]

    assert host_is_public("example.com", resolve_fn=weird) == (True, "")
    assert url_policy_error("http://") == "url names no host"
    assert "credentials" in (url_policy_error("https://user:secret@1.1.1.1/x") or "")


def test_redirect_handler_cap_policy_and_pass() -> None:
    req = urllib.request.Request("https://1.1.1.1/start")
    capped = _LimitedRedirectHandler(0)
    with pytest.raises(urllib.error.HTTPError, match="too many redirects"):
        capped.redirect_request(req, None, 302, "Found", {}, "https://1.1.1.1/next")

    limited = _LimitedRedirectHandler(2)
    with pytest.raises(FetchPolicyError, match="redirect refused"):
        limited.redirect_request(req, None, 302, "Found", {}, "file:///etc/passwd")

    hopped = limited.redirect_request(req, None, 302, "Found", {}, "http://1.1.1.1/ok")
    assert hopped is not None
    assert hopped.full_url.startswith("http://1.1.1.1/ok")


def test_html_skip_depth_blocks_and_non_http_links() -> None:
    html = "<script><b>hidden</b></script><span>word</span><p>next</p><a>nohref</a><a href='mailto:a@b.c'>m</a><a href='https://a.example/x'>x</a><a href='https://a.example/x'>again</a>"
    title, text, links = html_to_text(html, base_url="")
    assert "hidden" not in text
    assert "word" in text and "next" in text
    assert links == ["https://a.example/x"]
    assert title == ""


def test_fetch_skips_policy_when_disabled() -> None:
    def open_fn(request: urllib.request.Request, timeout: float) -> _CM:
        return _CM(b"body", url="http://example.test/a", content_type="text/plain")

    result = fetch_page("http://example.test/a", open_fn=open_fn, check_policy=False)
    assert result.ok and result.text == "body"


def test_html_extractor_skip_blocks_and_parser_fault(monkeypatch: pytest.MonkeyPatch) -> None:
    html = "<div><script>secret<script>nested</script></script><p>visible</p><div>more</div></div>"
    title, text, links = html_to_text(html, base_url="https://1.1.1.1/doc")
    assert "secret" not in text
    assert "visible" in text and "more" in text
    assert title == ""
    assert links == []

    def boom(self: _TextExtractor, data: str) -> None:
        raise RuntimeError("parser blew up")

    monkeypatch.setattr(_TextExtractor, "feed", boom)
    title, text, links = html_to_text("<p>ignored</p>", base_url="")
    assert (title, text, links) == ("", "", [])


def test_robots_wildcard_and_noise_lines() -> None:
    robots = "\n".join(
        [
            "# comment",
            "not a directive",
            "User-agent: *",
            "Disallow:",
            "Disallow: /secret*",
            "Disallow: /exact",
            "User-agent: otherbot",
            "Disallow: /only-them",
        ]
    )
    assert _robots_disallowed(robots, "dipcatcher-research/0.1", "/secret/page") is True
    assert _robots_disallowed(robots, "dipcatcher-research/0.1", "/exact/child") is True
    assert _robots_disallowed(robots, "dipcatcher-research/0.1", "/public") is False
    assert _robots_disallowed("User-agent: *\nDisallow: /", "anyone", "/anywhere") is True


def test_fetch_page_opener_plain_text_and_policy_error(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_robots_cache()

    class Opener:
        def open(self, request: urllib.request.Request, timeout: float | None = None) -> _CM:
            assert timeout == 0.05
            return _CM(b"plain body", url="https://1.1.1.1/a", content_type="text/plain")

    monkeypatch.setattr(urllib.request, "build_opener", lambda *a, **k: Opener())
    result = fetch_page(
        "http://example.com/a",
        timeout_s=0.05,
        resolve_fn=_public_resolve,
        check_policy=True,
    )
    assert result.ok
    assert result.text == "plain body"
    assert result.truncated is False

    def refuse(request: urllib.request.Request, timeout: float) -> object:
        raise FetchPolicyError("redirect refused by fetch policy: private")

    refused = fetch_page("http://1.1.1.1/x", open_fn=refuse, resolve_fn=_public_resolve)
    assert refused.error == "redirect refused by fetch policy: private"


def test_parse_results_swallows_parser_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class Boom(search_mod._DDGResultParser):
        def feed(self, data: str) -> None:
            raise RuntimeError("bad html")

    monkeypatch.setattr(search_mod, "_DDGResultParser", Boom)
    assert _parse_results("<html>", "ddg-lite") == []


def test_ddg_unwrap_without_uddg_and_urlopen(monkeypatch: pytest.MonkeyPatch) -> None:
    html = (
        "<a rel='nofollow' class='result-link' "
        "href='https://duckduckgo.com/l/?rut=abc'>Bare</a>"
        "<td class='result-snippet'>snippet text here</td>"
    )
    hits = _parse_results(html, "ddg-lite")
    assert hits and hits[0].url.startswith("https://duckduckgo.com/")

    seen: list[float] = []

    def urlopen(request: urllib.request.Request, timeout: float | None = None) -> _CM:
        seen.append(timeout or 0)
        assert request.full_url.startswith(DDG_LITE_URL)
        return _CM(b"<html></html>", url=request.full_url, content_type="text/html")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    searcher = DuckDuckGoLiteSearcher()
    body = searcher._get(DDG_LITE_URL, 0.01)
    assert body == b"<html></html>"
    assert seen == [0.01]


def test_search_pause_empty_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(search_mod.time, "sleep", lambda seconds: slept.append(seconds))
    clock = {"t": 10.0}

    def now() -> float:
        return clock["t"]

    calls: list[str] = []

    def open_fn(request: urllib.request.Request, timeout: float) -> _CM:
        calls.append(request.full_url)
        return _CM(b"<html>no results</html>", url=request.full_url, content_type="text/html")

    searcher = DuckDuckGoLiteSearcher(open_fn=open_fn, pause_s=0.4, now_fn=now)
    searcher._last_search_at = 0.0
    first = searcher.search("inflation", timeout_s=0.01)
    assert slept == []
    assert "no parseable" in first.error
    assert calls[0].startswith(DDG_LITE_URL) and calls[1].startswith(DDG_HTML_URL)

    clock["t"] = 10.1
    searcher.search("again", timeout_s=0.01)
    assert slept == [pytest.approx(0.3)]


def test_search_ddg_uses_fresh_searcher(monkeypatch: pytest.MonkeyPatch) -> None:
    class StandIn:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] = {}

        def search(self, query: str, **kwargs: object) -> SearchResult:
            self.kwargs = kwargs
            return SearchResult(query=query, hits=[SearchHit(url="https://a.example/x", title="t")])

    stand_in = StandIn()
    monkeypatch.setattr(search_mod, "DuckDuckGoLiteSearcher", lambda **kwargs: stand_in)
    result = search_ddg("q", max_results=2, timeout_s=0.01)
    assert result.hits[0].url == "https://a.example/x"
    assert stand_in.kwargs["max_results"] == 2


def test_markdown_claims_and_numeric_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    report = ResearchReport(
        goal="inflation",
        started_at="t0",
        finished_at="t1",
        redirected_from=["old goal"],
        errors=["search blew up"],
    )
    rendered = report.to_markdown()
    assert "redirected from: old goal" in rendered
    assert "search blew up" in rendered
    assert "## Sources" not in rendered

    left = Claim(
        text="Inflation will increase sharply across the year according to the report.",
        source_url="https://a.example/a",
        tokens=frozenset(
            eng.tokenize("Inflation will increase sharply across the year according to the report.")
        ),
    )
    right = Claim(
        text="Inflation will decrease sharply across the year according to the report.",
        source_url="https://b.example/b",
        tokens=frozenset(
            eng.tokenize("Inflation will decrease sharply across the year according to the report.")
        ),
    )
    groups = group_claims([left, right], threshold=0.5)
    assert groups and groups[0].contradictions
    with_sources = ResearchReport(
        goal="inflation",
        sources=[
            SourceDoc(url="https://a.example/a", title="", status=200, text="ok"),
            SourceDoc(url="https://b.example/b", status=500, error="HTTP 500"),
        ],
        groups=groups,
    )
    text = with_sources.to_markdown()
    assert "CONFLICT:" in text
    assert "failed: HTTP 500" in text

    skipped = extract_claims(
        "Purple elephants wander quietly across distant meadows tonight.",
        "inflation",
    )
    assert skipped == []
    assert _overlap(frozenset(), frozenset({"inflation"})) == 0.0

    same = ClaimGroup(key="k")
    _note_claim_conflict(
        same, left, Claim(text="no " + left.text, source_url=left.source_url, tokens=left.tokens)
    )
    assert same.contradictions == []

    def explode(value: str) -> Decimal:
        raise eng.InvalidOperation

    monkeypatch.setattr(eng, "Decimal", explode)
    assert (
        _numeric_facts(
            Claim(
                text="The rate was 3.5 percent in the latest report.",
                source_url="",
                tokens=frozenset(),
            )
        )
        == frozenset()
    )

    class MinusZero:
        def normalize(self) -> MinusZero:
            return self

        def __format__(self, spec: str) -> str:
            return "-0"

    monkeypatch.setattr(eng, "Decimal", lambda value: MinusZero())
    normalized = _numeric_facts(
        Claim(
            text="The change was -0 percent in the sample report.",
            source_url="",
            tokens=frozenset(),
        )
    )
    assert ("0", "%") in normalized


def test_engine_redirect_default_searcher_and_fetch_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_robots_cache()

    class Offline:
        def __init__(self, pause_s: float = 0.0) -> None:
            self.pause_s = pause_s

        def search(self, query: str, **kwargs: object) -> SearchResult:
            return SearchResult(query=query, error="offline")

    monkeypatch.setattr(search_mod, "DuckDuckGoLiteSearcher", Offline)
    events: list[str] = []
    bare = Researcher(now_fn=lambda: 0.0, sleep_fn=lambda _s: None)
    report = bare.run(
        "inflation",
        budget=ResearchBudget(max_searches=1, max_duration_s=5, search_pause_s=0, max_sources=1),
        queries=["inflation"],
        on_progress=lambda progress: events.append(progress.event),
    )
    assert report.errors and report.errors[0].startswith("search ")
    assert "started" in events

    researcher = Researcher(now_fn=lambda: 0.0, sleep_fn=lambda _s: None)
    researcher.redirect("fresh inflation question")

    class Redirecting:
        def search(self, query: str, **kwargs: object) -> SearchResult:
            return SearchResult(query=query, hits=[])

    researcher._searcher = Redirecting()
    redirected = researcher.run(
        "original",
        budget=ResearchBudget(max_searches=2, max_duration_s=5, search_pause_s=0, max_sources=1),
        queries=["original query"],
    )
    assert redirected.redirected_from == ["original"]
    assert redirected.goal == "fresh inflation question"

    cancel = threading.Event()
    cancel.set()
    empty = ResearchReport(goal="inflation")
    researcher._fetch_doc(
        "https://a.example/inflation",
        empty,
        set(),
        "inflation",
        ResearchBudget(),
        10.0,
        cancel,
        None,
        depth=0,
    )
    assert empty.sources == []

    def failing(url: str, **kwargs: object) -> FetchResult:
        if url.endswith("/robots.txt"):
            return FetchResult(url=url, status=404, error="missing", text="")
        return FetchResult(url=url, status=500, error="HTTP 500", text="")

    failed = Researcher(fetch_fn=failing, now_fn=lambda: 0.0)
    failed_report = ResearchReport(goal="inflation")
    notes: list[str] = []
    failed._fetch_doc(
        "https://a.example/page",
        failed_report,
        set(),
        "inflation",
        ResearchBudget(max_sources=2, max_depth=0),
        10.0,
        threading.Event(),
        lambda progress: notes.append(progress.event),
        depth=0,
    )
    assert failed_report.sources[0].error == "HTTP 500"
    assert "fetched" in notes

    link_cancel = threading.Event()

    def page(url: str, **kwargs: object) -> FetchResult:
        if url.endswith("/robots.txt"):
            return FetchResult(url=url, status=404, error="missing", text="")
        link_cancel.set()
        return FetchResult(
            url=url,
            final_url=url,
            status=200,
            content_type="text/plain",
            title="Inflation",
            text="The inflation rate rose 4 percent in the latest official market report.",
            links=["https://b.example/inflation-report"],
        )

    follower = Researcher(fetch_fn=page, now_fn=lambda: 0.0)
    followed = ResearchReport(goal="inflation")
    follower._fetch_doc(
        "https://a.example/start",
        followed,
        set(),
        "inflation",
        ResearchBudget(max_sources=4, max_depth=1),
        10.0,
        link_cancel,
        None,
        depth=0,
    )
    assert len(followed.sources) == 1

    def robots_block(url: str, **kwargs: object) -> FetchResult:
        return FetchResult(
            url=url,
            status=200,
            content_type="text/plain",
            text="User-agent: *\nDisallow: /\n",
        )

    clear_robots_cache()
    blocked = Researcher(fetch_fn=robots_block, now_fn=lambda: 0.0)
    blocked_report = ResearchReport(goal="inflation")
    skipped: list[str] = []
    blocked._fetch_doc(
        "https://c.example/hidden",
        blocked_report,
        set(),
        "inflation",
        ResearchBudget(),
        10.0,
        threading.Event(),
        lambda progress: skipped.append(progress.detail),
        depth=0,
    )
    assert blocked_report.sources == []
    assert any("robots.txt" in item for item in skipped)

    clock = {"t": 0.0}

    def budgeted(url: str, **kwargs: object) -> FetchResult:
        if url.endswith("/robots.txt"):
            return FetchResult(url=url, status=404, error="missing", text="")
        clock["t"] = 50.0
        return FetchResult(
            url=url,
            status=200,
            content_type="text/plain",
            text="Short.",
            links=["https://b.example/inflation-data"],
        )

    tight = Researcher(fetch_fn=budgeted, now_fn=lambda: clock["t"])
    tight_report = ResearchReport(goal="inflation")
    tight._fetch_doc(
        "https://a.example/start",
        tight_report,
        set(),
        "inflation",
        ResearchBudget(max_sources=1, max_depth=1),
        10.0,
        threading.Event(),
        None,
        depth=0,
    )
    assert len(tight_report.sources) == 1


def test_html_skip_block_links_and_policy_bypass() -> None:
    html = (
        "<script><b>hidden</b></script>"
        "<p>one</p><div>two</div>"
        "<a>bare</a><a href='https://1.1.1.1/a'>A</a><a href='mailto:a@b.c'>m</a>"
    )
    title, text, links = html_to_text(html, base_url="")
    assert "hidden" not in text
    assert "one" in text and "two" in text
    assert title == ""
    assert links == ["https://1.1.1.1/a"]
    nested = _TextExtractor()
    nested.handle_starttag("script", [])
    nested.handle_starttag("b", [])
    assert nested._skip_depth == 1 and nested._chunks == []
    nested.handle_endtag("script")
    nested.handle_data("one")
    nested.handle_starttag("div", [])
    assert nested._chunks[-1] == "\n"

    joined = html_to_text("<a href='/rel'>r</a>", base_url="https://1.1.1.1/root")
    assert joined[2] == ["https://1.1.1.1/rel"]

    clear_robots_cache()

    class Opener:
        def open(self, request: urllib.request.Request, timeout: float | None = None) -> _CM:
            return _CM(b"plain", url=request.full_url, content_type="text/plain")

    skipped = fetch_page(
        "http://1.1.1.1/plain",
        check_policy=False,
        open_fn=lambda request, timeout: Opener().open(request, timeout),
    )
    assert skipped.ok and skipped.text == "plain"


def test_negative_zero_numeric_fact_collapses() -> None:
    facts = _numeric_facts(
        Claim(
            text="The change was -0 percent in the sample report.",
            source_url="",
            tokens=frozenset(),
        )
    )
    assert ("0", "%") in facts
    assert ("-0", "%") not in facts
