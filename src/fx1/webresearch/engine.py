"""The deep web-research engine: iterative, budgeted, cancellable, honest.

Pipeline per run:
1. seed queries from the goal (caller can supply its own),
2. search → select unseen hits (snippet/goal overlap + domain diversity),
3. fetch (policy-checked, size-capped) → extract goal-relevant claims,
4. follow promising in-page references within the depth budget,
5. group claims across sources, detect explicit contradictions,
6. report every claim with its supporting/contradicting source counts.

Everything is interruptible: ``cancel_event`` is checked between every
network step and ``redirect(new_goal)`` re-targets the run at the next
step boundary. Budgets (wall-clock, search count, source count, bytes per
source) are checked at every step and reported honestly when they bind.

The report is *heuristic web verification*, labeled as such — it is not a
sealed lab receipt and must never be presented as one.
"""

from __future__ import annotations

import re
import threading
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation

from fx1.webresearch.fetch import FetchResult, fetch_page, robots_allowed
from fx1.webresearch.search import DEFAULT_MAX_RESULTS, Searcher, SearchHit

# Do not split after abbreviation dots ("U.S.", "e.g."): a claim that loses
# its subject groups and reads worse. Under-splitting is the safer error.
_SENTENCE_SPLIT = re.compile(r"(?<![A-Z]\.)(?<=[.!?])\s+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
        "which",
        "while",
        "you",
        "your",
    }
)
_NEGATION_HEAD = re.compile(
    r"^\W*(no|not|never|isn['’]t|aren['’]t|don['’]t|doesn['’]t|"
    r"won['’]t|refutes?|rejects?|fails?\s+to)\b",
    re.IGNORECASE,
)
_ANTONYMS: tuple[tuple[str, str], ...] = (
    ("increase", "decrease"),
    ("rise", "fall"),
    ("grow", "shrink"),
    ("bullish", "bearish"),
    ("support", "contradict"),
    ("agree", "disagree"),
    ("outperform", "underperform"),
    ("gain", "loss"),
)
_CLAIM_MARKERS = re.compile(
    r"\d|%|\b(?:more|less|than|since|after|before|now|latest|report(?:s|ed)?|"
    r"found|shows?|estimates?|according to|vs\.?|versus|compared)\b",
    re.IGNORECASE,
)
_NUMERIC_FACT = re.compile(
    r"(?<![\w.])"
    r"(?P<currency>[$€£¥])?\s*"
    r"(?P<value>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"\s*(?P<unit>%|percent(?:age)?(?:\s+points?)?|bp(?:s)?|basis\s+points?|"
    r"thousand|million|billion|trillion|[kmbt])?"
    r"(?!\w|[.]\d)",
    re.IGNORECASE,
)
_BOILERPLATE = re.compile(
    r"^(the following|this table|see (?:the |also )?|click|view |download|"
    r"note that|for additional|to calculate|about (?:this|the) )",
    re.IGNORECASE,
)
# Site-chrome phrases that carry no claim: navigation, account widgets,
# share links, legal footers. Any match disqualifies the sentence outright —
# these leak past goal-overlap because pages say "U.S." in their menus.
_NAV_CHROME = re.compile(
    r"\b(?:contact us|my account|sign (?:in|up)|log ?in|digital badges?|"
    r"explore our apps|tutorial|newsletter|privacy (?:policy|notice)|"
    r"terms (?:of|&|and)|cookie|share this page|link copied|skip to (?:main|content)|"
    r"menu|breadcrumb|follow us|subscribe|advertise|careers|all rights reserved|"
    r"photo by|published:|updated:|verified:|\u203a|"
    r"\u00a9|copyright)\b",
    re.IGNORECASE,
)


def _canonical_url(url: str) -> str:
    """Dedupe key for a URL: fragments identify the same document."""
    return urllib.parse.urldefrag(url).url


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ResearchBudget:
    """Hard limits for one research run; every field is checked mid-loop."""

    max_duration_s: float = 300.0
    max_searches: int = 10
    max_sources: int = 8
    max_bytes_per_source: int = 512 * 1024
    max_depth: int = 1
    max_results_per_search: int = DEFAULT_MAX_RESULTS
    search_pause_s: float = 0.6


@dataclass
class ResearchProgress:
    """One progress event for the console (event, human detail, wall time)."""

    event: str
    detail: str
    at: float = field(default_factory=time.time)


ProgressFn = Callable[[ResearchProgress], None]


@dataclass
class SourceDoc:
    url: str
    title: str = ""
    status: int | None = None
    text: str = ""
    links: list[str] = field(default_factory=list)
    error: str = ""
    truncated: bool = False
    fetched_at: str = ""

    @property
    def ok(self) -> bool:
        return self.status is not None and 200 <= self.status < 400 and not self.error


@dataclass
class Claim:
    text: str
    source_url: str
    tokens: frozenset[str] = frozenset()


@dataclass
class ClaimGroup:
    """One topic: the claims about it and who supports / contradicts whom."""

    key: str
    claims: list[Claim] = field(default_factory=list)
    contradictions: list[tuple[str, str]] = field(default_factory=list)

    @property
    def sources(self) -> set[str]:
        return {c.source_url for c in self.claims}

    @property
    def status(self) -> str:
        if self.contradictions:
            return "contested"
        if len(self.sources) >= 2:
            return "corroborated"
        return "single-source"


@dataclass
class ResearchReport:
    goal: str
    started_at: str = ""
    finished_at: str = ""
    cancelled: bool = False
    budget: ResearchBudget = field(default_factory=ResearchBudget)
    queries_used: list[str] = field(default_factory=list)
    sources: list[SourceDoc] = field(default_factory=list)
    groups: list[ClaimGroup] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    redirected_from: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return (
            "HEURISTIC WEB VERIFICATION — findings are cross-source checks of "
            "public web pages; not a sealed dipcatcher research receipt."
        )

    def to_markdown(self) -> str:
        lines = [
            f"# Web research: {self.goal}",
            self.label,
            "",
            f"started {self.started_at} — finished {self.finished_at}"
            + (" (cancelled)" if self.cancelled else ""),
            f"sources fetched: {len([s for s in self.sources if s.ok])} "
            f"of {len(self.sources)} attempted; queries used: {len(self.queries_used)}",
        ]
        if self.redirected_from:
            lines.append(f"redirected from: {', '.join(self.redirected_from)}")
        if self.sources:
            lines += ["", "## Sources", ""]
            for source in self.sources:
                state = "ok" if source.ok else f"failed: {source.error}"
                lines.append(f"- {source.title or source.url} <{source.url}> [{state}]")
        if self.groups:
            lines += ["", "## Findings (cross-source)", ""]
            for group in self.groups:
                lines.append(f"### {group.status}: {group.key}")
                lines.append(f"claims: {len(group.claims)}; sources: {len(group.sources)}")
                for claim in group.claims[:4]:
                    lines.append(f"- {claim.text} ({claim.source_url})")
                for left, right in group.contradictions:
                    lines.append(f"- CONFLICT: {left}  <->  {right}")
        if self.errors:
            lines += ["", "## Errors", ""]
            lines += [f"- {error}" for error in self.errors]
        return "\n".join(lines) + "\n"


def default_queries(goal: str) -> list[str]:
    """Deterministic seed queries anchored to the goal, including a
    deliberately adversarial angle so conflicting evidence is searched for,
    not avoided."""
    goal = goal.strip()
    return [
        goal,
        f"{goal} data and evidence",
        f"{goal} criticism OR contradictions OR alternative views",
    ]


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


def extract_claims(text: str, goal: str, *, max_claims: int = 12) -> list[Claim]:
    """Goal-relevant sentences as claims — numbers, comparatives, or goal
    vocabulary. Sentences are the honest unit: no generative rewriting."""
    goal_tokens = set(tokenize(goal))
    claims: list[Claim] = []
    for sentence in _SENTENCE_SPLIT.split(text):
        sentence = " ".join(sentence.split()).strip(" \n\t-•")
        tokens = tokenize(sentence)
        if len(tokens) < 5 or len(sentence) > 600:
            continue
        if _BOILERPLATE.match(sentence) or _NAV_CHROME.search(sentence):
            continue
        relevant = bool(goal_tokens & set(tokens)) or bool(_CLAIM_MARKERS.search(sentence))
        if not relevant:
            continue
        claims.append(Claim(text=sentence, source_url="", tokens=frozenset(tokens)))
        if len(claims) >= max_claims:
            break
    return claims


def _overlap(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _negated(claim: Claim) -> bool:
    return bool(_NEGATION_HEAD.search(claim.text))


def _polarity(claim: Claim) -> set[str]:
    words = set(claim.tokens)
    out: set[str] = set()
    for left, right in _ANTONYMS:
        if left in words or f"{left}s" in words:
            out.add(left)
        if right in words or f"{right}s" in words:
            out.add(right)
    return out


def _numeric_facts(claim: Claim) -> frozenset[tuple[str, str]]:
    """Return normalized value/unit facts mentioned by a claim.

    Numeric web claims are corroborated only when independently sourced text
    agrees on every explicit number and unit.  This deliberately includes
    dates and counts: text about different periods or sample sizes is not the
    same factual claim.  The parser is intentionally small and conservative;
    unsupported quantitative phrasing cannot manufacture corroboration.
    """
    facts: set[tuple[str, str]] = set()
    unit_aliases = {
        "percent": "%",
        "percentage": "%",
        "percentage point": "percentage-point",
        "percentage points": "percentage-point",
        "bp": "basis-point",
        "bps": "basis-point",
        "basis point": "basis-point",
        "basis points": "basis-point",
        "k": "thousand",
        "m": "million",
        "b": "billion",
        "t": "trillion",
    }
    for match in _NUMERIC_FACT.finditer(claim.text):
        raw_value = match.group("value").replace(",", "")
        try:
            value = format(Decimal(raw_value).normalize(), "f")
        except InvalidOperation:
            continue
        if value == "-0":
            value = "0"
        currency = match.group("currency") or ""
        raw_unit = " ".join((match.group("unit") or "").lower().split())
        unit = currency + unit_aliases.get(raw_unit, raw_unit)
        facts.add((value, unit))
    return frozenset(facts)


def _numeric_disagreement(left: Claim, right: Claim) -> bool:
    """Fail closed when grouped quantitative claims do not exactly agree."""
    left_facts = _numeric_facts(left)
    right_facts = _numeric_facts(right)
    return bool(left_facts or right_facts) and left_facts != right_facts


def _claims_conflict(left: Claim, right: Claim) -> bool:
    if left.source_url == right.source_url:
        return False
    if _negated(left) != _negated(right):
        return True
    left_pol = _polarity(left)
    right_pol = _polarity(right)
    if left_pol and right_pol and left_pol != right_pol:
        return True
    return _numeric_disagreement(left, right)


def _best_claim_group(
    claim: Claim, groups: list[ClaimGroup], *, threshold: float
) -> ClaimGroup | None:
    best: ClaimGroup | None = None
    best_score = 0.0
    for group in groups:
        for member in group.claims[:5]:
            score = _overlap(claim.tokens, member.tokens)
            if score > best_score:
                best, best_score = group, score
    if best is not None and best_score >= threshold:
        return best
    return None


def _record_group_contradictions(group: ClaimGroup) -> None:
    for i, left in enumerate(group.claims):
        for right in group.claims[i + 1 :]:
            pair = (left.text, right.text)
            if _claims_conflict(left, right) and pair not in group.contradictions:
                group.contradictions.append(pair)


def group_claims(claims: list[Claim], *, threshold: float = 0.6) -> list[ClaimGroup]:
    """Cluster claims by token overlap and detect contradictions inside each
    cluster: negation, or antonym polarity in opposite directions."""
    groups: list[ClaimGroup] = []
    for claim in claims:
        best = _best_claim_group(claim, groups, threshold=threshold)
        if best is not None:
            best.claims.append(claim)
        else:
            groups.append(ClaimGroup(key=" ".join(sorted(claim.tokens)[:6]), claims=[claim]))
    for group in groups:
        _record_group_contradictions(group)
    groups = [g for g in groups if g.claims]
    groups.sort(key=lambda g: (-len(g.sources), -len(g.claims)))
    return groups


class Researcher:
    """Iterative background-safe web researcher. All seams injectable."""

    def __init__(
        self,
        *,
        searcher: Searcher | None = None,
        fetch_fn: Callable[..., FetchResult] | None = None,
        now_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._searcher = searcher
        self._fetch_fn = fetch_fn or fetch_page
        self._now = now_fn
        self._sleep = sleep_fn
        self._redirect_lock = threading.Lock()
        self._redirect_goal: str | None = None

    def redirect(self, new_goal: str) -> None:
        """Re-target a running (or queued) research task at its next step."""
        with self._redirect_lock:
            self._redirect_goal = new_goal.strip()

    def _take_redirect(self) -> str | None:
        with self._redirect_lock:
            goal = self._redirect_goal
            self._redirect_goal = None
            return goal

    def _emit(self, on_progress: ProgressFn | None, event: str, detail: str) -> None:
        if on_progress is not None:
            on_progress(ResearchProgress(event=event, detail=detail))

    def run(
        self,
        goal: str,
        *,
        budget: ResearchBudget | None = None,
        queries: list[str] | None = None,
        cancel_event: threading.Event | None = None,
        on_progress: ProgressFn | None = None,
    ) -> ResearchReport:
        budget = budget or ResearchBudget()
        cancel_event = cancel_event or threading.Event()
        report = ResearchReport(goal=goal, started_at=_now_iso(), budget=budget)
        report.queries_used = list(queries) if queries else default_queries(goal)
        deadline = self._now() + budget.max_duration_s
        seen_urls: set[str] = set()
        searches_run = 0
        queries_used = list(report.queries_used)
        searcher = self._searcher
        if searcher is None:
            from fx1.webresearch.search import DuckDuckGoLiteSearcher

            searcher = DuckDuckGoLiteSearcher(pause_s=budget.search_pause_s)
        active_goal = goal
        query_index = 0

        self._emit(on_progress, "started", f"research goal: {active_goal}")

        def blocker() -> str | None:
            if cancel_event.is_set():
                return "cancelled"
            if self._now() > deadline:
                return "time budget exhausted"
            if searches_run >= budget.max_searches:
                return "search budget exhausted"
            if len(report.sources) >= budget.max_sources:
                return "source budget exhausted"
            return None

        while query_index < len(queries_used) and blocker() is None:
            redirect = self._take_redirect()
            if redirect and redirect != active_goal:
                report.redirected_from.append(active_goal)
                active_goal = redirect
                report.goal = redirect
                fresh = [q for q in default_queries(redirect) if q not in queries_used]
                queries_used = fresh + queries_used
                report.queries_used = queries_used
                self._emit(on_progress, "redirect", f"now researching: {redirect}")
                query_index = 0
                continue

            query = queries_used[query_index]
            query_index += 1
            searches_run += 1
            self._emit(on_progress, "searching", f"search: {query}")
            result = searcher.search(query, max_results=budget.max_results_per_search)
            if result.error:
                report.errors.append(f"search {query!r}: {result.error}")
                continue
            hits = _rank_hits(
                [hit for hit in result.hits if _canonical_url(hit.url) not in seen_urls],
                active_goal,
            )
            for hit in hits:
                reason = blocker()
                if reason:
                    self._emit(on_progress, "budget", reason)
                    break
                redirect = self._take_redirect()
                if redirect and redirect != active_goal:
                    report.redirected_from.append(active_goal)
                    active_goal = redirect
                    report.goal = redirect
                    fresh = [q for q in default_queries(redirect) if q not in queries_used]
                    queries_used = fresh + queries_used
                    report.queries_used = queries_used
                    self._emit(on_progress, "redirect", f"now researching: {redirect}")
                    query_index = 0
                    break
                self._fetch_doc(
                    hit.url,
                    report,
                    seen_urls,
                    active_goal,
                    budget,
                    deadline,
                    cancel_event,
                    on_progress,
                    depth=0,
                )

        report.cancelled = cancel_event.is_set()
        if report.cancelled:
            self._emit(on_progress, "cancelled", "research cancelled by operator")
        self._emit(on_progress, "verifying", "cross-source claim comparison")
        report.finished_at = _now_iso()
        self._emit(
            on_progress,
            "done",
            f"fetched {len([s for s in report.sources if s.ok])} sources, "
            f"{sum(len(g.claims) for g in report.groups)} claims across "
            f"{len(report.groups)} topics",
        )
        return report

    def _fetch_doc(
        self,
        url: str,
        report: ResearchReport,
        seen_urls: set[str],
        goal: str,
        budget: ResearchBudget,
        deadline: float,
        cancel_event: threading.Event,
        on_progress: ProgressFn | None,
        *,
        depth: int,
    ) -> None:
        canonical = _canonical_url(url)
        if canonical in seen_urls or len(report.sources) >= budget.max_sources:
            return
        seen_urls.add(canonical)
        if cancel_event.is_set() or self._now() > deadline:
            return
        if not robots_allowed(url, fetch_fn=self._fetch_fn):
            self._emit(on_progress, "skipped", f"robots.txt disallows: {url}")
            return
        self._emit(on_progress, "fetching", url)
        result = self._fetch_fn(url, max_bytes=budget.max_bytes_per_source)
        doc = SourceDoc(
            url=url,
            title=result.title,
            status=result.status,
            text=result.text,
            links=result.links,
            error=result.error,
            truncated=result.truncated,
            fetched_at=_now_iso(),
        )
        report.sources.append(doc)
        if not doc.ok:
            self._emit(on_progress, "fetched", f"failed {url}: {doc.error}")
            return
        claims = extract_claims(doc.text, goal)
        for claim in claims:
            claim.source_url = url
        if claims:
            prior = [c for group in report.groups for c in group.claims]
            report.groups = group_claims(prior + claims)
        self._emit(on_progress, "claim-extract", f"{url}: {len(claims)} goal-relevant claims")
        if depth < budget.max_depth:
            for link in _rank_links(doc.links, goal)[:3]:
                if len(report.sources) >= budget.max_sources:
                    return
                if cancel_event.is_set() or self._now() > deadline:
                    return
                self._fetch_doc(
                    link,
                    report,
                    seen_urls,
                    goal,
                    budget,
                    deadline,
                    cancel_event,
                    on_progress,
                    depth=depth + 1,
                )


def _rank_hits(hits: list[SearchHit], goal: str) -> list[SearchHit]:
    """Snippet/goal overlap first, then domain diversity: each domain's first
    hit keeps its rank position; later same-domain hits sink to the end."""
    goal_tokens = frozenset(tokenize(goal))
    scored: list[tuple[float, SearchHit]] = []
    for hit in hits:
        overlap = _overlap(goal_tokens, frozenset(tokenize(hit.title + " " + hit.snippet)))
        scored.append((overlap, hit))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    firsts: list[SearchHit] = []
    rest: list[SearchHit] = []
    seen_domains: set[str] = set()
    for _, hit in scored:
        domain = hit.url.split("/")[2] if "://" in hit.url else hit.url
        if domain in seen_domains:
            rest.append(hit)
        else:
            seen_domains.add(domain)
            firsts.append(hit)
    return firsts + rest


_BINARY_EXT = re.compile(
    r"\.(?:jpe?g|png|gif|webp|svg|ico|pdf|zip|gz|tar|mp[34]|wav|webm|mpg|exe|dmg)$",
    re.IGNORECASE,
)


def _rank_links(links: list[str], goal: str) -> list[str]:
    """Order in-page reference links by goal-term overlap in the URL *path*.

    Query strings are excluded from scoring so share-widget links (which
    echo the page title in a query param) do not outrank real references;
    binary assets are skipped entirely.
    """
    goal_tokens = frozenset(tokenize(goal))
    scored: list[tuple[float, str]] = []
    for link in links:
        path = urllib.parse.urlparse(link).path
        if _BINARY_EXT.search(path):
            continue
        text = path.rstrip("/").split("/")[-1].replace("-", " ").replace("_", " ")
        score = _overlap(goal_tokens, frozenset(tokenize(text)))
        if score > 0:
            scored.append((score, link))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [link for _, link in scored]
