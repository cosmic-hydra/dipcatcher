"""Contract tests for ``quant_fund.api.research_api`` (read-only evidence API).

Builds a labeled-SYNTHETIC evidence tree (research run notebook + immutable
twin, committed-style receipts, sealed benchmark dir, verifier docs, flat
artifacts) and asserts every route validates against its response model, the
OpenAPI schema is self-consistent, and no trading/order/broker surface exists.
"""

from __future__ import annotations

import json
import urllib.parse
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given
from hypothesis import settings as h_settings
from hypothesis import strategies as st

from quant_fund.api.research_api import (
    ArtifactDetail,
    ArtifactListResponse,
    HealthResponse,
    ReadyResponse,
    ReceiptDetail,
    ReceiptListResponse,
    ResearchApiSettings,
    ResultDetail,
    ResultListResponse,
    RunDetail,
    RunListResponse,
    RunMarkdownDetail,
    VerificationResult,
    VerifierRunDetail,
    VerifierRunListResponse,
    VerifierVersionDetail,
    VerifierVersionListResponse,
    create_app,
)
from quant_fund.research.catalog import (
    BENCHMARK_CATALOG_VERSION,
    REQUIRED_BENCHMARK_FAMILIES,
    RESEARCH_RECEIPT_SCHEMA_VERSION,
)
from quant_fund.research.receipt_schema import unavailable_overfitting_block
from quant_fund.research.verify import _receipt_digest
from quant_fund.utils.hashing import hash_file

_RUN_ID = "a" * 64
_OTHER_RUN_ID = "b" * 64

FORBIDDEN_ROUTE_TOKENS = (
    "order",
    "broker",
    "trade",
    "trading",
    "execute",
    "execution",
    "submit",
    "buy",
    "sell",
    "position",
    "alpaca",
    "paper",
    "live",
)


def _notebook_payload(research_root: Path, run_id: str) -> dict:
    """Minimal SYNTHETIC research notebook that passes the canonical verifier."""
    runs = research_root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / f"{run_id}.md").write_text("# research\n")
    payload: dict = {
        "schema_version": RESEARCH_RECEIPT_SCHEMA_VERSION,
        "firm": "Artificial Hedge",
        "product": "Dipcatcher",
        "version": "1.0.0",
        "generated_at": "2026-09-16T00:00:00+00:00",
        "data_source": "SYNTHETIC",
        "synthetic": True,
        "disclaimer": "research only",
        "ranking_target": "future_return_1",
        "claim": "research_only",
        "rankers": [],
        "hypotheses": [],
        "provenance": {
            "run_id": run_id,
            "git_revision": "HEAD",
            "git_worktree_sha256": "e" * 64,
            "config_sha256": "b" * 64,
            "dataset_sha256": "c" * 64,
            "dataset_content_sha256": "d" * 64,
            "northset_inputs_sha256": "e" * 64,
            "row_count": 10,
            "column_count": 3,
            "point_in_time": True,
            "execution_claim": "research_only",
            "benchmark_catalog_version": BENCHMARK_CATALOG_VERSION,
            "runtime": {
                "python": "3.12.0",
                "implementation": "CPython",
                "platform": "test",
                "machine": "test",
                "byteorder": "little",
                "packages": {
                    "numpy": "2.0.0",
                    "polars": "1.0.0",
                    "scipy": "1.0.0",
                    "scikit-learn": "1.0.0",
                },
            },
        },
        "scorecard": {
            name: {
                "executed": True,
                "nonempty": True,
                "finite_observation": True,
                "forbidden_metrics_absent": True,
                "claim": "research_metric_only",
            }
            for name in REQUIRED_BENCHMARK_FAMILIES
        },
        "families": {name: {"executed": True} for name in REQUIRED_BENCHMARK_FAMILIES},
        "backtest_overfitting": unavailable_overfitting_block(),
        "artifacts": {
            "json": "latest.json",
            "markdown": "latest.md",
            "immutable_json": f"runs/{run_id}.json",
            "immutable_markdown": f"runs/{run_id}.md",
            "immutable_markdown_sha256": hash_file(runs / f"{run_id}.md"),
        },
    }
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    return payload


def _write_run(research_root: Path, run_id: str) -> Path:
    payload = _notebook_payload(research_root, run_id)
    text = json.dumps(payload)
    (research_root / "runs" / f"{run_id}.json").write_text(text)
    return research_root / "runs" / f"{run_id}.json"


@pytest.fixture()
def evidence_tree(tmp_path: Path) -> dict[str, Path]:
    data_root = tmp_path / "data"
    research_root = data_root / "metadata" / "research"
    runs_dir = research_root / "runs"
    runs_dir.mkdir(parents=True)
    payload = _notebook_payload(research_root, _RUN_ID)
    text = json.dumps(payload)
    (runs_dir / f"{_RUN_ID}.json").write_text(text)
    (research_root / "latest.json").write_text(text)
    (research_root / "latest.md").write_text("# research\n")
    # A second, tampered run: same declared digests but mutated content.
    tampered = dict(payload)
    tampered["provenance"] = {**payload["provenance"], "run_id": _OTHER_RUN_ID}
    (runs_dir / f"{_OTHER_RUN_ID}.json").write_text(json.dumps(tampered))
    (runs_dir / f"{_OTHER_RUN_ID}.md").write_text("# research\n")

    result_dir = data_root / "metadata" / "real_benchmark" / "synthetic_run"
    result_dir.mkdir(parents=True)
    (result_dir / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "created_at": "2026-09-16T00:00:00+00:00"})
    )
    (result_dir / "validation.json").write_text("{}\n")

    receipts_dir = tmp_path / "receipts"
    receipts_dir.mkdir()
    (receipts_dir / "clean_bench.json").write_text(
        json.dumps(
            {
                "schema": "synthetic.bench/v1",
                "generated_at": "2026-09-16T00:00:00+00:00",
                "research_only": True,
                "live_pnl_claim": False,
                "metrics": {"brier": 0.1},
            }
        )
    )
    (receipts_dir / "dishonest.json").write_text(
        json.dumps(
            {
                "schema": "synthetic.bad/v1",
                "research_only": False,
                "live_pnl_claim": True,
            }
        )
    )
    (receipts_dir / "corrupt.json").write_text("{not json")

    verifier_dir = tmp_path / "verifier"
    (verifier_dir / "v1").mkdir(parents=True)
    (verifier_dir / "v1" / "acceptance.md").write_text("# acceptance v1\n")
    (verifier_dir / "v2").mkdir()
    (verifier_dir / "v2" / "acceptance.md").write_text("# acceptance v2\n")
    (verifier_dir / "runs").mkdir()
    (verifier_dir / "runs" / "2026-09-16T00-00Z_v1.md").write_text(
        "# Verifier run - v1\n\n- Time: 2026-09-16T00-00Z\n\nVerdict: PASS against verifier/v1/acceptance.md.\n"
    )
    (verifier_dir / "runs" / "not-a-version.md").write_text("# scratch\n")

    artifacts_dir = tmp_path / "artifacts"
    (artifacts_dir / "nested").mkdir(parents=True)
    (artifacts_dir / "top.json").write_text(json.dumps({"schema": "synthetic.a/v1"}))
    (artifacts_dir / "nested" / "inner.json").write_text(json.dumps({"ok": True}))
    (artifacts_dir / "weights.parquet").write_bytes(b"PAR1")  # non-JSON ignored

    return {
        "data_root": data_root,
        "receipts_dir": receipts_dir,
        "verifier_dir": verifier_dir,
        "artifacts_dir": artifacts_dir,
    }


@pytest.fixture()
def settings(evidence_tree: dict[str, Path]) -> ResearchApiSettings:
    return ResearchApiSettings(
        data_root=evidence_tree["data_root"],
        receipts_dir=evidence_tree["receipts_dir"],
        verifier_dir=evidence_tree["verifier_dir"],
        artifacts_dir=evidence_tree["artifacts_dir"],
    )


@pytest.fixture()
def client(settings: ResearchApiSettings) -> TestClient:
    return TestClient(create_app(settings))


# ---------------------------------------------------------------------------
# Contract: every route validates against its declared response model
# ---------------------------------------------------------------------------


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = HealthResponse.model_validate(resp.json())
    assert body.status == "ok"
    assert body.mode == "read_only"
    assert body.data_root_present and body.receipts_dir_present
    assert body.claim == "research_only"


def test_ready_when_roots_present(client: TestClient) -> None:
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = ReadyResponse.model_validate(resp.json())
    assert body.status == "ready"
    assert all(body.checks.values())
    assert body.claim == "research_only"


def test_ready_fails_closed_when_receipts_missing(
    settings: ResearchApiSettings, tmp_path: Path
) -> None:
    broken = ResearchApiSettings(
        data_root=settings.data_root,
        receipts_dir=tmp_path / "missing-receipts",
        verifier_dir=settings.verifier_dir,
        artifacts_dir=settings.artifacts_dir,
        api_key=settings.api_key,
    )
    resp = TestClient(create_app(broken)).get("/ready")
    assert resp.status_code == 503
    body = ReadyResponse.model_validate(resp.json())
    assert body.status == "not_ready"
    assert body.checks["receipts_dir"] is False


def test_runs_list_and_pagination(client: TestClient) -> None:
    resp = client.get("/runs")
    assert resp.status_code == 200
    body = RunListResponse.model_validate(resp.json())
    assert body.total == 2
    ids = {item.run_id for item in body.items}
    assert _RUN_ID in ids and _OTHER_RUN_ID in ids
    for item in body.items:
        assert item.sha256 and len(item.sha256) == 64
        assert item.synthetic is True
        assert item.has_markdown is True
        assert item.parse_error is None
    page = RunListResponse.model_validate(
        client.get("/runs", params={"limit": 1, "offset": 1}).json()
    )
    assert page.total == 2
    assert len(page.items) == 1
    assert page.limit == 1 and page.offset == 1


def test_run_detail_latest_and_markdown(client: TestClient) -> None:
    detail = RunDetail.model_validate(client.get(f"/runs/{_RUN_ID}").json())
    assert detail.run_id == _RUN_ID
    assert detail.receipt["provenance"]["run_id"] == _RUN_ID
    latest = RunDetail.model_validate(client.get("/runs/latest").json())
    assert latest.run_id == _RUN_ID
    md = RunMarkdownDetail.model_validate(client.get(f"/runs/{_RUN_ID}/markdown").json())
    assert md.markdown.startswith("# research")
    assert len(md.sha256) == 64


def test_run_verification_valid_and_tampered(client: TestClient) -> None:
    ok = VerificationResult.model_validate(client.get(f"/runs/{_RUN_ID}/verification").json())
    assert ok.valid is True
    assert ok.errors == []
    assert ok.verifier == "verify_research_artifact"
    assert ok.context == "staged_declared_layout"
    assert ok.live_pnl_claim is False
    bad = VerificationResult.model_validate(
        client.get(f"/runs/{_OTHER_RUN_ID}/verification").json()
    )
    # Tampered provenance run_id must fail some integrity/provenance check.
    assert bad.valid is False
    assert bad.errors


def test_receipts_list_detail_byhash(client: TestClient) -> None:
    listing = ReceiptListResponse.model_validate(client.get("/receipts").json())
    assert listing.total == 3
    by_id = {item.id: item for item in listing.items}
    assert by_id["clean_bench"].schema_tag == "synthetic.bench/v1"
    assert by_id["corrupt"].parse_error is not None
    detail = ReceiptDetail.model_validate(client.get("/receipts/clean_bench").json())
    assert detail.receipt["metrics"]["brier"] == 0.1
    via_hash = ReceiptDetail.model_validate(client.get(f"/receipts/by-hash/{detail.sha256}").json())
    assert via_hash.id == "clean_bench"


def test_receipt_verification_honesty_flags(client: TestClient) -> None:
    clean = VerificationResult.model_validate(
        client.get("/receipts/clean_bench/verification").json()
    )
    assert clean.valid is True and clean.verifier == "receipt_honesty_flags"
    dirty = VerificationResult.model_validate(client.get("/receipts/dishonest/verification").json())
    assert dirty.valid is False
    assert "research_only_flag_invalid" in dirty.errors
    assert "live_pnl_claim_invalid" in dirty.errors


def test_results_list_detail_verification(client: TestClient) -> None:
    listing = ResultListResponse.model_validate(client.get("/results").json())
    assert listing.total == 1
    item = listing.items[0]
    assert item.kind == "real_benchmark" and item.name == "synthetic_run"
    assert item.has_manifest and item.has_validation and not item.has_test
    detail = ResultDetail.model_validate(client.get("/results/real_benchmark/synthetic_run").json())
    assert detail.manifest is not None
    assert detail.manifest_sha256 and len(detail.manifest_sha256) == 64
    names = {f.name for f in detail.files}
    assert names == {"manifest.json", "validation.json"}
    verdict = VerificationResult.model_validate(
        client.get("/results/real_benchmark/synthetic_run/verification").json()
    )
    assert verdict.verifier == "verify_phase1_run"
    assert isinstance(verdict.valid, bool)


def test_verifier_versions_and_runs(client: TestClient) -> None:
    versions = VerifierVersionListResponse.model_validate(client.get("/verifier/versions").json())
    assert [v.version for v in versions.items] == ["v1", "v2"]
    v1 = VerifierVersionDetail.model_validate(client.get("/verifier/versions/v1").json())
    assert v1.acceptance_markdown.startswith("# acceptance v1")
    runs = VerifierRunListResponse.model_validate(client.get("/verifier/runs").json())
    assert runs.total == 2
    by_id = {item.id: item for item in runs.items}
    canonical = by_id["2026-09-16T00-00Z_v1"]
    assert canonical.version == "v1"
    assert canonical.time == "2026-09-16T00-00Z"
    assert canonical.verdict == "PASS"
    detail = VerifierRunDetail.model_validate(
        client.get("/verifier/runs/2026-09-16T00-00Z_v1").json()
    )
    assert "Verdict: PASS" in detail.markdown


def test_artifacts_list_and_detail(client: TestClient) -> None:
    listing = ArtifactListResponse.model_validate(client.get("/artifacts").json())
    ids = {item.id for item in listing.items}
    assert ids == {"top", "nested/inner"}
    detail = ArtifactDetail.model_validate(client.get("/artifacts/nested/inner").json())
    assert detail.artifact == {"ok": True}
    assert detail.path == "nested/inner.json"


# ---------------------------------------------------------------------------
# Errors, containment, auth
# ---------------------------------------------------------------------------


def test_not_found_and_malformed_ids(client: TestClient) -> None:
    assert client.get("/runs/" + "f" * 64).status_code == 404
    assert client.get("/runs/nothex").status_code == 400
    assert client.get("/receipts/ghost").status_code == 404
    assert client.get("/receipts/by-hash/" + "z" * 64).status_code == 400
    assert client.get("/receipts/by-hash/" + "0" * 64).status_code == 404
    assert client.get("/results/real_benchmark/ghost").status_code == 404
    assert client.get("/results/evil..kind/synthetic_run").status_code in (400, 404)
    assert client.get("/verifier/versions/v99").status_code == 404
    assert client.get("/verifier/versions/x1").status_code == 400
    assert client.get("/verifier/runs/ghost_v1").status_code == 404
    assert client.get("/artifacts/ghost").status_code == 404


def test_corrupt_receipt_detail_fails_closed(client: TestClient) -> None:
    resp = client.get("/receipts/corrupt")
    assert resp.status_code == 422


def test_path_traversal_rejected(client: TestClient) -> None:
    for url in (
        "/receipts/..%2F..%2Fetc%2Fpasswd",
        "/artifacts/..%2F..%2Fetc%2Fpasswd",
        "/artifacts/nested%2F..%2F..%2Fsecret",
        "/verifier/runs/..%2Fv1%2Facceptance",
    ):
        resp = client.get(url)
        assert resp.status_code != 200, url
        assert resp.status_code in (400, 404, 422)


def test_remote_client_refused_without_key(settings: ResearchApiSettings) -> None:
    remote = TestClient(create_app(settings), client=("203.0.113.7", 9))
    assert remote.get("/health").status_code == 200
    assert remote.get("/receipts").status_code == 403
    assert remote.get("/runs").status_code == 403


def test_api_key_gate(settings: ResearchApiSettings) -> None:
    keyed = ResearchApiSettings(
        data_root=settings.data_root,
        receipts_dir=settings.receipts_dir,
        verifier_dir=settings.verifier_dir,
        artifacts_dir=settings.artifacts_dir,
        api_key="s3cret",
    )
    remote = TestClient(create_app(keyed), client=("203.0.113.7", 9))
    assert remote.get("/health").status_code == 200
    assert remote.get("/receipts").status_code == 401
    assert remote.get("/receipts", headers={"X-API-Key": "wrong"}).status_code == 401
    ok = remote.get("/receipts", headers={"X-API-Key": "s3cret"})
    assert ok.status_code == 200


def test_security_headers(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["Cache-Control"] == "no-store"


# ---------------------------------------------------------------------------
# OpenAPI validity + route-surface audit
# ---------------------------------------------------------------------------


def _resolve_refs(node: object, components: dict, trail: tuple = ()) -> list[str]:
    """Collect unresolvable local ``$ref`` targets in an OpenAPI document."""
    missing: list[str] = []
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            if ref in trail:
                return missing
            if not ref.startswith("#/components/schemas/"):
                missing.append(ref)
            else:
                name = ref[len("#/components/schemas/") :]
                if name not in components:
                    missing.append(ref)
                else:
                    missing.extend(_resolve_refs(components[name], components, trail + (ref,)))
        for value in node.values():
            missing.extend(_resolve_refs(value, components, trail))
    elif isinstance(node, list):
        for item in node:
            missing.extend(_resolve_refs(item, components, trail))
    return missing


def test_openapi_schema_valid(settings: ResearchApiSettings) -> None:
    spec = create_app(settings).openapi()
    assert isinstance(spec, dict)
    assert spec["openapi"].startswith("3.")
    assert spec["info"]["title"] == "dipcatcher research API"
    json.dumps(spec)  # schema must round-trip
    components = spec.get("components", {}).get("schemas", {})
    for path, ops in spec["paths"].items():
        assert set(ops) <= {"get", "parameters"}, path
    assert _resolve_refs(spec["paths"], components) == []


def test_no_trading_or_write_routes(settings: ResearchApiSettings) -> None:
    """The service must expose read-only evidence routes only: every operation
    is GET and no path or operationId mentions order/broker/trade/execution/etc.
    """
    spec = create_app(settings).openapi()
    for path, ops in spec["paths"].items():
        lowered = path.lower()
        for token in FORBIDDEN_ROUTE_TOKENS:
            assert token not in lowered, f"forbidden token {token!r} in route {path}"
        for method, op in ops.items():
            assert method == "get", f"non-GET method {method} on {path}"
            if isinstance(op, dict):
                op_id = str(op.get("operationId", "")).lower()
                for token in FORBIDDEN_ROUTE_TOKENS:
                    assert token not in op_id, f"forbidden token {token!r} in {op_id}"


def test_registered_routes_are_get_only(settings: ResearchApiSettings) -> None:
    app = create_app(settings)
    for route in app.routes:
        methods = getattr(route, "methods", set()) or set()
        assert methods <= {"GET", "HEAD", "OPTIONS"}, getattr(route, "path", route)


# ---------------------------------------------------------------------------
# Property test: arbitrary ids never crash the service
# ---------------------------------------------------------------------------


def test_arbitrary_receipt_ids_never_500(client: TestClient) -> None:
    @given(st.text(max_size=200))
    @h_settings(max_examples=60, suppress_health_check=list(HealthCheck), deadline=None)
    def check(text: str) -> None:
        quoted = urllib.parse.quote(text, safe="")
        for url in (
            f"/receipts/{quoted}",
            f"/artifacts/{quoted}",
            f"/verifier/runs/{quoted}",
        ):
            resp = client.get(url)
            assert resp.status_code != 500, (url, resp.status_code)

    check()


def test_receipt_change_with_preserved_stat_is_not_cached(
    client: TestClient, settings: ResearchApiSettings
) -> None:
    import os

    path = settings.receipts_dir / "clean_bench.json"
    original = path.read_bytes()
    stat = path.stat()
    first = client.get("/receipts/clean_bench").json()
    changed = original.replace(b"0.1", b"0.9")
    assert changed != original and len(changed) == len(original)
    path.write_bytes(changed)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    second = client.get("/receipts/clean_bench").json()
    assert second["sha256"] != first["sha256"]
    assert second["receipt"]["metrics"]["brier"] == 0.9


def test_run_verification_rechecks_markdown_twin(
    client: TestClient, settings: ResearchApiSettings
) -> None:
    route = f"/runs/{_RUN_ID}/verification"
    assert client.get(route).json()["valid"] is True
    markdown = settings.runs_dir / f"{_RUN_ID}.md"
    markdown.write_text("# changed evidence\n")
    response = client.get(route).json()
    assert response["valid"] is False
    assert "immutable_markdown_hash_mismatch" in response["errors"]
    markdown.unlink()
    assert client.get(route).json()["valid"] is False


@pytest.mark.parametrize(
    ("root_name", "relative", "route"),
    [
        ("receipts_dir", "escaped.json", "/receipts"),
        ("artifacts_dir", "escaped.json", "/artifacts"),
        ("data_root", "metadata/research/latest.json", "/runs/latest"),
        ("data_root", f"metadata/research/runs/{'c' * 64}.json", "/runs"),
        ("verifier_dir", "v3/acceptance.md", "/verifier/versions"),
        ("verifier_dir", "runs/escape.md", "/verifier/runs"),
        ("data_root", "metadata/real_benchmark/synthetic_run/manifest.json", "/results"),
        (
            "data_root",
            "metadata/real_benchmark/synthetic_run/extra.json",
            "/results/real_benchmark/synthetic_run",
        ),
        ("data_root", f"metadata/research/runs/{_RUN_ID}.md", f"/runs/{_RUN_ID}/verification"),
    ],
)
def test_enumeration_and_verification_reject_escaping_symlinks(
    client: TestClient,
    settings: ResearchApiSettings,
    tmp_path: Path,
    root_name: str,
    relative: str,
    route: str,
) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text('{"private": "unrelated data"}')
    target = getattr(settings, root_name) / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    try:
        target.symlink_to(outside)
    except OSError:
        pytest.skip("filesystem does not permit symbolic links")
    response = client.get(route)
    assert response.status_code == 400
    assert "unrelated data" not in response.text
