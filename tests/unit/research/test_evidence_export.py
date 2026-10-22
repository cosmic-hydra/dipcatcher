"""Evidence-export: the producer half of ``verify-repo --evidence-only``.

The strongest test uses the live repo itself: export the real evidence
store and verify the bundle — if the export's member set diverges from the
pinned epoch state (a missing member, a stray extra file), the bundle
fails its own gates.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from quant_fund.research.evidence_export import (
    BUNDLE_MANIFEST,
    EVIDENCE_BUNDLE_SCHEMA,
    export_evidence_bundle,
)
from quant_fund.research.repo_integrity import EVIDENCE_CORPORA, verify_repo

REPO_ROOT = Path(__file__).resolve().parents[3]


def _has_git(root: Path) -> bool:
    try:
        subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


_HAS_REAL_TREE = (REPO_ROOT / "quality" / "epoch_heads.json").is_file() and _has_git(REPO_ROOT)


@pytest.fixture(scope="session")
def real_evidence_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One live-tree export per xdist worker.

    Each real-tree test used to call ``export_evidence_bundle(REPO_ROOT, …)``
    independently (~140MB, dominated by ``.dsh-24x7``). Concurrent copies under
    ``-n auto`` on a tight disk race native allocators into segfault / xdist
    ``node down``. Sharing one export per worker keeps coverage identical while
    cutting peak tmp by ~3× for this file.
    """
    if not _HAS_REAL_TREE:
        pytest.skip("requires the real repo tree with committed integrity state")
    out = tmp_path_factory.mktemp("real_evidence_bundle")
    export_evidence_bundle(REPO_ROOT, out)
    return out


def _minimal_evidence_tree(tmp_path: Path) -> Path:
    """A tree with just the evidence corpora + signature file."""
    from quant_fund.research.corpus_epoch import (
        corpus_epoch,
        update_heads_pin,
        write_epoch_receipt,
    )
    from quant_fund.research.repo_integrity import CORPORA

    root = tmp_path / "repo"
    pin = root / "quality" / "epoch_heads.json"
    # Stamp quality last: epoch_heads.json is itself a quality member, so the
    # quality epoch must see the final pin.
    for corpus_dir in sorted(EVIDENCE_CORPORA - {"quality"}) + ["quality"]:
        d = root / corpus_dir
        d.mkdir(parents=True, exist_ok=True)
        pattern = next(c[1] for c in CORPORA if c[0] == corpus_dir)
        ext = pattern.lstrip("*") or ".json"
        (d / f"evidence_0{ext}").write_text('{"n": 0}\n', encoding="utf-8")
        ep = write_epoch_receipt(corpus_epoch(d, pattern=pattern), d)
        update_heads_pin(pin, corpus_dir, pattern, ep)
    (root / "gate_pins.sig").write_text("sig-bytes\n", encoding="utf-8")
    return root


def test_export_reproduces_pinned_member_set(tmp_path: Path) -> None:
    root = _minimal_evidence_tree(tmp_path)
    manifest = export_evidence_bundle(root, tmp_path / "bundle")
    assert manifest["schema"] == EVIDENCE_BUNDLE_SCHEMA
    bundle = tmp_path / "bundle"
    for corpus_dir in EVIDENCE_CORPORA:
        assert (bundle / corpus_dir).is_dir(), corpus_dir
        # member count: the file + its epoch receipt only when the glob admits .json
        assert manifest["corpora"][corpus_dir]["members"] >= 1
    assert (bundle / "gate_pins.sig").is_file()
    assert (bundle / BUNDLE_MANIFEST).is_file()
    # The bundle's epoch chains re-derive the same heads — verify the corpus
    # state directly rather than through the full gate set (no crown jewels
    # or signature infrastructure exist in the synthetic tree).
    from quant_fund.research.corpus_epoch import member_digests
    from quant_fund.research.repo_integrity import CORPORA

    for corpus_dir in EVIDENCE_CORPORA:
        pattern = next(c[1] for c in CORPORA if c[0] == corpus_dir)
        assert member_digests(bundle / corpus_dir, pattern=pattern) == member_digests(
            root / corpus_dir, pattern=pattern
        )


def test_export_refuses_nonempty_target(tmp_path: Path) -> None:
    root = _minimal_evidence_tree(tmp_path)
    out = tmp_path / "bundle"
    out.mkdir()
    (out / "stale.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        export_evidence_bundle(root, out)


def test_export_fails_closed_on_missing_corpus(tmp_path: Path) -> None:
    root = _minimal_evidence_tree(tmp_path)
    shutil.rmtree(root / sorted(EVIDENCE_CORPORA)[0])
    with pytest.raises(FileNotFoundError, match="corpus missing"):
        export_evidence_bundle(root, tmp_path / "bundle")


def test_export_manifest_records_source_revision(tmp_path: Path) -> None:
    root = _minimal_evidence_tree(tmp_path)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x"],
        cwd=root,
        check=True,
    )
    manifest = export_evidence_bundle(root, tmp_path / "bundle")
    assert len(manifest["source_revision"]) == 40


@pytest.mark.skipif(
    not _HAS_REAL_TREE, reason="requires the real repo tree with committed integrity state"
)
def test_real_tree_export_verifies_evidence_only(real_evidence_bundle: Path) -> None:
    """Export the live repo's evidence store and run the real gate set on it:
    every epoch chain must re-derive its pinned head; crown_jewels reports
    the skipped marker; signature/witness gates verify offline."""
    import json

    manifest = json.loads((real_evidence_bundle / BUNDLE_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["source_revision"]

    result = verify_repo(real_evidence_bundle, evidence_only=True)
    fails = {
        name: gate["errors"]
        for name, gate in result["gates"].items()
        if not gate["ok"] and not gate.get("skipped")
    }
    assert fails == {}, fails
    assert result["gates"]["crown_jewels"].get("skipped") == "evidence_only"


@pytest.mark.skipif(
    not _HAS_REAL_TREE, reason="requires the real repo tree with committed integrity state"
)
def test_manifest_git_blob_ids_match_ls_tree(real_evidence_bundle: Path) -> None:
    """The manifest's per-member git blob ids must equal what git itself
    records at HEAD — the bundle is thereby bound to the commit's object
    database, not just to a directory that happened to be present."""
    import json

    manifest = json.loads((real_evidence_bundle / BUNDLE_MANIFEST).read_text(encoding="utf-8"))
    committed = manifest["corpora"]["receipts"]["git_members"]
    out = subprocess.run(
        ["git", "ls-tree", "-r", "HEAD", "--", "receipts"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    ls_tree = {
        line.split("\t", 1)[1].removeprefix("receipts/"): line.split()[2]
        for line in out.splitlines()
    }
    common = committed.keys() & ls_tree.keys()
    assert common, "no committed receipts overlapped the manifest"
    for rel in common:
        assert committed[rel] == ls_tree[rel], f"blob sha1 drift: {rel}"


@pytest.mark.skipif(
    not _HAS_REAL_TREE, reason="requires the real repo tree with committed integrity state"
)
def test_corrupted_bundle_member_fails_verification(
    real_evidence_bundle: Path, tmp_path: Path
) -> None:
    """A bundle with a tampered member must not verify — the export is only
    as trustworthy as the gates that re-check it."""
    bundle = tmp_path / "bundle"
    shutil.copytree(real_evidence_bundle, bundle)
    victim = bundle / "receipts"
    target = next(p for p in victim.glob("*.json") if not p.name.startswith("corpus_epoch_"))
    target.write_bytes(b'{"forged": true}\n')
    result = verify_repo(bundle, evidence_only=True)
    epoch_gate = result["gates"]["epoch:receipts"]
    assert not epoch_gate["ok"]
    assert not result["ok"]


BUNDLE_SCRIPT = REPO_ROOT / "scripts" / "verify_evidence_bundle.py"


def test_standalone_auditor_verifies_minimal_bundle(tmp_path: Path) -> None:
    """``scripts/verify_evidence_bundle.py`` — the zero-dependency third-party
    path — must reach the same verdict as ``verify-repo --evidence-only``."""
    import sys

    root = _minimal_evidence_tree(tmp_path)
    bundle = tmp_path / "bundle"
    export_evidence_bundle(root, bundle)
    proc = subprocess.run(
        [sys.executable, str(BUNDLE_SCRIPT), "--root", str(bundle)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_standalone_auditor_fails_closed_on_tamper(tmp_path: Path) -> None:
    """A forged bundle member fails the standalone chain audit."""
    import sys

    root = _minimal_evidence_tree(tmp_path)
    bundle = tmp_path / "bundle"
    export_evidence_bundle(root, bundle)
    victim = next(
        p for p in (bundle / "receipts").glob("*.json") if not p.name.startswith("corpus_epoch_")
    )
    victim.write_text('{"forged": true}\n', encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(BUNDLE_SCRIPT), "--root", str(bundle)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert "receipts" in proc.stdout
