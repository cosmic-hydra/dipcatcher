from __future__ import annotations

import json
from pathlib import Path

from quant_fund.research.crown_jewels import DEFAULT_JEWELS
from quant_fund.research.receipt_v2 import verify_receipt_file
from quant_fund.research.repo_integrity import (
    CORPORA,
    REPO_INTEGRITY_SCHEMA,
    verify_repo,
    write_repo_integrity_receipt,
)


def _git_repo(tmp_path: Path) -> Path:
    """Minimal tree satisfying every gate: the derived jewel set + 8 stamped corpora."""
    import subprocess

    from quant_fund.research.corpus_epoch import (
        HEADS_PIN_BASENAME,
        corpus_epoch,
        update_heads_pin,
        write_epoch_receipt,
    )
    from quant_fund.research.crown_jewels import (
        DEFAULT_JEWELS,
        write_crown_jewels_pin,
    )

    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    for jewel in DEFAULT_JEWELS:
        p = root / jewel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"# {jewel}\n")
    pin = root / "quality" / HEADS_PIN_BASENAME
    pin.parent.mkdir(parents=True, exist_ok=True)
    pin.write_text(json.dumps({"schema": "epoch_heads.v1", "heads": {}}))
    for corpus_dir, pattern, *_ in CORPORA:
        d = root / corpus_dir
        d.mkdir(parents=True, exist_ok=True)
        seed = f"seed.{pattern[2:]}" if len(pattern) > 1 else "seed"
        # receipts/ feeds the lattice gate — its member must parse as JSON
        body = json.dumps({"kind": "seed.v1"}) if corpus_dir == "receipts" else "seed"
        (d / seed).write_text(body)
        ep = write_epoch_receipt(corpus_epoch(d, pattern=pattern), d)
        update_heads_pin(pin, corpus_dir, pattern, ep)
    write_crown_jewels_pin(root)
    # Re-stamp quality: the pin and jewels file now exist as members.
    ep = write_epoch_receipt(corpus_epoch(root / "quality"), root / "quality")
    update_heads_pin(pin, "quality", "*.json", ep)
    # The committed pubkey is a crown jewel — sign the pins for real so the
    # gate isn't testing a stub. Priv key is test-local only.
    from quant_fund.research.gate_signatures import generate_keypair, sign_pins

    priv, pub = generate_keypair()
    sign_pins(root, priv, pub)
    write_crown_jewels_pin(root)
    ep = write_epoch_receipt(corpus_epoch(root / "quality"), root / "quality")
    update_heads_pin(pin, "quality", "*.json", ep)
    sign_pins(root, priv, pub)
    return root


def test_verify_repo_all_gates_green(tmp_path: Path) -> None:
    root = _git_repo(tmp_path)
    res = verify_repo(root)
    assert res["ok"], res
    # Fixture signs the pins for real (the pubkey is a jewel, so an unsigned
    # fixture would leave a stub pubkey failing the gate).
    assert res["gates"]["pin_signatures"] == {"ok": True, "signed": True, "errors": []}
    # No timestamp anchors committed — neutral gate state, same contract as
    # unsigned pins: absence is fine, a malformed one must fail closed.
    assert res["gates"]["timestamp_anchors"]["anchored"] is False
    assert res["gates"]["timestamp_anchors"]["ok"] is True
    # No checkpoint committed — same neutral contract as unsigned pins.
    assert res["gates"]["checkpoint"]["signed"] is False
    assert res["gates"]["checkpoint"]["ok"] is True
    # No witness proofs committed — the public-log layer is optional.
    assert res["gates"]["witness"]["witnessed"] is False
    assert res["gates"]["witness"]["ok"] is True
    # No checkpoint spine or key rotations — both neutral-absent.
    assert res["gates"]["spine"]["ok"] is True
    assert res["gates"]["key_rotation"]["ok"] is True
    assert res["gates"]["key_rotation"]["n_rotations"] == 0
    # No quorum registry or rotations — same neutral-absent contract.
    assert res["gates"]["quorum_rotations"]["ok"] is True
    assert res["gates"]["quorum_rotations"]["n_rotations"] == 0
    assert set(res["gates"]) == {
        "crown_jewels",
        "pin_signatures",
        "timestamp_anchors",
        "checkpoint",
        "witness",
        "spine",
        "key_rotation",
        "quorum_rotations",
        "lattice",
        *(f"epoch:{c[0]}" for c in CORPORA),
    }


def test_verify_repo_fails_on_any_gate(tmp_path: Path) -> None:
    root = _git_repo(tmp_path)
    (root / "pyproject.toml").write_text("tampered")
    res = verify_repo(root)
    assert not res["ok"]
    assert res["gates"]["crown_jewels"]["errors"] == ["jewel_mutated:pyproject.toml"]
    # Epochs untouched -> other gates still report clean.
    assert res["gates"]["epoch:receipts"]["ok"]


def test_repo_integrity_receipt_seals_and_verifies(tmp_path: Path) -> None:
    root = _git_repo(tmp_path)
    out = root / "quality" / "repo_integrity.json"
    write_repo_integrity_receipt(out, root)
    verdict = verify_receipt_file(out)
    assert verdict["valid"], verdict["errors"]
    body = json.loads(out.read_text())
    assert body["schema"] == REPO_INTEGRITY_SCHEMA
    assert body["ok"] is True
    assert body["pins"]["epoch_heads_sha256"]
    assert body["pins"]["crown_jewels_sha256"]


def test_verify_repo_missing_pin_fails_closed(tmp_path: Path) -> None:
    root = _git_repo(tmp_path)
    (root / "quality" / "epoch_heads.json").unlink()
    res = verify_repo(root)
    assert not res["ok"]
    # Quality chain loses its member + the pin is gone: jewel gate fine,
    # epoch gates fail closed.
    assert res["gates"]["crown_jewels"]["ok"]
    assert not res["gates"]["epoch:receipts"]["ok"]
    assert not res["gates"]["epoch:quality"]["ok"]


def test_verify_repo_local_only_corpus_skips_on_clone(tmp_path: Path) -> None:
    """data/metadata's epoch receipts are gitignored: a clone has the members
    but no chain. The gate must report an explicit skip, not fail closed on a
    checkout, and not silently pass — a machine with receipts verifies fully
    (covered by the all-green fixture)."""
    root = _git_repo(tmp_path)
    for f in (root / "data" / "metadata").glob("corpus_epoch_*.json"):
        f.unlink()
    res = verify_repo(root)
    gate = res["gates"]["epoch:data/metadata"]
    assert gate == {"ok": True, "skipped": "local_only_no_receipts", "errors": []}
    assert res["ok"], res


def test_repo_integrity_contract_clean_fixture(tmp_path: Path) -> None:
    """The contract accepts a fresh attestation — verify-receipt calls it."""
    from quant_fund.research.repo_integrity import (
        repo_integrity_contract_errors,
        repo_integrity_receipt,
    )

    root = _git_repo(tmp_path)
    body = repo_integrity_receipt(root)
    assert repo_integrity_contract_errors(body) == []
    # verify-receipt must dispatch to the contract — this is the bound that
    # makes a forged verdict catchable instead of structure-only.
    out = root / "quality" / "repo_integrity.json"
    write_repo_integrity_receipt(out, root)
    assert verify_receipt_file(out)["valid"]


def test_repo_integrity_contract_catches_forged_ok(tmp_path: Path) -> None:
    from quant_fund.research.repo_integrity import (
        repo_integrity_contract_errors,
        repo_integrity_receipt,
    )

    root = _git_repo(tmp_path)
    body = repo_integrity_receipt(root)
    # Forge: claim ok while a gate lists its failures.
    forged = dict(body)
    gates = dict(body["gates"])
    gates["crown_jewels"] = {"ok": False, "errors": ["jewel_mutated:x"]}
    forged["gates"] = gates
    forged["ok"] = True  # the lie: top-level ok despite a failed gate
    errs = repo_integrity_contract_errors(forged)
    assert "ok_incoherent" in errs
    # Also incoherent: gate ok=True while listing errors.
    forged2 = dict(body)
    g2 = dict(body["gates"])
    g2["epoch:receipts"] = {"ok": True, "errors": ["member_removed:'x.json'"]}
    forged2["gates"] = g2
    assert "gate_ok_incoherent:epoch:receipts" in repo_integrity_contract_errors(forged2)
    # Hidden surface: drop a required gate, keep ok=true.
    forged3 = dict(body)
    g3 = dict(body["gates"])
    del g3["pin_signatures"]
    forged3["gates"] = g3
    errs3 = repo_integrity_contract_errors(forged3)
    assert "gate_missing:pin_signatures" in errs3
    # Pin coherence: attestation can't disagree with its own gate.
    forged4 = dict(body)
    forged4["pins"] = dict(body["pins"], gate_pins_signed=not body["pins"]["gate_pins_signed"])
    assert "pins_sig_incoherent" in repo_integrity_contract_errors(forged4)


def test_verify_repo_evidence_only_bundle(tmp_path: Path) -> None:
    """An evidence bundle (5 evidence dirs + gate_pins.sig, no src/, no
    .git) verifies under evidence_only and honestly skips crown_jewels."""
    full = _git_repo(tmp_path)
    bundle = tmp_path / "bundle"
    from quant_fund.research.evidence_export import export_evidence_bundle

    export_evidence_bundle(full, bundle)

    res = verify_repo(bundle, evidence_only=True)
    assert res["ok"], res
    assert res["gates"]["crown_jewels"] == {
        "ok": True,
        "errors": [],
        "skipped": "evidence_only",
    }

    # Same bundle under the default mode fails closed — the bundle is not a
    # full tree and must not claim one.
    res_full = verify_repo(bundle)
    assert not res_full["ok"]
    assert res_full["gates"]["crown_jewels"]["errors"]


def test_repo_integrity_receipt_records_evidence_only_mode(tmp_path: Path) -> None:
    import shutil

    from quant_fund.research.repo_integrity import (
        repo_integrity_contract_errors,
        repo_integrity_receipt,
    )

    full = _git_repo(tmp_path)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for d in (
        "receipts",
        "verifier",
        "quality",
        "configs",
        "artifacts",
        ".dsh-24x7",
        "data/metadata",
    ):
        shutil.copytree(full / d, bundle / d)
    shutil.copytree(full / ".github", bundle / ".github")
    shutil.copy2(full / "gate_pins.sig", bundle / "gate_pins.sig")

    receipt = repo_integrity_receipt(bundle, evidence_only=True)
    assert receipt["mode"] == "evidence_only"
    assert repo_integrity_contract_errors(receipt) == []

    # A forged attestation can't relabel itself as a full-tree verdict —
    # the skipped crown_jewels gate betrays it.
    forged = dict(receipt, mode="full")
    assert "gate_skipped_in_full_mode:crown_jewels" in repo_integrity_contract_errors(forged)

    # Nor can an evidence_only attestation drop the skip marker.
    forged2 = json.loads(json.dumps(receipt))
    del forged2["gates"]["crown_jewels"]["skipped"]
    assert "evidence_only_skip_missing:crown_jewels" in repo_integrity_contract_errors(forged2)


REPO_ROOT = Path(__file__).resolve().parents[3]

# Tool-local state with no security surface — deliberately NOT corpora.
# data/ is absent from this list only because data/metadata IS a corpus.
# .mavis: agent plan-run notes, same class as .serena / .freebuff.
# attic: quarantine of non-qualifying template copies. attic/README.md
# states nothing there is imported, tested, packaged, or evidence.
# packaging: homebrew tap stub (sha256 FILL_AT_RELEASE). It is not an
# install path this repo executes; docker/ and deploy/ remain the
# supply-chain corpora.
NON_CORPUS_TOP_DIRS = frozenset({".freebuff", ".serena", ".mavis", "attic", "packaging"})

# The pinned set of tracked root-level files. Root files sit above every
# corpus dir, so the epoch chains can't see them — a new root file lands
# with zero integrity coverage. The pin forces the decision: jewel it,
# or document why it is inert here.
ROOT_FILES = frozenset(
    {
        ".dockerignore",
        ".env.example",
        ".gitattributes",
        ".gitignore",
        ".gitleaks.toml",
        ".pre-commit-config.yaml",
        ".python-version",
        ".test_durations",
        "AGENTS.md",
        "APPLY.md",
        "CHANGELOG.md",
        "CITATION.cff",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "Dockerfile",
        "Executive Summary.pdf",
        "FULL_COVERAGE.md",
        "HONESTY_RATING.md",
        "INFLIGHT",
        "LICENSE",
        "MATH_SPEC.md",
        "Makefile",
        "README.md",
        "RESEARCH_REFERENCES.md",
        "SECURITY.md",
        "conftest.py",
        "day_grind_progress.md",
        "docker-compose.yml",
        "fx1_seed_corpus.jsonl",
        "install.sh",
        "p",
        "gate_pins.sig",
        "mkdocs.yml",
        "pyproject.toml",
        "uv.lock",
    }
)


def _tracked_files() -> list[str]:
    import subprocess

    proc = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return proc.stdout.splitlines()


def test_closed_world_every_tracked_top_dir_is_a_corpus() -> None:
    """Every tracked top-level directory is epoch-chained or explicitly
    ruled out — a new dir that dodges both is silent uncovered territory."""
    tracked = _tracked_files()
    top_dirs = {p.split("/")[0] for p in tracked if "/" in p}
    corpus_tops = {c[0].split("/")[0] for c in CORPORA}
    assert top_dirs <= corpus_tops | NON_CORPUS_TOP_DIRS, (
        f"uncovered top-level dirs: {sorted(top_dirs - corpus_tops - NON_CORPUS_TOP_DIRS)}"
    )


def test_closed_world_root_files_pinned() -> None:
    """The root-level file set is pinned — files above the corpus dirs get
    no epoch coverage, so each must be deliberate (jewel or inert)."""
    tracked = _tracked_files()
    root_files = {p for p in tracked if "/" not in p}
    assert root_files == ROOT_FILES, (
        f"root file drift: +{sorted(root_files - ROOT_FILES)} -{sorted(ROOT_FILES - root_files)}"
    )


# The one root file that must NEVER be a crown jewel: the signature authenticates
# the pin manifest, so pinning its bytes inside that manifest is a self-reference
# (every re-sign would instantly stale its own jewel). Its integrity gate is the
# Ed25519 signature check itself.
SIGNATURE_FILE = "gate_pins.sig"


def test_closed_world_root_files_are_crown_jewels() -> None:
    """Name-pinning alone leaves content unpinned: a swapped pyproject dep or
    a weakened Makefile gate would change no name. Every root file is a
    crown jewel — the only content gate that reaches above the corpus dirs."""
    assert ROOT_FILES - {SIGNATURE_FILE} <= set(DEFAULT_JEWELS), (
        f"unpinned root files: {sorted(ROOT_FILES - {SIGNATURE_FILE} - set(DEFAULT_JEWELS))}"
    )
    assert SIGNATURE_FILE not in DEFAULT_JEWELS, (
        "gate_pins.sig must not be jewel-pinned — it is re-minted on every sign"
    )
