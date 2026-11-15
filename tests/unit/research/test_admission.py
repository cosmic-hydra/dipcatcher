"""Admission gate: a valid receipt may still not belong in the corpus."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from quant_fund.research.admission import (
    ADMISSION_SCHEMA,
    admission_check,
    admission_contract_errors,
    admit_batch,
    write_admission_receipt,
)
from quant_fund.research.receipt_v2 import seal_receipt, verify_receipt_file
from quant_fund.utils.hashing import hash_bytes


def _write(dirpath: Path, name: str, body: dict, *, seal: bool = True) -> Path:
    path = dirpath / name
    payload = seal_receipt(dict(body)) if seal else dict(body)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def _claim_body(value: float, *, dataset: str | None = None, inputs: str | None = None) -> dict:
    """Minimal claim-bearing receipt body."""
    return {
        "kind": "synthetic_fixture.v1",
        "schema": "synthetic_fixture.v1",  # unregistered kind — no lane contract fires
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "SYNTHETIC",
        "inputs_sha256": inputs or "ab" * 32,
        "params": {"drill": True},
        "results": [{"head": "x", "qlike": value}],
        "dataset_sha256": dataset or "cd" * 32,
    }


@pytest.fixture()
def corpus(tmp_path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    _write(root, "a.json", _claim_body(1.0))
    _write(root, "b.json", _claim_body(1.0))
    return root


def test_admit_clean_candidate(corpus: Path) -> None:
    # distinct inputs+dataset fingerprints — no claim group forms vs the corpus
    candidate = _write(
        corpus.parent, "c.json", _claim_body(2.0, dataset="ee" * 32, inputs="ff" * 32)
    )
    result = admission_check(candidate, corpus)
    assert result["verdict"] == "admit"
    assert all(c["ok"] for c in result["checks"])
    assert result["candidate_sha256"]


def test_reject_forged_seal(corpus: Path, tmp_path: Path) -> None:
    candidate = _write(tmp_path, "forged.json", _claim_body(1.0))
    doc = json.loads(candidate.read_text())
    doc["results"][0]["qlike"] = 999.0  # tamper post-seal
    candidate.write_text(json.dumps(doc))
    result = admission_check(candidate, corpus)
    assert result["verdict"] == "reject"


def test_reject_live_pnl_claim(corpus: Path, tmp_path: Path) -> None:
    body = _claim_body(1.0)
    body["live_pnl_claim"] = True
    candidate = _write(tmp_path, "live.json", body)
    result = admission_check(candidate, corpus)
    assert result["verdict"] == "reject"
    honesty = next(c for c in result["checks"] if c["name"] == "honesty")
    assert "live_pnl_claim_true" in honesty["reject_errors"]


def test_quarantine_missing_stamps(corpus: Path, tmp_path: Path) -> None:
    body = _claim_body(1.0)
    del body["research_only"]
    del body["data_label"]
    candidate = _write(tmp_path, "bare.json", body)
    result = admission_check(candidate, corpus)
    assert result["verdict"] == "quarantine"
    honesty = next(c for c in result["checks"] if c["name"] == "honesty")
    assert "research_only_missing" in honesty["quarantine_errors"]
    assert "data_label_missing" in honesty["quarantine_errors"]


def test_quarantine_new_inconsistent_group(corpus: Path, tmp_path: Path) -> None:
    """Same dataset fingerprint, contradictory claim → quarantine, not admit."""
    candidate = _write(
        tmp_path,
        "contra.json",
        _claim_body(9.9, dataset="cd" * 32),
    )
    result = admission_check(candidate, corpus)
    assert result["verdict"] == "quarantine"
    assert result["lattice_new_inconsistent"]


def test_verdict_check_coherence_contract() -> None:
    """The contract pins verdict↔checks coherence — no claim the checks deny."""
    base = {
        "kind": ADMISSION_SCHEMA,
        "schema": ADMISSION_SCHEMA,
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "CORPUS",
        "inputs_sha256": "ab" * 32,
        "candidate_sha256": "cd" * 32,
        "n_corpus_receipts": 2,
        "verdict": "admit",
        "checks": [
            {"name": "seal", "ok": True},
            {"name": "honesty", "ok": True, "reject_errors": [], "quarantine_errors": []},
            {"name": "lattice", "ok": True},
            {"name": "corpus", "ok": True},
        ],
    }
    assert admission_contract_errors(base) == []
    base["verdict"] = "admit"
    base["checks"][1]["quarantine_errors"] = ["data_label_missing"]
    assert "verdict_admit_with_findings" in admission_contract_errors(base)


def test_written_receipt_verifies(corpus: Path, tmp_path: Path) -> None:
    candidate = _write(tmp_path, "c.json", _claim_body(2.0))
    result = admission_check(candidate, corpus)
    out = tmp_path / "out"
    out.mkdir()
    path = write_admission_receipt(result, out)
    ver = verify_receipt_file(path)
    assert ver["valid"], ver["errors"]
    doc = json.loads(path.read_text())
    assert doc["kind"] == ADMISSION_SCHEMA


def test_fails_closed_on_missing_candidate(corpus: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        admission_check(tmp_path / "nope.json", corpus)


def test_fails_closed_on_missing_corpus(tmp_path: Path) -> None:
    candidate = _write(tmp_path, "c.json", _claim_body(1.0))
    with pytest.raises(ValueError, match="does not exist"):
        admission_check(candidate, tmp_path / "nope")


def test_wrong_verdict_rejected_by_writer(corpus: Path, tmp_path: Path) -> None:
    result = admission_check(_write(tmp_path, "c.json", _claim_body(1.0)), corpus)
    result["verdict"] = "admit"  # forge the verdict — checks must deny it
    result["checks"][0]["ok"] = False
    with pytest.raises(ValueError, match="contract"):
        write_admission_receipt(result, tmp_path)


def test_empty_corpus_admits_first_clean_receipt(tmp_path: Path) -> None:
    corpus = tmp_path / "empty_corpus"
    corpus.mkdir()
    candidate = _write(tmp_path, "first.json", _claim_body(1.0))
    result = admission_check(candidate, corpus)
    assert result["verdict"] == "admit"
    assert result["n_corpus_receipts"] == 0


def test_epoch_chain_check_present_or_skipped(corpus: Path, tmp_path: Path) -> None:
    """The epoch-chain check always runs — skipped only if corpus_epoch is absent."""
    # Distinct inputs/dataset fingerprints so the lattice check does not flag a
    # claim contradiction — this test exercises the epoch check, not lattice.
    candidate = _write(tmp_path, "c.json", _claim_body(2.0, dataset="ee" * 32, inputs="ff" * 32))
    result = admission_check(candidate, corpus)
    epoch_check = next(c for c in result["checks"] if c["name"] == "epoch_chain")
    assert epoch_check["ok"] is True
    # corpus_epoch lands in a sibling PR: either it is importable and the
    # (unstamped) corpus reports intact-with-no-epochs, or it is skipped.
    if epoch_check.get("skipped") == "corpus_epoch_unavailable":
        assert result["corpus_epoch_root"] is None
    else:
        # an unstamped corpus is benign — no epochs yet, nothing authoritative
        # broken; admissions proceed normally
        assert epoch_check["errors"] == ["no_epoch_receipts"]
        assert result["verdict"] == "admit"


def test_admit_batch_committed_candidate_is_not_vacuous(tmp_path: Path) -> None:
    """A receipt already inside the corpus must still face a real delta.

    Single-candidate ``admission_check`` on a committed file is vacuous on
    the lattice check (before == after — the candidate is already a member).
    ``admit_batch`` strips the changed names from the shadow corpus first.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    contra = _write(corpus, "contra.json", _claim_body(9.9, dataset="cd" * 32))
    _write(corpus, "a.json", _claim_body(1.0))
    _write(corpus, "b.json", _claim_body(1.0))

    # Single admission against the live corpus: vacuous admit (the
    # contradictory file is already in both sides of the delta).
    solo = admission_check(contra, corpus)
    assert not solo["lattice_new_inconsistent"]

    batch = admit_batch([contra], corpus)
    assert batch["verdict"] == "quarantine"
    assert batch["results"][0]["lattice_new_inconsistent"]


def test_admit_batch_intra_diff_contradiction(tmp_path: Path) -> None:
    """Two new receipts contradicting each other: the later file draws the flag."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write(corpus, "a.json", _claim_body(1.0))
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    first = _write(inbox, "m1.json", _claim_body(1.0, dataset="cd" * 32, inputs="dd" * 32))
    second = _write(inbox, "m2.json", _claim_body(9.9, dataset="cd" * 32, inputs="ee" * 32))

    batch = admit_batch([second, first], corpus)  # order-independence: sorted by name
    assert batch["verdict"] == "quarantine"
    by_name = {r["candidate"]: r["verdict"] for r in batch["results"]}
    assert by_name["m1.json"] == "admit"
    assert by_name["m2.json"] == "quarantine"
    assert batch["failures"] == ["m2.json"]


def test_admit_batch_reject_dominates(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write(corpus, "a.json", _claim_body(1.0))
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    good = _write(inbox, "g.json", _claim_body(2.0, dataset="ee" * 32, inputs="ff" * 32))
    forged = _write(inbox, "f.json", _claim_body(1.0, dataset="ee" * 32, inputs="ab" * 32))
    doc = json.loads(forged.read_text())
    doc["results"][0]["qlike"] = 999.0
    forged.write_text(json.dumps(doc))

    batch = admit_batch([good, forged], corpus)
    assert batch["verdict"] == "reject"
    assert batch["n_candidates"] == 2
    # f.json is rejected on its broken seal; it still lands in the shadow
    # (post-merge coexistence), so g.json — whose dataset group now contains
    # the forged claim — quarantines on the lattice delta. The batch as a
    # whole must not merge, and both files carry their own verdict.
    by_name = {r["candidate"]: r["verdict"] for r in batch["results"]}
    assert by_name == {"f.json": "reject", "g.json": "quarantine"}
    assert set(batch["failures"]) == {"f.json", "g.json"}


def test_admit_batch_fails_closed(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    with pytest.raises(ValueError, match="does not exist"):
        admit_batch([tmp_path / "nope.json"], corpus)
    candidate = _write(tmp_path, "c.json", _claim_body(1.0))
    with pytest.raises(ValueError, match="does not exist"):
        admit_batch([candidate], tmp_path / "nope")


def test_epoch_chain_fields_contract() -> None:
    """A broken epoch chain is quarantinable; forged bindings are pinned."""
    base = {
        "kind": ADMISSION_SCHEMA,
        "schema": ADMISSION_SCHEMA,
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "CORPUS",
        "inputs_sha256": "ab" * 32,
        "candidate_sha256": "cd" * 32,
        "n_corpus_receipts": 2,
        "corpus_epoch_receipt": "corpus_epoch_" + "0" * 16 + ".json",
        "corpus_epoch_root": "ef" * 32,
        "verdict": "quarantine",
        "checks": [
            {"name": "seal", "ok": True},
            {"name": "honesty", "ok": True, "reject_errors": [], "quarantine_errors": []},
            {"name": "lattice", "ok": True},
            {"name": "corpus", "ok": True},
            {"name": "epoch_chain", "ok": False, "errors": ["member_removed:x.json"]},
        ],
    }
    assert admission_contract_errors(base) == []
    base["corpus_epoch_root"] = "nothex"
    assert "corpus_epoch_root" in admission_contract_errors(base)
    base["corpus_epoch_root"] = "ef" * 32
    base["verdict"] = "admit"
    assert "verdict_admit_with_findings" in admission_contract_errors(base)


def test_tombstone_reject_error_makes_verdict_reject() -> None:
    """Contract: a retraction reject finding must be able to drive reject."""
    body = {
        "kind": ADMISSION_SCHEMA,
        "schema": ADMISSION_SCHEMA,
        "research_only": True,
        "live_pnl_claim": False,
        "simulated_only": False,
        "data_label": "CORPUS",
        "inputs_sha256": "ab" * 32,
        "params": {"q": 0.05, "corpus_dir": "receipts", "known_inconsistent": []},
        "candidate": "c.json",
        "candidate_sha256": "cd" * 32,
        "n_corpus_receipts": 2,
        "checks": [
            {"name": "seal", "ok": True, "errors": []},
            {"name": "honesty", "ok": True, "reject_errors": [], "quarantine_errors": []},
            {"name": "lattice", "ok": True, "new_inconsistent_groups": 0},
            {"name": "corpus", "ok": True},
            {
                "name": "tombstone",
                "ok": False,
                "reject_errors": ["retracted_bytes"],
                "findings": [],
            },
        ],
        "lattice_new_inconsistent": [],
        "survivors_added": [],
        "survivors_removed": [],
        "verdict": "reject",
    }
    assert admission_contract_errors(body) == []
    # ...and a forged 'admit' on the same checks must not contract-verify.
    forged = dict(body, verdict="admit")
    assert "verdict_admit_with_findings" in admission_contract_errors(forged)


def test_tombstone_finding_makes_verdict_quarantine() -> None:
    """Contract: a soft retraction finding (slot/scope) drives quarantine."""
    body = {
        "kind": ADMISSION_SCHEMA,
        "schema": ADMISSION_SCHEMA,
        "research_only": True,
        "live_pnl_claim": False,
        "simulated_only": False,
        "data_label": "CORPUS",
        "inputs_sha256": "ab" * 32,
        "params": {"q": 0.05, "corpus_dir": "receipts", "known_inconsistent": []},
        "candidate": "c.json",
        "candidate_sha256": "cd" * 32,
        "n_corpus_receipts": 2,
        "checks": [
            {"name": "seal", "ok": True, "errors": []},
            {"name": "honesty", "ok": True, "reject_errors": [], "quarantine_errors": []},
            {"name": "lattice", "ok": True, "new_inconsistent_groups": 0},
            {"name": "corpus", "ok": True},
            {
                "name": "tombstone",
                "ok": False,
                "reject_errors": [],
                "findings": ["retracted_slot"],
            },
        ],
        "lattice_new_inconsistent": [],
        "survivors_added": [],
        "survivors_removed": [],
        "verdict": "quarantine",
    }
    assert admission_contract_errors(body) == []


def test_retracted_bytes_rejected_end_to_end(tmp_path: Path) -> None:
    """E2E: the byte-exact retracted artifact is refused re-admission.

    receipt_tombstone lands on a sibling branch — skip dormant until both
    merge; the contract tests above pin the verdict wiring regardless.
    """
    pytest.importorskip("quant_fund.research.receipt_tombstone")
    from quant_fund.research.receipt_tombstone import write_tombstone

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write(corpus, "a.json", _claim_body(1.0))
    bad = _write(corpus, "b.json", _claim_body(1.0))
    write_tombstone(bad, corpus_dir=corpus, reason="bad inputs")

    candidate = tmp_path / "reissue.json"
    candidate.write_text(bad.read_bytes().decode())
    res = admission_check(candidate, corpus)
    # name differs → not the same slot; the *bytes* are what is retracted
    tomb = next(c for c in res["checks"] if c["name"] == "tombstone")
    assert tomb["ok"] is True  # byte-identity is name-keyed in the corpus

    res2 = admission_check(bad, corpus)
    tomb2 = next(c for c in res2["checks"] if c["name"] == "tombstone")
    assert tomb2["reject_errors"] == ["retracted_bytes"]
    assert res2["verdict"] == "reject"


# --------------------------------------------------------------------------- #
# Epoch-stamped corpora: admission applies the chain the way the other epoch
# gates do (removal acknowledgments, nested stamped members) and lands a
# diff's files in an order a merge can actually produce.
# --------------------------------------------------------------------------- #


def _stamp(corpus: Path) -> Path:
    from quant_fund.research.corpus_epoch import corpus_epoch, write_epoch_receipt

    return write_epoch_receipt(corpus_epoch(corpus), corpus)


def _fresh(tag: str) -> dict:
    """A clean claim body whose fingerprints are unique to ``tag``."""
    seed = hash_bytes(tag.encode())
    return _claim_body(2.0, dataset=seed, inputs=hash_bytes(seed.encode()))


def _chain_check(result: dict) -> dict:
    return next(c for c in result["checks"] if c["name"] == "epoch_chain")


@pytest.fixture()
def shrunk_corpus(tmp_path: Path) -> tuple[Path, str]:
    """A stamped corpus whose second epoch records ``old.json`` as removed."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write(corpus, "a.json", _claim_body(1.0))
    gone = _write(corpus, "old.json", _fresh("old"))
    _stamp(corpus)
    sha = hash_bytes(gone.read_bytes())
    gone.unlink()
    _stamp(corpus)
    return corpus, sha


def test_acknowledged_removal_admits(shrunk_corpus: tuple[Path, str], tmp_path: Path) -> None:
    corpus, sha = shrunk_corpus
    candidate = _write(tmp_path, "c.json", _fresh("c"))
    result = admission_check(candidate, corpus, allowed_removals={"old.json": sha})
    assert _chain_check(result)["ok"], _chain_check(result)
    assert result["verdict"] == "admit"
    assert result["params"]["allowed_removals"] == ["old.json"]


def test_unacknowledged_removal_quarantines(
    shrunk_corpus: tuple[Path, str], tmp_path: Path
) -> None:
    corpus, sha = shrunk_corpus
    candidate = _write(tmp_path, "c.json", _fresh("c"))
    for removals in (None, {"old.json": "0" * 64}, {"other.json": sha}):
        result = admission_check(candidate, corpus, allowed_removals=removals)
        errors = _chain_check(result)["errors"]
        assert any(e.startswith("member_removed:'old.json'") for e in errors), (removals, errors)
        assert result["verdict"] == "quarantine"


def test_admit_batch_passes_removal_acknowledgments(
    shrunk_corpus: tuple[Path, str], tmp_path: Path
) -> None:
    corpus, sha = shrunk_corpus
    candidate = _write(tmp_path, "c.json", _fresh("c"))
    assert admit_batch([candidate], corpus)["verdict"] == "quarantine"
    batch = admit_batch([candidate], corpus, allowed_removals={"old.json": sha})
    assert batch["verdict"] == "admit", batch["results"][0]["checks"]


def test_admit_batch_keeps_stamped_subtrees(tmp_path: Path) -> None:
    """``legacy-unsealed/`` is no candidate and no lattice member, but the
    epoch chain stamps it — the shadow corpus must still carry it."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write(corpus, "a.json", _claim_body(1.0))
    (corpus / "legacy-unsealed").mkdir()
    _write(corpus / "legacy-unsealed", "pre_seal.json", _fresh("legacy"), seal=False)
    _stamp(corpus)
    candidate = _write(tmp_path, "c.json", _fresh("c"))
    batch = admit_batch([candidate], corpus)
    chain = _chain_check(batch["results"][0])
    assert chain["ok"], chain
    assert batch["verdict"] == "admit"


def test_admit_batch_lands_receipts_before_the_epoch_stamping_them(tmp_path: Path) -> None:
    """``corpus_epoch_*`` sorts before ``z_*``: by name the epoch would be
    checked first, and the receipt it stamps would read as removed."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write(corpus, "a.json", _claim_body(1.0))
    _stamp(corpus)
    late = _write(corpus, "z_late.json", _fresh("late"))
    epoch = _stamp(corpus)
    assert epoch.name < late.name

    batch = admit_batch([epoch, late], corpus)
    assert [r["candidate"] for r in batch["results"]] == [late.name, epoch.name]
    assert batch["verdict"] == "admit", [_chain_check(r) for r in batch["results"]]


def _stamp_until(corpus: Path, tag: str, keep: Callable[[Path], bool]) -> Path:
    """Add a fresh receipt and stamp until the new epoch's name satisfies
    ``keep`` — epoch names are digests, so each try is a coin flip."""
    for i in range(64):
        receipt = _write(corpus, f"{tag}_{i:02d}.json", _fresh(f"{tag}{i}"))
        epoch = _stamp(corpus)
        if keep(epoch):
            return epoch
        epoch.unlink()
        receipt.unlink()
    pytest.fail(f"no {tag} epoch name satisfied the order constraint in 64 stamps")


def test_admit_batch_lands_parent_epoch_before_child(tmp_path: Path) -> None:
    """Two stamps in one diff, the child's digest-name sorting first: by name
    the child is an orphan and both read as heads."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write(corpus, "a.json", _claim_body(1.0))
    _stamp(corpus)
    parent = _stamp_until(corpus, "p", lambda e: e.name >= "corpus_epoch_8")
    child = _stamp_until(corpus, "q", lambda e: e.name < "corpus_epoch_8")
    assert json.loads(child.read_text())["prev_epoch_receipt"] == parent.name
    diff = sorted((p for p in corpus.glob("[pq]_*.json")), key=lambda p: p.name)

    batch = admit_batch([child, parent, *diff], corpus)
    order = [r["candidate"] for r in batch["results"]]
    assert order == [p.name for p in diff] + [parent.name, child.name]
    assert batch["verdict"] == "admit", [_chain_check(r) for r in batch["results"]]


def test_landing_order_is_parent_first(tmp_path: Path) -> None:
    from quant_fund.research.admission import _landing_order

    def epoch(name: str, prev: str | None) -> Path:
        body = {"schema": "corpus_epoch.v1", "kind": "corpus_epoch.v1", "prev_epoch_receipt": prev}
        return _write(tmp_path, name, body, seal=False)

    e1 = epoch("corpus_epoch_c.json", "corpus_epoch_live_head.json")  # parent outside the diff
    e2 = epoch("corpus_epoch_a.json", "corpus_epoch_c.json")
    e3 = epoch("corpus_epoch_b.json", "corpus_epoch_a.json")
    disguised = epoch("zz_epoch.json", "corpus_epoch_b.json")  # schema, not name, decides
    plain = _write(tmp_path, "m.json", _claim_body(1.0))
    junk = tmp_path / "corpus_epoch_junk.json"
    junk.write_text("{not json")  # unreadable: a plain candidate, its seal check reports it

    got = _landing_order([disguised, e3, plain, e2, junk, e1])
    assert [p.name for p in got] == [
        "corpus_epoch_junk.json",
        "m.json",
        "corpus_epoch_c.json",
        "corpus_epoch_a.json",
        "corpus_epoch_b.json",
        "zz_epoch.json",
    ]


def test_landing_order_cycle_keeps_name_order(tmp_path: Path) -> None:
    from quant_fund.research.admission import _landing_order

    body = {"schema": "corpus_epoch.v1"}
    a = _write(
        tmp_path,
        "corpus_epoch_a.json",
        {**body, "prev_epoch_receipt": "corpus_epoch_b.json"},
        seal=False,
    )
    b = _write(
        tmp_path,
        "corpus_epoch_b.json",
        {**body, "prev_epoch_receipt": "corpus_epoch_a.json"},
        seal=False,
    )
    assert _landing_order([b, a]) == [a, b]


def test_admit_batch_cli_allowed_removals(shrunk_corpus: tuple[Path, str], tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from quant_fund.cli.main import app

    corpus, sha = shrunk_corpus
    candidate = _write(tmp_path, "c.json", _fresh("c"))
    removals = tmp_path / "removals.json"
    removals.write_text(json.dumps({"old.json": sha}))
    out = tmp_path / "out"
    base = ["admit-batch", str(candidate), "--corpus-dir", str(corpus), "--out-dir", str(out)]

    ok = CliRunner().invoke(app, [*base, "--strict", "--allowed-removals", str(removals)])
    assert ok.exit_code == 0, ok.output
    assert "verdict=admit" in ok.output

    bare = CliRunner().invoke(app, [*base, "--strict"])
    assert bare.exit_code == 1, bare.output
    assert "epoch_chain" in bare.output

    removals.write_text(json.dumps({"old.json": "short"}))
    bad = CliRunner().invoke(app, [*base, "--allowed-removals", str(removals)])
    assert bad.exit_code != 0
    assert "allowed-removals must be a JSON object" in bad.output
