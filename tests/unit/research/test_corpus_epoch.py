"""Corpus epochs: hash-chained membership root makes the store tamper-evident."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_fund.research.corpus_epoch import (
    GENESIS_PREV,
    EpochStampLocked,
    _acquire_stamp_lock,
    check_epoch_chain,
    corpus_epoch,
    epoch_contract_errors,
    epoch_root,
    member_digests,
    write_epoch_receipt,
)
from quant_fund.research.receipt_v2 import seal_receipt, verify_receipt_file


def _receipt(dirpath: Path, name: str, marker: str) -> Path:
    body = {
        "kind": "synthetic_fixture.v1",
        "schema": "synthetic_fixture.v1",
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "SYNTHETIC",
        "inputs_sha256": "ab" * 32,
        "results": [{"marker": marker}],
    }
    path = dirpath / name
    path.write_text(json.dumps(seal_receipt(body), indent=2, sort_keys=True))
    return path


def _stamp(corpus: Path) -> Path:
    return write_epoch_receipt(corpus_epoch(corpus), corpus)


def test_genesis_epoch(corpus_dir: Path) -> None:
    receipt = corpus_epoch(corpus_dir)
    assert receipt["verdict"] == "genesis"
    assert receipt["prev_epoch_sha256"] == GENESIS_PREV
    assert receipt["prev_epoch_receipt"] is None
    assert receipt["n_members"] == 2
    assert epoch_contract_errors(receipt) == []


def test_epoch_root_deterministic_and_order_free() -> None:
    m1 = {"a": "ab" * 32, "b": "cd" * 32}
    m2 = {"b": "cd" * 32, "a": "ab" * 32}
    assert epoch_root(m1) == epoch_root(m2)
    assert epoch_root(m1) != epoch_root({"a": "ab" * 32})


def test_chain_advances_on_growth(corpus_dir: Path) -> None:
    _stamp(corpus_dir)
    _receipt(corpus_dir, "c.json", "3")
    receipt = corpus_epoch(corpus_dir)
    assert receipt["verdict"] == "advancing"
    # c.json joins, and the first epoch receipt is itself a corpus member.
    assert "c.json" in receipt["members_added"]
    assert any(n.startswith("corpus_epoch_") for n in receipt["members_added"])
    assert receipt["members_removed"] == []
    _stamp(corpus_dir)
    result = check_epoch_chain(corpus_dir)
    assert result["errors"] == [] and result["unstamped"] == []
    assert result["head"] is not None and result["head_epoch_root"] is not None


def test_deleted_member_detected(corpus_dir: Path) -> None:
    _stamp(corpus_dir)
    (corpus_dir / "a.json").unlink()
    receipt = corpus_epoch(corpus_dir)
    assert receipt["verdict"] == "shrinking"
    assert receipt["members_removed"] == ["a.json"]
    _stamp(corpus_dir)
    errors = check_epoch_chain(corpus_dir)["errors"]
    assert any("member_removed:" in e and "'a.json'" in e for e in errors)


def test_allowed_removal_pin(corpus_dir: Path) -> None:
    _stamp(corpus_dir)
    digest = member_digests(corpus_dir)["a.json"]
    (corpus_dir / "a.json").unlink()
    _stamp(corpus_dir)
    errors = check_epoch_chain(corpus_dir, allowed_removals={"a.json": digest})["errors"]
    assert not any("member_removed:" in e and "'a.json'" in e for e in errors)


def test_mutated_member_detected(corpus_dir: Path) -> None:
    _stamp(corpus_dir)
    (corpus_dir / "a.json").write_text('{"mutated": true}')
    _stamp(corpus_dir)  # same name, new digest — mutation, not removal
    errors = check_epoch_chain(corpus_dir)["errors"]
    assert any("member_mutated:" in e and "'a.json'" in e for e in errors)


def test_post_stamp_arrival_is_unstamped_not_error(corpus_dir: Path) -> None:
    _stamp(corpus_dir)
    _receipt(corpus_dir, "late.json", "sneaked in post-stamp")
    result = check_epoch_chain(corpus_dir)
    assert result["errors"] == []
    assert "late.json" in result["unstamped"]


def test_dishonest_delta_flagged(corpus_dir: Path) -> None:
    _stamp(corpus_dir)
    _receipt(corpus_dir, "c.json", "3")
    epoch = corpus_epoch(corpus_dir)
    epoch["members_added"] = []  # lie about the delta
    write_epoch_receipt(epoch, corpus_dir)
    errors = check_epoch_chain(corpus_dir)["errors"]
    assert any("members_added_dishonest" in e for e in errors)


def test_contract_rejects_tampered_members(corpus_dir: Path) -> None:
    epoch = corpus_epoch(corpus_dir)
    bad = dict(epoch)
    bad["members"] = bad["members"][:-1]
    errors = epoch_contract_errors(bad)
    assert "n_members_mismatch" in errors
    assert "epoch_root_mismatch" in errors
    with pytest.raises(ValueError, match="contract"):
        write_epoch_receipt(bad, corpus_dir)


def test_written_epoch_verifies(corpus_dir: Path) -> None:
    path = _stamp(corpus_dir)
    ver = verify_receipt_file(path)
    assert ver["valid"], ver["errors"]


def test_no_epochs_reports_missing(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    assert check_epoch_chain(corpus)["errors"] == ["no_epoch_receipts"]


def test_fails_closed_on_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        corpus_epoch(tmp_path / "nope")


def test_nested_members_use_posix_rel_paths(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "runs").mkdir(parents=True)
    _receipt(corpus, "a.json", "1")
    _receipt(corpus / "runs", "nested.json", "2")
    members = member_digests(corpus)
    assert set(members) == {"a.json", "runs/nested.json"}
    epoch = corpus_epoch(corpus)
    assert epoch["n_members"] == 2
    write_epoch_receipt(epoch, corpus)
    (corpus / "runs" / "nested.json").unlink()
    errors = check_epoch_chain(corpus)["errors"]
    assert any("head_member_missing_live:" in e and "runs/nested.json" in e for e in errors)


def test_chains_partition_by_pattern(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _receipt(corpus, "a.json", "1")
    (corpus / "report.md").write_text("# run report\n")
    # Two independent chains over the same dir: one over *.json, one *.md.
    # The second pattern mints a parallel chain — an explicit opt-in.
    json_epoch = corpus_epoch(corpus, pattern="*.json")
    md_epoch = corpus_epoch(corpus, pattern="*.md", allow_new_pattern=True)
    assert set(m["name"] for m in json_epoch["members"]) == {"a.json"}
    assert [m["name"] for m in md_epoch["members"]] == ["report.md"]
    write_epoch_receipt(json_epoch, corpus)
    md_path = write_epoch_receipt(md_epoch, corpus)
    json_check = check_epoch_chain(corpus, pattern="*.json")
    md_check = check_epoch_chain(corpus, pattern="*.md")
    assert json_check["errors"] == []
    # The md-chain's epoch file is a *.json arrival not yet stamped by the
    # json chain — reported as unstamped, not an error.
    assert json_check["unstamped"] == [md_path.name]
    assert md_check["errors"] == []
    assert md_check["unstamped"] == []
    # A new *.md arrival shows up only on the md chain.
    (corpus / "new_report.md").write_text("# more\n")
    assert check_epoch_chain(corpus, pattern="*.md")["unstamped"] == ["new_report.md"]
    assert check_epoch_chain(corpus, pattern="*.json")["errors"] == []


def test_foreign_pattern_stamp_refused(tmp_path: Path) -> None:
    """A mismatched --glob mints a parallel (dir, pattern) chain whose records
    read as unstamped members of the real one — refuse unless opted in."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _receipt(corpus, "a.json", "1")
    write_epoch_receipt(corpus_epoch(corpus, pattern="*"), corpus)
    with pytest.raises(ValueError, match="allow-new-pattern"):
        corpus_epoch(corpus, pattern="*.json")
    # Explicit opt-in mints the second chain.
    epoch = corpus_epoch(corpus, pattern="*.json", allow_new_pattern=True)
    assert epoch["params"]["pattern"] == "*.json"


def test_contract_rejects_bad_params(corpus_dir: Path) -> None:
    epoch = corpus_epoch(corpus_dir)
    bad = dict(epoch)
    bad["params"] = {"pattern": 42}
    assert "params_pattern_not_str" in epoch_contract_errors(bad)
    bad2 = dict(epoch)
    bad2["params"] = "nope"
    assert "params_not_mapping" in epoch_contract_errors(bad2)


@pytest.fixture()
def corpus_dir(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    _receipt(root, "a.json", "1")
    _receipt(root, "b.json", "2")
    return root


def test_heads_pin_round_trip(tmp_path: Path) -> None:
    """Stamp → pin → check: the pinned head passes, a descendant does too."""
    from quant_fund.research.corpus_epoch import (
        epoch_heads_key,
        load_heads_pin,
        update_heads_pin,
    )

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _receipt(corpus, "a.json", "1")
    first = write_epoch_receipt(corpus_epoch(corpus), corpus)
    pin = tmp_path / "epoch_heads.json"
    update_heads_pin(pin, corpus, "*.json", first)
    heads = load_heads_pin(pin)
    assert epoch_heads_key(corpus, "*.json") in heads
    exp = heads[epoch_heads_key(corpus, "*.json")]
    assert exp["receipt"] == first.name
    res = check_epoch_chain(corpus, expected_head=exp)
    assert res["errors"] == [] and res["head"] == first.name
    # A newer epoch re-heads the chain; the old pin must accept it as a
    # descendant (stamping is append-only, pin follows in the same commit).
    _receipt(corpus, "b.json", "2")
    second = write_epoch_receipt(corpus_epoch(corpus), corpus)
    res2 = check_epoch_chain(corpus, expected_head=exp)
    assert res2["errors"] == [] and res2["head"] == second.name


def test_heads_pin_missing_and_mutated(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _receipt(corpus, "a.json", "1")
    first = write_epoch_receipt(corpus_epoch(corpus), corpus)
    exp = {"receipt": first.name, "sha256": "0" * 64}
    errors = check_epoch_chain(corpus, expected_head=exp)["errors"]
    assert any(e.startswith("epoch_head_mutated:") for e in errors)
    (corpus / first.name).unlink()
    errors = check_epoch_chain(corpus, expected_head=exp)["errors"]
    assert any(e.startswith("epoch_head_missing:") for e in errors)
    assert any(e.startswith("epoch_head_rollback:") for e in errors)


def test_heads_pin_rollback_detected(tmp_path: Path) -> None:
    """Deleting the newest epoch silently reverts to the old head — pinned."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _receipt(corpus, "a.json", "1")
    first = write_epoch_receipt(corpus_epoch(corpus), corpus)
    _receipt(corpus, "b.json", "2")
    second = write_epoch_receipt(corpus_epoch(corpus), corpus)
    from quant_fund.utils.hashing import hash_bytes

    exp = {"receipt": second.name, "sha256": hash_bytes(second.read_bytes())}
    assert check_epoch_chain(corpus, expected_head=exp)["errors"] == []
    # Attack: delete the head epoch — without the pin this rewinds silently.
    second.unlink()
    errors = check_epoch_chain(corpus, expected_head=exp)["errors"]
    assert any(e.startswith("epoch_head_missing:") for e in errors)
    assert any(e.startswith("epoch_head_rollback:") for e in errors)
    # And unpinned, the same rollback is silent — documenting why the pin exists.
    unpinned = check_epoch_chain(corpus)
    assert unpinned["head"] == first.name and unpinned["errors"] == []


def test_heads_pin_bound_chain_fields(tmp_path: Path) -> None:
    """The pin's tree_root/chain_root/n_epochs bind the chain at the pinned
    head — a pin lying about them fails closed even when the head is live."""
    from quant_fund.research.corpus_epoch import (
        epoch_heads_key,
        load_heads_pin,
        update_heads_pin,
    )
    from quant_fund.utils.hashing import hash_bytes

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _receipt(corpus, "a.json", "1")
    write_epoch_receipt(corpus_epoch(corpus), corpus)
    _receipt(corpus, "b.json", "2")
    second = write_epoch_receipt(corpus_epoch(corpus), corpus)
    pin = tmp_path / "epoch_heads.json"
    update_heads_pin(pin, corpus, "*.json", second)
    exp = load_heads_pin(pin)[epoch_heads_key(corpus, "*.json")]
    assert exp["receipt"] == second.name and "chain_root" in exp and "n_epochs" in exp
    assert check_epoch_chain(corpus, expected_head=exp)["errors"] == []
    # Forge each bound field — head receipt and sha stay honest.
    sha = hash_bytes(second.read_bytes())
    for field, bogus, code in (
        ("tree_root", "0" * 64, "epoch_head_tree_root_mismatch"),
        ("chain_root", "0" * 64, "epoch_head_chain_root_mismatch"),
        ("n_epochs", exp["n_epochs"] + 1, "epoch_pin_n_epochs_mismatch"),
    ):
        forged = {**exp, "sha256": sha, field: bogus}
        errors = check_epoch_chain(corpus, expected_head=forged)["errors"]
        assert any(e.startswith(f"{code}:") for e in errors), (field, errors)


def test_heads_pin_validates_schema(tmp_path: Path) -> None:
    from quant_fund.research.corpus_epoch import load_heads_pin

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "other.v1", "heads": []}))
    with pytest.raises(ValueError, match="heads"):
        load_heads_pin(bad)
    bad2 = tmp_path / "bad2.json"
    bad2.write_text(json.dumps({"schema": "epoch_heads.v1", "heads": {"k/*": {"receipt": 1}}}))
    with pytest.raises(ValueError, match="malformed"):
        load_heads_pin(bad2)


def test_heads_pin_file_is_never_a_member(tmp_path: Path) -> None:
    """The pin file is chain bookkeeping — stamping it would self-invalidate."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _receipt(corpus, "a.json", "1")
    (corpus / "epoch_heads.json").write_text('{"schema": "epoch_heads.v1", "heads": {}}\n')
    members = member_digests(corpus)
    assert set(members) == {"a.json"}  # pin exempt under any pattern
    epoch = corpus_epoch(corpus)
    assert epoch["n_members"] == 1
    first = write_epoch_receipt(epoch, corpus)
    from quant_fund.research.corpus_epoch import update_heads_pin

    pin = corpus / "epoch_heads.json"  # pin can even live inside the corpus dir
    update_heads_pin(pin, corpus, "*.json", first)
    assert check_epoch_chain(corpus)["errors"] == []


def test_require_stamped_promotes_arrivals_to_errors(tmp_path: Path) -> None:
    """Critical corpora: an unstamped arrival is the tamper, not a pending stamp."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _receipt(corpus, "a.json", "1")
    write_epoch_receipt(corpus_epoch(corpus), corpus)
    assert check_epoch_chain(corpus, require_stamped=True)["errors"] == []
    _receipt(corpus, "rogue.json", "2")
    res = check_epoch_chain(corpus, require_stamped=True)
    assert res["errors"] == ["unstamped_member:'rogue.json'"]
    # Default mode stays informational for accumulative corpora.
    assert check_epoch_chain(corpus)["errors"] == []
    assert check_epoch_chain(corpus)["unstamped"] == ["rogue.json"]


def test_allow_member_updates_for_mutable_corpora(tmp_path: Path) -> None:
    """Mutable corpora (quality manifests, workflows) attest history, not
    immutability: a stamped member's legit edit between stamps must not be a
    tamper error — but post-stamp drift still fails via head_member_digest_drift
    until re-stamped."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _receipt(corpus, "manifest.json", "v1")
    _receipt(corpus, "pin.json", "p1")
    _stamp(corpus)
    # Legitimate member update between stamps.
    _receipt(corpus, "manifest.json", "v2")
    _stamp(corpus)
    # Strict (append-only) mode: digest change between epochs is tamper evidence.
    strict = check_epoch_chain(corpus)
    assert any(e.startswith("member_mutated:'manifest.json'") for e in strict["errors"])
    # Mutable mode: mutation recorded, not an error; head still covers live state.
    mutable = check_epoch_chain(corpus, allow_member_updates=True)
    assert mutable["errors"] == []
    # Drift after the newest stamp still fails even in mutable mode.
    _receipt(corpus, "manifest.json", "v3")
    drift = check_epoch_chain(corpus, allow_member_updates=True)
    assert "head_member_digest_drift:'manifest.json'" in drift["errors"]
    # Removals still error in mutable mode (use allowed_removals pins).
    (corpus / "pin.json").unlink()
    removed = check_epoch_chain(corpus, allow_member_updates=True)
    assert any("pin.json" in e for e in removed["errors"])


def test_witness_prefix_exempt_only_inside_quality(tmp_path: Path) -> None:
    """quality/witness/* proofs churn per checkpoint rewrite — exempt from
    membership so the chain doesn't require an endless restamp loop. The
    exemption is corpus-scoped: receipts/witness/* stays a normal member."""
    quality = tmp_path / "quality"
    (quality / "witness").mkdir(parents=True)
    _receipt(quality, "pin.json", "p1")
    _receipt(quality / "witness", "proof_1.json", "w1")
    members = member_digests(quality)
    assert "pin.json" in members
    assert "witness/proof_1.json" not in members

    # The same relative name under a different corpus is NOT exempt.
    receipts = tmp_path / "receipts"
    (receipts / "witness").mkdir(parents=True)
    _receipt(receipts, "main.json", "r1")
    _receipt(receipts / "witness", "proof_1.json", "w1")
    rmembers = member_digests(receipts)
    assert "witness/proof_1.json" in rmembers


def test_witness_replacement_is_not_member_removed(tmp_path: Path) -> None:
    """A proof stamped as a member, then exempted, must not be flagged as a
    removal when the chain rechecks history."""
    quality = tmp_path / "quality"
    (quality / "witness").mkdir(parents=True)
    _receipt(quality, "pin.json", "p1")
    _receipt(quality / "witness", "proof_1.json", "w1")
    _stamp(quality)  # head stamped while proof was still a member
    # Exemption now applies (same as post-rule code): proof stays on disk,
    # next stamp excludes it from membership, no removal error.
    _receipt(quality, "pin.json", "p2")
    _stamp(quality)
    res = check_epoch_chain(quality, allow_member_updates=True)
    assert res["errors"] == []


def test_stamp_lock_fails_closed_on_contention(tmp_path: Path) -> None:
    """A second stamp while one holds the lock must refuse, not fork the chain."""
    corpus = tmp_path / "receipts"
    corpus.mkdir()
    held = _acquire_stamp_lock(corpus)
    try:
        with pytest.raises(EpochStampLocked):
            _acquire_stamp_lock(corpus)
    finally:
        held.close()
    # Released: the next stamp proceeds.
    fd = _acquire_stamp_lock(corpus)
    fd.close()


def test_stamp_lockfile_is_not_a_member(tmp_path: Path) -> None:
    """The lockfile must never enter membership or trip coverage closure."""
    corpus = tmp_path / "receipts"
    corpus.mkdir()
    fd = _acquire_stamp_lock(corpus)
    fd.close()
    _receipt(corpus, "a.json", "r1")
    members = member_digests(corpus)
    assert ".epoch_stamp.lock" not in members


def test_symlink_members_refused_and_flagged(tmp_path: Path) -> None:
    """A symlink member is a target-swap mutation vector: ``is_file`` follows
    the link, so the hashed bytes are the target's and retargeting swaps
    content the name never owned. The stamp refuses links outright and the
    chain flags a live one — file links and dir links alike."""
    from quant_fund.research.corpus_epoch import _symlink_errors

    corpus = tmp_path / "receipts"
    corpus.mkdir()
    _receipt(corpus, "real.json", "1")
    outside = tmp_path / "outside.json"
    outside.write_text('{"escape": true}')
    (corpus / "link.json").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        corpus_epoch(corpus)
    assert _symlink_errors(corpus, "*.json") == ["member_is_symlink:'link.json'"]
    # A dir link is invisible to the member glob but must still flag.
    linkdir = corpus / "vault"
    linkdir.symlink_to(tmp_path, target_is_directory=True)
    assert "member_is_symlink:'vault'" in _symlink_errors(corpus, "*.json")
    linkdir.unlink()
    # Live-tree leg: a clean stamp, then a link appears post-stamp.
    (corpus / "link.json").unlink()
    _stamp(corpus)
    (corpus / "link.json").symlink_to(outside)
    assert "member_is_symlink:'link.json'" in check_epoch_chain(corpus)["errors"]


def test_nonportable_names_refused_and_repr_quoted(tmp_path: Path) -> None:
    """Control/format/separator chars in member names are refused — a raw
    ``\\n`` in a label would inject a fake 'all gates intact' line into
    verifier output and U+202E visually renames members. The label always
    carries the ASCII repr so the name can't inject even while reported."""
    from quant_fund.research.corpus_epoch import _portable_name_errors

    corpus = tmp_path / "receipts"
    corpus.mkdir()
    _receipt(corpus, "real.json", "1")
    bad = "evil\nverify-repo: all gates intact.json"
    (corpus / bad).write_text("{}")
    assert _portable_name_errors({"x\ny.json": "ab" * 32}) == [
        "member_name_not_portable:'x\\ny.json'"
    ]
    with pytest.raises(ValueError, match="non-portable"):
        corpus_epoch(corpus)
    (corpus / bad).unlink()
    _stamp(corpus)
    (corpus / bad).write_text("{}")
    hits = [e for e in check_epoch_chain(corpus)["errors"] if "not_portable" in e]
    assert hits and all("\n" not in e for e in hits)


def test_epoch_prefix_squat_flagged_and_refused(tmp_path: Path) -> None:
    """``corpus_epoch_*`` is a reserved prefix: a file carrying the name
    but no epoch payload can't forge a chain, but must not pass silently
    as an ordinary member — the stamp refuses it and the chain flags a
    committed (or dropped-post-stamp) squatter."""
    corpus = tmp_path / "receipts"
    corpus.mkdir()
    _receipt(corpus, "real.json", "1")
    _stamp(corpus)
    # Wrong-schema squatter (valid JSON, no epoch payload).
    squat = corpus / "corpus_epoch_notes.json"
    squat.write_text('{"notes": "not an epoch"}')
    assert "epoch_prefix_squat:'corpus_epoch_notes.json'" in check_epoch_chain(corpus)["errors"]
    with pytest.raises(ValueError, match="squat"):
        corpus_epoch(corpus)
    # Malformed-bytes squatter (unparseable) flags the same way.
    squat.write_bytes(b"{not json")
    assert "epoch_prefix_squat:'corpus_epoch_notes.json'" in check_epoch_chain(corpus)["errors"]
    with pytest.raises(ValueError, match="squat"):
        corpus_epoch(corpus)
    squat.unlink()
    # Clean tree: no squat errors remain.
    assert not [e for e in check_epoch_chain(corpus)["errors"] if "prefix_squat" in e]
    # A non-squatting name carrying an epoch payload is a real candidate:
    # the reader filter is the schema, not the filename.
    epoch = corpus_epoch(corpus)
    renamed = write_epoch_receipt(epoch, corpus)
    renamed.rename(corpus / "zebra.json")
    errors = check_epoch_chain(corpus)["errors"]
    assert not any("prefix_squat" in e for e in errors), errors


def test_pycache_dirs_never_members(tmp_path: Path) -> None:
    """``__pycache__`` dirs are machine-local bytecode caches — never members
    in ANY corpus (a `pattern="*"` chain must not differ across hosts or
    interpreter versions)."""
    corpus = tmp_path / "src"
    (corpus / "x" / "__pycache__").mkdir(parents=True)
    (corpus / "__pycache__").mkdir()
    _member_file = corpus / "a.py"
    _member_file.write_text("code")
    (corpus / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\x00")
    (corpus / "x" / "__pycache__" / "b.cpython-312.pyc").write_bytes(b"\x00")
    members = member_digests(corpus, pattern="*")
    assert set(members) == {"a.py"}


def test_github_workflows_subcorpus_exempt(tmp_path: Path) -> None:
    """``.github/workflows`` is its own corpus — the parent ``.github`` corpus
    must not double-chain it; the rest of ``.github`` stays members."""
    gh = tmp_path / ".github"
    (gh / "workflows").mkdir(parents=True)
    (gh / "workflows" / "ci.yml").write_text("on: push")
    (gh / "pull_request_template.md").write_text("## Summary")
    members = member_digests(gh, pattern="*")
    assert "pull_request_template.md" in members
    assert "workflows/ci.yml" not in members


def test_lake_runtime_sidecars_are_not_metadata_members(tmp_path: Path) -> None:
    """Ingest and the CLI drop gitignored sidecars beside the lake evidence.

    Those exact names are not chain members under ``data/metadata``. The same
    names in another corpus, and any other file in the metadata corpus, stay
    members — a dropped file still fails ``require_stamped``.
    """
    lake = tmp_path / "metadata"
    lake.mkdir()
    _receipt(lake, "evidence.json", "sealed")
    (lake / "data_manifest.json").write_text("{}\n")
    (lake / "resolved_config.json").write_text("{}\n")
    (lake / "proofcore.duckdb").write_bytes(b"db")
    members = member_digests(lake, pattern="*")
    assert set(members) == {"evidence.json"}
    write_epoch_receipt(corpus_epoch(lake, pattern="*"), lake)
    (lake / "rogue.json").write_text("{}\n")
    res = check_epoch_chain(lake, pattern="*", require_stamped=True)
    assert res["errors"] == ["unstamped_member:'rogue.json'"]

    other = tmp_path / "receipts"
    other.mkdir()
    (other / "data_manifest.json").write_text("{}\n")
    assert "data_manifest.json" in member_digests(other, pattern="*")


def test_script_membership_parity(tmp_path: Path) -> None:
    """The standalone auditor's membership must equal the library's byte-for-
    byte — it carries its own exemption rules and has diverged before."""
    from scripts import verify_epoch_chain as sv

    corpus = tmp_path / "quality"
    for sub in ("witness", "checkpoints", "quorum_rotations", "nested/__pycache__"):
        (corpus / sub).mkdir(parents=True)
    (corpus / "pin.json").write_text("{}")
    (corpus / "witness" / "p.json").write_text("{}")
    (corpus / "checkpoints" / "c.json").write_text("{}")
    (corpus / "quorum_rotations" / "rotation_x.json").write_text("{}")
    (corpus / "nested" / "__pycache__" / "m.pyc").write_bytes(b"\x00")
    gh = tmp_path / ".github"
    (gh / "workflows").mkdir(parents=True)
    (gh / "workflows" / "ci.yml").write_text("x")
    (gh / "tpl.md").write_text("t")
    lake = tmp_path / "metadata"
    lake.mkdir()
    (lake / "evidence.json").write_text("{}")
    (lake / "data_manifest.json").write_text("{}")
    (lake / "resolved_config.json").write_text("{}")
    (lake / "proofcore.duckdb").write_bytes(b"db")
    (lake / "nested" / "manifest.json").parent.mkdir()
    (lake / "nested" / "manifest.json").write_text("{}")
    for d in (corpus, gh, lake):
        assert sv._member_digests(d, "*") == member_digests(d, pattern="*")
