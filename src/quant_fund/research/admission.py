"""Receipt admission gate — decides whether a new receipt may join the corpus.

``verify-receipt`` answers "is this receipt valid"; ``dipcatcher admit``
answers "does this receipt deserve to sit in the evidence corpus next to
everything already there". A valid-but-contradictory receipt must not merge
quietly — it is quarantined until its lattice deltas are pinned or resolved.

Checks, in order:
1. ``seal``       — ``verify_receipt_file`` (signature, contract, integrity).
2. ``honesty``    — ``research_only is True`` and ``live_pnl_claim is False``
   declared somewhere the seal covers, ``data_label`` a non-empty string.
   Asserted dishonesty (``live_pnl_claim is True``) rejects outright.
3. ``lattice``    — run the receipt lattice over the corpus with and without
   the candidate; any *new* ``inconsistent`` claim group involving the
   candidate quarantines it.
4. ``corpus``     — corpus_audit on both corpora; the survivor-set delta is
   recorded as evidence (BH families legitimately shift as members arrive).
5. ``epoch_chain`` — the corpus's stamped epoch chain, with the same committed
   removal acknowledgments every other epoch gate applies; a broken chain
   quarantines.

Verdicts: ``admit`` | ``quarantine`` | ``reject``. ``--strict`` exits
nonzero on anything but ``admit``.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import hash_bytes

ADMISSION_SCHEMA = "receipt_admission.v1"


def _candidate_body(doc: Mapping[str, Any]) -> Mapping[str, Any]:
    """The claim-bearing body: v2 inner payload when present, else the doc."""
    inner = doc.get("payload")
    return inner if isinstance(inner, Mapping) else doc


def _honesty_errors(doc: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """(reject_errors, quarantine_errors) over the sealed honesty stamps."""
    reject: list[str] = []
    quarantine: list[str] = []
    bodies = [doc]
    inner = doc.get("payload")
    if isinstance(inner, Mapping):
        bodies.append(inner)
    for body in bodies:
        if body.get("live_pnl_claim") is True:
            reject.append("live_pnl_claim_true")
        if body.get("research_only") is False:
            reject.append("research_only_false")
    claim = _candidate_body(doc)
    if claim.get("research_only") is not True:
        quarantine.append("research_only_missing")
    if claim.get("live_pnl_claim") is not False:
        quarantine.append("live_pnl_claim_missing")
    label = claim.get("data_label")
    if not (isinstance(label, str) and label.strip()):
        quarantine.append("data_label_missing")
    return reject, quarantine


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return  # already linked — e.g. the candidate lives inside the corpus
    try:
        os.link(src, dst)
    except OSError:  # cross-filesystem corpus dirs still gate correctly
        shutil.copy2(src, dst)


def _shadow_corpus(corpus_dir: Path, extra: Path | None = None) -> Path:
    """Link the corpus (+ optional extra receipt) into a private dir."""
    shadow = Path(tempfile.mkdtemp(prefix="admit_corpus_"))
    for path in sorted(corpus_dir.glob("*.json")):
        if path.is_file():
            _link_or_copy(path, shadow / path.name)
    if extra is not None:
        _link_or_copy(extra, shadow / extra.name)
    return shadow


def _mirror_subtrees(corpus_dir: Path, shadow: Path) -> None:
    """Link every file below the corpus's subdirectories into ``shadow``.

    Subtrees such as ``legacy-unsealed/`` are never candidates and never
    lattice or FDR members (those read top-level ``*.json`` only), but the
    epoch chain stamps them: a shadow without them reads as a corpus whose
    stamped members vanished (``head_member_missing_live``)."""
    for path in sorted(corpus_dir.rglob("*")):
        rel = path.relative_to(corpus_dir)
        if len(rel.parts) > 1 and path.is_file():
            (shadow / rel).parent.mkdir(parents=True, exist_ok=True)
            _link_or_copy(path, shadow / rel)


def _epoch_parent(path: Path) -> tuple[bool, str | None]:
    """``(is_epoch, prev_epoch_receipt)`` for a candidate file — recognised by
    its ``corpus_epoch.v1`` schema like the chain reader does, not its name.
    Unreadable files are plain candidates; their seal check reports them."""
    from quant_fund.research.corpus_epoch import EPOCH_SCHEMA

    try:
        doc = json.loads(path.read_text())
    except (OSError, UnicodeError, ValueError):
        return False, None
    body = _candidate_body(doc) if isinstance(doc, Mapping) else {}
    if EPOCH_SCHEMA not in (body.get("schema"), body.get("kind")):
        return False, None
    prev = body.get("prev_epoch_receipt")
    return True, prev if isinstance(prev, str) else None


def _landing_order(candidates: Iterable[Path]) -> list[Path]:
    """The order a diff's files can have been written in: receipts by name,
    then corpus epoch receipts parent before child (ties by name).

    ``admit_batch`` gates each candidate against the corpus holding every
    candidate before it, so that order must be a real one. An epoch stamps
    the receipts around it and extends its parent; by name alone a second
    epoch in one diff, or an epoch sorting before a receipt it stamps, reads
    as an orphaned head or a removed member though the merged corpus is
    whole. The order only picks which intermediate states are checked: every
    candidate still sees all earlier ones, so an intra-diff contradiction is
    still caught on whichever file lands later. A cycle has no real order;
    its members keep name order and the chain check flags them."""
    ordered: list[Path] = []
    parents: dict[Path, str | None] = {}
    for path in sorted(candidates, key=lambda p: p.name):
        is_epoch, prev = _epoch_parent(path)
        if is_epoch:
            parents[path] = prev
        else:
            ordered.append(path)
    pending = list(parents)
    while pending:
        names = {p.name for p in pending}
        ready = [p for p in pending if parents[p] not in names] or pending
        ordered += ready
        pending = [p for p in pending if p not in ready]
    return ordered


def _inconsistent_groups(lattice: Mapping[str, Any]) -> set[str]:
    """Fingerprints of the inconsistent claim groups in a lattice receipt."""
    out: set[str] = set()
    for group in lattice.get("groups") or []:
        if isinstance(group, Mapping) and group.get("verdict") == "inconsistent":
            out.add(str(group.get("fingerprint", "?")))
    return out


def _survivors(corpus: Mapping[str, Any]) -> set[str]:
    return {
        f"{c.get('source')}|{c.get('path')}"
        for c in corpus.get("surviving_claims") or []
        if isinstance(c, Mapping)
    }


def admission_check(
    candidate: Path | str,
    corpus_dir: Path | str = Path("receipts"),
    *,
    q: float = 0.05,
    known_inconsistent: Mapping[str, str] | None = None,
    tombstone_dir: Path | str | None = None,
    allowed_removals: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Gate a candidate receipt against an existing corpus.

    ``allowed_removals`` (name -> sha256, ``quality/epoch_allowed_removals.json``)
    is handed to the epoch-chain check exactly as ``corpus-epoch --check``
    and ``verify-repo`` use it: a stamped member removed with acknowledged
    bytes is recorded history, any other removal still quarantines.

    Fails closed: a missing candidate or unreadable corpus dir raises;
    per-check findings are recorded rather than swallowed.
    """
    candidate = Path(candidate)
    corpus_dir = Path(corpus_dir)
    if not candidate.is_file():
        raise ValueError(f"candidate receipt {candidate} does not exist")
    if not corpus_dir.is_dir():
        raise ValueError(f"corpus dir {corpus_dir} does not exist")

    checks: list[dict[str, Any]] = []
    candidate_sha = hash_bytes(candidate.read_bytes())

    # -- 1. seal ---------------------------------------------------------------
    from quant_fund.research.receipt_v2 import verify_receipt_file

    verdict_result = verify_receipt_file(candidate)
    seal_ok = bool(verdict_result["valid"])
    checks.append(
        {
            "name": "seal",
            "ok": seal_ok,
            "errors": list(verdict_result["errors"]),
        }
    )

    try:
        doc = json.loads(candidate.read_text())
        if not isinstance(doc, Mapping):
            raise ValueError("receipt root is not an object")
    except (OSError, ValueError) as exc:
        # Narrowed from `except Exception` (quality ratchet): the parse block
        # reads bytes then json-loads — OSError/ValueError are the only fault
        # modes; exotic errors propagate. Recorded, never swallowed.
        doc = {}
        seal_ok = False
        checks.append({"name": "parse", "ok": False, "errors": [str(exc)]})

    # -- 2. honesty stamps ------------------------------------------------------
    reject_errors: list[str] = []
    quarantine_errors: list[str] = []
    if doc:
        reject_errors, quarantine_errors = _honesty_errors(doc)
    checks.append(
        {
            "name": "honesty",
            "ok": not reject_errors and not quarantine_errors,
            "reject_errors": reject_errors,
            "quarantine_errors": quarantine_errors,
        }
    )

    # -- 3. lattice delta -------------------------------------------------------
    from quant_fund.research.receipt_lattice import receipt_lattice

    before_shadow = _shadow_corpus(corpus_dir)
    after_shadow = _shadow_corpus(corpus_dir, extra=candidate)
    pins = dict(known_inconsistent or {})
    removals = dict(allowed_removals or {})
    before = receipt_lattice(before_shadow, known_inconsistent=pins)
    after = receipt_lattice(after_shadow, known_inconsistent=pins)
    new_inconsistent = _inconsistent_groups(after) - _inconsistent_groups(before)
    checks.append(
        {
            "name": "lattice",
            "ok": not new_inconsistent,
            "new_inconsistent_groups": len(new_inconsistent),
            "groups_after": after.get("n_claim_groups"),
        }
    )

    # -- 4. corpus FDR delta (evidence — never gate-driving on its own) ---------
    from quant_fund.research.corpus_inference import corpus_audit

    corpus_before = corpus_audit(before_shadow, q=q)
    corpus_after = corpus_audit(after_shadow, q=q)
    added = sorted(_survivors(corpus_after) - _survivors(corpus_before))
    removed = sorted(_survivors(corpus_before) - _survivors(corpus_after))
    checks.append(
        {
            "name": "corpus",
            "ok": True,
            "survivors_added": len(added),
            "survivors_removed": len(removed),
        }
    )

    corpus_files = sorted(p for p in corpus_dir.glob("*.json") if p.is_file())
    inputs_sha256 = hash_bytes(
        json.dumps(
            {
                "candidate": candidate.name,
                "candidate_sha256": candidate_sha,
                "corpus": {p.name: hash_bytes(p.read_bytes()) for p in corpus_files},
                "q": q,
            },
            sort_keys=True,
        ).encode()
    )

    # -- 5. epoch-chain integrity ------------------------------------------------
    # Binding to the stamped epoch chain is defense in depth on top of the
    # member-map digest: if the corpus was tampered *after* its last stamp,
    # admitting into it must not claim a clean pass. Feature-detected so this
    # module works whether or not corpus_epoch has merged/stamped yet.
    chain_errors: list[str] = []
    hard_chain_errors: list[str] = []
    epoch_head: str | None = None
    epoch_root_value: str | None = None
    try:
        from quant_fund.research.corpus_epoch import check_epoch_chain
    except ImportError:
        checks.append({"name": "epoch_chain", "ok": True, "skipped": "corpus_epoch_unavailable"})
    else:
        chain_result = check_epoch_chain(corpus_dir, allowed_removals=removals or None)
        chain_errors = list(chain_result["errors"])
        # An unstamped corpus ("no_epoch_receipts") is benign — the gate
        # works fine before the first epoch stamp exists. Only a *broken
        # stamped chain* (fork, member_removed/mutated, dishonest delta)
        # is a quarantine-class finding.
        hard_chain_errors = [e for e in chain_errors if e != "no_epoch_receipts"]
        head_name = chain_result.get("head")
        epoch_head = head_name if isinstance(head_name, str) else None
        root_value = chain_result.get("head_epoch_root")
        epoch_root_value = root_value if isinstance(root_value, str) else None
        checks.append(
            {
                "name": "epoch_chain",
                "ok": not hard_chain_errors,
                "errors": chain_errors,
                "unstamped": list(chain_result["unstamped"]),
            }
        )

    # -- 6. retraction -----------------------------------------------------------
    # An append-only corpus can't delete a bad receipt — it retracts it via a
    # sealed tombstone. Admission is the re-entry surface: the exact retracted
    # bytes must never be re-admitted, and the retracted name slot / claim
    # paths must not quietly re-fill. Feature-detected like epoch_chain.
    retract_reject: list[str] = []
    retract_findings: list[str] = []
    try:
        from quant_fund.research.receipt_tombstone import load_tombstones
    except ImportError:
        checks.append({"name": "tombstone", "ok": True, "skipped": "tombstone_unavailable"})
    else:
        # Resolve tombstones against the durable corpus, not the (possibly
        # shadowed) admission view — a retracted target must stay pinned to
        # real bytes even when admit_batch shadows it out mid-diff.
        tombs = load_tombstones(Path(tombstone_dir) if tombstone_dir is not None else corpus_dir)
        entry = tombs["active"].get(candidate.name)
        if entry is not None:
            if entry.get("target_sha256") == candidate_sha:
                retract_reject.append("retracted_bytes")
            elif entry.get("scope") == "all":
                retract_findings.append("retracted_slot")
            else:
                # Partial scope: the retracted claim paths must not re-enter.
                from quant_fund.research.corpus_inference import harvest_findings

                scoped = set(entry["scope"]) if isinstance(entry["scope"], list) else set()
                try:
                    cand_doc = json.loads(candidate.read_text())
                except (OSError, ValueError):  # parse already recorded above
                    cand_doc = {}
                paths = (
                    {f["path"] for f in harvest_findings(cand_doc, candidate.name)}
                    if isinstance(cand_doc, Mapping)
                    else set()
                )
                if paths & scoped:
                    retract_findings.append("retracted_scope_overlap")
        checks.append(
            {
                "name": "tombstone",
                "ok": not retract_reject and not retract_findings,
                "reject_errors": retract_reject,
                "findings": retract_findings,
            }
        )

    if not seal_ok or reject_errors or retract_reject:
        verdict = "reject"
    elif quarantine_errors or new_inconsistent or hard_chain_errors or retract_findings:
        verdict = "quarantine"
    else:
        verdict = "admit"

    return {
        "kind": ADMISSION_SCHEMA,
        "schema": ADMISSION_SCHEMA,
        "research_only": True,
        "live_pnl_claim": False,
        "simulated_only": False,
        "data_label": "CORPUS",
        "inputs_sha256": inputs_sha256,
        "params": {
            "q": q,
            "corpus_dir": str(corpus_dir),
            "known_inconsistent": sorted(pins),
            "allowed_removals": sorted(removals),
        },
        "candidate": candidate.name,
        "candidate_sha256": candidate_sha,
        "n_corpus_receipts": len(corpus_files),
        "checks": checks,
        "corpus_epoch_receipt": epoch_head,
        "corpus_epoch_root": epoch_root_value,
        "lattice_new_inconsistent": sorted(new_inconsistent),
        "survivors_added": added,
        "survivors_removed": removed,
        "verdict": verdict,
    }


def admit_batch(
    candidates: Iterable[Path | str],
    corpus_dir: Path | str = Path("receipts"),
    *,
    q: float = 0.05,
    known_inconsistent: Mapping[str, str] | None = None,
    allowed_removals: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Gate a set of incoming receipts the way a merge actually lands them.

    ``admission_check`` on an already-committed receipt has a vacuous lattice
    check: the candidate is already inside the corpus, so before == after and
    no delta is observable. For a diff of N changed receipts the honest gate
    is sequential — each candidate is checked against

        corpus − {all changed names} ∪ {changed files processed so far}

    which is exactly the intermediate state each file creates, so
    intra-diff contradictions are attributed to the file that introduces
    them and no commit sneaks a contradiction in inside a batch. Files are
    processed in ``_landing_order``: receipts by name, then epoch receipts
    parent first.

    Only top-level ``*.json`` members of the corpus are gated: quarantined
    subtrees (``legacy-unsealed/``) and non-receipt files are not corpus
    members and are skipped — they carry their own byte-pins elsewhere. The
    shadow still carries those subtrees, because the epoch chain stamps them.

    Fails closed: a missing candidate, a candidate inside the corpus under a
    *different* name, or an unreadable corpus dir raises.
    """
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        raise ValueError(f"corpus dir {corpus_dir} does not exist")
    ordered = [Path(c) for c in candidates]
    for cand in ordered:
        if not cand.is_file():
            raise ValueError(f"candidate receipt {cand} does not exist")
    ordered = _landing_order(ordered)

    changed_names = {c.name for c in ordered}
    shadow = Path(tempfile.mkdtemp(prefix="admit_batch_base_"))
    for path in sorted(corpus_dir.glob("*.json")):
        if path.is_file() and path.name not in changed_names:
            _link_or_copy(path, shadow / path.name)
    _mirror_subtrees(corpus_dir, shadow)

    results: list[dict[str, Any]] = []
    for cand in ordered:
        result = admission_check(
            cand,
            shadow,
            q=q,
            known_inconsistent=known_inconsistent,
            tombstone_dir=corpus_dir,
            allowed_removals=allowed_removals,
        )
        results.append(result)
        # Post-merge coexistence: a merged diff lands all of its files
        # together, so each later candidate must clear a corpus that already
        # contains the earlier ones — whatever verdict they drew.
        _link_or_copy(cand, shadow / cand.name)

    verdicts = [r["verdict"] for r in results]
    if "reject" in verdicts:
        verdict = "reject"
    elif "quarantine" in verdicts:
        verdict = "quarantine"
    else:
        verdict = "admit"
    return {
        "verdict": verdict,
        "n_candidates": len(ordered),
        "results": results,
        "failures": [r["candidate"] for r in results if r["verdict"] != "admit"],
    }


def admission_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """Re-derive the accounting of a ``receipt_admission.v1`` body."""
    errors: list[str] = []
    if payload.get("kind") != ADMISSION_SCHEMA and payload.get("schema") != ADMISSION_SCHEMA:
        return ["schema"]
    if payload.get("verdict") not in ("admit", "quarantine", "reject"):
        errors.append("verdict")
    for key in ("inputs_sha256", "candidate_sha256"):
        value = payload.get(key)
        if not (isinstance(value, str) and len(value) == 64):
            errors.append(key)
    checks = payload.get("checks")
    if not isinstance(checks, list) or not checks:
        errors.append("checks_missing")
    else:
        names = [c.get("name") for c in checks if isinstance(c, Mapping)]
        for required in ("seal", "honesty", "lattice", "corpus"):
            if required not in names:
                errors.append(f"check_{required}_missing")
    n_corpus = payload.get("n_corpus_receipts")
    if not isinstance(n_corpus, int) or n_corpus < 0:
        errors.append("n_corpus_receipts")
    # verdict↔checks coherence: reject requires a failed seal or honesty reject;
    # quarantine requires some quarantine-class finding; admit requires all ok.
    epoch_root = payload.get("corpus_epoch_root")
    if epoch_root is not None and not (isinstance(epoch_root, str) and len(epoch_root) == 64):
        errors.append("corpus_epoch_root")
    epoch_receipt = payload.get("corpus_epoch_receipt")
    if epoch_receipt is not None and not (
        isinstance(epoch_receipt, str) and epoch_receipt.startswith("corpus_epoch_")
    ):
        errors.append("corpus_epoch_receipt")
    if isinstance(checks, list) and errors == []:
        seal_check: Mapping[str, Any] = next((c for c in checks if c.get("name") == "seal"), {})
        honesty: Mapping[str, Any] = next((c for c in checks if c.get("name") == "honesty"), {})
        lattice: Mapping[str, Any] = next((c for c in checks if c.get("name") == "lattice"), {})
        epoch_chain: Mapping[str, Any] = next(
            (c for c in checks if c.get("name") == "epoch_chain"), {}
        )
        tombstone: Mapping[str, Any] = next((c for c in checks if c.get("name") == "tombstone"), {})
        verdict = payload["verdict"]
        rejectable = (
            not seal_check.get("ok", True)
            or bool(honesty.get("reject_errors"))
            or bool(tombstone.get("reject_errors"))
        )
        quarantinable = (
            bool(honesty.get("quarantine_errors"))
            or not lattice.get("ok", True)
            or not epoch_chain.get("ok", True)
            or bool(tombstone.get("findings"))
        )
        if verdict == "reject" and not rejectable:
            errors.append("verdict_reject_without_cause")
        elif verdict == "quarantine" and (rejectable or not quarantinable):
            errors.append("verdict_quarantine_mismatch")
        elif verdict == "admit" and (rejectable or quarantinable):
            errors.append("verdict_admit_with_findings")
    return errors


def write_admission_receipt(
    receipt: Mapping[str, Any],
    receipts_dir: Path | str = Path("receipts"),
    *,
    receipt_version: int = 1,
) -> Path:
    """Seal a receipt_admission receipt and write ``receipt_admission_<hash>.json``.

    Filename digest = canonical ``receipt_sha256``. Atomic, fail-closed on a
    malformed receipt. ``receipt_version=2`` wraps the body in the unified
    ``receipt.v2`` envelope.
    """
    from quant_fund.research.receipt_v2 import seal_receipt, wrap_receipt_v2
    from quant_fund.utils.atomicio import atomic_write_text
    from quant_fund.utils.hashing import canonical_json_bytes

    if (
        receipt.get("kind") != ADMISSION_SCHEMA
        or receipt.get("schema") != ADMISSION_SCHEMA
        or receipt.get("research_only") is not True
        or receipt.get("live_pnl_claim") is not False
        or not isinstance(receipt.get("inputs_sha256"), str)
        or not isinstance(receipt.get("params"), Mapping)
    ):
        raise ValueError("receipt_admission receipt violates its contract")
    errors = admission_contract_errors(receipt)
    if errors:
        raise ValueError(f"receipt_admission receipt violates its contract: {errors}")
    if receipt_version == 1:
        canonical = json.loads(canonical_json_bytes(dict(receipt)))
        digest = hash_bytes(canonical_json_bytes(canonical))
        payload = {**canonical, "receipt_sha256": digest}
    elif receipt_version == 2:
        payload = seal_receipt(
            wrap_receipt_v2(
                receipt,
                code_files=(Path(__file__),),
                verdict="pass" if receipt.get("verdict") == "admit" else "fail",
            )
        )
        digest = str(payload["receipt_sha256"])
    else:
        raise ValueError(f"receipt_version must be 1 or 2, got {receipt_version!r}")
    path = Path(receipts_dir) / f"receipt_admission_{digest[:16]}.json"
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
