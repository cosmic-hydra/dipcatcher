"""Repo integrity capstone: one sealed attestation over the whole evidence
integrity substrate.

The pieces each gate one surface — ``crown_jewels`` byte-pins the gate-defining
files, ``corpus_epoch`` chains the evidence corpora, ``check_epoch_chain``
verifies them. ``verify_repo`` composes them into a single verdict and seals it
as a ``repo_integrity.v1`` receipt: a point-in-time proof that the repo's
evidence corpus and the config that gates it were all intact *at the same
instant*.

The committed attestation is itself a ``quality/`` chain member, so a stale
attestation can't be silently edited — the next epoch stamp records the drift.

Provenance evidence only; never a market or P&L claim.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

from quant_fund.research.corpus_epoch import (
    check_epoch_chain,
    epoch_heads_key,
    load_heads_pin,
)
from quant_fund.research.crown_jewels import (
    DEFAULT_PIN_PATH as JEWELS_PIN,
)
from quant_fund.research.crown_jewels import (
    crown_jewels_errors,
)
from quant_fund.research.gate_signatures import verify_pin_signatures
from quant_fund.research.integrity_checkpoint import verify_checkpoint
from quant_fund.research.integrity_witness import verify_witnesses
from quant_fund.research.timestamp_anchor import verify_timestamps
from quant_fund.utils.atomicio import atomic_write_text
from quant_fund.utils.hashing import hash_bytes

REPO_INTEGRITY_SCHEMA = "repo_integrity.v1"

# (corpus_dir, member glob, require_stamped, allow_member_updates,
#  coverage_exempt globs) — the same policy the evidence-audit Makefile
# target enforces; keep in sync.
CORPORA: tuple[tuple[str, str, bool, bool, tuple[str, ...]], ...] = (
    # require_stamped=True everywhere: an unstamped member is a silent
    # injection vector — the fuzz drill demonstrated arrivals passed as
    # notes. Arrivals are errors until the epoch restamp commits them.
    # The inventory sidecar is a digest of the chained JSON, not a second
    # receipt. A new ``*.json`` pattern would fork the receipts chain.
    ("receipts", "*.json", True, False, ("legacy-unsealed/README.md", "*.sha256")),
    ("verifier", "*.md", True, False, ()),
    # quality's non-.json residents are jewel-pinned keys/certs,
    # self-authenticating TSA anchors, or ratchet notes — pinned elsewhere
    # or not corpus evidence, so they are exempt here.
    (
        "quality",
        "*.json",
        True,
        True,
        (
            "*.pub",
            "*.pem",
            "*.crt",
            "*.tsr",
            "*.txt",
            "*.md",
            "timestamps/*",
            "timestamps/ots/*",
        ),
    ),
    # Closed world: a `.yaml` workflow beside `*.yml` members would run on
    # GitHub while dodging the epoch chain — uncovered is an error.
    (".github/workflows", "*.yml", True, True, ("README.md",)),
    # Declared experiment inputs — a post-hoc config edit silently rewrites
    # what a sealed bench measured; mutable corpus, stamped arrivals only.
    ("configs", "*", True, True, ()),
    # Committed claim artifacts (champion selects, dev grids) — regenerated
    # between runs, so mutable; every committed member must be stamped.
    ("artifacts", "*", True, True, ("*.gitkeep",)),
    # Committed fleet evidence (the SOTA input stream). Live dir — arrivals
    # land between stamps (unstamped = informational, not error), but a
    # stamped member is immutable: .npz mutation after stamping is tamper.
    (".dsh-24x7", "*", False, False, ()),
    # Dataset manifests + validation outputs — inputs the data_manifest
    # receipts pin; write-once dated dirs, stamped arrivals.
    # Members are gitignored (only `.gitkeep` clones), so the epoch chain
    # exists only on machines that ran the stamps: LOCAL_ONLY_CORPORA.
    ("data/metadata", "*", True, False, ()),
    # ---- closed world: every tracked directory of code/evidence ----
    # The library itself — a non-jewel source file edited post-stamp is a
    # silent verdict-tampering vector; every byte is a chain member now.
    ("src", "*", True, True, ()),
    # Test code + committed fixtures: a swapped fixture or weakened oracle
    # launders a broken gate the same as a source tamper.
    ("tests", "*", True, True, ()),
    # Standalone auditors + generators: scripts/verify_*.py are the
    # third-party verification path — outside the package but just as
    # integrity-critical as the jewels among them.
    ("scripts", "*", True, True, ()),
    # Claim-bearing docs (audit ledgers, methodology, generated evidence
    # pages): an unchained doc could rewrite what the evidence asserts.
    ("docs", "*", True, True, ()),
    # Reality-study records (preregistrations, audit trails) — evidence.
    ("research", "*", True, True, ()),
    # Replay manifests + the session-replay harness.
    ("replay", "*", True, True, ()),
    ("reports", "*", True, True, ()),
    ("notebooks", "*", True, True, ()),
    ("examples", "*", True, True, ()),
    # Published client surface (TypeScript client + OpenAPI schema).
    ("clients", "*", True, True, ()),
    ("typings", "*", True, True, ()),
    # Formal specs (TLA+/configs) the formal lane checks against.
    ("spec", "*", True, True, ()),
    # Deployment surface — Dockerfiles and observability configs are
    # supply-chain inputs.
    ("docker", "*", True, True, ()),
    ("deploy", "*", True, True, ()),
    # Vendored third-party code — the highest-value supply-chain member.
    ("third_party", "*", True, True, ()),
    # Native extension source (Rust order-book core).
    ("rust", "*", True, True, ()),
    # Web explorer + its lockfile (published receipt export surface).
    ("web", "*", True, True, ()),
    # Verification snippet evidence used by the soft-verify gate.
    (".box-soft-verify", "*", True, True, ()),
    # Cursor agent env config — install.sh is an executed supply-chain file.
    (".cursor", "*", True, True, ()),
    # Remaining .github files (templates, plans); workflows/ is its own
    # corpus above and exempt here.
    (".github", "*", True, True, ("workflows/*",)),
)

# Corpora whose epoch receipts never clone (gitignored even when the member
# files themselves are committed). On a checkout that never ran the stamps
# the dir holds members but zero epoch receipts: the gate reports an
# explicit ``skipped`` state rather than a pass — a machine with receipts
# still verifies fully, and stamped-member immutability holds there.
LOCAL_ONLY_CORPORA: frozenset[str] = frozenset({"data/metadata"})

# Corpora an ``--evidence-only`` bundle is expected to carry — the evidence
# store, not the source tree. Under evidence_only an absent evidence corpus
# is still ``corpus_missing``; an absent code corpus reports a skip.
EVIDENCE_CORPORA: frozenset[str] = frozenset(c[0] for c in CORPORA[:8])

# Files every epoch gate reads besides its own corpus dir: the committed heads
# pin (each corpus's expected head) and the removal acknowledgments. A drill
# probe touching either re-runs every epoch gate (see ``epoch_scope``).
EPOCH_SHARED_INPUTS: frozenset[str] = frozenset(
    {"quality/epoch_heads.json", "quality/epoch_allowed_removals.json"}
)


def _checked_epoch_scope(epoch_scope: Collection[str] | None) -> frozenset[str] | None:
    """``verify_repo``'s ``epoch_scope`` as a set; unknown corpus names raise
    — a typo must not silently skip the gate it meant to run."""
    if epoch_scope is None:
        return None
    scope = frozenset(epoch_scope)
    unknown = sorted(scope - {c[0] for c in CORPORA})
    if unknown:
        raise ValueError(f"epoch_scope names unknown corpora: {unknown}")
    return scope


def _epoch_skip(
    corpus_dir: str, *, evidence_only: bool, scope: frozenset[str] | None
) -> str | None:
    """Why ``verify_repo`` skips a corpus's epoch gate, or ``None`` to run it.

    A non-evidence corpus dir can still be *present* in a bundle as the
    parent of an evidence corpus (``.github`` holding ``.github/workflows``)
    — presence is containment, not coverage."""
    if evidence_only and corpus_dir not in EVIDENCE_CORPORA:
        return "evidence_only"
    if scope is not None and corpus_dir not in scope:
        return "outside_probe_scope"
    return None


def verify_repo(
    root: Path | str = ".",
    *,
    heads_pin: Path | str = "quality/epoch_heads.json",
    evidence_only: bool = False,
    epoch_scope: Collection[str] | None = None,
) -> dict[str, Any]:
    """Run every integrity gate against the live tree.

    Returns ``{"gates": {name: {"ok": bool, "errors": [...]}}, "ok": bool}``.
    ``ok`` is true iff every gate reports zero errors. ``heads_pin`` is the
    committed epoch-heads pin required by every corpus chain.

    ``evidence_only`` verifies an evidence bundle — a tree carrying only the
    evidence dirs (receipts/, quality/, verifier/, configs/,
    .github/workflows/) with no source checkout: the crown-jewels byte pins
    cover ``src/`` and can't be checked, so that gate reports a skipped
    marker instead of a pass. The attestation records the mode so a partial
    verdict can never masquerade as a full-tree one.

    ``epoch_scope`` exists for the self-attack drills only. After a full run
    over a drill clone came back clean, a probe that touched files in corpora
    ``S`` re-runs every non-epoch gate plus the epoch gates of ``S``; every
    other corpus reports ``{"ok": True, "errors": [], "skipped":
    "outside_probe_scope"}``. Those gates were clean at baseline and nothing
    they read changed (an epoch gate reads its own corpus dir plus
    ``EPOCH_SHARED_INPUTS``; touching a shared input means a full run), and
    no gate has side effects, so a scoped run reports a subset of the full
    run's errors: a probe it catches, the full gate catches too. Scoped
    results can't be sealed as an attestation — the receipt contract only
    admits ``evidence_only`` skips. Unknown corpus names raise.
    """
    root = Path(root)
    scope = _checked_epoch_scope(epoch_scope)
    pin_path = root / heads_pin if not Path(heads_pin).is_absolute() else Path(heads_pin)
    # Committed removal acknowledgments — moving a stamped member (e.g. to
    # legacy-unsealed/) or a transient corpus blip is recorded history, not
    # an error, when name->sha256 is declared here. The file is itself a
    # corpus member: silently editing it still trips the epoch gate.
    allowed_removals: dict[str, str] = {}
    ar_path = root / "quality/epoch_allowed_removals.json"
    if ar_path.is_file():
        try:
            raw_ar = json.loads(ar_path.read_text())
            if isinstance(raw_ar, dict):
                allowed_removals = {str(k): str(v) for k, v in raw_ar.items() if isinstance(v, str)}
        except (OSError, json.JSONDecodeError):
            allowed_removals = {}
    gates: dict[str, dict[str, Any]] = {}

    if evidence_only:
        gates["crown_jewels"] = {
            "ok": True,
            "errors": [],
            "skipped": "evidence_only",
        }
    else:
        jewel_errs = crown_jewels_errors(root, pin_path=root / JEWELS_PIN)
        gates["crown_jewels"] = {"ok": not jewel_errs, "errors": jewel_errs}

    sig = verify_pin_signatures(root)
    gates["pin_signatures"] = {
        "ok": bool(sig["ok"]),
        "signed": bool(sig["signed"]),
        "errors": sig["errors"],
    }

    ts = verify_timestamps(root)
    from quant_fund.research.ots_anchor import verify_ots

    ots = verify_ots(root)
    gates["timestamp_anchors"] = {
        "ok": bool(ts["ok"] and ots["ok"]),
        "anchored": bool(ts["anchored"] or ots["anchored"]),
        "fresh": ts.get("fresh", {}),
        "ots": ots.get("attestations", {}),
        "errors": [*ts["errors"], *ots["errors"]],
    }

    cp = verify_checkpoint(root)
    gates["checkpoint"] = {
        "ok": bool(cp["ok"]),
        "signed": bool(cp["signed"]),
        "anchored": bool(cp.get("anchored", False)),
        "current": bool(cp.get("current", False)),
        "revision_ancestor": cp.get("revision_ancestor"),
        "errors": cp["errors"],
    }

    wit = verify_witnesses(root)
    gates["witness"] = {
        "ok": bool(wit["ok"]),
        "witnessed": bool(wit["witnessed"]),
        "errors": wit["errors"],
    }

    if evidence_only:
        # The spine audits continuity of the ``checkpoints/`` + ``witness/``
        # archives — exempt directories a bundle does not carry. The bundle's
        # trust anchor is the signed checkpoint tip instead, which the
        # ``checkpoint`` gate verifies.
        gates["spine"] = {
            "ok": True,
            "skipped": "evidence_only",
            "errors": [],
        }
    else:
        from quant_fund.research.checkpoint_chain import checkpoint_spine

        spine = checkpoint_spine(root)
        gates["spine"] = {
            "ok": bool(spine["ok"]),
            "signed": bool(spine.get("signed", False)),
            "spine_length": spine.get("spine_length", 0),
            "errors": spine["errors"],
        }

    from quant_fund.research.key_rotation import verify_rotations

    rot = verify_rotations(root)
    gates["key_rotation"] = {
        "ok": bool(rot["ok"]),
        "n_rotations": rot["n_rotations"],
        "n_resets": rot.get("n_resets", 0),
        "verdict": rot.get("verdict"),
        "errors": rot["errors"],
    }

    from quant_fund.research.quorum_rotation import verify_quorum_rotations

    qrot = verify_quorum_rotations(root)
    gates["quorum_rotations"] = {
        "ok": bool(qrot["ok"]),
        "n_rotations": qrot["n_rotations"],
        "n_resets": qrot.get("n_resets", 0),
        "verdict": qrot.get("verdict"),
        "errors": qrot["errors"],
    }

    pin_present = pin_path.is_file()
    pin_parse_error: str | None = None
    if pin_present:
        try:
            heads = load_heads_pin(pin_path)
        except (OSError, ValueError, json.JSONDecodeError):
            heads = {}
            pin_parse_error = f"heads_pin_malformed:{heads_pin}"
    else:
        heads = {}
    for corpus_dir, pattern, require_stamped, allow_updates, exempt in CORPORA:
        cdir = root / corpus_dir
        skip = _epoch_skip(corpus_dir, evidence_only=evidence_only, scope=scope)
        if skip is not None:
            gates[f"epoch:{corpus_dir}"] = {"ok": True, "skipped": skip, "errors": []}
            continue
        if not cdir.is_dir():
            gates[f"epoch:{corpus_dir}"] = {
                "ok": False,
                "errors": [f"corpus_missing:{corpus_dir}"],
            }
            continue
        if corpus_dir in LOCAL_ONLY_CORPORA and not any(cdir.glob("corpus_epoch_*.json")):
            # Members clone (committed) but the epoch receipts do not
            # (gitignored) — a checkout can't verify this chain, so the
            # gate reports an explicit skip, not a pass. A machine that
            # did run the stamps has receipts and verifies fully.
            gates[f"epoch:{corpus_dir}"] = {
                "ok": True,
                "skipped": "local_only_no_receipts",
                "errors": [],
            }
            continue
        if not pin_present:
            expected = None
            pin_errors = [f"heads_pin_missing:{heads_pin}"]
        elif pin_parse_error is not None:
            expected = None
            pin_errors = [pin_parse_error]
        else:
            expected = heads.get(epoch_heads_key(corpus_dir, pattern))
            pin_errors = (
                [] if expected is not None else [f"heads_pin_entry_missing:{corpus_dir}/{pattern}"]
            )
        res = check_epoch_chain(
            cdir,
            pattern=pattern,
            expected_head=expected,
            require_stamped=require_stamped,
            allow_member_updates=allow_updates,
            allowed_removals=allowed_removals,
        )
        errs = pin_errors + list(res["errors"])
        # Coverage closure at the corpus policy level: every file in the
        # dir must be a pattern member, an epoch record, a declared
        # exemption, or on the corpus's allowlist — a non-matching file
        # (e.g. a `.yaml` workflow beside `*.yml` members, which GitHub
        # would still run) carries zero chain evidence and is invisible to
        # every check above. Multi-chain dirs are fine here: coverage is
        # per-corpus entry, and each dir has exactly one.
        from fnmatch import fnmatch

        from quant_fund.research.corpus_epoch import (
            EXEMPT_BASENAMES,
            EXEMPT_RELPATHS,
            _exempt_member,
        )

        for f in sorted(cdir.rglob("*")):
            if not f.is_file() or f.name.startswith("corpus_epoch_"):
                continue
            rel = f.relative_to(cdir).as_posix()
            if (
                f.name in EXEMPT_BASENAMES
                or rel in EXEMPT_RELPATHS
                or _exempt_member(cdir, rel)
                or fnmatch(rel, pattern)
                or any(fnmatch(rel, g) for g in exempt)
            ):
                continue
            errs.append(f"uncovered_member:{rel!a}")
        gates[f"epoch:{corpus_dir}"] = {"ok": not errs, "errors": errs}

    if evidence_only:
        # The manifest is the exporter's claim about itself — schema,
        # covered corpora, and per-corpus globs must match this build's
        # policy table, else a forged manifest rides a valid chain.
        manifest_errs: list[str] = []
        mpath = root / "BUNDLE.json"
        if not mpath.is_file():
            manifest_errs.append("bundle_manifest_missing:BUNDLE.json")
        else:
            try:
                manifest = json.loads(mpath.read_text())
            except (OSError, json.JSONDecodeError):
                manifest = {}
                manifest_errs.append("bundle_manifest_malformed:BUNDLE.json")
            if manifest.get("schema") != "evidence_bundle.v1":
                manifest_errs.append(f"manifest_schema:{manifest.get('schema')!r}")
            declared = manifest.get("corpora")
            if not isinstance(declared, Mapping):
                manifest_errs.append("manifest_corpora_malformed")
            else:
                drift = sorted(set(declared) ^ EVIDENCE_CORPORA)
                if drift:
                    manifest_errs.append("manifest_corpora_mismatch:" + ",".join(drift))
                glob_by_corpus = {c[0]: c[1] for c in CORPORA}
                for cdir_name in set(declared) & EVIDENCE_CORPORA:
                    spec = declared[cdir_name]
                    m_glob = spec.get("glob") if isinstance(spec, Mapping) else None
                    if m_glob != glob_by_corpus[cdir_name]:
                        manifest_errs.append(f"manifest_glob_mismatch:{cdir_name}:{m_glob!r}")
        gates["bundle_manifest"] = {"ok": not manifest_errs, "errors": manifest_errs}

        # Closed world: a bundle may carry only the manifest, the signature
        # file, and files under the declared evidence corpora (nested
        # corpora claim by longest prefix — `.github/x` is foreign even
        # though `.github/workflows/` is a corpus). Inside a corpus, the
        # epoch gate's own uncovered-member sweep applies.
        foreign: list[str] = []
        evidence_prefixes = tuple(
            sorted((f"{d}/" for d in EVIDENCE_CORPORA), key=len, reverse=True)
        )
        for f in sorted(root.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(root).as_posix()
            if rel in ("BUNDLE.json", "gate_pins.sig") or rel.startswith(".git/"):
                continue
            if not rel.startswith(evidence_prefixes):
                foreign.append(f"foreign_member:{rel}")
        gates["bundle_coverage"] = {"ok": not foreign, "errors": foreign}

    # Semantic layer: byte-integrity says the corpus is untampered; the
    # lattice gate says its claims are coherent. A contradictory corpus
    # fails the capstone the same as a forged byte.
    lattice_errs: list[str] = []
    lattice_info: dict[str, Any] = {}
    lattice_dir = root / "receipts"
    if not lattice_dir.is_dir():
        lattice_errs.append("lattice_missing:receipts")
    else:
        try:
            from quant_fund.research.receipt_lattice import receipt_lattice

            ki_path = root / "quality" / "lattice_known_inconsistent.json"
            known: dict[str, str] = {}
            if ki_path.is_file():
                ki_doc = json.loads(ki_path.read_text())
                ki_map = ki_doc.get("known", ki_doc) if isinstance(ki_doc, Mapping) else {}
                if isinstance(ki_map, Mapping):
                    known = {str(k): str(v) for k, v in ki_map.items()}
            lat = receipt_lattice(lattice_dir, known_inconsistent=known)
            lattice_info = {
                "verdict": lat["verdict"],
                "n_claim_groups": lat["n_claim_groups"],
                "n_retracted": lat["n_retracted"],
            }
            if lat["n_parse_errors"]:
                lattice_errs.append(f"lattice_parse_errors:{lat['n_parse_errors']}")
            if lat["verdict"] == "inconsistent":
                lattice_errs.append("lattice_inconsistent")
        except Exception as exc:  # noqa: BLE001 — gate fails closed
            lattice_errs.append(f"lattice_error:{type(exc).__name__}: {exc}")
    gates["lattice"] = {
        "ok": not lattice_errs,
        "errors": lattice_errs,
        **lattice_info,
    }

    return {"gates": gates, "ok": all(g["ok"] for g in gates.values())}


# Gates every attestation must carry — the neutral/absent states are legal
# (unsigned pins, unanchored timestamps report ok=True) but the gate itself
# must be *run*: a receipt missing a gate hid the surface, not passed it.
_REQUIRED_GATES = (
    "crown_jewels",
    "pin_signatures",
    "timestamp_anchors",
    "spine",
    "key_rotation",
    "quorum_rotations",
    "lattice",
)


def repo_integrity_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """Contract check for a ``repo_integrity.v1`` receipt body.

    Re-verifying the live gates would need the whole repo; the contract
    instead proves the attestation is internally coherent — that the claimed
    ``ok`` verdict is what the recorded gate verdicts actually imply, and
    that the digest pins can't disagree with the gates they summarize.
    """
    errors: list[str] = []
    if payload.get("schema") != REPO_INTEGRITY_SCHEMA:
        errors.append("schema_mismatch")
    gates = payload.get("gates")
    if not isinstance(gates, Mapping) or not gates:
        errors.append("gates_missing")
        gates = {}
    for name in _REQUIRED_GATES:
        if name not in gates:
            errors.append(f"gate_missing:{name}")
    mode = payload.get("mode", "full")
    if mode not in ("full", "evidence_only"):
        errors.append(f"mode_unknown:{mode}")
        mode = "full"
    for name, gate in gates.items():
        if not isinstance(gate, Mapping):
            errors.append(f"gate_malformed:{name}")
            continue
        g_errors = gate.get("errors")
        if not isinstance(gate.get("ok"), bool) or not isinstance(g_errors, list):
            errors.append(f"gate_malformed:{name}")
            continue
        if not all(isinstance(e, str) for e in g_errors):
            errors.append(f"gate_errors_not_strings:{name}")
        # The gate's own verdict must match its listed errors — a receipt
        # can't claim ok while listing the errors it failed on.
        if bool(gate["ok"]) != (not g_errors):
            errors.append(f"gate_ok_incoherent:{name}")
        if "revision_ancestor" in gate and not (
            gate["revision_ancestor"] is None or isinstance(gate["revision_ancestor"], bool)
        ):
            errors.append(f"gate_field_malformed:{name}:revision_ancestor")
        if "skipped" in gate:
            # Only evidence_only may carry skipped gates, and only for
            # crown_jewels or epoch gates over non-evidence corpora (the
            # code tree isn't part of an evidence bundle): a full-tree
            # attestation silently skipping a gate would downgrade a
            # partial verdict into an implied full pass.
            skip_ok = gate["skipped"] == "evidence_only" and (
                name in ("crown_jewels", "spine")
                or (
                    name.startswith("epoch:")
                    and name.removeprefix("epoch:") not in EVIDENCE_CORPORA
                )
            )
            if not skip_ok:
                errors.append(f"gate_skip_invalid:{name}")
            elif mode != "evidence_only":
                errors.append(f"gate_skipped_in_full_mode:{name}")
    if mode == "evidence_only":
        cj = gates.get("crown_jewels")
        if not isinstance(cj, Mapping) or cj.get("skipped") != "evidence_only":
            errors.append("evidence_only_skip_missing:crown_jewels")
        # Both bundle gates must have run — an attestation over a bundle
        # without them can't claim the manifest agreed or foreign members
        # were absent.
        for req in ("bundle_coverage", "bundle_manifest"):
            bc = gates.get(req)
            if not isinstance(bc, Mapping) or "skipped" in bc:
                errors.append(f"evidence_only_skip_missing:{req}")
    expected_ok = bool(gates) and all(
        bool(g.get("ok")) for g in gates.values() if isinstance(g, Mapping)
    )
    if payload.get("ok") != expected_ok:
        errors.append("ok_incoherent")
    pins = payload.get("pins")
    if isinstance(pins, Mapping):
        sig_gate = gates.get("pin_signatures", {})
        ts_gate = gates.get("timestamp_anchors", {})
        if (
            isinstance(sig_gate, Mapping)
            and "signed" in sig_gate
            and pins.get("gate_pins_signed") != sig_gate["signed"]
        ):
            errors.append("pins_sig_incoherent")
        if (
            isinstance(ts_gate, Mapping)
            and "anchored" in ts_gate
            and pins.get("timestamps_anchored") != ts_gate["anchored"]
        ):
            errors.append("pins_anchor_incoherent")
        for key in ("epoch_heads_sha256", "crown_jewels_sha256"):
            v = pins.get(key)
            if v is not None and not (isinstance(v, str) and len(v) == 64):
                errors.append(f"pin_malformed:{key}")
    else:
        errors.append("pins_missing")
    params = payload.get("params")
    if isinstance(params, Mapping):
        corpora = params.get("corpora")
        if isinstance(corpora, list):
            for c in corpora:
                if f"epoch:{c}" not in gates:
                    errors.append(f"corpus_gate_missing:{c}")
    if not isinstance(payload.get("data_label"), str):
        errors.append("data_label_missing")
    return errors


def repo_integrity_receipt(
    root: Path | str = ".",
    *,
    heads_pin: Path | str = "quality/epoch_heads.json",
    evidence_only: bool = False,
) -> dict[str, Any]:
    """Sealed ``repo_integrity.v1`` attestation over the live gate verdicts."""
    from quant_fund.research.receipt_v2 import seal_receipt

    root = Path(root)
    verdict = verify_repo(root, heads_pin=heads_pin, evidence_only=evidence_only)
    pin_path = root / heads_pin
    body: dict[str, Any] = {
        "kind": "repo_integrity",
        "schema": REPO_INTEGRITY_SCHEMA,
        "data_label": "SYNTHETIC",
        "ok": verdict["ok"],
        "mode": "evidence_only" if evidence_only else "full",
        "gates": {
            # signed/anchored are verdict state, not metadata — an unsigned-
            # tree attestation must be distinguishable from a signed one.
            name: {
                "ok": g["ok"],
                "errors": sorted(g["errors"]),
                **{
                    k: g[k]
                    for k in ("signed", "anchored", "fresh", "skipped", "revision_ancestor")
                    if k in g
                },
            }
            for name, g in verdict["gates"].items()
        },
        "pins": {
            "epoch_heads_sha256": hash_bytes(pin_path.read_bytes()) if pin_path.is_file() else None,
            "crown_jewels_sha256": (
                hash_bytes((root / JEWELS_PIN).read_bytes())
                if (root / JEWELS_PIN).is_file()
                else None
            ),
            "gate_pins_signed": verdict["gates"]["pin_signatures"]["signed"],
            "timestamps_anchored": verdict["gates"]["timestamp_anchors"]["anchored"],
        },
        "params": {"corpora": [c[0] for c in CORPORA]},
    }
    return seal_receipt(body)


def write_repo_integrity_receipt(
    out_path: Path | str,
    root: Path | str = ".",
    *,
    heads_pin: Path | str = "quality/epoch_heads.json",
    evidence_only: bool = False,
) -> Path:
    """Write the sealed attestation to ``out_path`` (atomic)."""
    receipt = repo_integrity_receipt(root, heads_pin=heads_pin, evidence_only=evidence_only)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return out
