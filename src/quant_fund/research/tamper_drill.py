"""Tamper drill: end-to-end proof the integrity substrate is fail-closed.

The stack layers receipts → epoch chains → Merkle proofs → Ed25519 pins →
checkpoint → Rekor witness. Every layer has unit tests, but the claim that
matters is *systemic*: a real attacker rewrites bytes somewhere in the
tree — does ``verify_repo`` catch every reachable mutation?

This module runs that adversary for real. ``tamper_drill(root)`` clones the
integrity state (``quality/``, ``gate_pins.sig``, the corpora, the keys'
public halves) into a throwaway tree, applies each mutation probe, and runs
the same ``verify_repo`` CI runs. A drill receipt records per-probe the
verdict and which errors surfaced — anything that slips through is the
finding, not an assertion the author hoped for.

Fail-closed: any probe the clone does NOT catch is a drill failure; the
emitted receipt (``tamper_drill.v1``) is sealed like any other.

Each probe runs on a pristine clone (full snapshot restore after every
probe) and re-runs every non-epoch gate plus the epoch gates of the corpora
it touched — ``verify_repo(epoch_scope=...)``. The baseline is a full run
and must be clean, so skipping an untouched corpus can only drop errors,
never add them: a scoped catch is a full catch. Re-chaining all 28 corpora
for every probe made the drill O(probes x tree) — about half an hour on the
closed-world tree — for gates whose inputs the probe never touched.

Provenance evidence only; never a market or P&L claim.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import hash_bytes

DRILL_SCHEMA = "tamper_drill.v1"


# Integrity state paths the drill clones: every epoch-chained corpus (the
# whole closed world — an omitted dir reports corpus_missing and the drill
# proves nothing) plus the root signature file.
def _clone_members() -> tuple[str, ...]:
    from quant_fund.research.repo_integrity import CORPORA

    return tuple(dict.fromkeys([c[0] for c in CORPORA] + ["gate_pins.sig"]))


def _clone_state(root: Path, clone: Path) -> None:
    for rel in _clone_members():
        src = root / rel
        if src.is_dir():
            # Parent/child corpora overlap (e.g. .github re-enters
            # .github/workflows already cloned) — merge, don't fail.
            shutil.copytree(src, clone / rel, dirs_exist_ok=True)
        elif src.is_file():
            (clone / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, clone / rel)
    # Every crown-pinned file wherever it lives (root files like
    # .gitleaks.toml aren't under the cloned dirs).
    crown = root / "quality/crown_jewels.json"
    if crown.is_file():
        try:
            pinned = json.loads(crown.read_text()).get("files", {})
        except (OSError, ValueError):
            pinned = {}
        for rel in pinned:
            src = root / rel
            dst = clone / rel
            if src.is_file() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)


def _tree_state(clone: Path) -> dict[str, tuple[int, int, int, int, int]]:
    """``lstat`` fingerprint of every entry under ``clone`` — files, dirs and
    symlinks, links not followed. A delete drops an entry, a drop adds one,
    and any write or metadata change stamps the kernel-set ctime (userspace
    can't set it) — so once ``_await_fresh_ctime`` has run, the before/after
    diff names every path a probe touched."""
    state: dict[str, tuple[int, int, int, int, int]] = {}
    for dirpath, dirnames, filenames in os.walk(clone):
        for name in (*dirnames, *filenames):
            full = os.path.join(dirpath, name)
            st = os.lstat(full)
            rel = os.path.relpath(full, clone).replace(os.sep, "/")
            state[rel] = (st.st_mode, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino)
    return state


# Kernel file timestamps come from a clock that advances once per tick
# (<= 10 ms at HZ >= 100); waiting out two ticks past the newest ctime makes
# every later change stamp a strictly newer one.
_CTIME_SETTLE_NS = 25_000_000


def _await_fresh_ctime(state: dict[str, tuple[int, int, int, int, int]]) -> None:
    """Block until the clock is past every ctime in ``state`` by more than a
    timestamp tick, so a change can't reuse the ctime it replaces. A drill
    clone was last written a full verify run earlier — this almost never
    waits. Bounded: after a backwards clock step it gives up, which can only
    make a probe look like an escape (fail loud), never like a catch."""
    newest = max((v[3] for v in state.values()), default=0)
    deadline = time.monotonic() + 0.5
    while time.time_ns() <= newest + _CTIME_SETTLE_NS and time.monotonic() < deadline:
        time.sleep(0.005)


def _probe_scope(
    before: dict[str, tuple[int, int, int, int, int]],
    after: dict[str, tuple[int, int, int, int, int]],
) -> frozenset[str] | None:
    """Epoch corpora a probe touched, or ``None`` (re-run every epoch gate)
    when it touched a shared epoch input. Nested corpora both claim a path:
    ``.github/workflows/x.yml`` is under ``.github`` too."""
    from quant_fund.research.repo_integrity import CORPORA, EPOCH_SHARED_INPUTS

    touched = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
    if touched & EPOCH_SHARED_INPUTS:
        return None
    return frozenset(
        c[0] for c in CORPORA if any(t == c[0] or t.startswith(f"{c[0]}/") for t in touched)
    )


def _flip_first_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())
    if not data:
        raise ValueError(f"{path} is empty — probe undefined")
    data[0] ^= 0x01
    path.write_bytes(bytes(data))


def _reset_probes(quality: Path) -> list[tuple[str, Any]]:
    """Key-reset attacks: tamper with a committed reset record, and forge a
    fully valid reset that declares the live keys lost without the rest of
    the ceremony (keys installed, epochs stamped, pins and checkpoint
    re-signed). The forgery must still surface — terminus mismatch on both
    lineages, an unstamped quality member."""
    probes: list[tuple[str, Any]] = []
    records = sorted(quality.glob("key_reset_*.json"))
    if records:
        probes.append(("flip_key_reset_record", lambda v=records[0]: _flip_first_byte(v)))
    if (quality / "checkpoint.json").is_file() and (quality / "gate_signing.pub").is_file():

        def _forged_reset() -> None:
            from quant_fund.research.gate_signatures import generate_keypair
            from quant_fund.research.key_reset import build_reset

            name, raw, _ = build_reset(
                quality.parent,
                [generate_keypair()],
                reason="forged: keys declared lost by an attacker",
                authorized_by="nobody",
                at="2000-01-01T00:00:00+00:00",
            )
            (quality / name).write_bytes(raw)

        probes.append(("inject_forged_key_reset", _forged_reset))
    return probes


def _probes(clone: Path) -> list[tuple[str, Any]]:
    """The attack classes. Each returns a probe name + a mutation closure."""
    quality = clone / "quality"
    probes: list[tuple[str, Any]] = []

    heads = quality / "epoch_heads.json"
    if heads.is_file():
        probes.append(("flip_epoch_heads_pin", lambda: _flip_first_byte(heads)))
    crown = quality / "crown_jewels.json"
    if crown.is_file():
        probes.append(("flip_crown_jewels", lambda: _flip_first_byte(crown)))
    sig = clone / "gate_pins.sig"
    if sig.is_file():
        probes.append(("flip_gate_signature", lambda: _flip_first_byte(sig)))
    pub = quality / "gate_signing.pub"
    if pub.is_file():
        probes.append(("flip_gate_pubkey", lambda: _flip_first_byte(pub)))
    cp = quality / "checkpoint.json"
    if cp.is_file():
        probes.append(("flip_checkpoint", lambda: _flip_first_byte(cp)))

        def _rollback_prev() -> None:
            body = json.loads(cp.read_text())
            body["payload"]["prev_sha256"] = "f" * 64
            cp.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")

        probes.append(("rewrite_checkpoint_prev", _rollback_prev))
    witness_dir = quality / "witness"
    if witness_dir.is_dir():
        proofs = sorted(witness_dir.glob("*.json"))
        if proofs:
            probes.append(("flip_witness_proof", lambda p=proofs[0]: _flip_first_byte(p)))
    # Membership attacks on every corpus. The victim must be a *recorded*
    # member of the pinned head epoch — glob-matching alone picks up exempt
    # files the chain legitimately ignores (.epoch_stamp.lock, the heads
    # pin, mid-write checkpoint), and mutating those is correctly a no-op.
    from quant_fund.research.repo_integrity import CORPORA

    heads_pin = clone / "quality" / "epoch_heads.json"
    pin_heads: dict[str, Any] = {}
    if heads_pin.is_file():
        try:
            pin_heads = json.loads(heads_pin.read_text()).get("heads", {})
        except (OSError, ValueError):
            pin_heads = {}
    for corpus, pattern, _req, _upd, _ex in CORPORA:
        entry = pin_heads.get(f"{corpus}/{pattern}")
        head_receipt = clone / corpus / str(entry.get("receipt", "")) if entry else None
        victim = None
        if head_receipt is not None and head_receipt.is_file():
            try:
                head_members = json.loads(head_receipt.read_text()).get("members", [])
            except (OSError, ValueError):
                head_members = []
            for m in head_members:
                name = m.get("name") if isinstance(m, dict) else None
                if not isinstance(name, str):
                    continue
                candidate = clone / corpus / name
                # Prefer a non-self-referential member: flipping the chain's
                # own epoch receipts is caught too, but a plain member is
                # the unambiguous attack.
                # Non-empty only: flipping a byte in a 0-byte member is a
                # no-op that cannot drift its digest.
                if (
                    candidate.is_file()
                    and not name.startswith("corpus_epoch_")
                    and candidate.stat().st_size > 0
                ):
                    victim = candidate
                    break
        if victim is not None:
            probes.append((f"flip_member:{corpus}", lambda v=victim: _flip_first_byte(v)))
            probes.append((f"delete_member:{corpus}", lambda v=victim: v.unlink()))
    archive = quality / "checkpoints"
    if archive.is_dir() and cp.is_file():
        # Attack the predecessor the live checkpoint actually references —
        # unreferenced archive entries are content-addressed history a
        # verifier correctly ignores.
        try:
            prev = json.loads(cp.read_text()).get("payload", {}).get("prev_sha256")
        except (OSError, ValueError):
            prev = None
        archive_victim = next(
            (f for f in archive.glob("*.json") if hash_bytes(f.read_bytes()) == prev),
            None,
        )
        if archive_victim is not None:
            probes.append(
                ("flip_archived_prev_checkpoint", lambda v=archive_victim: _flip_first_byte(v))
            )
        # Drop the archive record the live checkpoint links to — the spine
        # gate must report a dangling predecessor.
        if archive_victim is not None:
            probes.append(("drop_archived_prev_checkpoint", lambda v=archive_victim: v.unlink()))

        # Inject a side-chain record: a copy of the live checkpoint with a
        # corrupted signature has a different digest but claims the same
        # prev_sha256 — the spine gate must flag the fork and the orphan.
        def _side_chain() -> None:
            forged = json.loads(cp.read_text())
            forged_sigs = forged.get("signatures")
            if isinstance(forged_sigs, list) and forged_sigs and isinstance(forged_sigs[0], dict):
                sig = str(forged_sigs[0].get("signature", ""))
                forged_sigs[0]["signature"] = ("0" if sig[:1] != "0" else "1") + sig[1:]
            else:
                sig = str(forged.get("signature", ""))
                forged["signature"] = ("0" if sig[:1] != "0" else "1") + sig[1:]
            (quality / "checkpoints" / "zz_injected_fork.json").write_text(json.dumps(forged))

        probes.append(("inject_side_chain_checkpoint", _side_chain))

        # Inject an UNAUTHORIZED key rotation: dual-signed, internally
        # consistent — but the 'old' key never signed a spine member. The
        # key_rotation gate must flag the unanchored genesis and (since the
        # forged terminus != live pub) the stale live key.
        def _evil_rotation() -> None:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )
            from cryptography.hazmat.primitives.serialization import (
                Encoding,
                PublicFormat,
            )

            from quant_fund.research.gate_signatures import generate_keypair, key_id
            from quant_fund.utils.hashing import canonical_json_bytes

            old_seed, _ = generate_keypair()
            new_seed, _ = generate_keypair()
            ok_ = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(old_seed))
            nk = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(new_seed))
            op = ok_.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
            np_ = nk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
            payload = {
                "at": "2000-01-01T00:00:00+00:00",
                "reason": "forged authorization",
                "old_key_id": key_id(op),
                "new_key_id": key_id(np_),
                "old_pubkey": op,
                "new_pubkey": np_,
            }
            body = {
                "schema": "key_rotation.v1",
                "payload": payload,
                "old_signature": ok_.sign(canonical_json_bytes(payload)).hex(),
                "new_signature": nk.sign(canonical_json_bytes(payload)).hex(),
            }
            (quality / "rotation_evil.json").write_text(json.dumps(body) + "\n")

        probes.append(("inject_unauthorized_rotation", _evil_rotation))
    probes += _reset_probes(quality)
    return probes


def tamper_drill(root: str | Path) -> dict[str, Any]:
    """Clone the integrity state, run every probe, record the verdicts.

    Returns a ``tamper_drill.v1`` payload: ``ok`` only when every probe was
    caught (``verify_repo`` returned errors). The receipt is evidence the
    stack fails closed *measured*, not asserted.
    """
    from quant_fund.research.repo_integrity import verify_repo

    root_path = Path(root)
    with tempfile.TemporaryDirectory(prefix="tamper_drill_") as tmp:
        clone = Path(tmp) / "clone"
        clone.mkdir(parents=True)
        _clone_state(root_path, clone)
        # Baseline must be clean first — a drill over a broken tree proves nothing.
        baseline = verify_repo(clone)
        baseline_errors = sorted(
            f"{g}:{e}"
            for g, gate in baseline.get("gates", {}).items()
            for e in (gate.get("errors") or [])
        )
        probes: list[dict[str, Any]] = []
        if baseline_errors:
            return {
                "schema": DRILL_SCHEMA,
                "research_only": True,
                "live_pnl_claim": False,
                "data_label": "CORPUS",
                "simulated_only": False,
                "ok": False,
                "baseline_errors": baseline_errors,
                "probes": [],
                "n_probes": 0,
                "n_caught": 0,
                "verdict": "baseline_dirty",
            }
        snapshot = Path(tmp) / "snapshot"
        for name, mutate in _probes(clone):
            shutil.rmtree(snapshot, ignore_errors=True)
            shutil.copytree(clone, snapshot)
            before = _tree_state(clone)
            _await_fresh_ctime(before)
            try:
                mutate()
            except (OSError, ValueError) as exc:
                probes.append({"probe": name, "error": f"probe_setup:{exc}", "caught": False})
            else:
                scope = _probe_scope(before, _tree_state(clone))
                res = verify_repo(clone, epoch_scope=scope)
                gate_errors = sorted(
                    f"{g}:{e}"
                    for g, gate in res.get("gates", {}).items()
                    for e in (gate.get("errors") or [])
                )
                probes.append(
                    {
                        "probe": name,
                        "caught": not res.get("ok", True),
                        "errors": gate_errors[:12],
                        "epoch_scope": "all" if scope is None else sorted(scope),
                    }
                )
            finally:
                # Restore the pristine clone for the next probe.
                shutil.rmtree(clone)
                shutil.move(str(snapshot), clone)
    n_caught = sum(1 for p in probes if p.get("caught"))
    return {
        "schema": DRILL_SCHEMA,
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "CORPUS",
        "simulated_only": False,
        "ok": n_caught == len(probes) and bool(probes),
        "baseline_errors": [],
        "probes": probes,
        "n_probes": len(probes),
        "n_caught": n_caught,
        "verdict": "fail_closed" if n_caught == len(probes) and probes else "probe_escaped",
    }


def drill_contract_errors(payload: Any) -> list[str]:
    """``tamper_drill.v1`` internal consistency; ``[]`` when clean."""
    if not isinstance(payload, dict) or payload.get("schema") != DRILL_SCHEMA:
        return ["schema_mismatch"]
    errors: list[str] = []
    if payload.get("research_only") is not True:
        errors.append("research_only_not_true")
    if payload.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    probes = payload.get("probes")
    n_probes = payload.get("n_probes")
    n_caught = payload.get("n_caught")
    if not isinstance(probes, list):
        errors.append("probes_not_list")
        probes = []
    for i, p in enumerate(probes):
        if not isinstance(p, dict) or not isinstance(p.get("caught"), bool):
            errors.append(f"probe_malformed:{i}")
    if not isinstance(n_probes, int) or n_probes != len(probes):
        errors.append("n_probes_mismatch")
    if not isinstance(n_caught, int) or n_caught != sum(
        1 for p in probes if p.get("caught") is True
    ):
        errors.append("n_caught_mismatch")
    verdict = payload.get("verdict")
    if verdict not in ("fail_closed", "probe_escaped", "baseline_dirty"):
        errors.append("verdict_unknown")
    elif verdict == "fail_closed" and n_caught != n_probes:
        errors.append("verdict_fail_closed_but_escaped")
    return errors


def write_drill_receipt(payload: dict[str, Any], out_dir: Path) -> Path:
    """Seal the drill outcome like any other evidence — even a failure."""
    from quant_fund.research.receipt_v2 import seal_receipt
    from quant_fund.utils.atomicio import atomic_write_text

    errors = drill_contract_errors(payload)
    if errors:
        raise ValueError(f"tamper_drill receipt violates contract: {errors}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sealed = seal_receipt(payload)
    path = out_dir / f"tamper_drill_{sealed['receipt_sha256'][:16]}.json"
    atomic_write_text(path, json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    return path
