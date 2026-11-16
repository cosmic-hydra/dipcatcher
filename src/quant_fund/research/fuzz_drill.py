"""Metamorphic fuzzing of the integrity verifier — the two-sided property.

``tamper_drill`` proves the verifier catches scripted attacks. A degenerate
verifier that fails *everything* would ace that drill. This lane closes the
blind spot: seeded random mutations are classified by what the gates
*should* say — mutations attacking covered state must fail (``escaped`` if
they pass); mutations to legitimately-uncovered or white-space-legal state
must still pass (``false_positive`` if they fail). Both directions are
measured against the real ``verify_repo`` on a cloned tree.

Mutation classes are sampled from the same surfaces the scripted drill
covers plus negative space (unpinned files, non-corpus writes, comments in
already-mutable corpora). ``expect`` is decided *before* the mutation runs
— the verifier's answer is then graded against it, making this a
metamorphic test: the oracle is the semantics of the layer, not a hard-coded
string match.

Seeded: the receipt carries ``seed`` + per-mutation outcomes; the same seed
replays the same attack sequence byte-for-byte.

Provenance evidence only; never a market or P&L claim.
"""

from __future__ import annotations

import json
import random
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from quant_fund.research.tamper_drill import (
    _await_fresh_ctime,
    _clone_state,
    _probe_scope,
    _tree_state,
)
from quant_fund.utils.hashing import hash_bytes

FUZZ_SCHEMA = "fuzz_drill.v1"

# (name, expected verify-repo ok) — "fail" for covered-state attacks,
# "ok" for mutations on state the gates deliberately do not cover.
_EXPECT_FAIL = "fail"
_EXPECT_OK = "ok"


def _flip_byte(path: Path, rng: random.Random) -> str:
    data = bytearray(path.read_bytes())
    if not data:
        raise ValueError("empty file")
    i = rng.randrange(len(data))
    data[i] ^= 1 << rng.randrange(8)
    path.write_bytes(bytes(data))
    return f"byte[{i}]^={data[i]:02x}"


def _corpus_file(root: Path, path: Path) -> Path:
    """Resolve *path* and require it to stay under *root* (fail closed)."""
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError(f"refusing to mutate outside corpus root: {path}")
    return resolved


def _truncate(root: Path, path: Path, rng: random.Random) -> str:
    path = _corpus_file(root, path)
    data = path.read_bytes()
    if len(data) < 8:
        raise ValueError("too small to truncate")
    keep = rng.randrange(4, len(data) - 4)
    path.write_bytes(data[:keep])
    return f"truncated_to:{keep}"


def _json_key_rename(root: Path, path: Path, rng: random.Random) -> str:
    path = _corpus_file(root, path)
    body = json.loads(path.read_bytes())
    if not isinstance(body, dict) or not body:
        raise ValueError("not a json object")
    key = rng.choice(sorted(body))
    body[f"{key}_x"] = body.pop(key)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return f"renamed:{key}"


def _corpus_members(clone: Path) -> list[Path]:
    """Every file any corpus covers (union, deduped)."""
    from quant_fund.research.repo_integrity import CORPORA

    out: list[Path] = []
    for corpus, pattern, _req, _upd, _ex in CORPORA:
        base = clone / corpus
        if not base.is_dir():
            continue
        for p in base.rglob(pattern):
            if p.is_file():
                out.append(p)
    return sorted(set(out))


def _covered_files(clone: Path) -> dict[str, list[Path]]:
    """Classify targets: pinned (jewel), corpus member, or uncovered."""
    crown = clone / "quality/crown_jewels.json"
    pinned: list[Path] = []
    if crown.is_file():
        try:
            pinned = [clone / rel for rel in json.loads(crown.read_text()).get("files", {})]
            pinned = [p for p in pinned if p.is_file()]
        except (OSError, ValueError):
            pinned = []
    return {"pinned": pinned, "corpus": _corpus_members(clone)}


def _mutations(clone: Path, rng: random.Random) -> list[tuple[str, str, Any]]:
    """Sample the attack surface: (name, expect, mutate-closure)."""
    cov = _covered_files(clone)
    corpus = [p for p in cov["corpus"] if p.is_file()]
    pinned = [p for p in cov["pinned"] if p.is_file()]
    out: list[tuple[str, str, Any]] = []

    # --- must-fail: covered state ---
    if corpus:
        v = rng.choice(corpus)
        out.append(("corpus_byte_flip", _EXPECT_FAIL, lambda v=v: _flip_byte(v, rng)))
        if v.suffix == ".json":
            out.append(
                ("corpus_key_rename", _EXPECT_FAIL, lambda v=v: _json_key_rename(clone, v, rng))
            )
        v2 = rng.choice(corpus)
        out.append(("corpus_truncate", _EXPECT_FAIL, lambda v=v2: _truncate(clone, v2, rng)))
    if pinned:
        v = rng.choice(pinned)
        out.append(("jewel_byte_flip", _EXPECT_FAIL, lambda v=v: _flip_byte(v, rng)))
        out.append(("jewel_delete", _EXPECT_FAIL, lambda v=v: v.unlink()))
    sig = clone / "gate_pins.sig"
    if sig.is_file():
        out.append(("pin_sig_flip", _EXPECT_FAIL, lambda: _flip_byte(sig, rng)))
    spine = sorted((clone / "quality/checkpoints").glob("*.json"))
    if spine:
        v = rng.choice(spine)
        out.append(("spine_record_flip", _EXPECT_FAIL, lambda v=v: _flip_byte(v, rng)))
        out.append(("spine_record_delete", _EXPECT_FAIL, lambda v=v: v.unlink()))
    wit = sorted((clone / "quality/witness").glob("*.json"))
    if wit:
        v = rng.choice(wit)
        out.append(("witness_proof_flip", _EXPECT_FAIL, lambda v=v: _flip_byte(v, rng)))
        out.append(("witness_proof_delete", _EXPECT_FAIL, lambda v=v: v.unlink()))
    live_cp = clone / "quality/checkpoint.json"
    if live_cp.is_file():
        out.append(("checkpoint_delete", _EXPECT_FAIL, lambda: live_cp.unlink()))
    # Unstamped corpus arrival (immutable corpus): must fail.
    receipts = clone / "receipts"
    if receipts.is_dir():

        def _inject() -> str:
            fake = receipts / f"fuzz_{rng.randrange(1 << 32):08x}.json"
            fake.write_text('{"schema": "injected", "ok": true}\n')
            return f"injected:{fake.name}"

        out.append(("unstamped_receipt_inject", _EXPECT_FAIL, _inject))

    # --- crafted-policy attacks: not byte noise, semantic forgeries ---
    heads_pin = clone / "quality/epoch_heads.json"
    if heads_pin.is_file():

        def _rollback() -> str:
            doc = json.loads(heads_pin.read_text())
            heads = doc.get("heads") if isinstance(doc, dict) else None
            if not isinstance(heads, dict):
                return "no_heads"
            for key, pin in heads.items():
                if not isinstance(pin, dict):
                    continue
                corpus = key.rsplit("/", 1)[0] if "/" in key else key
                pinned_name = str(pin.get("receipt") or "")
                epochs = sorted((clone / corpus).glob("corpus_epoch_*.json"))
                older = [e for e in epochs if e.name != pinned_name]
                if older:
                    victim = rng.choice(older)
                    from quant_fund.utils.hashing import hash_bytes

                    heads[key] = {
                        "receipt": victim.name,
                        "sha256": hash_bytes(victim.read_bytes()),
                    }
                    heads_pin.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
                    return f"rolled:{key}->{victim.name}"
            return "no_rollback_target"

        out.append(("heads_pin_rollback", _EXPECT_FAIL, _rollback))

        def _forge_epoch() -> str:
            target_dir = clone / "quality"
            from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

            body: dict[str, Any] = {
                "schema": "corpus_epoch.v1",
                "kind": "corpus_epoch.v1",
                "params": {"corpus": "quality", "pattern": "*.json"},
                "members": {"forged": "0" * 64},
                "n_members": 1,
                "members_added": ["forged"],
                "members_removed": [],
                "prev_epoch_receipt": None,
                "prev_epoch_sha256": None,
                "epoch_root_sha256": "0" * 64,
                "verdict": "genesis",
            }
            body["receipt_sha256"] = hash_bytes(canonical_json_bytes(body))
            path = target_dir / f"corpus_epoch_{body['receipt_sha256'][:16]}.json"
            path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
            return f"forged:{path.name}"

        out.append(("epoch_record_forge", _EXPECT_FAIL, _forge_epoch))

    allowed = clone / "quality/epoch_allowed_removals.json"
    quality_members = sorted(p for p in (clone / "quality").glob("*.json") if p.is_file())
    if allowed.is_file() and len(quality_members) > 2:

        def _launder() -> str:
            victim = rng.choice(
                [
                    p
                    for p in quality_members
                    if p.name not in ("epoch_allowed_removals.json", "epoch_heads.json")
                ]
            )
            victim.unlink()
            doc = json.loads(allowed.read_text())
            doc[victim.name] = "0" * 64  # forged allowance for the removal
            allowed.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
            return f"laundered:{victim.name}"

        out.append(("allowed_removals_launder", _EXPECT_FAIL, _launder))

    # Foreign-extension drop: a file matching no corpus pattern or exemption
    # (a `.yaml` workflow beside `*.yml` members — GitHub would still run it)
    # must surface as uncovered_member under the coverage-closure gate.
    wf_dir = clone / ".github" / "workflows"
    if wf_dir.is_dir():

        def _foreign_ext() -> str:
            p = wf_dir / f"fuzz_{rng.randrange(1 << 20)}.yaml"
            p.write_text("name: evil\n")
            return f"dropped:{p.name}"

        out.append(("uncovered_extension_drop", _EXPECT_FAIL, _foreign_ext))

    # --- name-semantics attacks: the member-name gate family must fire ---
    if receipts.is_dir():

        def _symlink_drop() -> str:
            import os

            link = receipts / f"fuzz_{rng.randrange(1 << 20)}.json"
            os.symlink("corpus_epoch.json", link)
            return f"symlinked:{link.name}"

        out.append(("member_symlink_drop", _EXPECT_FAIL, _symlink_drop))

        def _control_name() -> str:
            name = f"fuzz_{rng.randrange(1 << 20)}\nall_gates_intact.json"
            (receipts / name).write_text("{}")
            return "control-char-name"

        out.append(("member_name_control_char", _EXPECT_FAIL, _control_name))

        def _nfd_name() -> str:
            import unicodedata

            name = unicodedata.normalize("NFD", "café_fuzz.json")
            (receipts / name).write_text("{}")
            return "nfd-name"

        out.append(("member_name_nfd", _EXPECT_FAIL, _nfd_name))

        def _prefix_squat() -> str:
            name = f"corpus_epoch_fuzz_{rng.randrange(1 << 20)}.json"
            (receipts / name).write_text('{"not": "an epoch"}')
            return f"squatter:{name}"

        out.append(("epoch_prefix_squat_drop", _EXPECT_FAIL, _prefix_squat))

    # --- must-pass: legitimately uncovered state ---
    # Repo-root files sit above every corpus dir and aren't jewels — a
    # dropped scratch file there is the only honest benign mutation left
    # under closed-world coverage.
    target = clone / "FUZZ_NOTE.md"
    if not target.exists():

        def _write_uncovered() -> str:
            target.write_text("# fuzz probe — root files are unchained\n")
            return "created:FUZZ_NOTE.md"

        out.append(("uncovered_write", _EXPECT_OK, _write_uncovered))
    else:
        out.append(("uncovered_write", _EXPECT_OK, lambda: _flip_byte(target, rng)))
    return out


def fuzz_drill(root: str | Path, seed: int = 1, rounds: int | None = None) -> dict[str, Any]:
    """Seeded metamorphic drill: every mutation graded against its expected
    verdict — must-fail attacks that pass are ``escaped``, legal mutations
    that fail are ``false_positive``."""
    from quant_fund.research.repo_integrity import verify_repo

    root_path = Path(root)
    rng = random.Random(seed)
    with tempfile.TemporaryDirectory(prefix="fuzz_drill_") as tmp:
        clone = Path(tmp) / "clone"
        clone.mkdir(parents=True)
        _clone_state(root_path, clone)
        baseline = verify_repo(clone)
        baseline_errors = sorted(
            f"{g}:{e}"
            for g, gate in baseline.get("gates", {}).items()
            for e in (gate.get("errors") or [])
        )
        if baseline_errors:
            return {
                "schema": FUZZ_SCHEMA,
                "research_only": True,
                "live_pnl_claim": False,
                "data_label": "CORPUS",
                "simulated_only": False,
                "ok": False,
                "seed": seed,
                "baseline_errors": baseline_errors,
                "mutations": [],
                "n_mutations": 0,
                "n_escaped": 0,
                "n_false_positive": 0,
                "verdict": "baseline_dirty",
            }
        mutations = _mutations(clone, rng)
        rng.shuffle(mutations)
        if rounds is not None:
            mutations = mutations[:rounds]
        snapshot = Path(tmp) / "snapshot"
        results: list[dict[str, Any]] = []
        for name, expect, mutate in mutations:
            shutil.rmtree(snapshot, ignore_errors=True)
            shutil.copytree(clone, snapshot)
            before = _tree_state(clone)
            _await_fresh_ctime(before)
            try:
                try:
                    detail = mutate()
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    results.append({"mutation": name, "expect": expect, "skipped": f"setup:{exc}"})
                    continue
                if str(detail).startswith("no_"):
                    # Closure could not apply the mutation on this tree — a
                    # no-op must not be graded as an escape.
                    results.append({"mutation": name, "expect": expect, "skipped": str(detail)})
                    continue
                # Must-fail mutations re-verify scoped to the corpora they
                # touched (a scoped catch is a full catch — see
                # ``verify_repo(epoch_scope=...)``). Must-pass mutations run
                # every gate: a scoped pass can't prove the full verifier
                # stays quiet.
                scope = _probe_scope(before, _tree_state(clone)) if expect == _EXPECT_FAIL else None
                res = verify_repo(clone, epoch_scope=scope)
                got_ok = bool(res.get("ok", True))
                gate_errors = sorted(
                    f"{g}:{e}"
                    for g, gate in res.get("gates", {}).items()
                    for e in (gate.get("errors") or [])
                )
                entry: dict[str, Any] = {
                    "mutation": name,
                    "expect": expect,
                    "detail": detail,
                    "verifier_ok": got_ok,
                    "epoch_scope": "all" if scope is None else sorted(scope),
                }
                if expect == _EXPECT_FAIL and got_ok:
                    entry["outcome"] = "escaped"
                elif expect == _EXPECT_OK and not got_ok:
                    entry["outcome"] = "false_positive"
                    entry["errors"] = gate_errors[:8]
                else:
                    entry["outcome"] = "correct"
                    if gate_errors:
                        entry["errors"] = gate_errors[:8]
                results.append(entry)
            finally:
                # Restore after every mutation, skipped ones included — a
                # half-applied setup must not leak into the next mutation.
                shutil.rmtree(clone)
                shutil.move(str(snapshot), clone)
    n_escaped = sum(1 for r in results if r.get("outcome") == "escaped")
    n_fp = sum(1 for r in results if r.get("outcome") == "false_positive")
    n_ran = sum(1 for r in results if "outcome" in r)
    return {
        "schema": FUZZ_SCHEMA,
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "CORPUS",
        "simulated_only": False,
        "ok": n_escaped == 0 and n_fp == 0 and n_ran > 0,
        "seed": seed,
        "baseline_errors": [],
        "mutations": results,
        "n_mutations": len(results),
        "n_escaped": n_escaped,
        "n_false_positive": n_fp,
        "tree_sha256": hash_bytes(json.dumps(results, sort_keys=True).encode()),
        "verdict": "calibrated" if n_escaped == 0 and n_fp == 0 else "miscalibrated",
    }


RECEIPT_FUZZ_SCHEMA = "receipt_fuzz.v1"


def _claim_leaves(body: dict[str, Any]) -> list[tuple[list[str], Any]]:
    """Claim-bearing leaf paths: verdict/ok booleans, metrics, counts."""
    out: list[tuple[list[str], Any]] = []

    def walk(node: Any, path: list[str]) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, [*path, str(k)])
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, [*path, str(i)])
        else:
            out.append((path, node))

    walk(body, [])
    return [
        (p, v)
        for p, v in out
        if p
        and (
            p[-1] in ("ok", "verdict", "passed")
            or isinstance(v, bool)
            or (isinstance(v, (int, float)) and not isinstance(v, bool))
            and p[-1] != "receipt_sha256"
        )
    ]


def _set_path(body: dict[str, Any], path: list[str], value: Any) -> None:
    node: Any = body
    for key in path[:-1]:
        node = node[int(key)] if isinstance(node, list) else node[key]
    last = path[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def _receipt_variants(doc: dict[str, Any], rng: random.Random) -> list[tuple[str, dict[str, Any]]]:
    """Resealed forgeries: mutate a claim, recompute every seal honestly —
    an attacker with repo access can do no better."""
    from quant_fund.research.receipt_v2 import seal_receipt

    out: list[tuple[str, dict[str, Any]]] = []
    is_v2 = (
        isinstance(doc.get("payload"), dict)
        and "payload_sha256" in doc
        or doc.get("kind") == "receipt.v2"
    )

    def _mutate_leaf(base: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
        leaves = [
            (p, v)
            for p, v in _claim_leaves(base)
            if p[-1] not in ("receipt_sha256", "payload_sha256", "dataset_sha256")
        ]
        if not leaves:
            return None
        path, val = rng.choice(leaves)
        mutant = json.loads(json.dumps(base))
        if isinstance(val, bool):
            _set_path(mutant, path, not val)
            tag = "bool_flip"
        elif isinstance(val, (int, float)):
            _set_path(mutant, path, val * 1.5 + 1.0 if val else 0.5)
            tag = "metric_edit"
        else:
            _set_path(mutant, path, "forged")
            tag = "field_edit"
        return f"{tag}:{'.'.join(path[-2:])}", mutant

    if is_v2:
        inner = doc.get("payload")
        if isinstance(inner, dict):
            hit = _mutate_leaf(inner)
            if hit:
                tag, forged_inner = hit
                env = json.loads(json.dumps(doc))
                # Re-bind every digest honestly: inner seal, the envelope's
                # payload pin, then the outer seal — sha256 is integrity,
                # not authenticity, so a full forgery is free to mint.
                from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

                forged_inner = seal_receipt(forged_inner)
                env["payload"] = forged_inner
                if "payload_sha256" in env:
                    env["payload_sha256"] = hash_bytes(canonical_json_bytes(dict(forged_inner)))
                out.append((f"v2_{tag}", seal_receipt(env)))
    else:
        hit = _mutate_leaf(doc)
        if hit:
            tag, forged = hit
            out.append((f"v1_{tag}", seal_receipt(forged)))
    return out


def receipt_fuzz(root: str | Path, seed: int = 1) -> dict[str, Any]:
    """Forge-and-reseal drill over the committed receipt corpus: every
    mutation mints a *self-consistent* seal — only semantic contract
    re-derivation can catch it. ``escaped`` = a forged claim verified."""
    from quant_fund.research.receipt_v2 import verify_receipt_file

    rng = random.Random(seed)
    results: list[dict[str, Any]] = []
    receipts_dir = Path(root) / "receipts"
    for path in sorted(receipts_dir.glob("*.json")):
        if path.name.startswith("corpus_epoch_"):
            continue
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(doc, dict):
            continue
        for tag, forged in _receipt_variants(doc, rng):
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
                json.dump(forged, fh, indent=2, sort_keys=True)
                tmp_path = fh.name
            try:
                ver = verify_receipt_file(tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
            entry: dict[str, Any] = {
                "mutation": tag,
                "receipt": path.name,
                "expect": "fail",
                "verifier_ok": bool(ver.get("valid")),
            }
            if ver.get("valid"):
                entry["outcome"] = "escaped"
            else:
                entry["outcome"] = "correct"
                entry["errors"] = (ver.get("errors") or [])[:4]
            results.append(entry)
    n_escaped = sum(1 for r in results if r.get("outcome") == "escaped")
    return {
        "schema": RECEIPT_FUZZ_SCHEMA,
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "CORPUS",
        "simulated_only": False,
        "ok": n_escaped == 0 and len(results) > 0,
        "seed": seed,
        "mutations": results,
        "n_mutations": len(results),
        "n_escaped": n_escaped,
        "n_false_positive": 0,
        "verdict": "calibrated" if n_escaped == 0 else "miscalibrated",
    }


def fuzz_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("ok") and (payload.get("n_escaped") or payload.get("n_false_positive")):
        errors.append("ok_with_escapes")
    if payload.get("verdict") == "calibrated" and not payload.get("ok"):
        errors.append("calibrated_not_ok")
    for m in payload.get("mutations") or []:
        if (
            m.get("expect") == "fail"
            and m.get("verifier_ok") is True
            and m.get("outcome") not in (None, "escaped")
        ):
            errors.append(f"outcome_incoherent:{m.get('mutation')}")
    return errors


def write_fuzz_receipt(payload: dict[str, Any], out_dir: Path) -> Path:
    from quant_fund.research.receipt_v2 import seal_receipt
    from quant_fund.utils.atomicio import atomic_write_text

    errs = fuzz_contract_errors(payload)
    if errs:
        raise ValueError(f"fuzz receipt contract: {errs}")
    sealed = seal_receipt(payload)
    prefix = "receipt_fuzz" if payload.get("schema") == RECEIPT_FUZZ_SCHEMA else "fuzz_drill"
    name = f"{prefix}_{hash_bytes(json.dumps(payload, sort_keys=True).encode())[:16]}.json"
    out = out_dir / name
    atomic_write_text(out, json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    return out
