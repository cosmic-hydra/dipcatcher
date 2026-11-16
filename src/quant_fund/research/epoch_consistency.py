"""Chain-consistency proofs: prove a newer epoch head *extends* an older one.

``epoch_merkle`` gives the RFC 6962 inclusion half ("was member m in epoch
N?"); this is the consistency half ("does epoch N+K chain back to epoch N?").

The need: a history rewrite produces a perfectly valid *new* chain —
``check_epoch_chain`` verifies internal linkage, not that the live chain
reaches the head a verifier already holds (e.g. a timestamp-anchored pin from
an earlier clone). Walking ``prev_epoch_receipt`` links proves extension: a
rewritten chain can only attach to the held head by keeping every real
intermediate receipt verbatim, at which point it *is* the real history.

Proof shape: the ordered hop list ``[{name, sha256}]`` from the held head to
the current head. Verification re-reads the receipt files and checks per hop:
the successor's ``prev_epoch_receipt`` names the predecessor, the
successor's member map pins the predecessor's file digest, and each file's
bytes hash to the listed digest — so linkage and content both bind. When the
member glob excludes the receipts themselves (``*.md`` / ``*.yml`` corpora)
no successor can pin its predecessor as a member; the predecessor is bound
through its own seal instead: the seal must verify and the file name must
embed it (``corpus_epoch_<receipt_sha256[:16]>.json``). That hop is bound to
its successor only by the 16-hex (64-bit) name prefix, not a full digest.

Provenance evidence only; never a market or P&L claim.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from quant_fund.research.corpus_epoch import _epoch_receipts
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

CONSISTENCY_SCHEMA = "epoch_consistency.v1"


def _seal_binds_name(path: Path) -> bool:
    """True iff the receipt at ``path`` is validly sealed and named after
    its seal — the binding for a hop its successor cannot pin as a member."""
    from quant_fund.research.receipt_v2 import verify_receipt_file

    try:
        doc = json.loads(path.read_text())
    except (OSError, UnicodeError, ValueError):
        return False
    digest = doc.get("receipt_sha256") if isinstance(doc, Mapping) else None
    if not _digest_hex(digest) or path.name != f"corpus_epoch_{str(digest)[:16]}.json":
        return False
    return bool(verify_receipt_file(path)["valid"])


def _predecessor_bound(
    payload: Mapping[str, Any],
    prev: Mapping[str, Any],
    prev_path: Path | None,
    pattern: str,
) -> bool:
    """Does the successor ``payload`` bind its predecessor hop ``prev``?

    The successor's member map must pin the predecessor's bytes — linkage
    alone could point at a file that isn't this hop. Where the member glob
    excludes receipts, no successor can list one, so the predecessor binds
    through its own verified seal instead."""
    member_sha = next(
        (
            m.get("sha256")
            for m in payload.get("members") or []
            if isinstance(m, Mapping) and m.get("name") == prev["name"]
        ),
        None,
    )
    if member_sha is not None or fnmatch(str(prev["name"]), pattern):
        return bool(member_sha == prev["sha256"])
    return prev_path is not None and _seal_binds_name(prev_path)


def _digest_hex(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def chain_index(
    corpus_dir: Path | str, *, pattern: str = "*.json"
) -> dict[str, tuple[Path, Mapping[str, Any], str]]:
    """Name → (path, payload, file sha256) for the (dir, pattern) chain."""
    root = Path(corpus_dir)
    out: dict[str, tuple[Path, Mapping[str, Any], str]] = {}
    for path, payload in _epoch_receipts(root)[0]:
        if not isinstance(payload.get("params") or {}, Mapping):
            continue
        if (payload.get("params") or {}).get("pattern", "*.json") != pattern:
            continue
        out[path.name] = (path, payload, hash_bytes(path.read_bytes()))
    return out


def consistency_proof(
    corpus_dir: Path | str,
    from_receipt: str,
    *,
    pattern: str = "*.json",
    to_receipt: str | None = None,
) -> dict[str, Any]:
    """Proof that the (dir, pattern) chain extends ``from_receipt`` to head.

    Raises ``ValueError`` when ``from_receipt`` is unknown or unreachable from
    the head — no honest proof exists for a rewritten or forked history.
    """
    index = chain_index(corpus_dir, pattern=pattern)
    if from_receipt not in index:
        raise ValueError(f"unknown from_receipt: {from_receipt}")
    prevs = {e.get("prev_epoch_receipt") for _, e, _ in index.values()}
    heads = [name for name in index if name not in prevs]
    to = to_receipt or (sorted(heads)[-1] if heads else None)
    if to is None or to not in index:
        raise ValueError("no chain head to prove extension to")
    # Walk the chain back from `to`; collect the path to `from_receipt`.
    walk = [to]
    seen = {to}
    cur = to
    while cur != from_receipt:
        prev = index[cur][1].get("prev_epoch_receipt")
        if prev is None or prev not in index or prev in seen:
            raise ValueError(
                f"cannot reach {from_receipt} from head {to} — the chain does "
                "not extend it (rewrite or fork)"
            )
        walk.append(prev)
        seen.add(prev)
        cur = prev
    walk.reverse()
    hops = [{"name": n, "sha256": index[n][2]} for n in walk]
    return {
        "kind": CONSISTENCY_SCHEMA,
        "schema": CONSISTENCY_SCHEMA,
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "CORPUS",
        "corpus_dir": str(Path(corpus_dir)),
        "pattern": pattern,
        "from_receipt": hops[0],
        "to_receipt": hops[-1],
        "n_hops": len(hops),
        "hops": hops,
        "proof_sha256": hash_bytes(canonical_json_bytes({"hops": hops, "pattern": pattern})),
    }


def verify_consistency(
    proof: Mapping[str, Any],
    corpus_dir: Path | str,
    *,
    held_sha256: str | None = None,
) -> list[str]:
    """Verify a consistency proof against the live corpus dir.

    ``held_sha256`` is the digest a verifier already trusts for the old head —
    e.g. recovered from an anchored pin or an earlier clone. Passing it proves
    the live chain extends *that exact bytes state*; omitting it proves only
    internal linkage from the claimed start. Errors are fail-closed strings.
    """
    errors: list[str] = []
    errors.extend(consistency_contract_errors(proof))
    if errors:
        return errors
    hops = proof["hops"]
    pattern = str(proof.get("pattern") or "*.json")
    index = chain_index(corpus_dir, pattern=pattern)
    if held_sha256 is not None and hops[0]["sha256"] != held_sha256:
        errors.append("held_head_digest_mismatch")
    prev = None
    prev_path: Path | None = None
    for hop in hops:
        entry = index.get(hop["name"])
        if entry is None:
            errors.append(f"hop_missing:{hop['name']}")
            prev, prev_path = hop, None
            continue
        path, payload, digest = entry
        if digest != hop["sha256"]:
            errors.append(f"hop_digest_mismatch:{hop['name']}")
        if prev is not None:
            if payload.get("prev_epoch_receipt") != prev["name"]:
                errors.append(f"hop_link_broken:{hop['name']}")
            if not _predecessor_bound(payload, prev, prev_path, pattern):
                errors.append(f"hop_member_digest_mismatch:{hop['name']}")
        prev, prev_path = hop, path
    # The `to` endpoint must be a live chain head — proof of extension to a
    # mid-chain node is vacuous (a fork could still rewrite the suffix).
    if hops:
        prevs = {e.get("prev_epoch_receipt") for _, e, _ in index.values()}
        if hops[-1]["name"] in prevs:
            errors.append("to_not_chain_head")
    return errors


def consistency_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """``epoch_consistency.v1`` internal coherence; ``[]`` when clean."""
    errors: list[str] = []
    if payload.get("kind") != CONSISTENCY_SCHEMA:
        errors.append("kind_not_consistency")
    if payload.get("schema") != CONSISTENCY_SCHEMA:
        errors.append("schema_not_consistency")
    if payload.get("research_only") is not True:
        errors.append("research_only_not_true")
    if payload.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    hops = payload.get("hops")
    if not isinstance(hops, list) or not hops:
        errors.append("hops_missing")
        hops = []
    for hop in hops:
        if not isinstance(hop, Mapping) or not _digest_hex(hop.get("sha256")):
            errors.append("hop_malformed")
            break
        if not isinstance(hop.get("name"), str):
            errors.append("hop_name_not_string")
            break
    for endpoint in ("from_receipt", "to_receipt"):
        e = payload.get(endpoint)
        if not isinstance(e, Mapping):
            errors.append(f"{endpoint}_missing")
        elif hops:
            expected = hops[0] if endpoint == "from_receipt" else hops[-1]
            if e.get("name") != expected["name"] or e.get("sha256") != expected["sha256"]:
                errors.append(f"{endpoint}_incoherent")
    n_hops = payload.get("n_hops")
    if not isinstance(n_hops, int) or n_hops != len(hops):
        errors.append("n_hops_mismatch")
    if not _digest_hex(payload.get("proof_sha256")):
        errors.append("proof_sha256")
    return errors
