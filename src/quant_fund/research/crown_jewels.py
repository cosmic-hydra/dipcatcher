"""Crown-jewel byte pins for the gate-defining files.

The epoch chains cover ``receipts/``, ``verifier/``, ``quality/`` and
``.github/workflows/*.yml`` — but the files that *define* the gates live at
repo root: lint/type/test config, the dependency lock, the Makefile that
names the checks, the pre-commit hooks, the secret-scan allowlist, and
``AGENTS.md`` — the honesty contract itself. A silent edit there weakens
every check downstream of it.

``quality/crown_jewels.json`` pins ``relpath -> sha256`` for exactly
``DEFAULT_JEWELS``. The coverage set is code-defined on purpose: removing a
pin entry fails ``jewel_unpinned``, an entry outside the set fails
``jewel_unexpected`` — the pin cannot quietly shrink or grow. A tampered or
deleted file fails ``jewel_mutated`` / ``jewel_missing``; a pinned path that
is a symlink fails ``jewel_symlink`` (content substitution).

The pin file is itself a member of the ``quality/*.json`` epoch chain, so
its bytes are chain-stamped; the recursion bottoms out at git review of the
pin/epoch diff — the documented trust boundary, same as the epoch head pin.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import hash_bytes

CROWN_JEWELS_SCHEMA = "crown_jewels.v1"
DEFAULT_PIN_PATH = Path("quality/crown_jewels.json")

# The files that define or authorize every gate — byte-pinned exactly. Adding
# a jewel is a code change here (review-visible), not a pin edit. The
# verifier's own source is a jewel too: the strongest chain is useless if
# the code that checks it can be silently rewritten to `return ok`.
#
# The second block is the tracked root-file set. Root files sit above every
# corpus dir, so no epoch chain can reach them — this pin is the *only*
# content gate they get. ``gate_pins.sig`` is deliberately excluded: it is
# re-minted on every sign, and pinning its bytes would be self-referential
# (the signature authenticates the pins; the pins cannot pin the signature).
DEFAULT_JEWELS: tuple[str, ...] = (
    # --- tracked root files: no epoch coverage by construction ---
    ".dockerignore",  # what enters docker build contexts — supply-chain surface
    ".env.example",  # documents the secret names the tree expects
    ".gitattributes",  # line-ending/LFS rules — flips bytes at checkout
    ".gitignore",  # un-ignoring a path class can leak credentials silently
    ".python-version",  # interpreter pin
    ".test_durations",  # shard-balance timing data — skew = uneven shards
    "APPLY.md",  # fx-1 intake notes (tracked input)
    "CHANGELOG.md",  # release ledger
    "CITATION.cff",  # attribution metadata
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",  # contributor rules
    "Dockerfile",  # the container definition — a build-env tamper surface
    "HONESTY_RATING.md",  # the public honesty ledger
    "INFLIGHT",  # work ledger
    "LICENSE",  # the legal terms — a swapped license text is unreviewed relicensing
    "README.md",  # the front door — install instructions are a hijack surface
    "MATH_SPEC.md",  # fx-1 math contract (tracked input)
    "RESEARCH_REFERENCES.md",  # fx-1 bibliography (tracked input)
    "SECURITY.md",  # disclosure policy
    "day_grind_progress.md",  # progress tracker
    "docker-compose.yml",  # local service definitions
    # --- gate-defining files ---
    ".gitleaks.toml",  # secret-scan allowlist — removing a rule opens exfil lanes
    ".github/dependabot.yml",  # dep-bump policy — tampering injects malicious upgrades
    ".github/codeql/codeql-config.yml",  # CodeQL weakening — security-scan evasion
    ".pre-commit-config.yaml",  # local hook stack
    "AGENTS.md",  # the honesty contract
    "Makefile",  # the gate command definitions
    "conftest.py",  # global pytest fixtures/collection
    "mkdocs.yml",  # docs nav gate
    "pyproject.toml",  # deps, ruff/mypy/pytest config
    "tests/conftest.py",  # suite-level fixtures
    "uv.lock",  # the supply-chain lock
    "quality/gate_signing.pub",  # the pin-signature trust root
    "quality/witness_signing.pub",  # Rekor witness-signature key
    "quality/rekor_pubkey.pem",  # the public log's verification key — a swapped
    # copy would launder forged SET/checkpoint-note signatures
    "quality/timestamps/freetsa_cacert.pem",  # TSA chain verify root — a swapped
    "quality/timestamps/freetsa_tsa.crt",  # cert pair would launder forged anchors
    "fx1_seed_corpus.jsonl",  # tracked training input — a silent edit changes
    # what the model learns with no gate noticing
    # --- the verifier itself: a silent rewrite beats every layer above ---
    "src/quant_fund/research/admission.py",
    "src/quant_fund/research/auditor_bundle.py",
    "src/quant_fund/research/checkpoint_chain.py",
    "src/quant_fund/research/corpus_epoch.py",
    "src/quant_fund/research/corpus_inference.py",
    "src/quant_fund/research/crown_jewels.py",
    "src/quant_fund/research/custody.py",
    "src/quant_fund/research/epoch_consistency.py",
    "src/quant_fund/research/epoch_delta.py",
    "src/quant_fund/research/epoch_merkle.py",
    "src/quant_fund/research/fuzz_drill.py",
    "src/quant_fund/research/gate_signatures.py",
    "src/quant_fund/research/integrity_checkpoint.py",
    "src/quant_fund/research/integrity_witness.py",
    "src/quant_fund/research/key_rotation.py",
    "src/quant_fund/research/lane_contracts.py",
    "src/quant_fund/research/ots_anchor.py",
    "src/quant_fund/research/quorum_rotation.py",  # M-of-N registry lineage —
    # the authority a v2 checkpoint's signatures resolve under
    "src/quant_fund/research/receipt_graph.py",
    "src/quant_fund/research/receipt_lattice.py",
    "src/quant_fund/research/receipt_tombstone.py",
    "src/quant_fund/research/receipt_v2.py",
    "src/quant_fund/research/release_attestation.py",
    "src/quant_fund/research/repo_integrity.py",
    "src/quant_fund/research/selective_inference.py",
    "src/quant_fund/research/tamper_drill.py",
    "src/quant_fund/research/timestamp_anchor.py",
    "src/quant_fund/research/witness_scan.py",
    "src/quant_fund/utils/atomicio.py",
    "src/quant_fund/utils/hashing.py",
    "scripts/verify_auditor_bundle.py",  # the independent second verifier —
    # a silent patch to IT defeats the divergence-detection layer
    "scripts/verify_ots_auditor.py",  # the standalone OTS/Bitcoin auditor —
    # same class: the differential oracle must itself be pinned
    "scripts/verify_corpus_proof.py",  # standalone inclusion/absence-proof
    # auditor — the offline verification path's independent oracle
    "scripts/verify_epoch_chain.py",  # standalone full-chain auditor —
    # same class: a silent patch to the independent oracle defeats the
    # divergence-detection layer
)


# File-name vocabulary that marks a module as verifier-critical — the
# coverage check below derives the required-jewel set from it, so a future
# integrity module can't dodge the pin by simply not being listed here.
_VERIFIER_VOCABULARY = (
    "epoch",  # corpus_epoch, epoch_merkle, epoch_consistency
    "merkle",
    "signature",
    "integrity",
    "anchor",
    "crown",
    "lattice",
    "receipt_v2",
    "lane_contracts",
    "admission",
    "receipt_graph",
    "checkpoint",
    "tamper",
    "witness",
    "rotation",
    "fuzz",
    "custody",
    "attestation",
    "tombstone",
    "inference",  # corpus_inference — the FDR pool admission gates on
)


def verifier_coverage_errors(
    root: Path | str,
    jewels: tuple[str, ...] = DEFAULT_JEWELS,
) -> list[str]:
    """Every integrity-critical module must itself be a pinned jewel.

    A verifier module that isn't in ``DEFAULT_JEWELS`` is a hole: it could be
    silently rewritten without tripping ``jewel_mutated``. The covered set is
    *derived* — any ``research/`` module whose name matches the integrity
    vocabulary must be pinned — so new verifier modules can't dodge the
    pin by omission.
    """
    src = Path(root) / "src" / "quant_fund" / "research"
    required = {
        f"src/quant_fund/research/{p.name}"
        for p in src.glob("*.py")
        if any(tok in p.stem for tok in _VERIFIER_VOCABULARY)
    }
    return [f"verifier_unpinned:{j}" for j in sorted(required - set(jewels))]


def crown_jewel_digests(
    root: Path | str, jewels: tuple[str, ...] = DEFAULT_JEWELS
) -> dict[str, str]:
    """``{relpath: sha256}`` over the pinned set; fails closed on missing."""
    base = Path(root)
    out: dict[str, str] = {}
    for rel in jewels:
        path = base / rel
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"crown jewel not a regular file: {rel}")
        out[rel] = hash_bytes(path.read_bytes())
    return out


def write_crown_jewels_pin(
    root: Path | str = ".",
    pin_path: Path | str = DEFAULT_PIN_PATH,
) -> Path:
    """(Re)write the pin over ``DEFAULT_JEWELS``; overwrite-atomic."""
    from quant_fund.utils.atomicio import atomic_write_text

    payload = {"schema": CROWN_JEWELS_SCHEMA, "files": crown_jewel_digests(root)}
    path = Path(pin_path)
    if not path.is_absolute():
        path = Path(root) / path
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def load_crown_jewels_pin(pin_path: Path | str = DEFAULT_PIN_PATH) -> dict[str, str]:
    """Load the pin's files map; raises ValueError on malformed content."""
    path = Path(pin_path)
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema") != CROWN_JEWELS_SCHEMA:
        raise ValueError(f"{path} is not a {CROWN_JEWELS_SCHEMA} pin")
    files = raw.get("files")
    if not isinstance(files, Mapping):
        raise ValueError(f"{path} has no 'files' map")
    out: dict[str, str] = {}
    for name, sha in files.items():
        if not (
            isinstance(name, str)
            and isinstance(sha, str)
            and len(sha) == 64
            and all(c in "0123456789abcdef" for c in sha)
        ):
            raise ValueError(f"{path} entry {name!r} malformed")
        out[name] = sha
    return out


def crown_jewels_errors(
    root: Path | str = ".",
    pin_path: Path | str = DEFAULT_PIN_PATH,
    *,
    jewels: tuple[str, ...] = DEFAULT_JEWELS,
) -> list[str]:
    """Verify every jewel against the pin.

    Errors are sorted ``jewel_*`` codes. The pinned set must equal
    ``jewels`` exactly — a pin missing a jewel hides a removal
    (``jewel_unpinned``), an extra entry is an undeclared scope change
    (``jewel_unexpected``).
    """
    base = Path(root)
    pin = Path(pin_path)
    if not pin.is_absolute():
        pin = base / pin
    if not pin.is_file():
        return ["pin_missing"]
    try:
        pinned = load_crown_jewels_pin(pin)
    except (OSError, ValueError) as exc:
        return [f"pin_malformed:{exc}"]
    errors: list[str] = []
    # Coverage first: a verifier module that dodges DEFAULT_JEWELS is a
    # self-reference hole even when every pinned jewel is intact.
    errors.extend(verifier_coverage_errors(base, jewels))
    for name in sorted(set(jewels) - set(pinned)):
        errors.append(f"jewel_unpinned:{name}")
    for name in sorted(set(pinned) - set(jewels)):
        errors.append(f"jewel_unexpected:{name}")
    for name, expected in sorted(pinned.items()):
        path = base / name
        if name not in jewels:
            continue  # already reported as unexpected
        if path.is_symlink():
            errors.append(f"jewel_symlink:{name}")
        elif not path.is_file():
            errors.append(f"jewel_missing:{name}")
        elif hash_bytes(path.read_bytes()) != expected:
            errors.append(f"jewel_mutated:{name}")
    return sorted(errors)
