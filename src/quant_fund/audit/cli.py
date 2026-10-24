"""``verify-ledger`` command and the recording helpers it shares with the Typer CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from quant_fund.audit.errors import AuditError
from quant_fund.audit.ledger import AuditLedger
from quant_fund.audit.signing import Ed25519Signer, load_public_key_hex
from quant_fund.audit.verify import explain, verify_ledger


def run_verify(
    ledger: Path,
    *,
    trust_pub: Path | None = None,
    expect_size: int | None = None,
    expect_root: str | None = None,
    sigstore_identity: str | None = None,
    sigstore_issuer: str | None = None,
    sigstore_offline: bool = False,
) -> int:
    """Print a verification report and return 0 when the ledger is valid."""
    trust = None
    if trust_pub is not None:
        try:
            trust = load_public_key_hex(trust_pub)
        except (OSError, AuditError) as exc:
            sys.stdout.write(
                explain({"schema": "audit-ledger/1", "valid": False, "errors": [str(exc)]})
            )
            sys.stdout.write("\n")
            return 2
    report = verify_ledger(
        ledger,
        trust_public_key=trust,
        expect_size=expect_size,
        expect_root=expect_root,
        sigstore_identity=sigstore_identity,
        sigstore_issuer=sigstore_issuer,
        sigstore_offline=sigstore_offline,
    )
    sys.stdout.write(explain(report))
    sys.stdout.write("\n")
    return 0 if report["valid"] else 1


def build_parser() -> argparse.ArgumentParser:
    from quant_fund import __version__

    parser = argparse.ArgumentParser(
        prog="verify-ledger",
        description=(
            "Verify a hash-chained audit ledger. Detects edited, removed, and "
            "reordered entries against signed Merkle roots. Exit 0 is valid, "
            "1 is tamper or an inconsistent log, 2 is usage."
        ),
    )
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=f"verify-ledger {__version__}",
    )
    parser.add_argument(
        "ledger", type=Path, help="Directory with entries.jsonl and checkpoints.jsonl"
    )
    parser.add_argument(
        "--trust-pub",
        type=Path,
        default=None,
        help="Pinned Ed25519 public key (hex). Detects a rewritten log signed by another key.",
    )
    parser.add_argument(
        "--expect-size",
        type=int,
        default=None,
        help="Witness tree size. Detects truncation of the tip together with its checkpoint.",
    )
    parser.add_argument(
        "--expect-root",
        default=None,
        help="Witness Merkle root (hex) for --expect-size.",
    )
    parser.add_argument(
        "--sigstore-identity", default=None, help="Pinned Sigstore certificate identity"
    )
    parser.add_argument("--sigstore-issuer", default=None, help="Pinned Sigstore OIDC issuer")
    parser.add_argument(
        "--sigstore-offline",
        action="store_true",
        help="Verify Sigstore bundles from the local trust root only",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code = run_verify(
            args.ledger,
            trust_pub=args.trust_pub,
            expect_size=args.expect_size,
            expect_root=args.expect_root,
            sigstore_identity=args.sigstore_identity,
            sigstore_issuer=args.sigstore_issuer,
            sigstore_offline=args.sigstore_offline,
        )
    except OSError as exc:
        sys.stderr.write(f"error: {exc}\n")
        raise SystemExit(2) from None
    raise SystemExit(code)


def open_ledger(root: Path, signing_key: Path, *, sign_every: int = 1) -> AuditLedger:
    signer = Ed25519Signer.from_path(signing_key)
    return AuditLedger(root, signer=signer, sign_every=sign_every)


def generate_signing_key(path: Path) -> str:
    signer = Ed25519Signer.generate()
    signer.write(path)
    return signer.public_key_hex


def record_receipt(
    ledger_dir: Path,
    signing_key: Path,
    receipt: Path,
    *,
    sign_every: int = 1,
) -> dict[str, object]:
    from quant_fund.audit.record import record_research_receipt

    ledger = open_ledger(ledger_dir, signing_key, sign_every=sign_every)
    entry = record_research_receipt(ledger, receipt)
    return {"index": entry.index, "entry_hash": entry.entry_hash, "kind": entry.kind}


def record_paper(
    ledger_dir: Path,
    signing_key: Path,
    paper_dir: Path,
    *,
    sign_every: int = 1,
) -> dict[str, object]:
    from quant_fund.audit.record import record_paper_directory

    ledger = open_ledger(ledger_dir, signing_key, sign_every=sign_every)
    entries = record_paper_directory(ledger, paper_dir)
    return {"n": len(entries), "kinds": [entry.kind for entry in entries]}


def trace_number(
    ledger_dir: Path,
    receipt: Path,
    metric_path: str,
    *,
    trust_pub: Path | None = None,
    verify_receipt: bool = False,
) -> dict[str, object]:
    from quant_fund.audit.trace import trace_published_number

    trust = load_public_key_hex(trust_pub) if trust_pub is not None else None
    ledger = AuditLedger(ledger_dir, signer=None)
    report = trace_published_number(
        receipt,
        metric_path,
        ledger,
        trust_public_key=trust,
        verify_receipt=verify_receipt,
    )
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True, default=str))
    sys.stdout.write("\n")
    return report


if __name__ == "__main__":
    main()
