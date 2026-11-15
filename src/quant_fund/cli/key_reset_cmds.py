"""Key-reset command.

Split out of ``research_cmds``. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

from pathlib import Path

import typer

from .app import app
from .research_cmds import _load_signer_pairs


@app.command("key-reset")
def key_reset_cmd(
    root: Path = typer.Option(Path("."), "--root"),
    key_file: list[Path] = typer.Option(
        [],
        "--key-file",
        help="NEW signer key material ('<priv>:<pub>' or two lines). Repeatable; "
        "the first key becomes the gate key, all of them form the new registry "
        "and all of them sign the reset record.",
    ),
    threshold: int = typer.Option(1, "--threshold", min=1, help="New registry threshold."),
    witness_pub: Path | None = typer.Option(
        None,
        "--witness-pub",
        help="PEM ECDSA P-256 public key of the NEW witness key (installed as "
        "quality/witness_signing.pub). It does not sign the reset record.",
    ),
    reason: str = typer.Option(..., "--reason", help="Why the keys are lost (recorded verbatim)."),
    authorized_by: str = typer.Option(
        ..., "--authorized-by", help="Who authorized the reset (a recorded claim, not a proof)."
    ),
) -> None:
    """Declare the current signing keys LOST and install a new key set.

    Only for keys nobody can sign with any more: if an outgoing key still
    exists, use ``rotate-key`` / ``quorum-rotate``, which it authorizes.
    Writes ``quality/key_reset_<id>.json`` binding the current checkpoint
    (the boundary, verified under the lost keys) to the new keys (signed by
    every ``--key-file`` key; verifiers need the new registry's threshold,
    including the gate key), then installs ``gate_signing.pub``,
    ``gate_quorum.json`` and, with ``--witness-pub``, the witness key.
    ``checkpoint-chain``, ``verify-rotations`` and ``verify-repo`` report the
    reset; ``verify-bundle`` and ``scripts/verify_auditor_bundle.py`` re-check
    it. Finish the ceremony in the same commit: ``crown-jewels --write``,
    stamp epochs, ``quorum-sign``, ``checkpoint``, ``witness-checkpoint``.
    """
    from quant_fund.research.key_reset import reset_keys

    raws = [f.read_text().strip() for f in key_file]
    if not raws:
        typer.echo("key-reset: supply --key-file(s) of the NEW keys")
        raise typer.Exit(code=2)
    try:
        signers = [pair for raw in raws for pair in _load_signer_pairs(raw)]
        out = reset_keys(
            root,
            signers,
            reason=reason,
            authorized_by=authorized_by,
            threshold=threshold,
            witness_pubkey_pem=witness_pub.read_text() if witness_pub is not None else None,
        )
    except ValueError as exc:
        typer.echo(f"key-reset: {exc}")
        raise typer.Exit(code=2) from exc
    typer.echo(f"reset={out}")
    typer.echo(
        "next: crown-jewels --write, stamp epochs, quorum-sign, checkpoint, witness-checkpoint"
    )
