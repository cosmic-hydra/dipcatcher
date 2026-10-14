"""dipcatcher as the fx-1 harness.

The relationship, inverted: **fx-1 is the model; dipcatcher is the harness
that builds, evaluates, and verifies it.** This module is the typed bridge —
every lab capability fx-1 is allowed to touch is registered here as a
:class:`HarnessCommand` and executed through an injectable runner, so tests
never spawn real lab processes and fx-1 can never reach an unregistered
command.

Harness roles:
- *Data engine* — ingest/collect/features/labels feed ``fx1.data``.
- *Evaluation* — benches and books score fx-1's domain competence;
  ``fx1.eval`` scores its behavior.
- *Verification* — doctor / verify-research / validate / monitor gate every
  artifact the model is trained on or claims about.
- *Model training* — the lab's own train/optimize/paper surfaces, usable as
  curriculum and tool-use trace sources for fx-1.
"""

from __future__ import annotations

import subprocess  # noqa: S404 - runner is injectable; see Harness.run
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class HarnessRole(StrEnum):
    DATA_ENGINE = "data_engine"
    EVALUATION = "evaluation"
    VERIFICATION = "verification"
    MODEL_TRAINING = "model_training"


class HarnessCommand(BaseModel):
    """A registered lab command fx-1 may invoke through the harness."""

    name: str
    role: HarnessRole
    argv: list[str] = Field(description="dipcatcher CLI argv (no executable)")
    timeout_s: int = Field(default=300, ge=1, le=7200)
    description: str


def _cmd(name: str, role: HarnessRole, description: str, timeout_s: int = 300) -> HarnessCommand:
    return HarnessCommand(
        name=name, role=role, argv=[name], timeout_s=timeout_s, description=description
    )


# The complete registry of lab surfaces fx-1 can touch — mirrors
# ``quant_fund.cli.main``. Anything not listed here is unreachable by the
# model: fail-closed by construction. The hidden ``lab`` command and the
# network-serving ``api`` command are deliberately excluded.
HARNESS_REGISTRY: tuple[HarnessCommand, ...] = (
    # --- Verification ---
    _cmd(
        "doctor",
        HarnessRole.VERIFICATION,
        "Verify data manifest and latest research receipt before trusting state.",
    ),
    _cmd(
        "verify-research",
        HarnessRole.VERIFICATION,
        "Verify immutable provenance, scorecards, and research artifacts.",
    ),
    _cmd(
        "validate",
        HarnessRole.VERIFICATION,
        "Fail-closed causal / walk-forward / promotion gates for a model id.",
    ),
    _cmd(
        "monitor",
        HarnessRole.VERIFICATION,
        "Operational monitoring surface for the harness runtime.",
    ),
    # --- Data engine ---
    _cmd(
        "ingest",
        HarnessRole.DATA_ENGINE,
        "Bronze/silver lake ingest; writes data_manifest.json with SHA-256 hashes.",
        timeout_s=1800,
    ),
    _cmd(
        "collect",
        HarnessRole.DATA_ENGINE,
        "Explicit opt-in public-feed collection (binance, fred) with receipts.",
    ),
    _cmd(
        "build-features",
        HarnessRole.DATA_ENGINE,
        "Feature construction on the PIT lake.",
        timeout_s=1800,
    ),
    _cmd(
        "build-labels",
        HarnessRole.DATA_ENGINE,
        "Label construction on the PIT lake.",
        timeout_s=1800,
    ),
    # --- Evaluation ---
    _cmd(
        "research",
        HarnessRole.EVALUATION,
        "Scientific benches (proper scores; SYNTHETIC labeled).",
        timeout_s=3600,
    ),
    _cmd(
        "northset",
        HarnessRole.EVALUATION,
        "Order-book/candlestick slice: identities, OHLC vol, Kyle/Roll/OFI/VPIN.",
        timeout_s=1800,
    ),
    _cmd(
        "backtest",
        HarnessRole.EVALUATION,
        "Next-open fills, costs, participation; tamper-evident artifacts.",
        timeout_s=1800,
    ),
    _cmd("forecast", HarnessRole.EVALUATION, "Forecast surface for validated models."),
    _cmd(
        "kronos-forecast",
        HarnessRole.EVALUATION,
        "Research-only local Kronos candlestick adapter (no network).",
    ),
    _cmd(
        "candle-book",
        HarnessRole.EVALUATION,
        "Candle/order-book reconstruction benches.",
        timeout_s=900,
    ),
    _cmd(
        "kyle-ofi",
        HarnessRole.EVALUATION,
        "Kyle lambda / OFI microstructure benches.",
        timeout_s=900,
    ),
    _cmd(
        "session-book", HarnessRole.EVALUATION, "Session-level order-book benches.", timeout_s=900
    ),
    _cmd("vendor-book-map", HarnessRole.EVALUATION, "Vendor book mapping diagnostics."),
    _cmd("book-panel", HarnessRole.EVALUATION, "Book panel aggregation benches.", timeout_s=900),
    _cmd("report", HarnessRole.EVALUATION, "Research notebook/report generation."),
    _cmd(
        "tearsheet",
        HarnessRole.EVALUATION,
        "Backtest tearsheet diagnostics (analytics export, live_pnl_claim=false).",
    ),
    # --- Model training (lab surfaces; curriculum + tool-use traces) ---
    _cmd(
        "train",
        HarnessRole.MODEL_TRAINING,
        "Lab model training families (ranking/distribution/calibration/"
        "volatility/alpha/covariance/regime/tail/reinforcement/liquidity).",
        timeout_s=7200,
    ),
    _cmd(
        "optimize",
        HarnessRole.MODEL_TRAINING,
        "Hyperparameter optimization under fail-closed gates.",
        timeout_s=7200,
    ),
    _cmd(
        "paper",
        HarnessRole.MODEL_TRAINING,
        "Simulated broker shadow runs (no live fills).",
        timeout_s=1800,
    ),
    HarnessCommand(
        name="live",
        role=HarnessRole.MODEL_TRAINING,
        argv=["live", "status"],
        timeout_s=120,
        description=(
            "Read live-desk status. Does not transmit orders. Shadow is the "
            "default venue. live_pnl_claim is false and evidence conditions "
            "stay blocked."
        ),
    ),
)


class HarnessResult(BaseModel):
    command: str
    exit_code: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


# Runner signature: (argv, timeout_s) -> (exit_code, stdout, stderr)
Runner = Callable[[list[str], int], tuple[int, str, str]]


def _subprocess_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
    proc = subprocess.run(  # noqa: S603 - argv is registry-constrained
        ["dipcatcher", *argv],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


class Harness:
    """The fx-1 side of the dipcatcher harness."""

    def __init__(self, runner: Runner | None = None) -> None:
        self._runner = runner or _subprocess_runner
        self._registry = {c.name: c for c in HARNESS_REGISTRY}

    def list_commands(self, role: HarnessRole | None = None) -> list[HarnessCommand]:
        commands = list(self._registry.values())
        if role is not None:
            commands = [c for c in commands if c.role == role]
        return commands

    def get(self, name: str) -> HarnessCommand:
        if name not in self._registry:
            raise KeyError(
                f"{name!r} is not a registered harness command; fx-1 may only "
                f"use: {sorted(self._registry)}"
            )
        return self._registry[name]

    def run(
        self,
        name: str,
        extra_args: list[str] | None = None,
        *,
        config: Path | None = None,
    ) -> HarnessResult:
        """Execute a registered harness command, fail-closed on unknown names.

        Config paths are contained to the repo ``configs/`` directory, matching
        the lab API's allowlist posture.
        """
        command = self.get(name)
        argv = list(command.argv)
        configs_dir = (Path.cwd() / "configs").resolve()

        def check_contained(raw: str) -> Path:
            resolved = Path(raw).resolve()
            if configs_dir not in resolved.parents and resolved != configs_dir:
                raise ValueError(f"config path {resolved} escapes the configs/ allowlist")
            return resolved

        if config is not None:
            argv += ["--config", str(check_contained(str(config)))]
        if extra_args:
            # --config passed through extra_args must obey the same
            # containment as the keyword arg — otherwise it is a bypass.
            for i, arg in enumerate(extra_args):
                if arg == "--config" and i + 1 < len(extra_args):
                    check_contained(extra_args[i + 1])
                elif arg.startswith("--config="):
                    check_contained(arg.split("=", 1)[1])
            argv += list(extra_args)
        code, out, err = self._runner(argv, command.timeout_s)
        return HarnessResult(command=name, exit_code=code, stdout=out, stderr=err)
