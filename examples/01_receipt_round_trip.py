# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Receipt round trip
#
# Data label: SYNTHETIC
#
# Not investment advice. No live-trading claim.
# A small synthetic research run is sealed, verified, then rejected after a one-byte edit.

# %%
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
# Required families alone satisfy verify_research_artifact; the live-optional
# catalog is thousands of benches and makes this example unbounded in CI.
os.environ["DIPCATCHER_RESEARCH_REQUIRED_ONLY"] = "1"

from quant_fund.config import load_config
from quant_fund.research.agent import run_research
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.research.verify import verify_research_artifact

ROOT = Path(__file__).resolve().parents[1]
# Panel sized for required-family finite observations on the synthetic oracle.
N_ASSETS = 36
N_DAYS = 220
TRAIN_BARS = 80
VAL_BARS = 20
TEST_BARS = 20


def _flip_one_byte(payload: bytes) -> tuple[bytes, int]:
    token = b'"dataset_content_sha256": "'
    start = payload.find(token)
    if start < 0:
        raise SystemExit("receipt has no dataset_content_sha256 field to tamper")
    pos = start + len(token)
    original = payload[pos]
    replacement = b"0"[0] if original != b"0"[0] else b"1"[0]
    tampered = bytearray(payload)
    tampered[pos] = replacement
    changed = sum(left != right for left, right in zip(payload, tampered, strict=True))
    if changed != 1 or len(tampered) != len(payload):
        raise SystemExit("tamper did not change exactly one byte")
    return bytes(tampered), pos


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="dipcatcher-example-receipt-") as raw_root:
        root = Path(raw_root)
        os.environ["MLFLOW_TRACKING_URI"] = (root / "mlruns").as_uri()
        config = load_config(ROOT / "configs" / "research.yaml")
        config.data.root = root
        config.data.synthetic_n_assets = N_ASSETS
        config.data.synthetic_n_days = N_DAYS
        config.validation.train_bars = TRAIN_BARS
        config.validation.val_bars = VAL_BARS
        config.validation.test_bars = TEST_BARS
        # Required families plus the roughness panel. The live-optional canon
        # is a separate seeded catalog and is not part of this seal.
        notebook = run_research(config, include_live_optional=False)
        if notebook.data_source != "SYNTHETIC" or notebook.synthetic is not True:
            raise SystemExit("synthetic research run was not labeled SYNTHETIC")
        if notebook.claim != "research_only":
            raise SystemExit("synthetic research run is not research_only")
        if not family_blob_forbidden_metrics_absent(notebook.scorecard):
            raise SystemExit("scorecard contains a forbidden headline metric")
        run_id = notebook.provenance.get("run_id")
        if not isinstance(run_id, str) or len(run_id) != 64:
            raise SystemExit("receipt run_id is not a sha256 digest")
        receipt = root / "metadata" / "research" / "latest.json"
        verified = verify_research_artifact(receipt)
        print("data_label=SYNTHETIC")
        print("claim=research_only")
        print("not_investment_advice=true")
        print("no_live_trading_claim=true")
        print(f"synthetic={str(notebook.synthetic).lower()}")
        print(f"run_id={run_id}")
        print(f"scorecard_families={len(notebook.scorecard)}")
        print("forbidden_headline_metrics_absent=true")
        print(f"receipt_valid={str(verified['valid']).lower()}")
        if verified["valid"] is not True:
            errors = verified.get("errors")
            listed = errors if isinstance(errors, list) else [errors]
            for error in listed:
                print(f"receipt_error={error}")
            raise SystemExit("synthetic receipt failed verification")
        tampered, offset = _flip_one_byte(receipt.read_bytes())
        tampered_path = receipt.with_name("tampered.json")
        tampered_path.write_bytes(tampered)
        rejected = verify_research_artifact(tampered_path)
        print(f"tamper_offset={offset}")
        print("tamper_bytes_changed=1")
        print(f"tampered_valid={str(rejected['valid']).lower()}")
        errors = rejected.get("errors")
        if rejected["valid"] is not False or not isinstance(errors, list) or not errors:
            raise SystemExit("one-byte tamper was not rejected")
        print(f"tamper_error={errors[0]}")
        print("tamper_rejected=true")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"example_failed={type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
