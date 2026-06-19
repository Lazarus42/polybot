#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


def values(text: str) -> list[float]:
    return [float(value) for value in text.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run confirmation/entry-delay robustness sweeps.")
    parser.add_argument("--confirmations", default="1,5,15,30")
    parser.add_argument("--entry-delays", default="0,1,5")
    parser.add_argument("--output-root", type=Path, default=Path("reports/underdog_sweeps"))
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--fee-coefficient", type=float, default=0.07)
    parser.add_argument("--price-tick", type=float, default=0.01)
    parser.add_argument("--contract-step", type=float, default=0.01)
    parser.add_argument("--capital", type=float, default=5000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for confirmation in values(args.confirmations):
        for delay in values(args.entry_delays):
            name = f"confirm_{confirmation:g}m_delay_{delay:g}m".replace(".", "p")
            output = args.output_root / name
            command = [
                args.python,
                "scripts/optimize_underdog_bracket.py",
                "--confirmation-minutes",
                str(confirmation),
                "--entry-delay-minutes",
                str(delay),
                "--fee-coefficient",
                str(args.fee_coefficient),
                "--price-tick",
                str(args.price_tick),
                "--contract-step",
                str(args.contract_step),
                "--initial-capital",
                str(args.capital),
                "--output-dir",
                str(output),
            ]
            print(" ".join(command), flush=True)
            if args.dry_run:
                continue
            subprocess.run(command, check=True)
            summary = json.loads((output / "holdout_portfolio_summary.json").read_text())
            results.append(
                {
                    "confirmation_minutes": confirmation,
                    "entry_delay_minutes": delay,
                    "opportunities": summary["holdout_opportunities"],
                    "executed": summary["executed_trades"],
                    "portfolio_roi": summary["portfolio_roi"],
                    "profitable_trade_rate": summary["profitable_trade_rate"],
                }
            )
    if results:
        with (args.output_root / "sweep_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    main()
