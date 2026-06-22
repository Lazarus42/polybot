#!/usr/bin/env python3
"""One-shot: calibrate market impact, then sweep participation at the fitted coef.

Pipeline:
  1. Run `calibrate_market_impact.py` on the fill tape to fit (slippage_model, coef).
  2. Apply an optional `--stress-multiplier` to the coef (the calibration is a floor —
     it cannot see orders larger than any historically executed; see report §8).
  3. Run `walk_forward_residual_portfolio.py` (utility mode, fixed lambda/gamma) once per
     `--participations` value with that slippage setting.
  4. Assemble a net-of-impact frontier table + CSV and print the recommended operating point.

Run in the project venv:
    .venv/bin/python scripts/run_impact_calibrated_sweep.py \
        --external-signals reports/orthogonal_sleeves/orthogonal_signals.csv \
                           reports/cross_market_overround/cross_market_signals.csv

To re-run without recalibrating (or to test), pass --model-override / --coef-override.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PY = sys.executable


def run_calibration(args: argparse.Namespace) -> tuple[str, float]:
    out = args.output_dir / "impact_calibration"
    cmd = [PY, "scripts/calibrate_market_impact.py",
           "--fills", str(args.fills), "--output-dir", str(out),
           "--engine", "duckdb", "--sample", str(args.sample)]
    if args.max_markets is not None:
        cmd += ["--max-markets", str(args.max_markets)]
    print("→ calibrating:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    summary = json.loads((out / "impact_calibration.json").read_text())
    rec = summary["recommended"]
    return rec["slippage_model"], float(rec["slippage_coef"])


def run_one_participation(args: argparse.Namespace, model: str, coef: float, part: float) -> dict[str, Any]:
    out = args.output_dir / f"part_{part:g}"
    cmd = [PY, "scripts/walk_forward_residual_portfolio.py",
           "--output-dir", str(out),
           "--train-months", str(args.train_months),
           "--corr-threshold", str(args.corr_threshold),
           "--max-components", str(args.max_components),
           "--max-weight", str(args.max_weight),
           "--participation-fraction", str(part),
           "--modes", "utility",
           "--util-lambda", str(args.util_lambda),
           "--util-gamma", str(args.util_gamma),
           "--slippage-model", model,
           "--slippage-coef", str(coef)]
    if args.external_signals:
        cmd += ["--external-signals", *[str(p) for p in args.external_signals]]
    print(f"→ participation {part:g}:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    row = next(csv.DictReader((out / "portfolio_summary.csv").open()))
    return {
        "participation": part,
        "total_profit": round(float(row["total_profit"]), 2),
        "worst_month": round(float(row["worst_month"]), 2),
        "mean_deployed": round(float(row["mean_deployed"]), 1),
        "total_slippage_cost": round(float(row.get("total_slippage_cost", 0.0)), 2),
        "annualized_sharpe": round(float(row["annualized_sharpe"]), 2),
        "positive_rate": round(float(row["positive_rate"]), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fills", type=Path,
                    default=Path("archive/processed/underdog_events/fills_sorted.parquet"))
    ap.add_argument("--output-dir", type=Path, default=Path("reports/impact_calibrated_sweep"))
    ap.add_argument("--external-signals", type=Path, nargs="*", default=[])
    ap.add_argument("--participations", type=float, nargs="+",
                    default=[0.05, 0.10, 0.15, 0.20, 0.30])
    ap.add_argument("--stress-multiplier", type=float, default=1.0,
                    help="Multiply the fitted coef by this before sweeping (>=1 to be conservative).")
    # calibration knobs
    ap.add_argument("--sample", type=float, default=0.05)
    ap.add_argument("--max-markets", type=int, default=None)
    # override (skip calibration)
    ap.add_argument("--model-override", choices=["none", "linear", "sqrt"], default=None)
    ap.add_argument("--coef-override", type=float, default=None)
    # portfolio knobs (match prior runs)
    ap.add_argument("--util-lambda", type=float, default=1.0)
    ap.add_argument("--util-gamma", type=float, default=0.5)
    ap.add_argument("--train-months", type=int, default=12)
    ap.add_argument("--corr-threshold", type=float, default=0.6)
    ap.add_argument("--max-components", type=int, default=12)
    ap.add_argument("--max-weight", type=float, default=0.35)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.model_override is not None and args.coef_override is not None:
        model, coef = args.model_override, args.coef_override
        print(f"→ using override: model={model} coef={coef}", flush=True)
    else:
        model, coef = run_calibration(args)
        print(f"→ fitted: model={model} coef={coef}", flush=True)
    coef *= args.stress_multiplier
    if args.stress_multiplier != 1.0:
        print(f"→ stressed coef (x{args.stress_multiplier}) = {coef:.4f}", flush=True)

    rows = [run_one_participation(args, model, coef, p) for p in args.participations]

    # frontier CSV + table
    fields = ["participation", "total_profit", "worst_month", "mean_deployed",
              "total_slippage_cost", "annualized_sharpe", "positive_rate"]
    with (args.output_dir / "net_frontier.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    best = max(rows, key=lambda r: r["total_profit"])
    best_sharpe = max(rows, key=lambda r: r["annualized_sharpe"])
    print("\n=== NET-OF-IMPACT FRONTIER (utility λ={}, γ={}, model={}, coef={:.4f}) ===".format(
        args.util_lambda, args.util_gamma, model, coef))
    hdr = f"{'part':>6} {'profit':>9} {'worst':>9} {'deployed':>9} {'slip':>9} {'sharpe':>7} {'pos':>6}"
    print(hdr)
    for r in rows:
        print(f"{r['participation']:>6g} {r['total_profit']:>9.2f} {r['worst_month']:>9.2f} "
              f"{r['mean_deployed']:>9.1f} {r['total_slippage_cost']:>9.2f} "
              f"{r['annualized_sharpe']:>7.2f} {r['positive_rate']:>6.2f}")
    print(f"\nmax profit  → participation {best['participation']:g} "
          f"(${best['total_profit']:.2f}, worst ${best['worst_month']:.2f})")
    print(f"max Sharpe  → participation {best_sharpe['participation']:g} "
          f"(Sharpe {best_sharpe['annualized_sharpe']:.2f}, ${best_sharpe['total_profit']:.2f})")
    print(f"\nartifacts: {args.output_dir}/net_frontier.csv, "
          f"{args.output_dir}/impact_calibration/impact_calibration.json")


if __name__ == "__main__":
    main()
