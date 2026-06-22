#!/usr/bin/env python3
"""Combine walk-forward OOS diagnostic outputs into summary tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/oos_diagnostic_comparison"))
    args = parser.parse_args()

    summary_frames = []
    period_frames = []
    selected_frames = []
    for report_dir in args.report_dirs:
        label = report_dir.name.removeprefix("walk_forward_oos_").removeprefix("oos_")
        summary = load_csv(report_dir / "oos_summary.csv")
        if not summary.empty:
            summary.insert(0, "experiment", label)
            summary.insert(1, "report_dir", str(report_dir))
            summary_frames.append(summary)
        periods = load_csv(report_dir / "oos_period_results.csv")
        if not periods.empty:
            periods.insert(0, "experiment", label)
            periods.insert(1, "report_dir", str(report_dir))
            period_frames.append(periods)
        selected = load_csv(report_dir / "selected_strategies.csv")
        if not selected.empty:
            selected.insert(0, "experiment", label)
            selected.insert(1, "report_dir", str(report_dir))
            selected_frames.append(selected)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    combined_periods = pd.concat(period_frames, ignore_index=True) if period_frames else pd.DataFrame()
    combined_selected = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()

    if not combined_summary.empty:
        combined_summary.to_csv(args.output_dir / "combined_oos_summary.csv", index=False)
        ranking = combined_summary.copy()
        if "summary_scope" in ranking:
            ranking = ranking[ranking["summary_scope"].eq("all")]
        ranking = ranking.sort_values(
            ["mean_account_return", "mean_without_top1", "mean_max_drawdown"],
            ascending=[False, False, True],
        )
        ranking.to_csv(args.output_dir / "ranked_experiments.csv", index=False)
    if not combined_periods.empty:
        combined_periods.to_csv(args.output_dir / "combined_oos_period_results.csv", index=False)
    if not combined_selected.empty:
        combined_selected.to_csv(args.output_dir / "combined_selected_strategies.csv", index=False)

    print({
        "output_dir": str(args.output_dir),
        "summary_rows": len(combined_summary),
        "period_rows": len(combined_periods),
        "selected_rows": len(combined_selected),
    })


if __name__ == "__main__":
    main()
