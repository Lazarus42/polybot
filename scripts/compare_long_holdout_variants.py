#!/usr/bin/env python3
"""Compare multiple long-holdout experiment directories."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DIRS = [
    "reports/long_holdout_weighting",
    "reports/long_holdout_variant_conservative_pacing",
    "reports/long_holdout_variant_small_stakes",
    "reports/long_holdout_variant_lower_minfit",
    "reports/long_holdout_variant_more_time_to_exit",
]


def load_variant(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path / "account_summary.csv")
    df["variant"] = path.name.replace("long_holdout_", "")
    df["top1_contribution"] = df["realized_profit"] - df["without_top_1"]
    return df


def plot_variant_summary(summary: pd.DataFrame, output: Path) -> None:
    ordered = summary.sort_values("mean_profit", ascending=False)
    x = np.arange(len(ordered))
    fig, axis = plt.subplots(figsize=(14, 7))
    axis.bar(x - 0.2, ordered["mean_profit"], width=0.4, label="mean profit")
    axis.bar(x + 0.2, ordered["mean_without_top1"], width=0.4, label="mean without top 1")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels(ordered["variant"], rotation=35, ha="right")
    axis.set_ylabel("profit ($)")
    axis.set_title("Variant Mean Profit vs. Top-Winner-Robust Profit")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "variant_mean_profit_vs_robust.png", dpi=160)
    plt.close(fig)


def plot_best_runs(df: pd.DataFrame, output: Path) -> None:
    best = df[df["entries"] > 0].sort_values("realized_profit", ascending=False).head(30)
    labels = (
        best["variant"]
        + "\n"
        + best["sizing_policy"]
        + " "
        + best["gate"]
        + " "
        + best["holdout_window"]
        + " c"
        + best["cut_fraction"].astype(str)
    )
    x = np.arange(len(best))
    fig, axis = plt.subplots(figsize=(18, 8))
    axis.bar(x - 0.2, best["realized_profit"], width=0.4, label="realized")
    axis.bar(x + 0.2, best["without_top_1"], width=0.4, label="without top 1")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    axis.set_ylabel("profit ($)")
    axis.set_title("Best Realized Runs Across Strategy Variants")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "best_runs_across_variants.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dirs", nargs="+", default=DEFAULT_DIRS)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/long_holdout_variant_comparison"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = [load_variant(Path(item)) for item in args.dirs if (Path(item) / "account_summary.csv").exists()]
    df = pd.concat(frames, ignore_index=True)
    active = df[df["entries"] > 0].copy()
    summary = active.groupby("variant").agg(
        runs=("realized_profit", "size"),
        mean_profit=("realized_profit", "mean"),
        median_profit=("realized_profit", "median"),
        positive_rate=("realized_profit", lambda values: float((values > 0).mean())),
        mean_without_top1=("without_top_1", "mean"),
        best_profit=("realized_profit", "max"),
        best_without_top1=("without_top_1", "max"),
        mean_deployed=("deployed", "mean"),
        mean_drawdown=("max_drawdown", "mean"),
    ).reset_index().sort_values("mean_profit", ascending=False)

    df.to_csv(args.output_dir / "combined_account_summary.csv", index=False)
    summary.to_csv(args.output_dir / "variant_summary.csv", index=False)
    active.sort_values("realized_profit", ascending=False).head(50).to_csv(
        args.output_dir / "top_realized_runs.csv", index=False
    )
    active.sort_values("without_top_1", ascending=False).head(50).to_csv(
        args.output_dir / "top_without_top1_runs.csv", index=False
    )
    plot_variant_summary(summary, args.output_dir)
    plot_best_runs(active, args.output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
