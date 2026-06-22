#!/usr/bin/env python3
"""Visual summaries for long-holdout underdog allocation experiments."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_counts(value: object) -> dict[str, int]:
    if pd.isna(value):
        return {}
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
        return {str(key): int(val) for key, val in parsed.items()}
    except (SyntaxError, ValueError, TypeError):
        try:
            parsed = json.loads(text)
            return {str(key): int(val) for key, val in parsed.items()}
        except (json.JSONDecodeError, TypeError):
            return {}


def annotate_bars(axis) -> None:
    for patch in axis.patches:
        height = patch.get_height()
        if not np.isfinite(height):
            continue
        axis.annotate(
            f"{height:,.0f}",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=7,
            rotation=90,
            xytext=(0, 2 if height >= 0 else -2),
            textcoords="offset points",
        )


def plot_profit_by_policy(df: pd.DataFrame, output: Path) -> None:
    active = df[df["entries"] > 0].copy()
    windows = ["3m", "6m", "12m", "all"]
    gates = sorted(active["gate"].unique())
    fig, axes = plt.subplots(len(gates), len(windows), figsize=(18, 4.8 * len(gates)), sharey=True)
    if len(gates) == 1:
        axes = np.asarray([axes])
    for row, gate in enumerate(gates):
        for col, window in enumerate(windows):
            axis = axes[row, col]
            subset = active[(active["gate"] == gate) & (active["holdout_window"] == window)]
            order = (
                subset.groupby("sizing_policy")["realized_profit"]
                .median()
                .sort_values(ascending=False)
                .index
            )
            values = subset.set_index(["cut_fraction", "sizing_policy"])["realized_profit"].unstack()
            values = values.reindex(columns=order)
            values.plot(kind="bar", ax=axis, width=0.8)
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_title(f"{gate} / {window}")
            axis.set_xlabel("cut")
            axis.set_ylabel("profit ($)")
            axis.tick_params(axis="x", rotation=0)
            if row != 0 or col != len(windows) - 1:
                axis.legend().remove()
    fig.suptitle("Realized Profit by Policy, Gate, Window, and Cut", y=0.995)
    fig.tight_layout()
    fig.savefig(output / "profit_by_policy_window.png", dpi=160)
    plt.close(fig)


def plot_tail_dependence(df: pd.DataFrame, output: Path) -> None:
    active = df[df["entries"] > 0].copy()
    active["top1_contribution"] = active["realized_profit"] - active["without_top_1"]
    active["tail_dependence_ratio"] = np.where(
        active["realized_profit"].abs() > 1e-9,
        active["top1_contribution"] / active["realized_profit"].abs(),
        np.nan,
    )
    best = active.sort_values("realized_profit", ascending=False).head(24).copy()
    labels = (
        best["cut_fraction"].astype(str)
        + " "
        + best["gate"].str.replace("_", " ")
        + " "
        + best["holdout_window"]
        + "\n"
        + best["sizing_policy"]
    )
    x = np.arange(len(best))
    fig, axis = plt.subplots(figsize=(18, 7))
    width = 0.35
    axis.bar(x - width / 2, best["realized_profit"], width, label="realized")
    axis.bar(x + width / 2, best["without_top_1"], width, label="without top 1")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    axis.set_ylabel("profit ($)")
    axis.set_title("Top Results vs. Same Runs With Largest Winner Removed")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "tail_dependence_top_results.png", dpi=160)
    plt.close(fig)


def plot_profit_distribution(df: pd.DataFrame, output: Path) -> None:
    active = df[df["entries"] > 0].copy()
    policies = sorted(active["sizing_policy"].unique())
    data = [active.loc[active["sizing_policy"] == policy, "realized_profit"].to_numpy() for policy in policies]
    fig, axis = plt.subplots(figsize=(12, 7))
    axis.boxplot(data, labels=policies, showmeans=True)
    for idx, values in enumerate(data, start=1):
        jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) else []
        axis.scatter(np.full(len(values), idx) + jitter, values, alpha=0.55, s=24)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("realized profit ($)")
    axis.set_title("Profit Distribution Across Cuts, Gates, and Holdout Windows")
    axis.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output / "profit_distribution_by_policy.png", dpi=160)
    plt.close(fig)


def plot_deployment_vs_profit(df: pd.DataFrame, output: Path) -> None:
    active = df[df["entries"] > 0].copy()
    fig, axis = plt.subplots(figsize=(11, 7))
    for policy, group in active.groupby("sizing_policy"):
        axis.scatter(group["deployed"], group["realized_profit"], label=policy, s=55, alpha=0.75)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("capital deployed ($)")
    axis.set_ylabel("realized profit ($)")
    axis.set_title("Deployment vs. Profit")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "deployment_vs_profit.png", dpi=160)
    plt.close(fig)


def plot_exit_family_mix(df: pd.DataFrame, output: Path) -> None:
    rows = []
    for _, row in df[df["entries"] > 0].iterrows():
        counts = parse_counts(row.get("exit_family_counts"))
        for family, count in counts.items():
            rows.append({
                "sizing_policy": row["sizing_policy"],
                "exit_family": family,
                "count": count,
            })
    if not rows:
        return
    mix = pd.DataFrame(rows).groupby(["sizing_policy", "exit_family"])["count"].sum().unstack(fill_value=0)
    axis = mix.plot(kind="bar", stacked=True, figsize=(12, 7))
    axis.set_title("Exit Family Usage Across Active Runs")
    axis.set_ylabel("positions")
    axis.tick_params(axis="x", rotation=35)
    fig = axis.figure
    fig.tight_layout()
    fig.savefig(output / "exit_family_mix.png", dpi=160)
    plt.close(fig)


def write_summary_tables(df: pd.DataFrame, output: Path) -> None:
    active = df[df["entries"] > 0].copy()
    active["top1_contribution"] = active["realized_profit"] - active["without_top_1"]
    active["robust_profit"] = active["without_top_1"]
    best = active.sort_values("realized_profit", ascending=False).head(25)
    robust = active.sort_values("robust_profit", ascending=False).head(25)
    by_policy = active.groupby("sizing_policy").agg(
        runs=("realized_profit", "size"),
        mean_profit=("realized_profit", "mean"),
        median_profit=("realized_profit", "median"),
        positive_rate=("realized_profit", lambda values: float((values > 0).mean())),
        mean_without_top1=("without_top_1", "mean"),
        mean_deployed=("deployed", "mean"),
        mean_drawdown=("max_drawdown", "mean"),
    ).reset_index().sort_values("mean_profit", ascending=False)
    best.to_csv(output / "top_realized_runs.csv", index=False)
    robust.to_csv(output / "top_without_top1_runs.csv", index=False)
    by_policy.to_csv(output / "policy_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("reports/long_holdout_weighting/account_summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/long_holdout_weighting/visuals"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    plot_profit_by_policy(df, args.output_dir)
    plot_tail_dependence(df, args.output_dir)
    plot_profit_distribution(df, args.output_dir)
    plot_deployment_vs_profit(df, args.output_dir)
    plot_exit_family_mix(df, args.output_dir)
    write_summary_tables(df, args.output_dir)
    print(json.dumps({
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "files": sorted(path.name for path in args.output_dir.iterdir()),
    }, indent=2))


if __name__ == "__main__":
    main()
