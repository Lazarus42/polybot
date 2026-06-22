#!/usr/bin/env python3
"""Risk summaries and bootstrap diagnostics for OOS period-result CSVs."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_INPUTS = [
    Path("reports/oos_best_strategy_duration_sweep/oos_period_results.csv"),
    Path("reports/oos_ensemble_tests/ensemble_period_results.csv"),
    Path("reports/oos_best_strategy_throttle_comparison/combined_oos_period_results.csv"),
    Path("reports/market_making_oos/market_making_period_results.csv"),
    Path("reports/market_making_selection_sweep/market_making_period_results.csv"),
]

DEFAULT_GROUP_COLS = ["experiment", "selected_strategy", "test_months"]
AUTO_GROUP_COLS = ["throttle_variant", "strategy", "ensemble", "feature_mode", "selection_score"]


def infer_label(path: Path) -> str:
    parent = path.parent.name
    return parent.removeprefix("oos_").removeprefix("market_making_")


def load_period_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.insert(0, "source_file", str(path))
    if "experiment" not in df.columns:
        df.insert(1, "experiment", infer_label(path))
    if "account_return" not in df.columns:
        if {"total_account_value", "initial_cash"}.issubset(df.columns):
            df["account_return"] = df["total_account_value"] / df["initial_cash"] - 1.0
        elif {"realized_profit", "initial_cash"}.issubset(df.columns):
            df["account_return"] = df["realized_profit"] / df["initial_cash"]
        else:
            raise ValueError(f"{path} has no account_return or enough columns to infer it")
    if "selected_strategy" not in df.columns:
        if "strategy" in df.columns:
            df["selected_strategy"] = df["strategy"]
        elif "ensemble" in df.columns:
            df["selected_strategy"] = df["ensemble"]
        elif {"feature_mode", "selection_score"}.issubset(df.columns):
            df["selected_strategy"] = df["feature_mode"].astype(str) + ":" + df["selection_score"].astype(str)
        else:
            df["selected_strategy"] = df["experiment"].astype(str)
    for column in ("without_top_1", "profit_capped_at_5x_cost", "profit_capped_at_10x_cost", "profit_capped_at_20x_cost", "max_drawdown"):
        if column not in df.columns:
            df[column] = np.nan
    return df


def annualization_factor(test_months: float) -> float:
    return 12.0 / test_months if test_months > 0 else 1.0


def downside_std(returns: np.ndarray) -> float:
    downside = np.minimum(returns, 0.0)
    if len(downside) <= 1:
        return 0.0
    return float(np.std(downside, ddof=1))


def sharpe(returns: np.ndarray, periods_per_year: float) -> float:
    if len(returns) <= 1:
        return np.nan
    std = float(np.std(returns, ddof=1))
    if std <= 1e-12:
        return np.nan
    return float(np.mean(returns) / std * math.sqrt(periods_per_year))


def sortino(returns: np.ndarray, periods_per_year: float) -> float:
    ds = downside_std(returns)
    if ds <= 1e-12:
        return np.nan
    return float(np.mean(returns) / ds * math.sqrt(periods_per_year))


def compound_return(returns: np.ndarray) -> float:
    return float(np.prod(1.0 + returns) - 1.0) if len(returns) else 0.0


def annualized_compound(returns: np.ndarray, test_months: float) -> float:
    if not len(returns):
        return 0.0
    total = compound_return(returns)
    years = max(1e-9, len(returns) * test_months / 12.0)
    if total <= -1.0:
        return -1.0
    return float((1.0 + total) ** (1.0 / years) - 1.0)


def max_drawdown_from_returns(returns: np.ndarray) -> float:
    if not len(returns):
        return 0.0
    curve = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(curve)
    drawdowns = (peaks - curve) / np.maximum(peaks, 1e-12)
    return float(np.max(drawdowns))


def summarize_group(group: pd.DataFrame, keys: dict[str, object]) -> dict[str, object]:
    returns = group["account_return"].astype(float).to_numpy()
    profits = group["realized_profit"].astype(float).to_numpy()
    months = float(group["test_months"].iloc[0]) if "test_months" in group.columns else 1.0
    periods_per_year = annualization_factor(months)
    ann_return = annualized_compound(returns, months)
    observed_drawdown = float(group["max_drawdown"].astype(float).max()) if group["max_drawdown"].notna().any() else 0.0
    series_drawdown = max_drawdown_from_returns(returns)
    worst_drawdown = max(observed_drawdown, series_drawdown)
    calmar = ann_return / worst_drawdown if worst_drawdown > 1e-12 else np.nan
    return {
        **keys,
        "periods": int(len(group)),
        "mean_profit": float(np.mean(profits)),
        "median_profit": float(np.median(profits)),
        "mean_return": float(np.mean(returns)),
        "median_return": float(np.median(returns)),
        "return_std": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
        "downside_std": downside_std(returns),
        "sharpe_ann": sharpe(returns, periods_per_year),
        "sortino_ann": sortino(returns, periods_per_year),
        "compound_return": compound_return(returns),
        "annualized_compound_return": ann_return,
        "max_drawdown": worst_drawdown,
        "calmar": calmar,
        "positive_rate": float(np.mean(profits > 0)),
        "return_positive_rate": float(np.mean(returns > 0)),
        "worst_profit": float(np.min(profits)),
        "best_profit": float(np.max(profits)),
        "p05_return": float(np.quantile(returns, 0.05)),
        "p25_return": float(np.quantile(returns, 0.25)),
        "p75_return": float(np.quantile(returns, 0.75)),
        "p95_return": float(np.quantile(returns, 0.95)),
        "mean_without_top1": float(group["without_top_1"].astype(float).mean()) if group["without_top_1"].notna().any() else np.nan,
        "mean_capped_5x": float(group["profit_capped_at_5x_cost"].astype(float).mean()) if group["profit_capped_at_5x_cost"].notna().any() else np.nan,
        "mean_capped_10x": float(group["profit_capped_at_10x_cost"].astype(float).mean()) if group["profit_capped_at_10x_cost"].notna().any() else np.nan,
        "mean_capped_20x": float(group["profit_capped_at_20x_cost"].astype(float).mean()) if group["profit_capped_at_20x_cost"].notna().any() else np.nan,
        "mean_entries": float(group["entries"].astype(float).mean()) if "entries" in group.columns else np.nan,
        "mean_deployed": float(group["deployed"].astype(float).mean()) if "deployed" in group.columns else np.nan,
    }


def bootstrap_group(
    group: pd.DataFrame,
    keys: dict[str, object],
    iterations: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    returns = group["account_return"].astype(float).to_numpy()
    profits = group["realized_profit"].astype(float).to_numpy()
    months = float(group["test_months"].iloc[0]) if "test_months" in group.columns else 1.0
    periods_per_year = annualization_factor(months)
    if len(returns) == 0:
        return []
    rows = []
    for _ in range(iterations):
        sample = rng.integers(0, len(returns), len(returns))
        sampled_returns = returns[sample]
        sampled_profits = profits[sample]
        rows.append({
            **keys,
            "sample_mean_profit": float(np.mean(sampled_profits)),
            "sample_mean_return": float(np.mean(sampled_returns)),
            "sample_sharpe_ann": sharpe(sampled_returns, periods_per_year),
            "sample_sortino_ann": sortino(sampled_returns, periods_per_year),
            "sample_compound_return": compound_return(sampled_returns),
            "sample_annualized_compound_return": annualized_compound(sampled_returns, months),
            "sample_max_drawdown": max_drawdown_from_returns(sampled_returns),
            "sample_positive_rate": float(np.mean(sampled_profits > 0)),
        })
    return rows


def summarize_bootstrap(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        return samples
    metric_columns = [
        "sample_mean_profit",
        "sample_mean_return",
        "sample_sharpe_ann",
        "sample_sortino_ann",
        "sample_compound_return",
        "sample_annualized_compound_return",
        "sample_max_drawdown",
        "sample_positive_rate",
    ]
    key_columns = [column for column in samples.columns if column not in metric_columns]
    rows = []
    for key, group in samples.groupby(key_columns, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(key_columns, key))
        for metric in metric_columns:
            values = group[metric].replace([np.inf, -np.inf], np.nan).dropna()
            if values.empty:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_p05"] = np.nan
                row[f"{metric}_median"] = np.nan
                row[f"{metric}_p95"] = np.nan
                row[f"{metric}_prob_gt_0"] = np.nan
            else:
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_p05"] = float(values.quantile(0.05))
                row[f"{metric}_median"] = float(values.median())
                row[f"{metric}_p95"] = float(values.quantile(0.95))
                row[f"{metric}_prob_gt_0"] = float((values > 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def default_existing_inputs() -> list[Path]:
    return [path for path in DEFAULT_INPUTS if path.exists()]


def resolve_group_cols(combined: pd.DataFrame, requested: Iterable[str]) -> list[str]:
    group_cols = [column for column in requested if column in combined.columns]
    using_default = list(requested) == DEFAULT_GROUP_COLS
    if using_default:
        for column in AUTO_GROUP_COLS:
            if column in combined.columns and column not in group_cols:
                group_cols.append(column)
    if "test_months" in combined.columns and "test_months" not in group_cols:
        group_cols.append("test_months")
    if not group_cols:
        group_cols = ["experiment"]
    return group_cols


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="*")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/oos_risk_summary"))
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--group-cols",
        nargs="+",
        default=DEFAULT_GROUP_COLS,
        help=(
            "Columns used to form risk-summary groups when present. The default "
            "also auto-adds variant columns such as throttle_variant when present."
        ),
    )
    args = parser.parse_args()

    inputs = args.inputs or default_existing_inputs()
    if not inputs:
        raise SystemExit("No input files provided and no default OOS period-result files exist.")

    frames = []
    for path in inputs:
        if not path.exists():
            print(f"Skipping missing input: {path}")
            continue
        frames.append(load_period_file(path))
    if not frames:
        raise SystemExit("No readable input files.")
    combined = pd.concat(frames, ignore_index=True, sort=False)

    group_cols = resolve_group_cols(combined, args.group_cols)

    summary_rows = []
    bootstrap_rows = []
    rng = np.random.default_rng(args.seed)
    for key, group in combined.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        keys = dict(zip(group_cols, key))
        summary_rows.append(summarize_group(group, keys))
        if args.bootstrap_iterations > 0:
            bootstrap_rows.extend(bootstrap_group(group, keys, args.bootstrap_iterations, rng))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_dir / "combined_period_results.csv", index=False)
    risk_summary = pd.DataFrame(summary_rows).sort_values(
        [column for column in ("test_months", "sharpe_ann", "mean_return") if column in group_cols or column in summary_rows[0]],
        ascending=[True, False, False][:len([column for column in ("test_months", "sharpe_ann", "mean_return") if column in group_cols or column in summary_rows[0]])],
    )
    risk_summary.to_csv(args.output_dir / "risk_summary.csv", index=False)

    bootstrap_samples = pd.DataFrame(bootstrap_rows)
    if not bootstrap_samples.empty:
        bootstrap_samples.to_csv(args.output_dir / "bootstrap_samples.csv", index=False)
        bootstrap_summary = summarize_bootstrap(bootstrap_samples)
        bootstrap_summary.to_csv(args.output_dir / "bootstrap_summary.csv", index=False)

    print({
        "output_dir": str(args.output_dir),
        "inputs": [str(path) for path in inputs],
        "period_rows": len(combined),
        "group_cols": group_cols,
        "summary_rows": len(risk_summary),
        "bootstrap_rows": len(bootstrap_samples),
    })


if __name__ == "__main__":
    main()
