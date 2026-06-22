#!/usr/bin/env python3
"""Walk-forward residual-combination portfolio over strategy-family components.

At each monthly step we use only a trailing window to:
  1. estimate each candidate component's edge (mean monthly per-dollar return),
     volatility, and pairwise correlation;
  2. greedily select a low-correlation subset of positive-edge components
     (trade the residuals, not overlapping bets);
  3. risk-weight the survivors by inverse volatility, capped per name; and
  4. allocate the monthly budget by those weights and replay the *next* month
     out-of-sample under the liquidity participation cap.

We compare this against an equal-weight-all baseline and a single-best baseline.
Everything is causal: the fit for month t uses only months strictly before t.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from long_holdout_weighting_experiments import add_months
from realistic_underdog_account import write_csv
from rank_components_tune import assign_sleeve
from replay_family_ensemble_oos import load_component_signals, replay_ensemble
from replay_strategy_family_oos import period_key  # noqa: F401  (kept for parity/debugging)
from walk_forward_oos import month_floor


FAMILY_LANE_DEFAULTS = {
    "single_market_existing": 0.55,
    "cross_market_overround": 0.15,
    "complete_set_underround": 0.15,
    "duplicate_gap": 0.075,
    "ladder_violation": 0.075,
}


def month_list(first: datetime, last: datetime) -> list[datetime]:
    months = []
    cur = first
    while cur <= last:
        months.append(cur)
        cur = add_months(cur, 1)
    return months


def component_family(component: str) -> str:
    if component.startswith("complete_set_underround"):
        return "complete_set_underround"
    if component.startswith("cross_market_overround"):
        return "cross_market_overround"
    if component.startswith("duplicate_gap"):
        return "duplicate_gap"
    if component.startswith("ladder_violation"):
        return "ladder_violation"
    return "single_market_existing"


def normalize_weights(weights: dict[str, float], max_weight: float | None = None) -> dict[str, float]:
    weights = {k: float(v) for k, v in weights.items() if np.isfinite(v) and v > 0}
    if not weights:
        return {}
    if max_weight is not None and max_weight > 0:
        weights = {k: min(v, max_weight) for k, v in weights.items()}
    total = sum(weights.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in weights.items()}


def parse_family_lanes(value: str) -> dict[str, float]:
    if not value:
        return dict(FAMILY_LANE_DEFAULTS)
    lanes: dict[str, float] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        name, weight = item.split("=", 1)
        lanes[name.strip()] = float(weight)
    return normalize_weights(lanes)


def candidate_components(
    signals: pd.DataFrame,
    pool_file: Path | None,
    min_total_signals: int,
    core_stake: float,
    tail_stake: float,
    external_components: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Build candidate component dicts, restricted to a ranked pool file if given."""
    external_components = external_components or set()
    wanted: set[tuple[str, str]] | None = None
    if pool_file and pool_file.exists():
        specs = json.loads(pool_file.read_text())["components"]
        wanted = {(s.split(":")[0], s.split(":")[1]) for s in specs}

    counts = signals.groupby(["strategy", "exit_rule"]).size()
    components = []
    for (strategy, exit_rule), n in counts.items():
        is_external = (strategy, exit_rule) in external_components
        if wanted is not None and (strategy, exit_rule) not in wanted and not is_external:
            continue
        if wanted is None and n < min_total_signals and not is_external:
            continue
        sleeve = assign_sleeve(exit_rule)
        components.append({
            "strategy": strategy,
            "exit_rule": exit_rule,
            "sleeve": "external" if is_external else sleeve,
            "stake": tail_stake if sleeve == "tail" else core_stake,
            "monthly_cap": 0.0,   # set per period from weights
            "priority": 0,        # set per period from weights
            "component": f"{strategy}:{exit_rule}",
            "family": component_family(f"{strategy}:{exit_rule}"),
        })
    return components


def component_monthly_matrices(signals: pd.DataFrame, components: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    """Return month x component matrices for unit return, profit dollars, and capacity."""
    keys = {(c["strategy"], c["exit_rule"]): c["component"] for c in components}
    sub = signals[signals.set_index(["strategy", "exit_rule"]).index.isin(keys)].copy()
    sub = sub[np.isfinite(sub["unit_return"])]
    sub["month"] = sub["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M")
    sub["component"] = [keys[(s, e)] for s, e in zip(sub["strategy"], sub["exit_rule"])]
    sub["entry_fill_usd"] = pd.to_numeric(sub.get("entry_fill_usd", 0.0), errors="coerce").fillna(0.0)
    sub["unit_return"] = pd.to_numeric(sub["unit_return"], errors="coerce")
    sub["profit_capacity"] = sub["entry_fill_usd"] * sub["unit_return"]
    return {
        "unit": sub.groupby(["month", "component"])["unit_return"].mean().unstack("component").sort_index(),
        "profit": sub.groupby(["month", "component"])["profit_capacity"].sum().unstack("component").sort_index(),
        "capacity": sub.groupby(["month", "component"])["entry_fill_usd"].sum().unstack("component").sort_index(),
        "count": sub.groupby(["month", "component"]).size().unstack("component").sort_index(),
    }


def component_diagnostics(
    matrices: dict[str, pd.DataFrame],
    components: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    unit = matrices["unit"]
    profit = matrices["profit"].fillna(0.0)
    capacity = matrices["capacity"].fillna(0.0)
    count = matrices["count"].fillna(0.0)
    component_map = {c["component"]: c for c in components}
    rows: list[dict[str, Any]] = []
    for comp in sorted(component_map):
        u = unit[comp].dropna() if comp in unit else pd.Series(dtype=float)
        p = profit[comp] if comp in profit else pd.Series(dtype=float)
        cap = capacity[comp] if comp in capacity else pd.Series(dtype=float)
        cnt = count[comp] if comp in count else pd.Series(dtype=float)
        active = cap[cap > 0]
        rows.append({
            "component": comp,
            "family": component_map[comp]["family"],
            "months_with_return": int(len(u)),
            "months_with_capacity": int(len(active)),
            "total_capacity": float(cap.sum()) if len(cap) else 0.0,
            "mean_monthly_capacity": float(active.mean()) if len(active) else 0.0,
            "median_monthly_capacity": float(active.median()) if len(active) else 0.0,
            "total_profit_capacity": float(p.sum()) if len(p) else 0.0,
            "mean_monthly_profit_capacity": float(p.mean()) if len(p) else 0.0,
            "mean_unit_return": float(u.mean()) if len(u) else 0.0,
            "median_unit_return": float(u.median()) if len(u) else 0.0,
            "unit_positive_rate": float((u > 0).mean()) if len(u) else 0.0,
            "total_signals": int(cnt.sum()) if len(cnt) else 0,
        })
    return rows


def fit_stat_weights(
    train: pd.DataFrame,
    mode: str,
    min_train_obs: int,
    corr_threshold: float,
    max_components: int,
    max_weight: float,
    score_frame: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Return component -> weight from trailing monthly component statistics."""
    stats = {}
    for col in train.columns:
        series = train[col].dropna()
        if len(series) < min_train_obs:
            continue
        mean = float(series.mean())
        std = float(series.std(ddof=1)) if len(series) > 1 else float("nan")
        if mean <= 0 or not np.isfinite(std) or std <= 0:
            continue
        score = mean / std
        if mode == "capacity_weighted" and score_frame is not None and col in score_frame:
            score = mean * float(score_frame[col].fillna(0.0).mean())
        stats[col] = {"mean": mean, "std": std, "score": score}
    if not stats:
        return {}

    ranked = sorted(stats, key=lambda c: stats[c]["score"], reverse=True)
    if mode == "top1":
        return {ranked[0]: 1.0}

    if mode == "equal_all":
        selected = ranked[:max_components]
        return {c: 1.0 / len(selected) for c in selected}

    # mode == "decorrelated_riskparity"
    corr = train[ranked].corr()
    selected: list[str] = []
    for c in ranked:
        if all(abs(float(corr.loc[c, s])) < corr_threshold for s in selected if np.isfinite(corr.loc[c, s])):
            selected.append(c)
        if len(selected) >= max_components:
            break
    if not selected:
        return {}
    inv_vol = np.array([1.0 / stats[c]["std"] for c in selected])
    weights = inv_vol / inv_vol.sum()
    weights = np.minimum(weights, max_weight)
    weights = weights / weights.sum()  # renormalize after capping
    return dict(zip(selected, weights))


def fit_utility_weights(
    profit: pd.DataFrame,
    capacity: pd.DataFrame,
    min_train_obs: int,
    corr_threshold: float,
    max_components: int,
    max_weight: float,
    util_lambda: float,
    util_gamma: float,
) -> dict[str, float]:
    """Select & weight components on a unified dollar-utility objective.

    Per component, on the trailing window:
      ep  = mean monthly profit dollars            (expected profit)
      dd  = downside deviation of monthly profit    (risk; losing months only)
      cap = mean monthly deployable capacity dollars(idle proxy: more = less idle)

    risk_adjusted = ep - util_lambda * dd
    score         = max(risk_adjusted, 0) * cap ** util_gamma

    util_lambda controls downside aversion; util_gamma controls idle aversion
    (gamma=0 ignores capacity; higher gamma tilts the budget toward components
    that actually absorb capital so less of it sits idle). Selection greedily
    de-correlates on the *profit* series; survivors are weighted by score, capped.
    """
    profit = profit.fillna(0.0)
    capacity = capacity.fillna(0.0)
    stats: dict[str, dict[str, float]] = {}
    for col in profit.columns:
        series = profit[col]
        # require the component to have actually traded in enough trailing months
        active = series[series != 0.0]
        if len(active) < min_train_obs:
            continue
        ep = float(series.mean())
        downside = series.clip(upper=0.0)
        dd = float(np.sqrt((downside ** 2).mean()))
        cap = float(capacity[col].mean()) if col in capacity else 0.0
        risk_adjusted = ep - util_lambda * dd
        if risk_adjusted <= 0 or cap <= 0:
            continue
        score = risk_adjusted * (cap ** util_gamma)
        if not np.isfinite(score) or score <= 0:
            continue
        stats[col] = {"score": score, "ep": ep, "dd": dd, "cap": cap}
    if not stats:
        return {}

    ranked = sorted(stats, key=lambda c: stats[c]["score"], reverse=True)
    corr = profit[ranked].corr()
    selected: list[str] = []
    for c in ranked:
        if all(abs(float(corr.loc[c, s])) < corr_threshold for s in selected if np.isfinite(corr.loc[c, s])):
            selected.append(c)
        if len(selected) >= max_components:
            break
    if not selected:
        return {}
    scores = np.array([stats[c]["score"] for c in selected], dtype=float)
    weights = scores / scores.sum()
    weights = np.minimum(weights, max_weight)
    weights = weights / weights.sum()
    return dict(zip(selected, weights))


def fit_weights(
    matrices: dict[str, pd.DataFrame],
    mode: str,
    min_train_obs: int,
    corr_threshold: float,
    max_components: int,
    max_weight: float,
    family_lanes: dict[str, float],
    util_lambda: float = 0.0,
    util_gamma: float = 0.0,
) -> dict[str, float]:
    """Select weights for one monthly fit under the requested allocation objective."""
    unit = matrices["unit"]
    profit = matrices["profit"].fillna(0.0)
    capacity = matrices["capacity"].fillna(0.0)
    if mode in {"decorrelated_riskparity", "unit_sharpe", "equal_all", "top1"}:
        stat_mode = "decorrelated_riskparity" if mode in {"decorrelated_riskparity", "unit_sharpe"} else mode
        return fit_stat_weights(unit, stat_mode, min_train_obs, corr_threshold, max_components, max_weight)
    if mode == "profit_sharpe":
        return fit_stat_weights(profit, "decorrelated_riskparity", min_train_obs, corr_threshold, max_components, max_weight)
    if mode == "capacity_weighted":
        return fit_stat_weights(unit, "capacity_weighted", min_train_obs, corr_threshold, max_components, max_weight, capacity)
    if mode == "utility":
        return fit_utility_weights(profit, capacity, min_train_obs, corr_threshold,
                                   max_components, max_weight, util_lambda, util_gamma)
    if mode == "family_lanes":
        combined: dict[str, float] = {}
        components_by_family: dict[str, list[str]] = defaultdict(list)
        for col in unit.columns:
            components_by_family[component_family(col)].append(col)
        for family, lane_weight in family_lanes.items():
            cols = components_by_family.get(family, [])
            if not cols or lane_weight <= 0:
                continue
            lane_unit = unit[cols]
            lane_capacity = capacity[cols]
            lane_components = max(1, math.ceil(max_components * lane_weight))
            lane_weights = fit_stat_weights(
                lane_unit,
                "capacity_weighted",
                min_train_obs,
                corr_threshold,
                lane_components,
                max_weight,
                lane_capacity,
            )
            for comp, weight in lane_weights.items():
                combined[comp] = lane_weight * weight
        return normalize_weights(combined, max_weight)
    raise ValueError(f"Unknown allocation mode: {mode}")


def replay_month(
    signals: pd.DataFrame,
    components: list[dict[str, Any]],
    weights: dict[str, float],
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    args: argparse.Namespace,
) -> dict[str, Any]:
    chosen = []
    ranked_by_weight = sorted(weights, key=lambda c: weights[c], reverse=True)
    for rank, comp_key in enumerate(ranked_by_weight):
        base = next(c for c in components if c["component"] == comp_key)
        item = dict(base)
        item["monthly_cap"] = float(weights[comp_key] * args.deploy_budget)
        item["priority"] = len(ranked_by_weight) - rank
        chosen.append(item)
    if not chosen:
        return {"realized_profit": 0.0, "deployed": 0.0, "slippage_cost": 0.0, "max_drawdown": 0.0, "entries": 0, "n_components": 0}
    combined = load_component_signals(signals, chosen)
    period_signals = combined[(combined["timestamp"] >= test_start) & (combined["timestamp"] < test_end)]
    result = replay_ensemble(
        period_signals, test_end, args.initial_cash, args.period_budget, "month",
        args.reserve_fraction, args.min_stake, args.max_trades_per_market,
        args.max_components_per_market, args.participation_fraction,
        args.min_stake_fill_fraction, args.slippage_model, args.slippage_coef,
        args.stake_fill_fraction, args.max_stake,
    )
    return {
        "realized_profit": result["realized_profit"],
        "deployed": result["deployed"],
        "slippage_cost": result.get("slippage_cost", 0.0),
        "max_drawdown": result["max_drawdown"],
        "entries": result["entries"],
        "n_components": len(chosen),
    }


def summarize(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for mode, g in df.groupby("mode"):
        p = g["realized_profit"].to_numpy(dtype=float)
        std = float(np.std(p, ddof=1)) if len(p) > 1 else 0.0
        out.append({
            "mode": mode,
            "months": int(len(g)),
            "total_profit": float(p.sum()),
            "mean_monthly_profit": float(p.mean()),
            "monthly_profit_std": std,
            "monthly_sharpe": float(p.mean() / std) if std > 0 else 0.0,
            "annualized_sharpe": float(p.mean() / std * np.sqrt(12)) if std > 0 else 0.0,
            "positive_rate": float(np.mean(p > 0)),
            "worst_month": float(p.min()),
            "best_month": float(p.max()),
            "worst_monthly_drawdown": float(g["max_drawdown"].max()),
            "mean_deployed": float(g["deployed"].mean()),
            "mean_slippage_cost": float(g["slippage_cost"].mean()) if "slippage_cost" in g else 0.0,
            "total_slippage_cost": float(g["slippage_cost"].sum()) if "slippage_cost" in g else 0.0,
            "mean_n_components": float(g["n_components"].mean()),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", type=Path, default=Path("reports/strategy_family_diagnostics/strategy_family_signals.csv"))
    parser.add_argument("--external-signals", type=Path, nargs="*", default=[],
                        help="Additional component-format signal CSVs to append, e.g. cross-market sleeves.")
    parser.add_argument("--pool-file", type=Path, default=Path("reports/component_ranking/selected_components.json"),
                        help="Ranked pool to restrict candidates to; ignored if missing.")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/residual_portfolio"))
    parser.add_argument("--first-month", default="2022-11-01")
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--min-train-obs", type=int, default=4, help="Min trailing months a component must trade in.")
    parser.add_argument("--min-total-signals", type=int, default=40, help="Used only when no pool file is given.")
    parser.add_argument("--corr-threshold", type=float, default=0.6)
    parser.add_argument("--max-components", type=int, default=8)
    parser.add_argument("--max-weight", type=float, default=0.40)
    parser.add_argument("--core-stake", type=float, default=5.0)
    parser.add_argument("--tail-stake", type=float, default=2.0)
    parser.add_argument("--initial-cash", type=float, default=5000.0)
    parser.add_argument("--period-budget", type=float, default=5000.0)
    parser.add_argument("--deploy-budget", type=float, default=5000.0, help="Total monthly cap split across components by weight.")
    parser.add_argument("--reserve-fraction", type=float, default=0.30)
    parser.add_argument("--min-stake", type=float, default=0.25,
                        help="Absolute minimum stake floor below which a trade is skipped.")
    parser.add_argument("--max-trades-per-market", type=int, default=1)
    parser.add_argument("--max-components-per-market", type=int, default=1)
    parser.add_argument("--participation-fraction", type=float, default=0.10)
    parser.add_argument("--min-stake-fill-fraction", type=float, default=0.02,
                        help="Market-dependent minimum: floor scales with this fraction of entry_fill_usd (0 disables).")
    parser.add_argument("--modes", nargs="+",
                        default=["decorrelated_riskparity", "profit_sharpe", "capacity_weighted", "family_lanes", "equal_all", "top1"],
                        choices=["decorrelated_riskparity", "unit_sharpe", "profit_sharpe", "capacity_weighted", "utility", "family_lanes", "equal_all", "top1"])
    parser.add_argument("--util-lambda", type=float, default=1.0,
                        help="utility mode: downside-deviation aversion (higher = more downside-protective).")
    parser.add_argument("--util-gamma", type=float, default=0.5,
                        help="utility mode: idle-capital aversion (0 ignores capacity; higher tilts to high-absorption components).")
    parser.add_argument("--slippage-model", choices=["none", "linear", "sqrt"], default="none",
                        help="Market-impact model: effective entry price worsens by coef*(debit/fill)^alpha (alpha=1 linear, 0.5 sqrt).")
    parser.add_argument("--slippage-coef", type=float, default=0.0,
                        help="Impact coefficient; 0 disables. Effective return = (1+unit_return)/(1+s)-1 with s = coef*(debit/fill)^alpha.")
    parser.add_argument("--stake-fill-fraction", type=float, default=0.0,
                        help="Liquidity-scaled sizing: target stake = clip(this*entry_fill_usd, base stake, --max-stake). "
                             "0 keeps the flat base stake. Set <= --participation-fraction so participation stays the hard ceiling.")
    parser.add_argument("--max-stake", type=float, default=float("inf"),
                        help="Absolute per-trade ceiling for liquidity-scaled sizing (risk cap on single-name exposure).")
    parser.add_argument("--family-lanes", default="",
                        help="Comma-separated lane weights for family_lanes mode, e.g. single_market_existing=0.55,cross_market_overround=0.15,complete_set_underround=0.15,duplicate_gap=0.075,ladder_violation=0.075")
    parser.add_argument("--exclude-categories", nargs="*", default=[],
                        help="Drop these market categories before anything else (e.g. crypto).")
    parser.add_argument("--min-horizon-days", type=float, default=None,
                        help="Drop signals whose horizon-to-deadline is below this (e.g. 1 to skip sub-daily markets).")
    args = parser.parse_args()

    signals = pd.read_csv(args.signals, low_memory=False)
    signals["timestamp"] = pd.to_datetime(signals["timestamp"], utc=True, format="mixed")
    external_keys: set[tuple[str, str]] = set()
    external_files = []
    for path in args.external_signals:
        if not path.exists():
            raise SystemExit(f"External signals file not found: {path}")
        ext = pd.read_csv(path, low_memory=False)
        if ext.empty:
            continue
        required = {"timestamp", "strategy", "exit_rule", "unit_return", "market_id", "entry_fill_usd"}
        missing = required.difference(ext.columns)
        if missing:
            raise SystemExit(f"External signals file {path} missing required columns: {sorted(missing)}")
        ext["timestamp"] = pd.to_datetime(ext["timestamp"], utc=True, format="mixed")
        if "category" not in ext.columns:
            ext["category"] = "external"
        if "horizon_days" not in ext.columns:
            ext["horizon_days"] = np.nan
        external_keys.update(set(zip(ext["strategy"].astype(str), ext["exit_rule"].astype(str))))
        external_files.append(str(path))
        signals = pd.concat([signals, ext], ignore_index=True, sort=False)
    if args.exclude_categories:
        signals = signals[~signals["category"].isin(args.exclude_categories)]
    if args.min_horizon_days is not None:
        signals = signals[signals["horizon_days"].astype(float) >= args.min_horizon_days]
    components = candidate_components(
        signals, args.pool_file, args.min_total_signals, args.core_stake, args.tail_stake, external_keys
    )
    if not components:
        raise SystemExit("No candidate components matched.")
    matrices_all = component_monthly_matrices(signals, components)
    family_lanes = parse_family_lanes(args.family_lanes)

    first_month = datetime.fromisoformat(args.first_month).replace(tzinfo=timezone.utc)
    last_month = month_floor(int(signals["timestamp"].max().timestamp()))
    months = month_list(first_month, last_month)

    period_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    for i, test_month in enumerate(months):
        if i < args.train_months:
            continue
        train_index = [pd.Period(m, freq="M") for m in months[i - args.train_months:i]]
        train_matrices = {name: frame.loc[frame.index.isin(train_index)] for name, frame in matrices_all.items()}
        if train_matrices["unit"].empty:
            continue
        test_start = pd.Timestamp(test_month)
        test_end = pd.Timestamp(add_months(test_month, 1))
        for mode in args.modes:
            weights = fit_weights(
                train_matrices,
                mode,
                args.min_train_obs,
                args.corr_threshold,
                args.max_components,
                args.max_weight,
                family_lanes,
                args.util_lambda,
                args.util_gamma,
            )
            res = replay_month(signals, components, weights, test_start, test_end, args)
            period_rows.append({
                "mode": mode,
                "test_month": test_month.date(),
                **res,
                "weights": json.dumps({k: round(v, 4) for k, v in weights.items()}, sort_keys=True),
            })
            for comp, w in weights.items():
                weight_rows.append({"mode": mode, "test_month": test_month.date(), "component": comp, "weight": w})

    df = pd.DataFrame(period_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "portfolio_period_results.csv", period_rows)
    write_csv(args.output_dir / "portfolio_weights_by_month.csv", weight_rows)
    diagnostic_rows = component_diagnostics(matrices_all, components)
    write_csv(args.output_dir / "component_capacity_diagnostics.csv", diagnostic_rows)
    summary = summarize(df) if not df.empty else []
    write_csv(args.output_dir / "portfolio_summary.csv", summary)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "signals": str(args.signals),
        "external_signals": external_files,
        "pool_file": str(args.pool_file) if args.pool_file.exists() else None,
        "candidates": len(components),
        "family_lanes": family_lanes,
        "util_lambda": args.util_lambda,
        "util_gamma": args.util_gamma,
        "slippage_model": args.slippage_model,
        "slippage_coef": args.slippage_coef,
        "stake_fill_fraction": args.stake_fill_fraction,
        "max_stake": args.max_stake if math.isfinite(args.max_stake) else None,
        "test_months": int(df["test_month"].nunique()) if not df.empty else 0,
        "summary": summary,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "candidates": len(components),
                      "period_rows": len(period_rows), "modes": args.modes}, indent=2))


if __name__ == "__main__":
    main()
