#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

from polymarket_backtest.analysis import summarize_by_category, summarize_trades, trades_to_dataframe
from polymarket_backtest.backtest import load_config_from_env, run_backtest


def main() -> None:
    config = load_config_from_env(os.environ)
    results = run_backtest(config)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trades = results["trades"]
    counters = results["counters"]

    summary = summarize_trades(trades, counters)
    summary["config"] = results["config"]

    trades_df = trades_to_dataframe(trades)
    by_category = summarize_by_category(trades_df)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "data_issues.json").write_text(json.dumps(results["issues"], indent=2))

    if not trades_df.empty:
        trades_df.sort_values(by="pnl", ascending=False).to_csv(
            output_dir / "trades.csv", index=False
        )
        trades_df.nlargest(10, "pnl").to_csv(output_dir / "top_winners.csv", index=False)
        trades_df.nsmallest(10, "pnl").to_csv(output_dir / "top_losers.csv", index=False)

    if not by_category.empty:
        by_category.to_csv(output_dir / "by_category.csv", index=False)


if __name__ == "__main__":
    main()
