"""Run the SP2L strategy over every symbol/timeframe export and report.

    python -m backtest.run_grid --data <dir> --out results

Each named configuration is applied to every dataset; the script writes a
per-run summary table, the individual trade lists, and a markdown report.
"""

import argparse
import json
import os

import pandas as pd

from .config import Config
from .data import discover_datasets, load_ohlc, TIMEFRAME_ORDER
from .engine import run_backtest


def build_configs() -> list[Config]:
    """The runs the report compares.

    ``upstream`` is the reference: the settings shipped in
    ``SP2L_Advanced_Bot.py``, with every point-denominated threshold scaled by
    each symbol's contract point. The rest change one assumption at a time so
    the effect of each is readable on its own.

    A ``spread_points`` of -1 is a placeholder replaced per symbol with that
    symbol's typical spread.
    """

    upstream = Config(
        label="upstream",
        notes=(
            "Upstream settings verbatim. Fills at the extreme the entry bar "
            "eventually reaches, and charges nothing to trade."
        ),
    )

    # The bot fires on the first poll after price crosses the trigger level,
    # so it captures only the start of the move into the bar's extreme.
    live = upstream.with_(entry_mode="fill_fraction", fill_fraction=0.2)

    return [
        upstream,
        live.with_(
            label="live_fill",
            notes=(
                "Same signals, filled a fifth of the way from the trigger "
                "level into the bar's extreme, which is the kind of fill a "
                "ten-second poll can actually reach."
            ),
        ),
        live.with_(
            label="live_fill_spread",
            spread_points=-1.0,
            notes="The live fill with each symbol's typical spread charged.",
        ),
        upstream.with_(
            label="trigger_fill_spread",
            entry_mode="fill_fraction",
            fill_fraction=0.0,
            spread_points=-1.0,
            notes=(
                "Filled exactly at the trigger level, with spread. The "
                "conservative end of what the bot can reach."
            ),
        ),
        upstream.with_(
            label="bar_close_spread",
            entry_mode="market_close",
            spread_points=-1.0,
            notes=(
                "As if the bot only ever acted on closed bars: filled at the "
                "close while the target stays priced off the bar extreme. "
                "The pessimistic bound."
            ),
        ),
        upstream.with_(
            label="atr_upstream",
            threshold_mode="atr",
            notes=(
                "Upstream fills, but the gap and stop-distance thresholds "
                "become ATR multiples so they mean the same thing on every "
                "symbol and timeframe."
            ),
        ),
        live.with_(
            label="atr_live_spread",
            threshold_mode="atr",
            spread_points=-1.0,
            notes="ATR thresholds with the live fill and spread.",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        required=True,
        help="Directory of <symbol>/<export>.csv files",
    )
    parser.add_argument("--out", default="results", help="Output directory")
    parser.add_argument(
        "--save-trades",
        action="store_true",
        help="Write the full trade list of every run",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    datasets = discover_datasets(args.data)
    configs = build_configs()

    rows: list[dict] = []

    for dataset in datasets:
        df = load_ohlc(dataset.path)

        for cfg in configs:
            if cfg.spread_points < 0:
                cfg = cfg.with_(
                    spread_points=dataset.spec.typical_spread_points
                )

            result = run_backtest(
                df, cfg, dataset.spec, dataset.symbol, dataset.timeframe
            )

            rows.append(
                {
                    "config": cfg.label,
                    "symbol": dataset.symbol,
                    "timeframe": dataset.timeframe,
                    **result.summary,
                }
            )

            if args.save_trades and not result.trades.empty:
                trades_dir = os.path.join(args.out, "trades", cfg.label)
                os.makedirs(trades_dir, exist_ok=True)
                result.trades.to_csv(
                    os.path.join(
                        trades_dir,
                        f"{dataset.symbol}_{dataset.timeframe}.csv",
                    ),
                    index=False,
                )

        print(f"done {dataset.symbol} {dataset.timeframe}", flush=True)

    summary = pd.DataFrame(rows)
    summary["tf_order"] = summary["timeframe"].map(TIMEFRAME_ORDER.index)
    summary = summary.sort_values(
        ["config", "symbol", "tf_order"]
    ).drop(columns="tf_order")

    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)

    with open(os.path.join(args.out, "configs.json"), "w") as handle:
        json.dump(
            {c.label: {**c.__dict__} for c in configs}, handle, indent=2
        )

    print(f"\nwrote {os.path.join(args.out, 'summary.csv')}")


if __name__ == "__main__":
    main()
