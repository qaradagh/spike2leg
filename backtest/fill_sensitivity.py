"""How much of the strategy's result comes from the assumed fill price.

The upstream backtest fills at the extreme the entry bar eventually reaches;
the live bot fills within one poll interval of price crossing the previous
bar's extreme. This sweeps the whole range between those two assumptions and
reports where the edge disappears.

    python -m backtest.fill_sensitivity --data <dir> --out results
"""

import argparse
import os

import numpy as np
import pandas as pd

from .config import Config
from .data import discover_datasets, load_ohlc
from .engine import run_backtest

FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="results")
    parser.add_argument(
        "--spread",
        action="store_true",
        help="Also charge each symbol's typical spread",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    rows: list[dict] = []

    for dataset in discover_datasets(args.data):
        df = load_ohlc(dataset.path)

        for fraction in FRACTIONS:
            cfg = Config(
                entry_mode="fill_fraction",
                fill_fraction=fraction,
                spread_points=(
                    dataset.spec.typical_spread_points if args.spread else 0.0
                ),
                label=f"f={fraction}",
            )

            result = run_backtest(
                df, cfg, dataset.spec, dataset.symbol, dataset.timeframe
            )

            rows.append(
                {
                    "fill_fraction": fraction,
                    "symbol": dataset.symbol,
                    "timeframe": dataset.timeframe,
                    **result.summary,
                }
            )

        print(f"done {dataset.symbol} {dataset.timeframe}", flush=True)

    detail = pd.DataFrame(rows)

    suffix = "_spread" if args.spread else ""
    detail.to_csv(
        os.path.join(args.out, f"fill_sensitivity{suffix}.csv"), index=False
    )

    summary = (
        detail.groupby("fill_fraction")
        .apply(
            lambda g: pd.Series(
                {
                    "trades": int(g["trades"].sum()),
                    "win_rate": (
                        100 * g["wins"].sum() / g["trades"].sum()
                        if g["trades"].sum()
                        else np.nan
                    ),
                    "total_R": g["total_R"].sum(),
                    "expectancy_R": (
                        g["total_R"].sum() / g["trades"].sum()
                        if g["trades"].sum()
                        else np.nan
                    ),
                    "profitable_datasets": int((g["total_R"] > 0).sum()),
                    "datasets": len(g),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    print()
    print(summary.round(3).to_string(index=False))

    summary.to_csv(
        os.path.join(args.out, f"fill_sensitivity_summary{suffix}.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
