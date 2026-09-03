"""Does the setup pay better at a target beyond 1R?

The bot holds one position per symbol at a time, so a wider target does not
only change the payoff of each trade -- it also blocks signals while the
trade is open. This sweeps the target and reports both, so a higher total R
earned from far fewer trades is not mistaken for a better strategy.

    python -m backtest.tp_sweep --data <dir> --out results
"""

import argparse
import os

import numpy as np
import pandas as pd

from .config import Config
from .data import discover_datasets, load_ohlc, TIMEFRAME_ORDER
from .engine import run_backtest

TARGETS = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]

# Each regime is (label, entry fill fraction, charge spread).
REGIMES = [
    ("live", 0.2, True),
    ("upstream", 1.0, False),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    rows: list[dict] = []

    for dataset in discover_datasets(args.data):
        df = load_ohlc(dataset.path)

        for regime, fraction, charge_spread in REGIMES:
            for target in TARGETS:
                cfg = Config(
                    entry_mode="fill_fraction",
                    fill_fraction=fraction,
                    tp_r=target,
                    spread_points=(
                        dataset.spec.typical_spread_points
                        if charge_spread
                        else 0.0
                    ),
                    label=f"{regime}_tp{target}",
                )

                result = run_backtest(
                    df, cfg, dataset.spec, dataset.symbol, dataset.timeframe
                )

                rows.append(
                    {
                        "regime": regime,
                        "tp_r": target,
                        "symbol": dataset.symbol,
                        "timeframe": dataset.timeframe,
                        **result.summary,
                    }
                )

        print(f"done {dataset.symbol} {dataset.timeframe}", flush=True)

    detail = pd.DataFrame(rows)
    detail["tf_order"] = detail["timeframe"].map(TIMEFRAME_ORDER.index)
    detail = detail.sort_values(
        ["regime", "tp_r", "symbol", "tf_order"]
    ).drop(columns="tf_order")

    detail.to_csv(os.path.join(args.out, "tp_sweep.csv"), index=False)

    summary = []

    for (regime, target), group in detail.groupby(["regime", "tp_r"]):
        trades = int(group["trades"].sum())
        bars = int(group["bars"].sum())

        summary.append(
            {
                "regime": regime,
                "tp_r": target,
                "trades": trades,
                "win_rate": (
                    100 * group["wins"].sum() / trades if trades else np.nan
                ),
                # A target of nR needs this win rate just to break even.
                "breakeven_win_rate": 100 / (1 + target),
                "total_R": group["total_R"].sum(),
                "expectancy_R": (
                    group["total_R"].sum() / trades if trades else np.nan
                ),
                # Opportunity-adjusted: R earned per 10k bars of market time.
                "R_per_10k_bars": group["total_R"].sum() / bars * 10_000,
                "avg_bars_held": float(
                    (group["avg_bars_held"] * group["trades"]).sum() / trades
                )
                if trades
                else np.nan,
                "profitable_datasets": int((group["total_R"] > 0).sum()),
                "unresolved": int(group["unresolved"].sum()),
            }
        )

    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(args.out, "tp_sweep_summary.csv"), index=False)

    for regime in summary["regime"].unique():
        print()
        print(f"=== {regime} ===")
        print(
            summary[summary.regime == regime]
            .drop(columns="regime")
            .round(3)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
