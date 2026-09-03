"""Search the strategy's settings, and measure how much of the winner is luck.

Trying thousands of settings on one price history will always turn up
something profitable, so this splits every series in two by date: the search
ranks settings on the earlier part only, and the later part is never used to
choose anything -- it only reports what the chosen settings then did.

Two diagnostics say whether a result is worth anything:

  IS/OOS correlation  across all settings tried. Near zero means the search is
                      fitting noise and the winner's out-of-sample number is a
                      coin flip.
  chance benchmark    the best out-of-sample result among all settings, which
                      is what picking the luckiest one would have looked like.

    python -m backtest.optimise --data <dir> --symbols xauusd us30 \\
        --timeframes M1 M5 M15 H1 --out results
"""

import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd

from .config import Config
from .data import discover_datasets, load_ohlc
from .engine import Signal, detect_signals, simulate

# Values swept for each setting. The first entry of each is what the bot
# ships with, so the shipped configuration is always inside the grid.
GRID = {
    "spike_candle_size": [1.5, 1.0, 2.0, 2.5, 3.0],
    "pgap_points": [100, 0, 50, 200, 400],
    "max_sl_distance_points": [1000, 300, 500, 2000, 5000],
    "ema_period": [60, 0, 30, 120, 200],  # 0 turns the filter off
    "max_opposite_moves": [1, -1, 2],  # -1 turns the filter off
    "time_window_minutes": [0, 30, 60],  # 0 turns the filter off
}

TARGETS = [1.0, 2.0, 3.0, 5.0]

IN_SAMPLE_FRACTION = 0.65
MIN_IN_SAMPLE_TRADES = 25
MIN_OUT_OF_SAMPLE_TRADES = 15


def retarget(signals: list[Signal], tp_r: float) -> list[Signal]:
    """Re-price each signal's target without re-running detection.

    Which signals exist does not depend on the target: under the fill models
    used here quote and fill are one price, so the "stop or target already on
    the wrong side of the fill" rejection can never trigger. Only the target
    level itself moves, so it can be recomputed from the stop distance.
    """

    return [
        Signal(
            position=s.position,
            signal_pos=s.signal_pos,
            entry_pos=s.entry_pos,
            entry=s.entry,
            quoted_entry=s.quoted_entry,
            stop=s.stop,
            target=s.quoted_entry
            + s.position * tp_r * abs(s.quoted_entry - s.stop),
            risk=s.risk,
            spike_body=s.spike_body,
        )
        for s in signals
    ]


def stats(r: pd.Series) -> tuple[int, float, float]:
    """Trade count, R per trade, and its t-statistic."""

    n = len(r)

    if n < 2:
        return n, np.nan, np.nan

    spread = r.std(ddof=1)

    return (
        n,
        float(r.mean()),
        float(r.mean() / (spread / np.sqrt(n))) if spread > 0 else np.nan,
    )


def search_one(df: pd.DataFrame, spec, symbol: str, timeframe: str) -> list[dict]:
    split_at = df.index[int(len(df) * IN_SAMPLE_FRACTION)]

    base = Config(
        entry_mode="fill_fraction",
        fill_fraction=0.2,
        spread_points=spec.typical_spread_points,
    )

    rows: list[dict] = []
    keys = list(GRID)

    for values in itertools.product(*(GRID[k] for k in keys)):
        settings = dict(zip(keys, values))

        cfg = base.with_(
            spike_candle_size=settings["spike_candle_size"],
            pgap_points=settings["pgap_points"],
            max_sl_distance_points=settings["max_sl_distance_points"],
            use_ema_filter=settings["ema_period"] > 0,
            ema_period=max(settings["ema_period"], 2),
            use_trend_filter=settings["max_opposite_moves"] >= 0,
            max_opposite_moves=max(settings["max_opposite_moves"], 0),
            use_time_filter=settings["time_window_minutes"] > 0,
            time_window_minutes=max(settings["time_window_minutes"], 1),
        )

        signals, _ = detect_signals(df, cfg, spec)

        if len(signals) < MIN_IN_SAMPLE_TRADES:
            continue

        for target in TARGETS:
            trades = simulate(
                df, retarget(signals, target), cfg.with_(tp_r=target), spec
            )

            if trades.empty:
                continue

            early = trades[trades["entry_time"] < split_at]["R"]
            late = trades[trades["entry_time"] >= split_at]["R"]

            n_is, exp_is, t_is = stats(early)
            n_oos, exp_oos, t_oos = stats(late)

            if n_is < MIN_IN_SAMPLE_TRADES or n_oos < MIN_OUT_OF_SAMPLE_TRADES:
                continue

            rows.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    **settings,
                    "tp_r": target,
                    "n_is": n_is,
                    "exp_is": exp_is,
                    "t_is": t_is,
                    "n_oos": n_oos,
                    "exp_oos": exp_oos,
                    "t_oos": t_oos,
                    "total_R": float(trades["R"].sum()),
                }
            )

    return rows


def report(results: pd.DataFrame, symbol: str, timeframe: str) -> None:
    subset = results[
        (results.symbol == symbol) & (results.timeframe == timeframe)
    ]

    print(f"\n{'=' * 78}\n{symbol.upper()} {timeframe}  --  {len(subset)} settings survived the trade minimums")

    if subset.empty:
        return

    # Spearman without pulling in scipy: Pearson over the ranks.
    correlation = subset["exp_is"].rank().corr(subset["exp_oos"].rank())

    print(f"  IS/OOS rank correlation across all settings : {correlation:+.3f}")
    print(
        f"  best OOS of any setting (the luck ceiling)  : "
        f"{subset['exp_oos'].max():+.3f} R"
    )
    print(
        f"  median OOS across all settings              : "
        f"{subset['exp_oos'].median():+.3f} R"
    )

    best = subset.sort_values("exp_is", ascending=False).head(8)

    print("\n  ranked by in-sample only; the OOS columns are what happened next")
    print(
        "  spike  gap   maxSL  ema  trend  window   TP |"
        "   n_is   exp_is   t_is |  n_oos  exp_oos  t_oos"
    )

    for _, row in best.iterrows():
        ema = "off" if row.ema_period == 0 else int(row.ema_period)
        trend = "off" if row.max_opposite_moves < 0 else int(row.max_opposite_moves)
        window = "off" if row.time_window_minutes == 0 else f"{int(row.time_window_minutes)}m"

        print(
            f"  {row.spike_candle_size:5.2f} {int(row.pgap_points):4d} "
            f"{int(row.max_sl_distance_points):6d} {str(ema):>4s} "
            f"{str(trend):>6s} {window:>7s} {row.tp_r:4.1f} |"
            f" {int(row.n_is):5d} {row.exp_is:+8.3f} {row.t_is:6.2f} |"
            f" {int(row.n_oos):5d} {row.exp_oos:+8.3f} {row.t_oos:6.2f}"
        )

    chosen = best.iloc[0]
    beat = (subset["exp_oos"] > chosen["exp_oos"]).mean() * 100

    print(
        f"\n  the in-sample winner's OOS result is beaten by {beat:.0f}% of "
        f"all settings tried"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="results")
    parser.add_argument("--symbols", nargs="+", default=["xauusd", "us30"])
    parser.add_argument("--timeframes", nargs="+", default=["M1", "M5", "M15", "H1"])
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    combinations = int(np.prod([len(v) for v in GRID.values()])) * len(TARGETS)
    print(
        f"grid: {combinations:,} settings per symbol/timeframe "
        f"(in-sample {IN_SAMPLE_FRACTION:.0%} / out-of-sample "
        f"{1 - IN_SAMPLE_FRACTION:.0%})"
    )

    rows: list[dict] = []

    for dataset in discover_datasets(args.data, args.timeframes):
        if dataset.symbol not in args.symbols:
            continue

        started = time.time()
        df = load_ohlc(dataset.path)
        found = search_one(df, dataset.spec, dataset.symbol, dataset.timeframe)
        rows.extend(found)

        print(
            f"  {dataset.symbol} {dataset.timeframe}: {len(found):,} kept "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(args.out, "optimise.csv"), index=False)

    for symbol in args.symbols:
        for timeframe in args.timeframes:
            report(results, symbol, timeframe)

    print(f"\nwrote {os.path.join(args.out, 'optimise.csv')}")


if __name__ == "__main__":
    main()
