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

# Picking from tens of thousands of settings on a couple of dozen trades
# selects for small samples, not for good settings: the highest in-sample
# expectancy in the grid will be whichever rare setting happened to catch a
# good run. These floors, and ranking by t rather than by expectancy, are
# what keep the search from doing that.
MIN_IN_SAMPLE_TRADES = 120
MIN_OUT_OF_SAMPLE_TRADES = 60


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


def config_for(base: Config, settings: dict) -> Config:
    return base.with_(
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


def search(datasets, frames: dict, timeframe: str) -> list[dict]:
    """Score every setting on all the given symbols at once.

    One setting has to earn its keep on gold and the Dow together, which is
    what the strategy claims anyway and halves the room to fit either one's
    noise. Each symbol is split at its own date, since their histories start
    and end in different places.
    """

    splits = {
        d.symbol: frames[d.symbol].index[
            int(len(frames[d.symbol]) * IN_SAMPLE_FRACTION)
        ]
        for d in datasets
    }

    rows: list[dict] = []
    keys = list(GRID)

    for values in itertools.product(*(GRID[k] for k in keys)):
        settings = dict(zip(keys, values))

        found = {}

        for dataset in datasets:
            base = Config(
                entry_mode="fill_fraction",
                fill_fraction=0.2,
                spread_points=dataset.spec.typical_spread_points,
            )
            cfg = config_for(base, settings)
            signals, _ = detect_signals(frames[dataset.symbol], cfg, dataset.spec)
            found[dataset.symbol] = (cfg, signals)

        for target in TARGETS:
            early: list[np.ndarray] = []
            late: list[np.ndarray] = []
            per_symbol: dict[str, float] = {}

            for dataset in datasets:
                cfg, signals = found[dataset.symbol]

                if not signals:
                    continue

                trades = simulate(
                    frames[dataset.symbol],
                    retarget(signals, target),
                    cfg.with_(tp_r=target),
                    dataset.spec,
                )

                if trades.empty:
                    continue

                is_early = trades["entry_time"] < splits[dataset.symbol]
                early.append(trades["R"].to_numpy()[is_early.to_numpy()])
                late.append(trades["R"].to_numpy()[~is_early.to_numpy()])
                per_symbol[dataset.symbol] = float(trades["R"].mean())

            if not early or not late:
                continue

            a = pd.Series(np.concatenate(early))
            b = pd.Series(np.concatenate(late))

            n_is, exp_is, t_is = stats(a)
            n_oos, exp_oos, t_oos = stats(b)

            if n_is < MIN_IN_SAMPLE_TRADES or n_oos < MIN_OUT_OF_SAMPLE_TRADES:
                continue

            # Every symbol has to pull its weight, not be carried.
            worst = min(per_symbol.values()) if per_symbol else np.nan

            rows.append(
                {
                    "timeframe": timeframe,
                    **settings,
                    "tp_r": target,
                    "n_is": n_is,
                    "exp_is": exp_is,
                    "t_is": t_is,
                    "n_oos": n_oos,
                    "exp_oos": exp_oos,
                    "t_oos": t_oos,
                    "worst_symbol_exp": worst,
                    **{f"exp_{k}": v for k, v in per_symbol.items()},
                }
            )

    return rows


def report(results: pd.DataFrame, timeframe: str, rank_by: str) -> None:
    subset = results[results.timeframe == timeframe]

    print(f"\n{'=' * 92}")
    print(
        f"{timeframe}  --  {len(subset):,} settings cleared "
        f"{MIN_IN_SAMPLE_TRADES}/{MIN_OUT_OF_SAMPLE_TRADES} trades in and out of sample"
    )

    if subset.empty:
        print("  nothing cleared the trade floors")
        return

    # Spearman without pulling in scipy: Pearson over the ranks.
    correlation = subset["exp_is"].rank().corr(subset["exp_oos"].rank())

    # With this many settings tried, an in-sample t only means something if
    # it clears the multiple-comparison bar, not the usual 2.
    print(f"  IS/OOS rank correlation over every setting : {correlation:+.3f}")
    print(f"  median OOS across all settings             : {subset['exp_oos'].median():+.3f} R")
    print(f"  best OOS of any setting (the luck ceiling) : {subset['exp_oos'].max():+.3f} R")
    print(f"  settings with OOS above zero               : {(subset['exp_oos'] > 0).mean() * 100:.0f}%")

    best = subset.sort_values(rank_by, ascending=False).head(8)

    print(f"\n  ranked by {rank_by} in-sample only; OOS columns are what happened next")
    print(
        "  spike  gap   maxSL  ema  trend  window   TP |"
        "   n_is   exp_is   t_is |  n_oos  exp_oos  t_oos | worst sym"
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
            f" {int(row.n_oos):5d} {row.exp_oos:+8.3f} {row.t_oos:6.2f} |"
            f" {row.worst_symbol_exp:+9.3f}"
        )

    chosen = best.iloc[0]
    beaten = (subset["exp_oos"] > chosen["exp_oos"]).mean() * 100

    print(
        f"\n  the chosen setting's OOS result is beaten by {beaten:.0f}% of all "
        f"settings tried"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="results")
    parser.add_argument("--symbols", nargs="+", default=["xauusd", "us30"])
    parser.add_argument("--timeframes", nargs="+", default=["M1", "M5", "M15", "H1"])
    parser.add_argument(
        "--rank-by",
        default="t_is",
        choices=["t_is", "exp_is"],
        help="t_is ranks by in-sample t, which does not reward tiny samples",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    combinations = int(np.prod([len(v) for v in GRID.values()])) * len(TARGETS)
    print(
        f"grid: {combinations:,} settings, scored on "
        f"{' + '.join(s.upper() for s in args.symbols)} together "
        f"(in-sample {IN_SAMPLE_FRACTION:.0%} / out-of-sample "
        f"{1 - IN_SAMPLE_FRACTION:.0%})"
    )

    rows: list[dict] = []
    datasets = [
        d
        for d in discover_datasets(args.data, args.timeframes)
        if d.symbol in args.symbols
    ]

    for timeframe in args.timeframes:
        group = [d for d in datasets if d.timeframe == timeframe]

        if not group:
            continue

        started = time.time()
        frames = {d.symbol: load_ohlc(d.path) for d in group}
        found = search(group, frames, timeframe)
        rows.extend(found)

        print(
            f"  {timeframe}: {len(found):,} settings kept "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(args.out, "optimise.csv"), index=False)

    for timeframe in args.timeframes:
        report(results, timeframe, args.rank_by)

    print(f"\nwrote {os.path.join(args.out, 'optimise.csv')}")


if __name__ == "__main__":
    main()
