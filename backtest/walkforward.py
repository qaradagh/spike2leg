"""Choose settings the way you would have to live: only from the past.

A single in-sample/out-of-sample split gives one out-of-sample number, and
with thousands of settings tried, one number is easy to get lucky on. This
cuts each series into equal folds by date and steps forward through them:
settings are picked on every fold before the current one, then scored on the
current one alone. Stringing those scores together gives a record where no
choice ever saw the data it was judged on.

The same walk is run against a shuffled control -- settings picked at random
instead of by past performance -- so the selection rule is measured against
picking blind rather than against zero.

    python -m backtest.walkforward --data <dir> --symbols xauusd us30 \\
        --timeframes M1 M5 M15 --folds 5 --out results
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd

from .config import Config
from .data import discover_datasets, load_ohlc
from .engine import detect_signals, simulate
from .optimise import GRID, TARGETS, config_for, retarget

MIN_TRADES_PER_FOLD = 25


def fold_of(times: pd.Series, edges: pd.DatetimeIndex) -> np.ndarray:
    return np.searchsorted(edges.to_numpy(), times.to_numpy(), side="right") - 1


def collect(datasets, frames: dict, folds: int) -> tuple[pd.DataFrame, list]:
    """Per-setting, per-fold trade count and total R, pooled over symbols.

    A setting is scored on gold and the Dow together, so it cannot be chosen
    for fitting one of them. Each symbol is cut into folds at its own dates,
    since their histories do not line up.
    """

    edges = {
        d.symbol: pd.DatetimeIndex(
            [
                frames[d.symbol].index[int(len(frames[d.symbol]) * i / folds)]
                for i in range(folds)
            ]
        )
        for d in datasets
    }

    keys = list(GRID)
    rows: list[dict] = []
    labels: list[tuple] = []

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
            row = {f"n{f}": 0 for f in range(folds)}
            row.update({f"R{f}": 0.0 for f in range(folds)})

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

                which = fold_of(trades["entry_time"], edges[dataset.symbol])
                r = trades["R"].to_numpy()

                for fold in range(folds):
                    picked = r[which == fold]
                    row[f"n{fold}"] += len(picked)
                    row[f"R{fold}"] += float(picked.sum())

            if any(row[f"n{f}"] < MIN_TRADES_PER_FOLD for f in range(folds)):
                continue

            rows.append(row)
            labels.append((*values, target))

    return pd.DataFrame(rows), labels


def walk(table: pd.DataFrame, folds: int, rng) -> dict:
    """Step forward, picking on the past and scoring on the next fold."""

    picked_R = picked_n = 0.0
    blind_R = blind_n = 0.0
    chosen: list[int] = []

    for fold in range(1, folds):
        past_R = table[[f"R{i}" for i in range(fold)]].sum(axis=1)
        past_n = table[[f"n{i}" for i in range(fold)]].sum(axis=1)

        expectancy = past_R / past_n.replace(0, np.nan)
        winner = int(expectancy.idxmax())
        chosen.append(winner)

        picked_R += table.at[winner, f"R{fold}"]
        picked_n += table.at[winner, f"n{fold}"]

        blind = int(rng.integers(len(table)))
        blind_R += table.at[blind, f"R{fold}"]
        blind_n += table.at[blind, f"n{fold}"]

    return {
        "picked_R": picked_R,
        "picked_n": int(picked_n),
        "picked_exp": picked_R / picked_n if picked_n else np.nan,
        "blind_R": blind_R,
        "blind_n": int(blind_n),
        "blind_exp": blind_R / blind_n if blind_n else np.nan,
        "chosen": chosen,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="results")
    parser.add_argument("--symbols", nargs="+", default=["xauusd", "us30"])
    parser.add_argument("--timeframes", nargs="+", default=["M1", "M5", "M15", "H1"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--blind-runs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    datasets = [
        d
        for d in discover_datasets(args.data, args.timeframes)
        if d.symbol in args.symbols
    ]

    summary: list[dict] = []

    for timeframe in args.timeframes:
        group = [d for d in datasets if d.timeframe == timeframe]

        if not group:
            continue

        frames = {d.symbol: load_ohlc(d.path) for d in group}
        table, labels = collect(group, frames, args.folds)

        if table.empty:
            print(f"\n{timeframe}: no setting traded enough in every fold")
            continue

        result = walk(table, args.folds, rng)

        blind = np.array(
            [
                walk(table, args.folds, rng)["blind_exp"]
                for _ in range(args.blind_runs)
            ]
        )

        keys = list(GRID) + ["tp_r"]
        picks = [dict(zip(keys, labels[i])) for i in result["chosen"]]

        print(f"\n{'=' * 78}")
        print(
            f"{timeframe}  --  {len(table):,} settings, {args.folds} folds, "
            f"{' + '.join(s.upper() for s in args.symbols)}"
        )
        print(
            f"  walk-forward   : {result['picked_exp']:+.4f} R over "
            f"{result['picked_n']} trades  (total {result['picked_R']:+.1f} R)"
        )
        print(
            f"  picking blind  : {np.mean(blind):+.4f} R "
            f"± {np.std(blind, ddof=1):.4f}  over {args.blind_runs} draws"
        )
        print(
            f"  selection gain : "
            f"{(result['picked_exp'] - np.mean(blind)) / np.std(blind, ddof=1):+.2f} sd"
            f"   ({(blind >= result['picked_exp']).mean() * 100:.0f}% of blind draws did better)"
        )
        print("  settings chosen at each step:")

        for fold, pick in enumerate(picks, start=1):
            print(
                f"    fold {fold + 1} <- spike {pick['spike_candle_size']}, "
                f"gap {pick['pgap_points']}, maxSL {pick['max_sl_distance_points']}, "
                f"ema {pick['ema_period'] or 'off'}, "
                f"trend {pick['max_opposite_moves'] if pick['max_opposite_moves'] >= 0 else 'off'}, "
                f"window {pick['time_window_minutes'] or 'off'}, "
                f"TP {pick['tp_r']}R"
            )

        summary.append(
            {
                "timeframe": timeframe,
                "settings": len(table),
                "walkforward_exp": result["picked_exp"],
                "walkforward_trades": result["picked_n"],
                "walkforward_R": result["picked_R"],
                "blind_exp": float(np.mean(blind)),
                "blind_sd": float(np.std(blind, ddof=1)),
                "gain_sd": float(
                    (result["picked_exp"] - np.mean(blind))
                    / np.std(blind, ddof=1)
                ),
            }
        )

    pd.DataFrame(summary).to_csv(
        os.path.join(args.out, "walkforward.csv"), index=False
    )
    print(f"\nwrote {os.path.join(args.out, 'walkforward.csv')}")


if __name__ == "__main__":
    main()
