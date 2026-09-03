"""Does the setup's timing and direction earn the result, or does the market?

A wide target on a trending instrument can pay off from almost any entry, so a
positive result at a 3R target is not by itself evidence that the spike pattern
works. This runs the same exit rules against two null models:

  random timing    same trade count and same direction mix, entered at
                   randomly chosen bars. Stop distances are redrawn from the
                   real signals' stop-to-ATR ratios and rescaled to the
                   volatility at the random bar, so a stop is never mismatched
                   to its moment -- a raw stop distance dropped into a busier
                   stretch would be stopped out by the bar it opened in.
  random direction same entry bars and stop distances as the real trades,
                   with each direction decided by a coin flip

If the real strategy cannot beat these, the pattern is not what produces the
result.

    python -m backtest.entry_control --data <dir> --tp 3.0 --runs 30
"""

import argparse

import numpy as np
import pandas as pd

from .config import Config
from .data import discover_datasets, load_ohlc
from .engine import Signal, atr, detect_signals, simulate


def synthetic(pos, direction, price, risk, tp_r) -> Signal:
    return Signal(
        position=direction,
        signal_pos=pos,
        entry_pos=pos,
        entry=price,
        quoted_entry=price,
        stop=price - direction * risk,
        target=price + direction * tp_r * risk,
        risk=risk,
        spike_body=np.nan,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tp", type=float, default=3.0)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    real_total = 0.0
    real_trades = 0
    timing = np.zeros(args.runs)
    flipped = np.zeros(args.runs)

    for dataset in discover_datasets(args.data):
        df = load_ohlc(dataset.path)

        cfg = Config(
            entry_mode="fill_fraction",
            fill_fraction=0.2,
            tp_r=args.tp,
            spread_points=dataset.spec.typical_spread_points,
        )

        signals, _ = detect_signals(df, cfg, dataset.spec)

        if not signals:
            continue

        trades = simulate(df, signals, cfg, dataset.spec)
        real_total += trades["R"].sum()
        real_trades += len(trades)

        close = df["close"].to_numpy(dtype=float)
        n = len(df)

        bar_atr = atr(
            df["high"].to_numpy(dtype=float),
            df["low"].to_numpy(dtype=float),
            close,
            cfg.atr_period,
        )

        directions = np.array([s.position for s in signals])
        risks = np.array([s.risk for s in signals])
        positions = np.array([s.entry_pos for s in signals])

        # How wide the real stops are relative to the volatility they were
        # set in. Redrawing this ratio, rather than the raw distance, keeps
        # the null model's stops as reachable as the strategy's.
        risk_in_atr = risks / bar_atr[positions]
        risk_in_atr = risk_in_atr[np.isfinite(risk_in_atr)]

        for run in range(args.runs):
            # Random timing: the same trades, offered at random moments.
            where = np.sort(rng.choice(np.arange(4, n - 1), len(signals), replace=False))
            order = rng.permutation(len(signals))

            ratios = rng.choice(risk_in_atr, len(signals))

            drawn = [
                synthetic(
                    int(pos), int(directions[j]), float(close[pos]),
                    float(ratio * bar_atr[pos]), args.tp,
                )
                for pos, j, ratio in zip(where, order, ratios)
            ]
            timing[run] += simulate(df, drawn, cfg, dataset.spec)["R"].sum()

            # Random direction: the real moments, with the side coin-flipped.
            coins = rng.choice([-1, 1], len(signals))

            drawn = [
                synthetic(
                    int(pos), int(coin), float(close[pos]), float(risk), args.tp
                )
                for pos, coin, risk in zip(positions, coins, risks)
            ]
            flipped[run] += simulate(df, drawn, cfg, dataset.spec)["R"].sum()

    print(f"target {args.tp}R · live fill · spread charged · {args.runs} runs\n")
    print(f"  strategy         total R {real_total:+9.1f}   ({real_trades} trades)")

    for name, sample in (("random timing", timing), ("random direction", flipped)):
        z = (real_total - sample.mean()) / sample.std(ddof=1)
        beat = int((sample >= real_total).sum())
        print(
            f"  {name:16s} total R {sample.mean():+9.1f} ± {sample.std(ddof=1):.1f}"
            f"   strategy is {z:.1f} sd above; {beat}/{args.runs} runs beat it"
        )


if __name__ == "__main__":
    main()
