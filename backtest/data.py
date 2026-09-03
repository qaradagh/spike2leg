"""Loading OHLC data exported from TradingView.

The live bot pulls candles through ``Meta.GetRates``; offline we read the CSV
exports instead. Both paths hand the engine the same four columns indexed by
timestamp, so the strategy code does not care which one was used.
"""

import os
import re
from dataclasses import dataclass

import pandas as pd

from .config import SYMBOL_SPECS, SymbolSpec

# TradingView writes the timeframe into the file name, e.g.
# "OANDA_EURUSD, 240_aa433.csv".
_FILENAME_RE = re.compile(r"^(?P<feed>[^,]+),\s*(?P<tf>\w+)_")

TIMEFRAME_LABELS = {
    "1": "M1",
    "5": "M5",
    "15": "M15",
    "30": "M30",
    "60": "H1",
    "240": "H4",
    "1D": "D1",
    "1W": "W1",
}

TIMEFRAME_ORDER = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]

REQUIRED_COLUMNS = ["open", "high", "low", "close"]


@dataclass(frozen=True)
class Dataset:
    symbol: str
    timeframe: str
    feed: str
    path: str
    spec: SymbolSpec


def discover_datasets(root: str) -> list[Dataset]:
    """Find every ``<root>/<symbol>/<feed>, <tf>_<hash>.csv`` export."""

    datasets: list[Dataset] = []

    for symbol in sorted(os.listdir(root)):
        symbol_dir = os.path.join(root, symbol)

        if not os.path.isdir(symbol_dir):
            continue

        spec = SYMBOL_SPECS.get(symbol.lower())

        if spec is None:
            raise KeyError(
                f"No SymbolSpec declared for '{symbol}'. Add one to "
                f"backtest/config.py before backtesting it."
            )

        for name in sorted(os.listdir(symbol_dir)):
            if not name.lower().endswith(".csv"):
                continue

            match = _FILENAME_RE.match(name)

            if match is None:
                raise ValueError(f"Unrecognised export file name: {name}")

            timeframe = TIMEFRAME_LABELS.get(match.group("tf"))

            if timeframe is None:
                raise ValueError(
                    f"Unknown timeframe '{match.group('tf')}' in {name}"
                )

            datasets.append(
                Dataset(
                    symbol=symbol.lower(),
                    timeframe=timeframe,
                    feed=match.group("feed").strip(),
                    path=os.path.join(symbol_dir, name),
                    spec=spec,
                )
            )

    datasets.sort(
        key=lambda d: (d.symbol, TIMEFRAME_ORDER.index(d.timeframe))
    )

    return datasets


def load_ohlc(path: str, drop_flat: bool = True) -> pd.DataFrame:
    """Read one export into a sorted, numeric, timestamp-indexed frame.

    ``drop_flat`` removes bars where open == high == low == close. Those are
    backfilled placeholders in the long daily histories (XAUUSD/D1 carries
    thousands of them before real intraday data begins) and they would
    otherwise register as zero-body candles in the spike comparison.
    """

    df = pd.read_csv(path)
    df.columns = [str(c).lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    time_column = next(
        (c for c in ("time", "datetime", "date") if c in df.columns), None
    )

    if time_column is None:
        raise ValueError(f"{path} has no recognisable time column")

    df[time_column] = pd.to_datetime(
        df[time_column], format="mixed", utc=True
    )
    df = df.set_index(time_column).sort_index()

    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLUMNS)

    if drop_flat:
        flat = (
            (df["open"] == df["high"])
            & (df["high"] == df["low"])
            & (df["low"] == df["close"])
        )
        df = df[~flat]

    df = df[~df.index.duplicated(keep="first")]

    return df[REQUIRED_COLUMNS].copy()
