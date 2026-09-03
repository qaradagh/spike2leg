"""Standalone backtest engine for the SP2L (spike-to-leg) strategy."""

from .config import Config, SYMBOL_SPECS, SymbolSpec
from .data import load_ohlc, discover_datasets, Dataset
from .engine import run_backtest, BacktestResult

__all__ = [
    "Config",
    "SYMBOL_SPECS",
    "SymbolSpec",
    "load_ohlc",
    "discover_datasets",
    "Dataset",
    "run_backtest",
    "BacktestResult",
]
