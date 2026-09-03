"""SP2L spike-to-leg strategy: signal detection, simulation and statistics.

This is an offline port of ``SP2L2_Advanced_Backtest.ipynb``. The trading rules
are unchanged; the MetaTrader 5 calls are replaced by CSV input and the row-wise
pandas lookups by NumPy arrays so a full symbol/timeframe grid runs in seconds.

The setup, reading candles right-to-left from the signal bar ``c``:

    c-3  "before"  bar 3
    c-2  "spike"   bar 2 -- the large body; its extreme becomes the stop
    c-1  "after"   bar 1
    c              signal bar

A long setup needs three consecutive rising bullish candles, a gap between
bar 1's low and bar 3's high, and a bar-2 body at least ``spike_candle_size``
times both neighbouring bodies. The entry is then the first pullback that
breaks the previous bar's low; the target is ``tp_r`` times the stop distance.
"""

from dataclasses import dataclass, field
from datetime import time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import Config, SymbolSpec


# ============================================================
# INDICATORS
# ============================================================

def _shift(values: np.ndarray, lag: int) -> np.ndarray:
    out = np.full_like(values, np.nan)
    out[lag:] = values[:-lag]
    return out


def ema(close: np.ndarray, period: int) -> np.ndarray:
    return (
        pd.Series(close).ewm(span=period, adjust=False).mean().to_numpy()
    )


def true_range(
    high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    previous_close = _shift(close, 1)

    return np.nanmax(
        np.vstack(
            [
                high - low,
                np.abs(high - previous_close),
                np.abs(low - previous_close),
            ]
        ),
        axis=0,
    )


def atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """Wilder-smoothed average true range."""

    return (
        pd.Series(true_range(high, low, close))
        .ewm(alpha=1 / period, adjust=False)
        .mean()
        .to_numpy()
    )


def adx(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """Wilder-smoothed ADX, matching the notebook's implementation."""

    up_move = np.diff(high, prepend=np.nan)
    down_move = -np.diff(low, prepend=np.nan)

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    smoothed_atr = atr(high, low, close, period)

    def wilder(values: np.ndarray) -> np.ndarray:
        return (
            pd.Series(values)
            .ewm(alpha=1 / period, adjust=False)
            .mean()
            .to_numpy()
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100 * wilder(plus_dm) / smoothed_atr
        minus_di = 100 * wilder(minus_dm) / smoothed_atr

        denominator = plus_di + minus_di
        dx = np.where(
            denominator > 0,
            100 * np.abs(plus_di - minus_di) / denominator,
            np.nan,
        )

    return wilder(dx)


# ============================================================
# SETUP DETECTION
# ============================================================

def _session_mask(index: pd.DatetimeIndex, cfg: Config) -> np.ndarray:
    """True where the bar opens inside the configured session window."""

    if not cfg.use_session_filter:
        return np.ones(len(index), dtype=bool)

    stamps = index

    if stamps.tz is None:
        # Upstream treats naive timestamps as the broker's UTC+3 server time.
        stamps = stamps.tz_localize("Etc/GMT-3")

    local = stamps.tz_convert(ZoneInfo(cfg.session_timezone))

    start = dtime(cfg.session_start_hour, 0)
    end = dtime(cfg.session_end_hour, 0)

    local_times = np.array([t.time() for t in local])

    if start <= end:
        return (local_times >= start) & (local_times < end)

    # Window wraps past midnight.
    return (local_times >= start) | (local_times < end)


def _setup_masks(
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    gap_price: np.ndarray,
    spike_multiple: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised long/short setup conditions evaluated at the signal bar."""

    o1, h1, l1, c1 = (_shift(x, 1) for x in (o, h, l, c))
    o2, h2, l2, c2 = (_shift(x, 2) for x in (o, h, l, c))
    o3, h3, l3, c3 = (_shift(x, 3) for x in (o, h, l, c))

    with np.errstate(invalid="ignore"):
        buy_body_spike = c2 - o2
        buy_body_before = c3 - o3
        buy_body_after = c1 - o1

        buy = (
            (c1 > c2)
            & (o1 > o2)
            & (c2 > c3)
            & (o2 > o3)
            & (c1 > o1)
            & (c2 > o2)
            & (c3 > o3)
            & (l1 > h3 + gap_price)
            & (buy_body_spike > spike_multiple * buy_body_before)
            & (buy_body_spike > spike_multiple * buy_body_after)
        )

        sell_body_spike = o2 - c2
        sell_body_before = o3 - c3
        sell_body_after = o1 - c1

        sell = (
            (c1 < c2)
            & (o1 < o2)
            & (c2 < c3)
            & (o2 < o3)
            & (c1 < o1)
            & (c2 < o2)
            & (c3 < o3)
            & (h1 < l3 - gap_price)
            & (sell_body_spike > spike_multiple * sell_body_before)
            & (sell_body_spike > spike_multiple * sell_body_after)
        )

    return np.nan_to_num(buy, nan=False), np.nan_to_num(sell, nan=False)


@dataclass
class Signal:
    position: int  # +1 long, -1 short
    signal_pos: int  # bar where the setup completed
    entry_pos: int  # bar where the pullback filled
    entry: float  # price actually filled
    quoted_entry: float  # price the bot records and prices the target from
    stop: float
    target: float
    risk: float  # stop distance from the filled price
    spike_body: float


def detect_signals(
    df: pd.DataFrame, cfg: Config, spec: SymbolSpec
) -> tuple[list[Signal], dict]:
    """Find every setup and the first pullback entry that follows it.

    Returns the signals plus a diagnostics dict describing what was discarded
    along the way.
    """

    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(df)

    if n < 5:
        return [], {"setups": 0, "rejected_stops": 0}

    if cfg.threshold_mode == "points":
        gap_price = np.full(n, cfg.pgap_points * spec.point)
        max_sl_price = np.full(n, cfg.max_sl_distance_points * spec.point)
    elif cfg.threshold_mode == "atr":
        bar_atr = atr(h, l, c, cfg.atr_period)
        gap_price = cfg.pgap_atr * bar_atr
        max_sl_price = cfg.max_sl_atr * bar_atr
    else:
        raise ValueError(f"Unknown threshold_mode: {cfg.threshold_mode}")

    buy_setup, sell_setup = _setup_masks(
        o, h, l, c, gap_price, cfg.spike_candle_size
    )

    trend_ema = ema(c, cfg.ema_period) if cfg.use_ema_filter else None
    bar_adx = adx(h, l, c, cfg.adx_period) if cfg.use_range_filter else None
    in_session = _session_mask(df.index, cfg)

    rejected_stops = 0

    def filters_pass(pos: int, direction: int) -> bool:
        if trend_ema is not None:
            if direction > 0 and not c[pos] > trend_ema[pos]:
                return False
            if direction < 0 and not c[pos] < trend_ema[pos]:
                return False

        if bar_adx is not None:
            value = bar_adx[pos]
            if not np.isfinite(value) or value < cfg.min_adx:
                return False

        if not in_session[pos]:
            return False

        return True

    def find_entry(
        signal_pos: int, stop: float, direction: int, sl_cap: float
    ) -> tuple[int, float, float, float, float] | None:
        """First pullback bar after the setup, or None if the setup dies.

        The trend filter counts consecutive bars that fail to extend the move.
        Once it is breached it can never recover for a later entry, so the
        search stops there rather than scanning to the end of the series as
        the original loop does; the outcome is identical.

        Returns ``(bar, fill, quoted, target, risk)``. The bot always prices
        the stop and target off ``quoted`` -- the bar extreme it records --
        while ``fill`` is where the order really executes, so the two come
        apart under the market-order model.
        """

        consecutive_opposite = 0

        for pos in range(signal_pos + 1, n):
            if cfg.use_trend_filter:
                extended = (
                    h[pos] > h[pos - 1] if direction > 0 else l[pos] < l[pos - 1]
                )

                if extended:
                    consecutive_opposite = 0
                else:
                    consecutive_opposite += 1

                    if consecutive_opposite > cfg.max_opposite_moves:
                        return None

            bar_extreme = l[pos] if direction > 0 else h[pos]
            previous_extreme = l[pos - 1] if direction > 0 else h[pos - 1]

            triggered = (
                bar_extreme < previous_extreme
                if direction > 0
                else bar_extreme > previous_extreme
            )

            if not triggered:
                continue

            # ``quoted`` is the price the stop distance and target are priced
            # from; ``fill`` is where the order actually executes.
            if cfg.entry_mode == "market_close":
                # Acting only once the bar has closed: the bot would still
                # price its stop and target off the bar extreme it recorded.
                quoted = bar_extreme
                fill = c[pos]
            else:
                if cfg.entry_mode == "fill_fraction":
                    fraction = cfg.fill_fraction
                elif cfg.entry_mode == "prev_low":
                    fraction = 0.0
                elif cfg.entry_mode == "bar_low":
                    fraction = 1.0
                else:
                    raise ValueError(
                        f"Unknown entry_mode: {cfg.entry_mode}"
                    )

                # Firing inside the bar: the order goes in this far past the
                # trigger level, and the stop and target are priced from the
                # same instant, so quote and fill are one price.
                excursion = abs(bar_extreme - previous_extreme)
                quoted = fill = (
                    previous_extreme - direction * fraction * excursion
                )

            # The bot's own risk check runs on the price it recorded.
            quoted_risk = (
                (quoted - stop) if direction > 0 else (stop - quoted)
            )

            if quoted_risk <= 0 or quoted_risk > sl_cap:
                return None

            if not filters_pass(pos, direction):
                continue

            target = quoted + direction * cfg.tp_r * quoted_risk

            risk = (fill - stop) if direction > 0 else (stop - fill)

            # A market order whose stop or target already sits on the wrong
            # side of the fill is rejected by the terminal ("Invalid stops"),
            # so no trade is taken. Only reachable once the fill price is
            # allowed to differ from the quoted one.
            if risk <= 0 or direction * (target - fill) <= 0:
                nonlocal rejected_stops
                rejected_stops += 1
                return None

            return pos, fill, quoted, target, risk

        return None

    signals: dict[int, Signal] = {}

    # Longs are resolved before shorts, matching the write order upstream:
    # when both point at the same entry bar, the long wins.
    for direction, setup_mask in ((1, buy_setup), (-1, sell_setup)):
        for signal_pos in np.flatnonzero(setup_mask):
            signal_pos = int(signal_pos)
            spike_pos = signal_pos - 2

            stop = l[spike_pos] if direction > 0 else h[spike_pos]

            found = find_entry(
                signal_pos, stop, direction, float(max_sl_price[signal_pos])
            )

            if found is None:
                continue

            entry_pos, entry, quoted, target, risk = found

            if entry_pos in signals:
                continue

            signals[entry_pos] = Signal(
                position=direction,
                signal_pos=signal_pos,
                entry_pos=entry_pos,
                entry=entry,
                quoted_entry=quoted,
                stop=stop,
                target=target,
                risk=risk,
                # Upstream reads the body of bar 1 here; the spike is bar 2.
                # Only the diagnostic bucketing uses it, never a trade
                # decision, so the corrected bar is recorded.
                spike_body=abs(c[spike_pos] - o[spike_pos]),
            )

    diagnostics = {
        "setups": int(buy_setup.sum() + sell_setup.sum()),
        "rejected_stops": rejected_stops,
    }

    return [signals[pos] for pos in sorted(signals)], diagnostics


# ============================================================
# SIMULATION
# ============================================================

def simulate(
    df: pd.DataFrame,
    signals: list[Signal],
    cfg: Config,
    spec: SymbolSpec,
) -> pd.DataFrame:
    """Walk the bars once, holding at most one setup at a time.

    A bar that touches both stop and target is resolved as a loss: without
    tick data the order is unknowable, so the pessimistic reading is taken.
    """

    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    index = df.index

    by_entry = {s.entry_pos: s for s in signals}
    spread = cfg.spread_points * spec.point

    trades: list[dict] = []
    active: dict | None = None

    def close_all(exit_price: float, exit_pos: int, reason: str) -> None:
        for leg in active["legs"]:
            direction = leg["direction"]

            # The spread is paid on entry: a long really buys at ask and
            # exits at bid, so the effective fill is worse by one spread.
            effective_entry = leg["entry"] + direction * spread

            move = direction * (exit_price - effective_entry)
            r_multiple = (
                move
                / active["base_risk"]
                * leg["volume_multiplier"]
            )

            trades.append(
                {
                    "setup_id": active["setup_id"],
                    "entry_number": leg["entry_number"],
                    "signal_time": index[active["signal_pos"]],
                    "entry_time": index[leg["entry_pos"]],
                    "exit_time": index[exit_pos],
                    "bars_held": exit_pos - leg["entry_pos"],
                    "direction": "BUY" if direction > 0 else "SELL",
                    "entry": leg["entry"],
                    "sl": active["sl"],
                    "tp": active["tp"],
                    "risk": leg["risk"],
                    "base_risk": active["base_risk"],
                    "sl_points": active["base_risk"] / spec.point,
                    "volume_multiplier": leg["volume_multiplier"],
                    "exit": exit_price,
                    "R": r_multiple,
                    "PnL": r_multiple * cfg.risk_per_trade,
                    "exit_reason": reason,
                    "spike_body": active["spike_body"],
                }
            )

    for i in range(len(df)):
        if active is not None:
            direction = active["direction"]
            sl = active["sl"]
            tp = active["tp"]

            hit_sl = l[i] <= sl if direction > 0 else h[i] >= sl
            hit_tp = h[i] >= tp if direction > 0 else l[i] <= tp

            if hit_sl and hit_tp:
                close_all(sl, i, "SL_and_TP_same_bar_SL_first")
                active = None
                continue

            if hit_sl:
                close_all(sl, i, "SL")
                active = None
                continue

            if hit_tp:
                close_all(tp, i, "TP")
                active = None
                continue

            if cfg.use_second_entry and not active["second_entry_active"]:
                second = active["second_entry"]

                touched = (
                    l[i] <= second if direction > 0 else h[i] >= second
                )

                if touched:
                    second_risk = abs(second - sl)

                    if second_risk > 0:
                        active["legs"].append(
                            {
                                "entry_number": 2,
                                "entry_pos": i,
                                "direction": direction,
                                "entry": second,
                                "risk": second_risk,
                                "volume_multiplier": (
                                    cfg.second_entry_volume_multiplier
                                    * second_risk
                                    / active["base_risk"]
                                ),
                            }
                        )
                        active["second_entry_active"] = True

            continue

        signal = by_entry.get(i)

        if signal is None:
            continue

        base_risk = signal.risk
        direction = signal.position

        tp = signal.target

        active = {
            "setup_id": signal.signal_pos,
            "signal_pos": signal.signal_pos,
            "direction": direction,
            "base_risk": base_risk,
            "sl": signal.stop,
            "tp": tp,
            "second_entry": (
                signal.quoted_entry - direction * base_risk / 2
            ),
            "second_entry_active": False,
            "spike_body": signal.spike_body,
            "legs": [
                {
                    "entry_number": 1,
                    "entry_pos": i,
                    "direction": direction,
                    "entry": signal.entry,
                    "risk": base_risk,
                    "volume_multiplier": 1.0,
                }
            ],
        }

    if active is not None:
        close_all(float(c[-1]), len(df) - 1, "END_OF_DATA")

    return pd.DataFrame(trades)


# ============================================================
# STATISTICS
# ============================================================

def summarise(
    trades: pd.DataFrame, df: pd.DataFrame, cfg: Config
) -> dict:
    """Headline performance numbers for one run."""

    summary = {
        "bars": len(df),
        "start": df.index[0] if len(df) else None,
        "end": df.index[-1] if len(df) else None,
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": np.nan,
        "total_R": 0.0,
        "expectancy_R": np.nan,
        "profit_factor": np.nan,
        "total_pnl": 0.0,
        "return_pct": 0.0,
        "max_drawdown": 0.0,
        "max_drawdown_pct": 0.0,
        "avg_bars_held": np.nan,
        "buy_trades": 0,
        "sell_trades": 0,
        "buy_R": 0.0,
        "sell_R": 0.0,
        "tp_exits": 0,
        "sl_exits": 0,
        "unresolved": 0,
        "avg_sl_points": np.nan,
    }

    if trades.empty:
        return summary

    wins = trades["R"] > 0

    gross_profit = trades.loc[trades["PnL"] > 0, "PnL"].sum()
    gross_loss = abs(trades.loc[trades["PnL"] < 0, "PnL"].sum())

    equity = cfg.initial_cash + trades["PnL"].cumsum()
    drawdown = equity - equity.cummax()

    summary.update(
        {
            "trades": len(trades),
            "wins": int(wins.sum()),
            "losses": int((~wins).sum()),
            "win_rate": float(wins.mean() * 100),
            "total_R": float(trades["R"].sum()),
            "expectancy_R": float(trades["R"].mean()),
            "profit_factor": (
                float(gross_profit / gross_loss)
                if gross_loss > 0
                else np.inf
            ),
            "total_pnl": float(trades["PnL"].sum()),
            "return_pct": float(
                trades["PnL"].sum() / cfg.initial_cash * 100
            ),
            "max_drawdown": float(drawdown.min()),
            "max_drawdown_pct": float(
                drawdown.min() / cfg.initial_cash * 100
            ),
            "avg_bars_held": float(trades["bars_held"].mean()),
            "buy_trades": int((trades["direction"] == "BUY").sum()),
            "sell_trades": int((trades["direction"] == "SELL").sum()),
            "buy_R": float(
                trades.loc[trades["direction"] == "BUY", "R"].sum()
            ),
            "sell_R": float(
                trades.loc[trades["direction"] == "SELL", "R"].sum()
            ),
            "tp_exits": int((trades["exit_reason"] == "TP").sum()),
            "sl_exits": int(
                trades["exit_reason"].str.startswith("SL").sum()
            ),
            "unresolved": int(
                (trades["exit_reason"] == "END_OF_DATA").sum()
            ),
            "avg_sl_points": float(trades["sl_points"].mean()),
        }
    )

    return summary


@dataclass
class BacktestResult:
    symbol: str
    timeframe: str
    config: Config
    trades: pd.DataFrame
    summary: dict
    signals: list[Signal] = field(default_factory=list)


def run_backtest(
    df: pd.DataFrame,
    cfg: Config,
    spec: SymbolSpec,
    symbol: str = "",
    timeframe: str = "",
) -> BacktestResult:
    signals, diagnostics = detect_signals(df, cfg, spec)
    trades = simulate(df, signals, cfg, spec)
    summary = summarise(trades, df, cfg)
    summary["setups_detected"] = diagnostics["setups"]
    summary["setups_entered"] = len(signals)
    summary["rejected_stops"] = diagnostics["rejected_stops"]

    return BacktestResult(
        symbol=symbol,
        timeframe=timeframe,
        config=cfg,
        trades=trades,
        summary=summary,
        signals=signals,
    )
