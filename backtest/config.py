"""Configuration objects for the SP2L backtest engine.

The live bot reads ``point``/``digits`` from ``mt5.symbol_info()``. Offline we
have no terminal, so contract sizes are declared here and every threshold that
the original code expressed in broker *points* is converted with the same
``points * point`` formula the bot uses.
"""

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class SymbolSpec:
    """Contract details needed to translate broker points into prices."""

    name: str
    point: float
    digits: int
    # Round-turn cost charged on entry, expressed in broker points. The
    # original backtest assumes zero; these are typical retail values used
    # only when a run explicitly asks for a cost model.
    typical_spread_points: float = 0.0


SYMBOL_SPECS = {
    # FX majors: 5-digit pricing, so 1 point = 1/10 pip.
    "eurusd": SymbolSpec("EURUSD", 0.00001, 5, typical_spread_points=10),
    "gbpusd": SymbolSpec("GBPUSD", 0.00001, 5, typical_spread_points=15),
    "audusd": SymbolSpec("AUDUSD", 0.00001, 5, typical_spread_points=12),
    "eurgbp": SymbolSpec("EURGBP", 0.00001, 5, typical_spread_points=15),
    # Metals.
    "xauusd": SymbolSpec("XAUUSD", 0.01, 2, typical_spread_points=25),
    # Cash index CFDs.
    "us30": SymbolSpec("US30", 0.1, 1, typical_spread_points=20),
    "us100": SymbolSpec("US100", 0.1, 1, typical_spread_points=15),
    "us500": SymbolSpec("US500", 0.1, 1, typical_spread_points=5),
}


@dataclass(frozen=True)
class Config:
    """Strategy and simulation settings.

    Defaults mirror the SETTINGS block of ``SP2L_Advanced_Bot.py`` so a run
    with no overrides reproduces the upstream configuration.
    """

    # --- Setup geometry ---------------------------------------------------
    spike_candle_size: float = 1.5
    pgap_points: float = 100.0
    max_sl_distance_points: float = 1000.0
    tp_r: float = 1.0

    # --- Threshold scaling ------------------------------------------------
    # "points": gap and max-SL are fixed multiples of the broker point, as
    #           upstream does. Calibrated for XAUUSD M1.
    # "atr":    the same two thresholds become multiples of ATR(atr_period),
    #           which keeps them comparable across symbols and timeframes.
    threshold_mode: str = "points"
    atr_period: int = 14
    pgap_atr: float = 0.25
    max_sl_atr: float = 2.5

    # --- Entry model ------------------------------------------------------
    # Meta.GetRates calls copy_rates_from with a timestamp three hours in the
    # future, so the frame the bot reads ends with the *forming* bar. The
    # pullback test compares that bar's running low against the last closed
    # bar's low, which means the bot fires part-way through the bar -- within
    # one poll interval of price crossing the previous bar's extreme -- and
    # sends a market order. Stop and target are priced off whatever low was
    # running at that instant, so quote and fill coincide.
    #
    # "fill_fraction": fill this far into the move past the trigger level,
    #                  where 0.0 is the crossing itself and 1.0 is the bar's
    #                  eventual extreme. The live fill sits near the low end;
    #                  the exact value depends on how fast price moves inside
    #                  one poll interval.
    # "prev_low":      shorthand for fill_fraction 0.0.
    # "bar_low":       shorthand for fill_fraction 1.0 -- the upstream
    #                  backtest, which fills at the best price the bar ever
    #                  reached and cannot be known until the bar closes.
    # "market_close":  fill at the bar's close instead, as if the bot only
    #                  ever acted on closed bars. A pessimistic bound.
    entry_mode: str = "bar_low"
    fill_fraction: float = 0.0

    # --- Filters ----------------------------------------------------------
    use_ema_filter: bool = True
    ema_period: int = 60

    use_trend_filter: bool = True
    max_opposite_moves: int = 1

    use_range_filter: bool = False
    adx_period: int = 14
    min_adx: float = 20.0

    use_session_filter: bool = False
    session_start_hour: int = 1
    session_end_hour: int = 5
    session_timezone: str = "America/New_York"

    # The strategy's author trades it only at a fixed list of clock times,
    # given in Tehran local time. Entries are allowed from each listed time
    # until ``time_window_minutes`` later, and blocked otherwise. Tehran has
    # had no daylight saving since 2022, so the offset is a flat UTC+3:30
    # and the windows do not shift through the year; the New York times they
    # line up with do shift.
    use_time_filter: bool = False
    entry_times: tuple[str, ...] = (
        "09:00",  # 01:30 New York
        "10:00",  # 02:30
        "14:00",  # 06:30
        "15:30",  # 08:00
        "16:30",  # 09:00
        "17:00",  # 09:30, the NYSE open
        "18:00",  # 10:30
        "18:30",  # 11:00
        "21:00",  # 13:30
        "23:00",  # 15:30, half an hour before the close
    )
    time_window_minutes: int = 60
    entry_timezone: str = "Asia/Tehran"

    # --- Second entry -----------------------------------------------------
    use_second_entry: bool = False
    second_entry_volume_multiplier: float = 2.0

    # --- Money management -------------------------------------------------
    initial_cash: float = 10_000.0
    risk_per_trade: float = 100.0

    # --- Costs ------------------------------------------------------------
    # Charged once per leg, in broker points, worsening entry and exit.
    spread_points: float = 0.0

    label: str = "upstream"
    notes: str = ""

    def with_(self, **changes) -> "Config":
        return replace(self, **changes)
