"""Behavioural tests for the SP2L engine, on hand-built bar sequences.

Run with ``python -m backtest.test_engine`` or under pytest.

Every fixture is built around one long setup so the expected stop, target and
R-multiple can be worked out on paper:

    bar 1  99.8 .. 101.2   small bullish body   "before"
    bar 2 100.9 .. 110.5   nine-point body      "spike"  -> stop = 100.9
    bar 3 111.5 .. 113.5   small bullish body   "after", gapped above bar 1
    bar 4                  signal bar
    bar 5  low 112.0       first lower low      -> entry
"""

import pandas as pd

from .config import Config, SymbolSpec
from .engine import detect_signals, run_backtest

SPEC = SymbolSpec("TEST", point=0.01, digits=2)

# Wide enough that the stop-distance cap never fires in these fixtures.
BASE = Config(
    max_sl_distance_points=1_000_000,
    use_ema_filter=False,
    use_trend_filter=False,
    label="test",
)

SETUP_BARS = [
    #  open,   high,    low,   close
    (99.0, 99.5, 98.5, 99.0),  # 0 filler
    (100.0, 101.2, 99.8, 101.0),  # 1 before
    (101.0, 110.5, 100.9, 110.0),  # 2 spike
    (112.0, 113.5, 111.5, 113.0),  # 3 after
    (113.0, 114.0, 112.8, 113.5),  # 4 signal bar
    (113.5, 113.8, 112.0, 113.2),  # 5 first lower low -> entry
]

EXPECTED_STOP = 100.9
EXPECTED_ENTRY = 112.0
EXPECTED_RISK = EXPECTED_ENTRY - EXPECTED_STOP  # 11.1
EXPECTED_TARGET = EXPECTED_ENTRY + EXPECTED_RISK  # 123.1


def frame_of(bars) -> pd.DataFrame:
    bars = list(bars)

    return pd.DataFrame(
        bars,
        columns=["open", "high", "low", "close"],
        index=pd.date_range(
            "2026-01-01", periods=len(bars), freq="5min", tz="UTC"
        ),
    )


def frame(extra_bars) -> pd.DataFrame:
    """The reference setup followed by ``extra_bars``."""

    return frame_of(SETUP_BARS + list(extra_bars))


def approx(value: float, expected: float, tol: float = 1e-9) -> bool:
    return abs(value - expected) < tol


def test_setup_is_detected_with_the_spike_low_as_stop():
    signals, _ = detect_signals(frame([(113.2, 114.0, 113.0, 113.8)]),
                                BASE, SPEC)

    assert len(signals) == 1

    signal = signals[0]

    assert signal.position == 1
    assert signal.signal_pos == 4
    assert signal.entry_pos == 5
    assert approx(signal.stop, EXPECTED_STOP)
    assert approx(signal.entry, EXPECTED_ENTRY)
    assert approx(signal.risk, EXPECTED_RISK)
    assert approx(signal.target, EXPECTED_TARGET)


def test_target_hit_pays_one_r():
    result = run_backtest(
        frame([(113.2, 124.0, 113.0, 123.5)]), BASE, SPEC
    )

    assert len(result.trades) == 1

    trade = result.trades.iloc[0]

    assert trade["exit_reason"] == "TP"
    assert approx(trade["exit"], EXPECTED_TARGET)
    assert approx(trade["R"], 1.0)


def test_stop_hit_loses_one_r():
    result = run_backtest(
        frame([(113.2, 113.5, 100.0, 100.5)]), BASE, SPEC
    )

    trade = result.trades.iloc[0]

    assert trade["exit_reason"] == "SL"
    assert approx(trade["exit"], EXPECTED_STOP)
    assert approx(trade["R"], -1.0)


def test_bar_touching_both_levels_is_read_as_a_loss():
    result = run_backtest(
        frame([(113.2, 124.0, 100.0, 120.0)]), BASE, SPEC
    )

    trade = result.trades.iloc[0]

    assert trade["exit_reason"] == "SL_and_TP_same_bar_SL_first"
    assert approx(trade["R"], -1.0)


def test_market_fill_earns_less_than_one_r_at_the_same_target():
    """The live bot targets off the bar low but fills at the bar close."""

    result = run_backtest(
        frame([(113.2, 124.0, 113.0, 123.5)]),
        BASE.with_(entry_mode="market_close"),
        SPEC,
    )

    trade = result.trades.iloc[0]

    fill = 113.2  # close of the entry bar
    expected_r = (EXPECTED_TARGET - fill) / (fill - EXPECTED_STOP)

    assert approx(trade["entry"], fill)
    assert approx(trade["tp"], EXPECTED_TARGET)
    assert trade["exit_reason"] == "TP"
    assert approx(trade["R"], expected_r)
    assert trade["R"] < 1.0


def test_limit_fill_prices_its_target_off_the_limit():
    """The limit variant quotes and fills at the same knowable price."""

    result = run_backtest(
        frame([(113.2, 130.0, 113.0, 129.0)]),
        BASE.with_(entry_mode="prev_low"),
        SPEC,
    )

    trade = result.trades.iloc[0]
    fill = 112.8  # low of bar 4, where the resting limit sat

    assert approx(trade["entry"], fill)
    assert approx(trade["tp"], fill + (fill - EXPECTED_STOP))
    assert approx(trade["R"], 1.0)


def test_stop_cap_rejects_a_setup_whose_stop_is_too_far():
    cfg = BASE.with_(max_sl_distance_points=100)  # 1.00 price units

    signals, _ = detect_signals(
        frame([(113.2, 114.0, 113.0, 113.8)]), cfg, SPEC
    )

    assert signals == []


def test_gap_requirement_rejects_an_ungapped_setup():
    bars = list(SETUP_BARS)
    # Drop bar 3 onto bar 1's high so the two legs no longer gap apart.
    bars[3] = (101.3, 102.0, 101.25, 101.9)
    bars[4] = (101.9, 102.2, 101.6, 102.0)
    bars[5] = (102.0, 102.1, 101.0, 101.5)

    frame_ = pd.DataFrame(
        bars + [(101.5, 110.0, 101.0, 109.0)],
        columns=["open", "high", "low", "close"],
        index=pd.date_range("2026-01-01", periods=len(bars) + 1, freq="5min",
                            tz="UTC"),
    )

    signals, _ = detect_signals(frame_, BASE, SPEC)

    assert signals == []


# Bars 5-7 replace the reference entry bar: the move stalls for two bars
# before the first lower low finally prints.
STALLING_BARS = [
    (113.5, 113.9, 113.4, 113.6),  # 5 lower high, no lower low
    (113.6, 113.7, 113.5, 113.6),  # 6 lower high again
    (113.6, 113.8, 110.0, 111.0),  # 7 the first lower low
]


def test_trend_filter_kills_a_setup_that_stalls_for_too_long():
    df = frame_of(SETUP_BARS[:5] + STALLING_BARS)

    # Without the filter the stall is irrelevant and bar 7 is the entry.
    relaxed, _ = detect_signals(df, BASE, SPEC)

    assert len(relaxed) == 1
    assert relaxed[0].entry_pos == 7

    # With it, two consecutive bars that fail to extend the move exceed
    # max_opposite_moves and the setup is abandoned.
    strict, _ = detect_signals(
        df, BASE.with_(use_trend_filter=True, max_opposite_moves=1), SPEC
    )

    assert strict == []


def test_ema_filter_blocks_a_long_below_its_average():
    # Bar 5 makes the lower low that triggers the entry but closes back
    # under the fast average; bar 6 never trades below it, so nothing else
    # can trigger.
    df = frame_of(
        SETUP_BARS[:5]
        + [
            (113.5, 113.8, 110.5, 111.0),
            (111.0, 111.2, 110.8, 111.0),
        ]
    )

    without_filter, _ = detect_signals(df, BASE, SPEC)

    assert len(without_filter) == 1
    assert without_filter[0].entry_pos == 5

    with_filter, _ = detect_signals(
        df, BASE.with_(use_ema_filter=True, ema_period=3), SPEC
    )

    assert with_filter == []


def test_only_one_setup_is_held_at_a_time():
    """A second signal while a trade is open is skipped, not stacked."""

    df = frame(
        [
            (113.2, 113.4, 113.0, 113.3),
            (113.3, 113.5, 113.1, 113.4),
        ]
        + SETUP_BARS[1:]
        + [(113.2, 124.0, 113.0, 123.5)]
    )

    result = run_backtest(df, BASE, SPEC)

    assert result.summary["setups_entered"] >= 1
    assert result.trades["setup_id"].nunique() == len(result.trades)


def test_spread_is_charged_against_the_entry():
    spread_points = 50  # 0.50 price units at point 0.01

    without = run_backtest(
        frame([(113.2, 124.0, 113.0, 123.5)]), BASE, SPEC
    ).trades.iloc[0]

    with_cost = run_backtest(
        frame([(113.2, 124.0, 113.0, 123.5)]),
        BASE.with_(spread_points=spread_points),
        SPEC,
    ).trades.iloc[0]

    expected_drop = (spread_points * SPEC.point) / EXPECTED_RISK

    assert approx(without["R"] - with_cost["R"], expected_drop)


def test_flat_series_produces_nothing():
    bars = [(100.0, 100.5, 99.5, 100.0)] * 50

    df = pd.DataFrame(
        bars,
        columns=["open", "high", "low", "close"],
        index=pd.date_range("2026-01-01", periods=len(bars), freq="5min",
                            tz="UTC"),
    )

    result = run_backtest(df, BASE, SPEC)

    assert result.summary["setups_detected"] == 0
    assert result.trades.empty


def _main() -> None:
    tests = [
        (name, value)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]

    failures = 0

    for name, test in tests:
        try:
            test()
        except AssertionError as error:
            failures += 1
            print(f"FAIL {name}: {error or 'assertion failed'}")
        else:
            print(f"ok   {name}")

    print(f"\n{len(tests) - failures}/{len(tests)} passed")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
