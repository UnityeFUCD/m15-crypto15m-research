"""Tests for the HCR production stack. WRITTEN BEFORE THE IMPLEMENTATION.

Each test encodes a requirement from the specification. Several deliberately
probe cases where a naive implementation is wrong:

  - r_common must use COMPLETED returns only (no look-ahead)
  - contiguity must be enforced across every window used
  - the calm window needs 24 CONTIGUOUS observations, not 24 rows
  - dynamic sizing must bind on the TIGHTEST of four constraints
  - sizing must never round up, and never go below 1 without disabling
  - quantity may fall intraday but may only rise at a UTC day boundary
  - circuit-breaker thresholds are on DAY-START equity, and the giveback is
    measured from the intraday HIGH, not from day-start
  - the kill floor is max($211, 75% of strategy-era HWM), which can EXCEED
    the static floor once the account grows

Run:  python -m pytest tests/test_hcr_stack.py -q
"""
from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capture.hcr import (           # noqa: E402
    State, breaker_state, common_return, dynamic_qty, hcr_qualifies,
    calm_mean, portfolio_admits,
)

# ---------------------------------------------------------------- signal
def test_common_return_is_simple_mean_of_six():
    vals = [0.001, 0.002, 0.003, -0.001, 0.000, 0.001]
    # sum by hand: 0.006 over 6 coins = 0.001 exactly
    assert common_return(vals) == pytest.approx(sum(vals) / 6)
    assert common_return(vals) == pytest.approx(0.001)


def test_common_return_requires_all_inputs_present():
    """Spec: 'all six common-factor inputs available'. A missing coin is a
    SKIP, not a mean over five."""
    with pytest.raises(ValueError):
        common_return([0.001, 0.002, None, 0.001, 0.0, 0.001])
    with pytest.raises(ValueError):
        common_return([0.001, 0.002, 0.003])


def test_common_return_rejects_nan():
    with pytest.raises(ValueError):
        common_return([0.001, float("nan"), 0.003, 0.0, 0.0, 0.0])


def test_calm_mean_needs_24_contiguous_windows():
    """24 ROWS is not 24 CONTIGUOUS windows. A gap must invalidate."""
    good = [(i * 900, 0.001) for i in range(24)]
    assert calm_mean(good) == pytest.approx(0.001)
    gapped = [(i * 900, 0.001) for i in range(23)] + [(23 * 900 + 900, 0.001)]
    with pytest.raises(ValueError):
        calm_mean(gapped)


def test_calm_mean_rejects_short_history():
    with pytest.raises(ValueError):
        calm_mean([(i * 900, 0.001) for i in range(23)])


def test_calm_mean_uses_absolute_values():
    alt = [(i * 900, 0.002 if i % 2 else -0.002) for i in range(24)]
    assert calm_mean(alt) == pytest.approx(0.002)


# ------------------------------------------------------------ qualification
def test_hcr_no_favourite_needs_common_move_UP():
    """NO favourite (d=-1) opposes an UP move: -1 * r <= -0.0015 -> r >= +15bp"""
    assert hcr_qualifies("no", r_common=+0.0020, calm=0.0020)
    assert not hcr_qualifies("no", r_common=-0.0020, calm=0.0020)


def test_hcr_yes_favourite_needs_common_move_DOWN():
    assert hcr_qualifies("yes", r_common=-0.0020, calm=0.0020)
    assert not hcr_qualifies("yes", r_common=+0.0020, calm=0.0020)


def test_hcr_threshold_is_inclusive_at_exactly_15bp():
    assert hcr_qualifies("no", r_common=0.0015, calm=0.0020)
    assert not hcr_qualifies("no", r_common=0.00149, calm=0.0020)


def test_hcr_calm_threshold_inclusive_at_30bp():
    assert hcr_qualifies("no", r_common=0.0020, calm=0.0030)
    assert not hcr_qualifies("no", r_common=0.0020, calm=0.00301)


def test_hcr_has_no_minute_filter():
    """Spec: 'Do not use the minute-00 filter.'"""
    for minute in (0, 15, 30, 45):
        assert hcr_qualifies("no", r_common=0.0020, calm=0.0020,
                             close_minute=minute)


def test_hcr_rejects_bad_side():
    with pytest.raises(ValueError):
        hcr_qualifies("maybe", r_common=0.002, calm=0.002)


# --------------------------------------------------------------- sizing
def test_dynamic_qty_matches_spec_table_at_300():
    """Spec table at E=$300: 65c->18, 70c->17, 75c->16, 80c->15."""
    for price, expect in ((0.65, 18), (0.70, 17), (0.75, 16), (0.80, 15)):
        assert dynamic_qty(equity=300.0, price=price, floor=211.0,
                           launch_equity=300.0, q_max=20) == expect


def test_dynamic_qty_never_exceeds_cap():
    assert dynamic_qty(equity=100000.0, price=0.65, floor=211.0,
                       launch_equity=300.0, q_max=20) == 20


def test_dynamic_qty_floors_never_rounds_up():
    q = dynamic_qty(equity=300.0, price=0.80, floor=211.0,
                    launch_equity=300.0, q_max=20)
    per_window = 2 * q * 0.80
    assert per_window <= 0.08 * 300.0 + 1e-9


def test_dynamic_qty_binds_on_tightest_constraint():
    """Near the floor, the floor-room term must dominate."""
    q = dynamic_qty(equity=230.0, price=0.70, floor=211.0,
                    launch_equity=300.0, q_max=20)
    room = 230.0 - 211.0 - 0.05 * 300.0
    assert q <= max(0, int(room / (4 * 0.70)))


def test_dynamic_qty_returns_zero_when_no_room():
    assert dynamic_qty(equity=215.0, price=0.70, floor=211.0,
                       launch_equity=300.0, q_max=20) == 0


def test_dynamic_qty_uses_min_of_current_and_daystart_equity():
    """Spec: E = min(current, day-start). An intraday run-up must NOT raise q."""
    up = dynamic_qty(equity=300.0, price=0.70, floor=211.0,
                     launch_equity=300.0, q_max=20, day_start_equity=250.0)
    flat = dynamic_qty(equity=250.0, price=0.70, floor=211.0,
                       launch_equity=300.0, q_max=20, day_start_equity=250.0)
    assert up == flat


def test_dynamic_qty_falls_immediately_after_losses():
    hi = dynamic_qty(equity=300.0, price=0.70, floor=211.0,
                     launch_equity=300.0, q_max=20)
    lo = dynamic_qty(equity=260.0, price=0.70, floor=211.0,
                     launch_equity=300.0, q_max=20)
    assert lo < hi


# ------------------------------------------------------------- breakers
def test_normal_state_when_flat():
    s = State(equity=300.0, day_start=300.0, day_high=300.0,
              strategy_hwm=300.0, launch_equity=300.0)
    assert breaker_state(s) == "NORMAL"


def test_throttle_on_8pct_day_loss():
    s = State(equity=276.0, day_start=300.0, day_high=300.0,
              strategy_hwm=300.0, launch_equity=300.0)
    assert breaker_state(s) == "THROTTLED"


def test_throttle_on_10pct_giveback_from_intraday_high():
    """Giveback is measured from the HIGH, not from day-start. Equity here is
    ABOVE day-start, so a day-loss rule would never fire."""
    s = State(equity=340.0, day_start=300.0, day_high=372.0,
              strategy_hwm=372.0, launch_equity=300.0)
    assert breaker_state(s) == "THROTTLED"


def test_halt_on_15pct_day_loss():
    s = State(equity=255.0, day_start=300.0, day_high=300.0,
              strategy_hwm=300.0, launch_equity=300.0)
    assert breaker_state(s) == "HALTED"


def test_halt_on_12pct_giveback():
    s = State(equity=336.0, day_start=300.0, day_high=372.0,
              strategy_hwm=372.0, launch_equity=300.0)
    assert breaker_state(s) == "HALTED"


def test_kill_uses_static_floor_when_account_is_small():
    s = State(equity=210.0, day_start=300.0, day_high=300.0,
              strategy_hwm=300.0, launch_equity=300.0)
    assert breaker_state(s) == "KILL"


def test_kill_floor_RISES_with_strategy_hwm():
    """75% of a $1,000 HWM is $750, well above the static $211. An account at
    $700 must KILL even though it is far above $211."""
    s = State(equity=700.0, day_start=760.0, day_high=760.0,
              strategy_hwm=1000.0, launch_equity=300.0)
    assert breaker_state(s) == "KILL"


def test_kill_dominates_other_states():
    s = State(equity=100.0, day_start=300.0, day_high=300.0,
              strategy_hwm=300.0, launch_equity=300.0)
    assert breaker_state(s) == "KILL"


def test_integrity_failures_force_kill():
    s = State(equity=300.0, day_start=300.0, day_high=300.0,
              strategy_hwm=300.0, launch_equity=300.0,
              side_mapping_ok=False)
    assert breaker_state(s) == "KILL"
    s2 = State(equity=300.0, day_start=300.0, day_high=300.0,
               strategy_hwm=300.0, launch_equity=300.0,
               unmanaged_orders=1)
    assert breaker_state(s2) == "KILL"


def test_three_api_failures_halt_the_day():
    s = State(equity=300.0, day_start=300.0, day_high=300.0,
              strategy_hwm=300.0, launch_equity=300.0,
              consecutive_api_failures=3)
    assert breaker_state(s) == "HALTED"


# ------------------------------------------------------------ portfolio
def test_cap_two_per_close_window():
    assert portfolio_admits(in_window=0, total_open=0)
    assert portfolio_admits(in_window=1, total_open=1)
    assert not portfolio_admits(in_window=2, total_open=2)


def test_total_open_cap_of_four():
    assert portfolio_admits(in_window=0, total_open=3)
    assert not portfolio_admits(in_window=0, total_open=4)


def test_throttled_allows_only_one_per_window():
    assert portfolio_admits(in_window=0, total_open=0, throttled=True)
    assert not portfolio_admits(in_window=1, total_open=1, throttled=True)
