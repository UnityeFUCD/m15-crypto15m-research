"""Property tests for the HCR stack: invariants that must hold for ALL inputs.

test_hcr_stack.py pins specific spec cases. This file sweeps whole input
ranges and asserts properties that must never break, because the expensive
failures are not the ones on the examples you thought of.

Every property here was verified against the implementation and found holding;
they exist so a later edit cannot silently break one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capture.hcr import (           # noqa: E402
    STATIC_FLOOR, State, breaker_state, dynamic_qty, hcr_qualifies,
    kill_floor, portfolio_admits,
)

EQUITIES = range(211, 3000, 7)
PRICES = [p / 100.0 for p in range(50, 96)]
SIGNALS = [x / 10000.0 for x in range(-60, 61)]


# ------------------------------------------------------- risk cannot be breached
@pytest.mark.parametrize("frac,positions,label", [
    (0.08, 2, "per-window"),
    (0.18, 4, "total"),
])
def test_notional_never_exceeds_risk_fraction(frac, positions, label):
    """Sizing must satisfy its constraints for EVERY (equity, price), not just
    the spec table. Rounding up anywhere would breach these."""
    for E in EQUITIES:
        for p in PRICES:
            q = dynamic_qty(float(E), p, 211.0, 300.0, q_max=20)
            if q:
                assert positions * q * p <= frac * E + 1e-9, (
                    f"{label} breach at E={E} p={p} q={q}")


def test_worst_case_loss_can_never_breach_the_floor():
    """If all four positions lose, equity must still clear floor+buffer."""
    for E in EQUITIES:
        for p in PRICES:
            q = dynamic_qty(float(E), p, 211.0, 300.0, q_max=20)
            if q:
                assert 4 * q * p <= (E - 211.0 - 15.0) + 1e-9


# ------------------------------------------------------------------ monotonicity
def test_size_is_monotone_in_equity():
    """More money must never produce a smaller position."""
    for E in range(212, 3000):
        assert (dynamic_qty(float(E), 0.70, 211.0, 300.0)
                >= dynamic_qty(float(E - 1), 0.70, 211.0, 300.0))


def test_size_is_monotone_in_price():
    """A more expensive contract must never produce a larger position."""
    for cents in range(51, 96):
        assert (dynamic_qty(600.0, cents / 100.0, 211.0, 300.0)
                <= dynamic_qty(600.0, (cents - 1) / 100.0, 211.0, 300.0))


# ------------------------------------------------------------- the live account
def test_live_account_state_is_refused():
    """The real account sits at $136.27 against a $531 strategy HWM. It must
    size to zero AND kill - this is the case that actually matters."""
    floor = kill_floor(531.0)
    assert dynamic_qty(136.27, 0.70, floor, 531.0, q_max=20) == 0
    assert breaker_state(State(equity=136.27, day_start=136.27,
                               day_high=136.27, strategy_hwm=531.0,
                               launch_equity=531.0)) == "KILL"


def test_kill_floor_tracks_the_high_water_mark():
    assert kill_floor(531.0) == pytest.approx(398.25)   # not $211
    assert kill_floor(0.0) == STATIC_FLOOR


# ------------------------------------------------------------------ breakers
def test_kill_dominates_every_other_trigger():
    s = State(equity=50.0, day_start=300.0, day_high=300.0,
              strategy_hwm=300.0, launch_equity=300.0,
              consecutive_api_failures=9)
    assert breaker_state(s) == "KILL"


@pytest.mark.parametrize("mult,expect", [
    (0.92, "THROTTLED"),    # exactly -8%
    (0.85, "HALTED"),       # exactly -15%
])
def test_thresholds_are_inclusive_at_the_boundary(mult, expect):
    s = State(equity=300.0 * mult, day_start=300.0, day_high=300.0,
              strategy_hwm=300.0, launch_equity=300.0)
    assert breaker_state(s) == expect


# -------------------------------------------------------------------- signal
def test_signal_is_symmetric_under_side_and_sign_flip():
    """A YES favourite against -r must qualify exactly when a NO favourite
    against +r does. Asymmetry would mean a hidden directional bias."""
    for r in SIGNALS:
        assert hcr_qualifies("yes", -r, 0.002) == hcr_qualifies("no", r, 0.002)


def test_no_signal_qualifies_both_sides_at_once():
    for r in SIGNALS:
        assert not (hcr_qualifies("yes", r, 0.002)
                    and hcr_qualifies("no", r, 0.002))


def test_calm_gate_blocks_even_an_extreme_signal():
    assert not hcr_qualifies("no", r_common=0.05, calm=0.0031)


# ----------------------------------------------------------------- portfolio
def test_either_cap_alone_is_sufficient_to_refuse():
    assert not portfolio_admits(in_window=2, total_open=0)
    assert not portfolio_admits(in_window=0, total_open=4)
