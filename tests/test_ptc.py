"""PTC state machine tests. WRITTEN BEFORE capture/ptc.py, per Part 6.

Each test encodes one requirement from the brief. Several deliberately probe
cases where the obvious implementation is wrong, because every expensive error
in this project was invisible in the final number:

  * the orders table inverts side semantics on action=sell (agreement with the
    fills table is 1.0000 derived, 0.4765 naive)
  * pandas datetimes arrive at s/ms/us/ns and dividing by the wrong power of
    ten invalidated PTC v1
  * a fill can land DURING the cancel race, after the request and before the
    confirmation, and must suppress the commitment
  * state must never advance on an optimistic local guess; only an
    authoritative exchange acknowledgement may move it

Run:  python -m pytest tests/test_ptc.py -q
"""
from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capture.ptc import (                                    # noqa: E402
    FROZEN, PtcConfig, PtcMachine, S, TimestampUnitError,
    commit_quantity, detect_epoch_unit, held_side, held_quote_micros,
    ioc_fee_micros, to_epoch_seconds,
)

M = 1_000_000       # one dollar in micros


def cfg(**kw):
    base = dict(enabled=True, live_enabled=False, wait_seconds=60,
                commit_qty=5, equity_micros=500 * M, kill=False)
    base.update(kw)
    return PtcConfig(**base)


def machine(**kw) -> PtcMachine:
    return PtcMachine(cfg(**kw))


def opp(mach, tid="KXBTC15M-26AUG051900-00", close_ts=1786000800, ask=700_000):
    return mach.register(opportunity_id=tid + "|1", ticker=tid,
                         close_ts=close_ts, coin="BTC", side="yes",
                         bid_micros=690_000, ask_micros=ask)


# ------------------------------------------------------------ side semantics
def test_held_side_inverts_on_sell():
    """orders table: action=sell on side=yes means the position HELD is NO."""
    assert held_side("sell", "yes") == "no"
    assert held_side("sell", "no") == "yes"
    assert held_side("buy", "yes") == "yes"
    assert held_side("buy", "no") == "no"


def test_held_side_rejects_unknown():
    with pytest.raises(ValueError):
        held_side("cancel", "yes")
    with pytest.raises(ValueError):
        held_side("buy", "maybe")


def test_held_quote_conversion_yes_and_no():
    yb, ya = 700_000, 730_000
    assert held_quote_micros("yes", yb, ya) == (700_000, 730_000)
    # holding NO: bid = 1 - yes_ask, ask = 1 - yes_bid
    assert held_quote_micros("no", yb, ya) == (270_000, 300_000)


def test_held_quote_round_trips():
    """The NO conversion is an involution: applying it twice is identity."""
    for yb in range(510_000, 900_000, 37_000):
        ya = yb + 20_000
        nb, na = held_quote_micros("no", yb, ya)
        assert (nb, na) == (M - ya, M - yb)
        assert held_quote_micros("no", nb, na) == (yb, ya)


# ------------------------------------------------------------- timestamps
@pytest.mark.parametrize("raw,unit", [
    (1_786_000_800, "s"), (1_786_000_800_000, "ms"),
    (1_786_000_800_000_000, "us"), (1_786_000_800_000_000_000, "ns"),
])
def test_detect_epoch_unit(raw, unit):
    assert detect_epoch_unit(raw) == unit


@pytest.mark.parametrize("raw", [
    1_786_000_800, 1_786_000_800_000,
    1_786_000_800_000_000, 1_786_000_800_000_000_000,
])
def test_all_epoch_units_normalise_to_the_same_second(raw):
    assert to_epoch_seconds(raw) == pytest.approx(1_786_000_800.0, abs=1e-3)


def test_absurd_epoch_is_rejected_not_silently_scaled():
    """PTC v1 turned a 60s wait into 1.8 billion seconds. That must raise."""
    with pytest.raises(TimestampUnitError):
        to_epoch_seconds(10 ** 25)


# ---------------------------------------------------------------- lifecycle
def test_probe_is_exactly_one_contract():
    m = machine()
    o = opp(m)
    assert m.state(o) is S.TREATMENT_ASSIGNED
    req = m.build_probe(o)
    assert req["count"] == 1
    assert req["post_only"] is True
    assert req["action"] == "buy"


def test_client_order_id_is_idempotent_and_phase_scoped():
    m = machine()
    o = opp(m)
    a = m.client_order_id(o, "probe")
    b = m.client_order_id(o, "probe")
    c = m.client_order_id(o, "commit")
    assert a == b            # same opportunity+phase -> identical id
    assert a != c            # probe and commit never collide
    assert a.startswith("ptc-")


def test_state_advances_only_on_authoritative_ack():
    m = machine()
    o = opp(m)
    m.mark_submitting(o)
    assert m.state(o) is S.PROBE_SUBMITTING
    # a local guess must not advance the machine
    m.on_api_error(o, "timeout")
    assert m.state(o) is not S.PROBE_RESTING
    m.on_probe_ack(o, order_id="X1", exchange_created_ts=1_786_000_000.0)
    assert m.state(o) is S.PROBE_RESTING


def test_timer_starts_from_exchange_created_time_not_local_clock():
    m = machine()
    o = opp(m)
    m.mark_submitting(o)
    # local clock is 30s ahead of the exchange ack
    m.on_probe_ack(o, order_id="X1", exchange_created_ts=1_000.0)
    assert not m.should_cancel(o, now=1_030.0)
    assert not m.should_cancel(o, now=1_059.9)
    assert m.should_cancel(o, now=1_060.0)


def test_fill_before_timeout_blocks_commitment():
    m = machine()
    o = opp(m)
    m.mark_submitting(o)
    m.on_probe_ack(o, "X1", 1_000.0)
    m.on_fill(o, qty=1, price_micros=690_000, ts=1_010.0)
    assert m.state(o) is S.PROBE_FILLED
    assert not m.commit_allowed(o)


def test_fill_during_cancel_race_blocks_commitment():
    """The decisive case: request sent, fill lands, confirmation arrives."""
    m = machine()
    o = opp(m)
    m.mark_submitting(o)
    m.on_probe_ack(o, "X1", 1_000.0)
    m.on_cancel_requested(o, ts=1_060.0)
    assert m.state(o) is S.CANCEL_REQUESTED
    m.on_fill(o, qty=1, price_micros=690_000, ts=1_060.4)   # the race
    m.on_cancel_confirmed(o, ts=1_060.9, filled_qty=1)
    assert m.state(o) is S.CANCEL_RACE_RECONCILED
    assert not m.commit_allowed(o)


def test_clean_no_fill_permits_commitment():
    m = machine()
    o = opp(m)
    m.mark_submitting(o)
    m.on_probe_ack(o, "X1", 1_000.0)
    m.on_cancel_requested(o, ts=1_060.0)
    m.on_cancel_confirmed(o, ts=1_060.5, filled_qty=0)
    assert m.state(o) is S.COMMIT_ELIGIBLE
    assert m.commit_allowed(o)


def test_commit_requires_confirmed_zero_fill_not_merely_a_request():
    m = machine()
    o = opp(m)
    m.mark_submitting(o)
    m.on_probe_ack(o, "X1", 1_000.0)
    m.on_cancel_requested(o, ts=1_060.0)
    assert not m.commit_allowed(o)          # no confirmation yet -> fail closed


def test_cancellation_rejection_is_not_a_confirmation():
    m = machine()
    o = opp(m)
    m.mark_submitting(o)
    m.on_probe_ack(o, "X1", 1_000.0)
    m.on_cancel_requested(o, ts=1_060.0)
    m.on_cancel_rejected(o, ts=1_061.0, reason="already_filled")
    assert m.state(o) is S.ERROR
    assert not m.commit_allowed(o)


# ------------------------------------------------------------ commit branch
def test_ask_band_is_enforced_on_the_refreshed_book():
    m = machine()
    for ask, ok in ((590_000, False), (600_000, True), (800_000, True),
                    (800_001, False)):
        o = opp(m, tid=f"KXBTC15M-26AUG05190{ask % 7}-00", ask=ask)
        m.mark_submitting(o)
        m.on_probe_ack(o, "X", 1_000.0)
        m.on_cancel_requested(o, 1_060.0)
        m.on_cancel_confirmed(o, 1_060.5, filled_qty=0)
        m.refresh_book(o, ask_micros=ask)
        assert m.commit_allowed(o) is ok


def test_only_one_commitment_per_close_window():
    m = machine()
    ready = []
    for i in range(3):
        o = m.register(opportunity_id=f"W|{i}", ticker=f"KX{i}15M-X-00",
                       close_ts=1786000800, coin="BTC", side="yes",
                       bid_micros=690_000, ask_micros=700_000)
        m.mark_submitting(o); m.on_probe_ack(o, f"X{i}", 1_000.0)
        m.on_cancel_requested(o, 1_060.0)
        m.on_cancel_confirmed(o, 1_060.5, filled_qty=0)
        ready.append(o)
    assert sum(m.try_commit(o) is not None for o in ready) == 1


def test_probes_still_allowed_on_other_coins_in_the_same_window():
    """Only the FULL commitment is capped at one; q1 probes are not."""
    m = machine()
    for i in range(4):
        o = m.register(opportunity_id=f"P|{i}", ticker=f"KX{i}15M-X-00",
                       close_ts=1786000800, coin="BTC", side="yes",
                       bid_micros=690_000, ask_micros=700_000)
        assert m.build_probe(o)["count"] == 1


def test_no_duplicate_commitment_for_one_opportunity():
    m = machine()
    o = opp(m)
    m.mark_submitting(o); m.on_probe_ack(o, "X1", 1_000.0)
    m.on_cancel_requested(o, 1_060.0); m.on_cancel_confirmed(o, 1_060.5, 0)
    assert m.try_commit(o) is not None
    assert m.try_commit(o) is None


def test_ioc_partial_fill_is_accepted_and_recorded():
    m = machine()
    o = opp(m)
    m.mark_submitting(o); m.on_probe_ack(o, "X1", 1_000.0)
    m.on_cancel_requested(o, 1_060.0); m.on_cancel_confirmed(o, 1_060.5, 0)
    m.try_commit(o)
    m.on_ioc_result(o, filled_qty=2, price_micros=700_000)
    assert m.state(o) is S.IOC_COMPLETE
    assert m.record(o).ioc_filled == 2


def test_ioc_zero_fill_completes_without_retry():
    m = machine()
    o = opp(m)
    m.mark_submitting(o); m.on_probe_ack(o, "X1", 1_000.0)
    m.on_cancel_requested(o, 1_060.0); m.on_cancel_confirmed(o, 1_060.5, 0)
    m.try_commit(o)
    m.on_ioc_result(o, filled_qty=0, price_micros=700_000)
    assert m.state(o) is S.IOC_COMPLETE
    assert m.try_commit(o) is None          # never retry, never chase


def test_ioc_fee_rounding_matches_the_exchange():
    """ceil to 4dp of 0.07 * C * p * (1-p), in micros."""
    assert ioc_fee_micros(700_000, 1) == 14_700
    assert ioc_fee_micros(700_000, 15) == 220_500
    assert ioc_fee_micros(500_000, 1) == 17_500
    for q in (1, 5, 15, 20):
        for p in (600_000, 650_000, 700_000, 800_000):
            assert ioc_fee_micros(p, q) >= 0
            assert isinstance(ioc_fee_micros(p, q), int)


# ------------------------------------------------------------------ sizing
def test_commit_quantity_respects_window_and_total_budgets():
    for eq in range(300, 5000, 137):
        for px in range(600_000, 800_001, 25_000):
            q = commit_quantity(eq * M, px, FROZEN)
            if q:
                assert q * px <= FROZEN.window_risk_frac * eq * M + 1
                assert q <= FROZEN.max_commit_qty


def test_quantity_is_monotone_in_equity_and_price():
    prev = -1
    for eq in range(200, 4000, 25):
        q = commit_quantity(eq * M, 700_000, FROZEN)
        assert q >= prev
        prev = q
    prev = 10 ** 9
    for px in range(600_000, 800_001, 5_000):
        q = commit_quantity(2000 * M, px, FROZEN)
        assert q <= prev
        prev = q


def test_quantity_is_zero_below_the_floor():
    assert commit_quantity(100 * M, 700_000, FROZEN) == 0


def test_quantity_never_hardcodes_a_constant():
    """Size is derived from equity. Below ~$233 the 6% window budget binds and
    the answer varies; above it the 20-contract capacity cap binds and the
    answer is deliberately constant. Both regimes must be reachable."""
    budget_bound = [commit_quantity(e * M, 700_000, FROZEN)
                    for e in (215, 220, 225, 230)]
    assert len(set(budget_bound)) > 1          # varies where the budget binds
    assert budget_bound == sorted(budget_bound)
    cap_bound = [commit_quantity(e * M, 700_000, FROZEN)
                 for e in (400, 4000, 40000)]
    assert set(cap_bound) == {FROZEN.max_commit_qty}


def test_capacity_cap_binds_above_the_documented_threshold():
    """Where the two regimes meet: 0.06*E/p == max_qty at E = 20*0.70/0.06."""
    threshold = FROZEN.max_commit_qty * 0.70 / FROZEN.window_risk_frac
    assert commit_quantity(int((threshold - 20) * M), 700_000, FROZEN) < \
        FROZEN.max_commit_qty
    assert commit_quantity(int((threshold + 20) * M), 700_000, FROZEN) == \
        FROZEN.max_commit_qty


def test_live_account_equity_sizes_to_zero():
    """$136.27 is below the $211 floor; PTC must not size a commitment."""
    assert commit_quantity(136_270_000, 700_000, FROZEN) == 0


# ------------------------------------------------------------------- KILL
def test_kill_produces_zero_new_orders():
    m = machine(kill=True)
    o = opp(m)
    assert m.state(o) is S.KILLED
    assert m.build_probe(o) is None
    assert m.try_commit(o) is None


def test_disabled_by_default_produces_no_orders():
    m = PtcMachine(PtcConfig(enabled=False, live_enabled=False,
                             wait_seconds=60, commit_qty=5,
                             equity_micros=500 * M))
    o = opp(m)
    assert m.build_probe(o) is None


def test_side_integrity_failure_forces_kill():
    m = machine()
    m.report_integrity_failure("side mapping disagrees with fills table")
    o = opp(m)
    assert m.state(o) is S.KILLED
    assert m.build_probe(o) is None


def test_timestamp_integrity_failure_forces_kill():
    m = machine()
    m.report_integrity_failure("order-to-close outside 0-900s")
    assert m.killed


def test_api_error_fails_closed_never_optimistic():
    m = machine()
    o = opp(m)
    m.mark_submitting(o)
    m.on_api_error(o, "network timeout after POST")
    assert m.state(o) is S.ERROR
    assert not m.commit_allowed(o)
    assert m.record(o).needs_reconciliation


# ------------------------------------------------------------ denominator
def test_every_outcome_stays_in_the_denominator():
    m = machine()
    a = m.register("A|1", "KXA15M-X-00", 1786000800, "BTC", "yes", 690_000, 700_000)
    b = m.register("B|1", "KXB15M-X-00", 1786000800, "ETH", "yes", 690_000, 700_000)
    c = m.register("C|1", "KXC15M-X-00", 1786000800, "SOL", "yes", 690_000, 900_000)
    m.mark_blocked(b, "risk")
    m.mark_submitting(a); m.on_api_error(a, "boom")
    m.refresh_book(c, ask_micros=900_000)
    assert m.denominator() == 3          # skipped, blocked and errored all count


# --------------------------------------------------------------- restart
@pytest.mark.parametrize("upto", [
    S.TREATMENT_ASSIGNED, S.PROBE_SUBMITTING, S.PROBE_RESTING,
    S.CANCEL_REQUESTED, S.CANCEL_CONFIRMED, S.COMMIT_ELIGIBLE,
])
def test_restart_preserves_state_and_assignment(upto):
    m = machine()
    o = opp(m)
    if upto >= S.PROBE_SUBMITTING:
        m.mark_submitting(o)
    if upto >= S.PROBE_RESTING:
        m.on_probe_ack(o, "X1", 1_000.0)
    if upto >= S.CANCEL_REQUESTED:
        m.on_cancel_requested(o, 1_060.0)
    if upto >= S.CANCEL_CONFIRMED:
        m.on_cancel_confirmed(o, 1_060.5, filled_qty=0)
    snap = m.snapshot()
    m2 = PtcMachine(cfg())
    m2.restore(snap)
    assert m2.state(o) == m.state(o)
    assert m2.client_order_id(o, "probe") == m.client_order_id(o, "probe")


def test_restart_after_submit_before_persist_reconciles_not_resubmits():
    """Process death between POST and local write must not double-submit."""
    m = machine()
    o = opp(m)
    m.mark_submitting(o)
    snap = m.snapshot()                     # died here; ack never recorded
    m2 = PtcMachine(cfg())
    m2.restore(snap)
    assert m2.state(o) is S.PROBE_SUBMITTING
    assert m2.needs_startup_reconciliation()
    # the exchange did accept it; reconciliation adopts it by client_order_id
    m2.adopt_from_exchange(o, order_id="X1", exchange_created_ts=1_000.0,
                           filled_qty=0)
    assert m2.state(o) is S.PROBE_RESTING
    assert m2.build_probe(o) is None        # never a second probe


def test_duplicate_exchange_response_is_idempotent():
    m = machine()
    o = opp(m)
    m.mark_submitting(o)
    m.on_probe_ack(o, "X1", 1_000.0)
    m.on_probe_ack(o, "X1", 1_000.0)        # duplicate delivery
    assert m.record(o).probe_submitted == 1


def test_out_of_order_fills_do_not_double_count():
    m = machine()
    o = opp(m)
    m.mark_submitting(o); m.on_probe_ack(o, "X1", 1_000.0)
    m.on_fill(o, qty=1, price_micros=690_000, ts=1_010.0, fill_id="F1")
    m.on_fill(o, qty=1, price_micros=690_000, ts=1_005.0, fill_id="F1")
    assert m.record(o).probe_filled == 1


def test_duplicate_runner_is_refused():
    m = machine()
    m.report_integrity_failure("duplicate runner detected")
    assert m.killed


# -------------------------------------------------------------- accounting
def test_money_identities_reconcile_exactly():
    m = machine(commit_qty=5)
    o = opp(m)
    m.mark_submitting(o); m.on_probe_ack(o, "X1", 1_000.0)
    m.on_cancel_requested(o, 1_060.0); m.on_cancel_confirmed(o, 1_060.5, 0)
    m.try_commit(o)
    m.on_ioc_result(o, filled_qty=5, price_micros=700_000)
    r = m.record(o)
    expected = 5 * 700_000 + ioc_fee_micros(700_000, 5)
    assert r.cash_out_micros == expected
    assert isinstance(r.cash_out_micros, int)


def test_money_is_never_binary_float():
    m = machine()
    o = opp(m)
    with pytest.raises((TypeError, ValueError)):
        m.refresh_book(o, ask_micros=0.70)
