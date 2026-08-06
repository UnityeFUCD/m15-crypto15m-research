"""Deterministic replay and fault injection for PTC (Part 6).

DETERMINISTIC REPLAY drives the real 303-order LSM history through the state
machine twice and requires byte-identical output. Any hidden dependence on
dict ordering, wall-clock time or RNG state shows up here rather than in
production.

FAULT INJECTION asserts the invariant that matters: no fault may produce
duplicate or excess exposure. Each scenario is a real failure this
architecture must survive, not a hypothetical.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from capture.ptc import (                                    # noqa: E402
    PtcConfig, PtcMachine, S, held_side, to_epoch_seconds,
)

M = 1_000_000
DATA = ROOT / "data"


def cfg(**kw):
    base = dict(enabled=True, live_enabled=False, wait_seconds=60,
                commit_qty=5, equity_micros=500 * M)
    base.update(kw)
    return PtcConfig(**base)


# ------------------------------------------------------------------ replay
def _load_history():
    pd = pytest.importorskip("pandas")
    f = DATA / "orders_history.parquet"
    if not f.exists():
        pytest.skip("orders_history.parquet not present")
    o = pd.read_parquet(f)
    lsm = o[o.client_order_id.astype(str).str.startswith("lsm")].copy()
    lsm = lsm.sort_values(["ticker", "order_id"]).reset_index(drop=True)
    return lsm


def _drive(lsm) -> str:
    """Run every historical order through the machine; return a digest."""
    import pandas as pd
    m = PtcMachine(cfg())
    for i, r in enumerate(lsm.itertuples()):
        side = held_side(r.action, r.side)
        close = (pd.to_datetime(r.ticker.split("-")[1], format="%y%b%d%H%M",
                                utc=True) + pd.Timedelta(hours=4))
        cts = int(close.timestamp())
        oid = m.register(f"{r.ticker}|{r.order_id}", r.ticker, cts,
                         r.ticker[2:5], side, 690_000, 700_000)
        m.mark_submitting(oid)
        ack = float(pd.Timestamp(r.created_time).timestamp())
        m.on_probe_ack(oid, str(r.order_id), ack)
        filled = float(r.fill_count_fp or 0) > 0
        if filled:
            m.on_fill(oid, 1, 690_000, ack + 1.0, fill_id=f"F{i}")
        else:
            m.on_cancel_requested(oid, ack + 60.0)
            m.on_cancel_confirmed(oid, ack + 60.4, filled_qty=0)
            if m.try_commit(oid):
                m.on_ioc_result(oid, 5, 700_000)
    snap = m.snapshot()
    blob = json.dumps(snap, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def test_replay_of_the_full_history_is_deterministic():
    lsm = _load_history()
    assert len(lsm) == 303
    assert _drive(lsm) == _drive(lsm)


def test_replay_respects_one_commitment_per_close_window():
    lsm = _load_history()
    import pandas as pd
    m = PtcMachine(cfg())
    per_window: dict[int, int] = {}
    for i, r in enumerate(lsm.itertuples()):
        close = (pd.to_datetime(r.ticker.split("-")[1], format="%y%b%d%H%M",
                                utc=True) + pd.Timedelta(hours=4))
        cts = int(close.timestamp())
        oid = m.register(f"{r.ticker}|{r.order_id}", r.ticker, cts,
                         r.ticker[2:5], held_side(r.action, r.side),
                         690_000, 700_000)
        m.mark_submitting(oid)
        m.on_probe_ack(oid, str(r.order_id), 1_000.0 + i)
        m.on_cancel_requested(oid, 1_060.0 + i)
        m.on_cancel_confirmed(oid, 1_060.5 + i, filled_qty=0)
        if m.try_commit(oid):
            per_window[cts] = per_window.get(cts, 0) + 1
    assert per_window, "no commitments produced - replay is not exercising it"
    assert max(per_window.values()) == 1


# --------------------------------------------------------- fault injection
def _resting(m, oid, ack=1_000.0):
    m.mark_submitting(oid)
    m.on_probe_ack(oid, "X1", ack)


def _opp(m, tid="KXBTC15M-26AUG051900-00", close=1786000800):
    return m.register(tid + "|1", tid, close, "BTC", "yes", 690_000, 700_000)


def test_death_after_submit_before_persist_yields_one_order_not_two():
    m = PtcMachine(cfg())
    o = _opp(m)
    m.mark_submitting(o)
    snap = m.snapshot()                       # crash here
    m2 = PtcMachine(cfg())
    m2.restore(snap)
    assert m2.needs_startup_reconciliation()
    m2.adopt_from_exchange(o, "X1", 1_000.0, filled_qty=0)
    assert m2.build_probe(o) is None
    assert m2.record(o).probe_submitted == 1


def test_death_after_cancel_request_reconciles_the_race():
    m = PtcMachine(cfg())
    o = _opp(m)
    _resting(m, o)
    m.on_cancel_requested(o, 1_060.0)
    snap = m.snapshot()                       # crash before confirmation
    m2 = PtcMachine(cfg())
    m2.restore(snap)
    assert m2.needs_startup_reconciliation()
    assert not m2.commit_allowed(o)           # fail closed until confirmed
    m2.on_fill(o, 1, 690_000, 1_060.3, fill_id="F1")
    m2.on_cancel_confirmed(o, 1_060.9, filled_qty=1)
    assert m2.state(o) is S.CANCEL_RACE_RECONCILED
    assert m2.try_commit(o) is None


def test_delayed_cancel_ack_never_authorises_a_commit_early():
    m = PtcMachine(cfg())
    o = _opp(m)
    _resting(m, o)
    m.on_cancel_requested(o, 1_060.0)
    for _ in range(50):                       # ack never arrives
        assert not m.commit_allowed(o)


def test_duplicate_exchange_responses_are_idempotent():
    m = PtcMachine(cfg())
    o = _opp(m)
    m.mark_submitting(o)
    for _ in range(5):
        m.on_probe_ack(o, "X1", 1_000.0)
    assert m.record(o).probe_submitted == 1
    m.on_cancel_requested(o, 1_060.0)
    for _ in range(5):
        m.on_cancel_confirmed(o, 1_060.5, filled_qty=0)
    assert m.try_commit(o) is not None
    assert m.try_commit(o) is None


def test_network_timeout_after_possibly_accepted_post_fails_closed():
    m = PtcMachine(cfg())
    o = _opp(m)
    m.mark_submitting(o)
    m.on_api_error(o, "timeout after POST - acceptance unknown")
    assert m.state(o) is S.ERROR
    assert not m.commit_allowed(o)
    assert m.record(o).needs_reconciliation
    assert m.needs_startup_reconciliation()


def test_stale_book_snapshot_cannot_authorise_an_out_of_band_commit():
    m = PtcMachine(cfg())
    o = _opp(m, close=1786000800)
    _resting(m, o)
    m.on_cancel_requested(o, 1_060.0)
    m.on_cancel_confirmed(o, 1_060.5, filled_qty=0)
    m.refresh_book(o, ask_micros=930_000)     # moved out of band
    assert not m.commit_allowed(o)
    assert m.try_commit(o) is None


def test_exchange_pause_blocks_and_stays_in_the_denominator():
    m = PtcMachine(cfg())
    o = _opp(m)
    m.mark_blocked(o, "exchange paused")
    assert m.state(o) is S.BLOCKED
    assert m.build_probe(o) is None
    assert m.denominator() == 1


def test_out_of_order_fills_are_deduplicated_by_fill_id():
    m = PtcMachine(cfg())
    o = _opp(m)
    _resting(m, o)
    m.on_fill(o, 1, 690_000, 1_010.0, fill_id="F1")
    m.on_fill(o, 1, 690_000, 1_002.0, fill_id="F1")
    m.on_fill(o, 1, 690_000, 1_020.0, fill_id="F1")
    assert m.record(o).probe_filled == 1
    assert m.record(o).cash_out_micros == 690_000


def test_duplicate_runner_kills_and_produces_no_orders():
    m = PtcMachine(cfg())
    o = _opp(m)
    m.report_integrity_failure("duplicate runner process detected")
    assert m.killed
    assert m.state(o) is S.KILLED
    assert m.build_probe(o) is None
    assert m.try_commit(o) is None


def test_paginated_truncation_is_detectable_as_missing_reconciliation():
    """A truncated resting-orders page leaves a submitted order unadopted.
    The machine must still report that reconciliation is outstanding."""
    m = PtcMachine(cfg())
    a = _opp(m, "KXA15M-26AUG051900-00")
    b = _opp(m, "KXB15M-26AUG051900-00")
    m.mark_submitting(a)
    m.mark_submitting(b)
    m.adopt_from_exchange(a, "X1", 1_000.0, filled_qty=0)   # page 1 only
    assert m.needs_startup_reconciliation()                # b still pending
    assert m.state(b) is S.PROBE_SUBMITTING


# ------------------------------------------------------- exposure property
@pytest.mark.parametrize("n_opportunities", [1, 3, 8, 25])
def test_no_fault_sequence_creates_excess_exposure(n_opportunities):
    """Across a hostile interleaving, total committed contracts never exceed
    one commitment per close window."""
    m = PtcMachine(cfg(commit_qty=5))
    windows = [1786000800, 1786001700]
    committed: dict[int, int] = {}
    for i in range(n_opportunities):
        w = windows[i % len(windows)]
        o = m.register(f"O|{i}", f"KX{i}15M-X-00", w, "BTC", "yes",
                       690_000, 700_000)
        m.mark_submitting(o)
        m.on_probe_ack(o, f"X{i}", 1_000.0 + i)
        m.on_probe_ack(o, f"X{i}", 1_000.0 + i)          # duplicate
        m.on_cancel_requested(o, 1_060.0 + i)
        m.on_cancel_confirmed(o, 1_060.5 + i, filled_qty=0)
        m.on_cancel_confirmed(o, 1_060.5 + i, filled_qty=0)   # duplicate
        for _ in range(3):                                # repeated attempts
            if m.try_commit(o):
                committed[w] = committed.get(w, 0) + 1
    assert all(v == 1 for v in committed.values())
    assert m.denominator() == n_opportunities
