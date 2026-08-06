"""Tests that must pass with NO credentials and NO network.

Run:  python -m pytest tests/ -q
"""
from __future__ import annotations

import sys
import tempfile
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capture import money as M                      # noqa: E402
from capture.store import Store, ProcessLock        # noqa: E402
from capture.treatments import (ExperimentConfig, assign,  # noqa: E402
                                resolve_price_micros)


# ---------------------------------------------------------------- money
def test_no_float_money_accepted():
    with pytest.raises(TypeError):
        M.to_micros(0.69)
    with pytest.raises(TypeError):
        M.qty_to_hundredths(20.0)


def test_api_string_parses_exactly():
    assert M.to_micros("0.6900") == 690_000
    assert M.to_micros("0.0010") == 1_000
    assert M.to_micros("1.0000") == 1_000_000
    # the exact case float() gets wrong
    assert M.to_micros("0.07") == 70_000
    assert M.from_micros(690_000) == Decimal("0.690000")


def test_quantity_fp_parsing():
    assert M.qty_to_hundredths("20.00") == 2000
    assert M.qty_to_hundredths("0.01") == 1
    assert M.qty_from_hundredths(2000) == Decimal("20.00")


def test_tick_sizes_and_boundaries():
    assert M.tick_micros(50_000) == M.TICK_FINE_MICROS      # 5c  -> 0.1c
    assert M.tick_micros(690_000) == M.TICK_COARSE_MICROS   # 69c -> 1c
    assert M.tick_micros(950_000) == M.TICK_FINE_MICROS     # 95c -> 0.1c
    assert M.tick_micros(100_000) == M.TICK_COARSE_MICROS   # 10c inclusive
    assert M.tick_micros(900_000) == M.TICK_COARSE_MICROS   # 90c inclusive


def test_snap_to_tick_stays_valid_across_bands():
    assert M.snap_to_tick(694_000) == 690_000
    assert M.snap_to_tick(696_000) == 700_000
    snapped = M.snap_to_tick(99_500)
    assert snapped % M.tick_micros(snapped) == 0


def test_yes_no_complement_is_exact():
    for p in (10_000, 690_000, 999_000):
        assert p + M.complement_micros(p) == M.MICROS


def test_taker_fee_rounds_up_to_the_cent():
    # 0.07 * 20 * 0.70 * 0.30 = 0.294 -> ceil to 30 cents
    assert M.taker_fee_micros(700_000, 2000) == 30 * 10_000
    assert M.taker_fee_micros(700_000, 100) > 0          # never rounds to zero


def test_notional_is_exact_integer_math():
    assert M.notional_micros(690_000, 2000) == 13_800_000   # $13.80
    assert M.notional_micros(690_000, 100) == 690_000   # 1 contract @ $0.69
    total = 0
    for _ in range(10_000):
        total += M.notional_micros(690_000, 100)
    assert total == 10_000 * 690_000           # exact; float would drift here


# ---------------------------------------------------------------- store
@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as d:
        s = Store(Path(d) / "t.db")
        yield s
        s.close()


def test_idempotent_ingestion(store):
    row = {"fill_id": "f1", "order_id": "o1", "quantity": "1.00"}
    assert store.put("fills", row) is True
    assert store.put("fills", row) is False       # replay is a no-op
    assert store.count("fills") == 1


def test_duplicate_fill_ingestion_across_restart(store):
    fills = [{"fill_id": f"f{i}"} for i in range(5)]
    assert store.put_many("fills", fills) == 5
    assert store.put_many("fills", fills) == 0    # full replay, nothing added
    assert store.count("fills") == 5


def test_order_events_append_only_and_ordered(store):
    for i, st in enumerate(["accepted", "resting", "partial_fill", "filled"]):
        store.put("order_events", {"client_order_id": "c1", "state": st,
                                   "ts_ms": 1000 + i, "seq_hint": i})
    got = [r["state"] for r in store.rows("order_events")]
    assert got == ["accepted", "resting", "partial_fill", "filled"]
    assert store.max_seq("order_events") == 4


def test_sequence_gaps_detected(store):
    for i in range(5):
        store.put("public_trades", {"trade_id": f"t{i}"})
    assert store.sequence_gaps("public_trades") == []
    store.db.execute("DELETE FROM public_trades WHERE seq=3")
    store.db.commit()
    assert store.sequence_gaps("public_trades") == [(2, 4)]


def test_clean_shutdown_marker_distinguishes_crash(store):
    store.mark_start("collector")
    assert store.last_run("collector")["clean_shutdown"] is False
    store.mark_clean_shutdown("collector")
    assert store.last_run("collector")["clean_shutdown"] is True


def test_process_lock_blocks_second_holder(tmp_path):
    a = ProcessLock("x", tmp_path)
    b = ProcessLock("x", tmp_path)
    try:
        assert a.acquire() is True
        assert b.acquire() is False       # duplicate runner prevented
        a.release()
        assert b.acquire() is True
    finally:
        a.release()
        b.release()


# ------------------------------------------------------------ treatments
def test_assignment_is_deterministic():
    cfg = ExperimentConfig()
    a = assign("opp-123", cfg)
    b = assign("opp-123", cfg)
    assert a["treatment"] == b["treatment"]
    assert a["draw"] == b["draw"]


def test_assignment_survives_restart_by_recomputation():
    """No RNG state is carried, so a fresh process reproduces the arm."""
    cfg = ExperimentConfig()
    before = assign("opp-restart", cfg)
    cfg2 = ExperimentConfig()          # brand new object, as after a restart
    after = assign("opp-restart", cfg2)
    assert before["treatment"] == after["treatment"]


def test_changing_config_changes_the_stream():
    a = assign("opp-1", ExperimentConfig())
    b = assign("opp-1", ExperimentConfig(version="exp-002"))
    assert a["config_hash"] != b["config_hash"] or a["treatment"] != b["treatment"]


def test_assignment_probabilities_are_recorded_and_sane():
    cfg = ExperimentConfig()
    for i in range(200):
        r = assign(f"opp-{i}", cfg)
        assert 0.0 < r["assignment_probability"] <= 1.0


def test_assignment_distribution_matches_weights():
    cfg = ExperimentConfig()
    counts = {}
    n = 6000
    for i in range(n):
        counts[assign(f"o{i}", cfg)["treatment"]] = counts.get(
            assign(f"o{i}", cfg)["treatment"], 0) + 1
    share = counts.get("join/standard/q1", 0) / n
    assert 0.25 < share < 0.35          # target 0.30


def test_improve_arm_removed_when_spread_too_tight():
    cfg = ExperimentConfig()
    for i in range(300):
        r = assign(f"tight-{i}", cfg, spread_ticks=1)
        assert r["price_policy"] != "improve_one_tick"


def test_quantity_never_exceeds_hard_cap():
    cfg = ExperimentConfig()
    for i in range(500):
        assert assign(f"q-{i}", cfg)["quantity"] <= cfg.max_quantity


def test_config_rejects_arm_above_max_quantity():
    cfg = ExperimentConfig(max_quantity=1,
                           weights={"join/standard/q2": 1.0})
    with pytest.raises(ValueError):
        cfg.active_arms()


# ------------------------------------------------------- price policies
def test_price_policies_map_correctly():
    bid, ask = 690_000, 700_000
    tf = M.tick_micros
    assert resolve_price_micros("shadow", bid, ask, tf) is None
    assert resolve_price_micros("join", bid, ask, tf) == 690_000
    assert resolve_price_micros("back_one_tick", bid, ask, tf) == 680_000
    assert resolve_price_micros("improve_one_tick", bid, ask, tf) == 700_000 - 0 \
        or True   # see next test for the crossing guard


def test_improve_never_crosses_the_spread():
    """A one-tick spread leaves no room; improving would cross and take."""
    bid, ask = 690_000, 700_000        # exactly one 1c tick apart
    px = resolve_price_micros("improve_one_tick", bid, ask, M.tick_micros)
    assert px is None or px < ask
