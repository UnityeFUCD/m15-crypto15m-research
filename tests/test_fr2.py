"""Tests for the frozen FR2 historical candidate.

These tests validate rule semantics only. They do not establish profitability
or permit live trading.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fr2_reproduce", ROOT / "research" / "fr2_reproduce.py"
)
assert SPEC and SPEC.loader
FR2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FR2)


def candle(ml: float, bid: float, ask: float) -> dict:
    return {"ml": ml, "bc": bid, "bh": bid, "ac": ask, "al": ask}


def row(
    *,
    ticker: str,
    coin: str,
    result: str,
    entry_bid: float,
    entry_ask: float,
    delayed_bid: float,
    delayed_ask: float,
    include_delay: bool = True,
) -> dict:
    path = [candle(14.0, entry_bid, entry_ask)]
    if include_delay:
        path.append(candle(12.0, delayed_bid, delayed_ask))
    return {
        "ticker": ticker,
        "coin": coin,
        "date": "2026-05-25",
        "result": result,
        "path": json.dumps(path),
    }


def run(tmp_path: Path, rows: list[dict]) -> pd.DataFrame:
    source = tmp_path / "paths.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    return FR2.build_candidates(source)


def test_fee_is_rounded_once_for_the_block() -> None:
    assert FR2.taker_fee(20, 0.70) == pytest.approx(0.2940)


def test_yes_favorite_uses_delayed_yes_ask(tmp_path: Path) -> None:
    data = run(
        tmp_path,
        [
            row(
                ticker="KXBTC15M-26MAY242000-00",
                coin="BTC",
                result="yes",
                entry_bid=0.70,
                entry_ask=0.72,
                delayed_bid=0.73,
                delayed_ask=0.75,
            )
        ],
    )
    item = data.iloc[0]
    assert item.side == "yes"
    assert item.initial_bid == pytest.approx(0.70)
    assert item.delayed_ask == pytest.approx(0.75)
    assert bool(item.selected)


def test_no_favorite_uses_complement_of_delayed_yes_bid(tmp_path: Path) -> None:
    data = run(
        tmp_path,
        [
            row(
                ticker="KXETH15M-26MAY242000-00",
                coin="ETH",
                result="no",
                entry_bid=0.28,
                entry_ask=0.30,
                delayed_bid=0.27,
                delayed_ask=0.29,
            )
        ],
    )
    item = data.iloc[0]
    assert item.side == "no"
    assert item.initial_bid == pytest.approx(0.70)
    assert item.delayed_ask == pytest.approx(0.73)
    assert bool(item.selected)


def test_original_side_is_retained_after_book_flip(tmp_path: Path) -> None:
    data = run(
        tmp_path,
        [
            row(
                ticker="KXSOL15M-26MAY242000-00",
                coin="SOL",
                result="yes",
                entry_bid=0.70,
                entry_ask=0.72,
                delayed_bid=0.28,
                delayed_ask=0.72,
            )
        ],
    )
    item = data.iloc[0]
    assert item.side == "yes"
    assert item.delayed_ask == pytest.approx(0.72)
    assert bool(item.selected)


@pytest.mark.parametrize(
    ("ask", "expected"),
    [(0.5999, False), (0.60, True), (0.80, True), (0.8001, False)],
)
def test_delayed_ask_band_is_frozen_and_inclusive(
    tmp_path: Path, ask: float, expected: bool
) -> None:
    data = run(
        tmp_path,
        [
            row(
                ticker="KXXRP15M-26MAY242000-00",
                coin="XRP",
                result="yes",
                entry_bid=0.70,
                entry_ask=0.72,
                delayed_bid=max(0.01, ask - 0.02),
                delayed_ask=ask,
            )
        ],
    )
    assert bool(data.iloc[0].selected) is expected


def test_exact_two_minute_observation_is_required(tmp_path: Path) -> None:
    data = run(
        tmp_path,
        [
            row(
                ticker="KXDOGE15M-26MAY242000-00",
                coin="DOGE",
                result="yes",
                entry_bid=0.70,
                entry_ask=0.72,
                delayed_bid=0.70,
                delayed_ask=0.72,
                include_delay=False,
            ),
            row(
                ticker="KXBTC15M-26MAY242030-30",
                coin="BTC",
                result="yes",
                entry_bid=0.70,
                entry_ask=0.72,
                delayed_bid=0.70,
                delayed_ask=0.72,
            ),
        ],
    )
    assert set(data.ticker) == {"KXBTC15M-26MAY242030-30"}


def test_max_one_market_per_close_uses_existing_series_order(
    tmp_path: Path,
) -> None:
    data = run(
        tmp_path,
        [
            row(
                ticker="KXETH15M-26MAY242000-00",
                coin="ETH",
                result="yes",
                entry_bid=0.70,
                entry_ask=0.72,
                delayed_bid=0.70,
                delayed_ask=0.72,
            ),
            row(
                ticker="KXBTC15M-26MAY242000-00",
                coin="BTC",
                result="yes",
                entry_bid=0.70,
                entry_ask=0.72,
                delayed_bid=0.70,
                delayed_ask=0.72,
            ),
        ],
    )
    selected = data[data.selected]
    assert list(selected.ticker) == ["KXBTC15M-26MAY242000-00"]


def test_initial_band_is_65_to_below_80(tmp_path: Path) -> None:
    data = run(
        tmp_path,
        [
            row(
                ticker="KXBTC15M-26MAY242000-00",
                coin="BTC",
                result="yes",
                entry_bid=0.65,
                entry_ask=0.67,
                delayed_bid=0.68,
                delayed_ask=0.70,
            ),
            row(
                ticker="KXETH15M-26MAY242030-30",
                coin="ETH",
                result="yes",
                entry_bid=0.80,
                entry_ask=0.82,
                delayed_bid=0.75,
                delayed_ask=0.77,
            ),
        ],
    )
    assert set(data.ticker) == {"KXBTC15M-26MAY242000-00"}


def test_pnl_charges_price_and_fee_at_actual_quantity(tmp_path: Path) -> None:
    data = run(
        tmp_path,
        [
            row(
                ticker="KXBTC15M-26MAY242000-00",
                coin="BTC",
                result="yes",
                entry_bid=0.70,
                entry_ask=0.72,
                delayed_bid=0.70,
                delayed_ask=0.70,
            )
        ],
    )
    item = data.iloc[0]
    expected = 20 * (1.0 - 0.70) - FR2.taker_fee(20, 0.70)
    assert item.pnl == pytest.approx(expected)
    assert item.net_edge_per_contract == pytest.approx(expected / 20)
