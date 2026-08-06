"""Current Kalshi incentive audit for CRYPTO15M.

This study checks whether live volume or liquidity rewards can change the
negative fill-corrected economics of the broad 15-minute crypto maker strategy.
It uses only public Kalshi endpoints and never places orders.

Why this matters
----------------
The unsampled population estimate is approximately -0.43 cents per submitted
contract after maker fill selection. A reward paid for resting liquidity or
eligible volume is a separate cash flow that earlier backtests did not include.
This script first establishes whether any CRYPTO15M market has an active or
upcoming program, then calculates conservative reward break-even bounds.

The liquidity score share is not historically observable because participant
identity and future snapshots are private. Therefore the script reports:
- exact program terms from GET /incentive_programs;
- exact current aggregate books;
- a conservative target-size share bound;
- a current-book share estimate;
- reward required to offset the measured trading leak.

A positive point estimate is not a deployment PASS. It only justifies a
prospective score-and-reward capture study.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "results"
OUT.mkdir(parents=True, exist_ok=True)

BASE = "https://external-api.kalshi.com/trade-api/v2"
SERIES = {
    "KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M",
    "KXDOGE15M", "KXHYPE15M", "KXBNB15M", "KXSHIB15M",
}
QTY_GRID = [1, 5, 10, 15, 20, 30, 50, 100]
BASE_EV_SUBMITTED_DOLLARS = -0.0043  # -0.43c, unsampled population estimate
VOLUME_REWARD_CAP_DOLLARS = 0.005   # official program cap: $0.005 / contract
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "m15-incentive-research/1.0"})


def get_json(path: str, params: dict | None = None, retries: int = 4) -> dict:
    url = BASE + path
    last: Exception | None = None
    for attempt in range(retries):
        try:
            response = SESSION.get(url, params=params, timeout=25)
            response.raise_for_status()
            value = response.json()
            if not isinstance(value, dict):
                raise RuntimeError(f"non-object response from {response.url}")
            return value
        except Exception as exc:  # public read; bounded retry is safe
            last = exc
            if attempt + 1 < retries:
                time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


def get_incentives(status: str) -> list[dict]:
    programs: list[dict] = []
    cursor = ""
    seen: set[str] = set()
    while True:
        params: dict[str, Any] = {
            "status": status, "type": "all", "limit": 10000,
        }
        if cursor:
            params["cursor"] = cursor
        body = get_json("/incentive_programs", params)
        batch = body.get("incentive_programs") or []
        if not isinstance(batch, list):
            raise RuntimeError("incentive_programs is not a list")
        programs.extend(item for item in batch if isinstance(item, dict))
        nxt = str(body.get("next_cursor") or body.get("cursor") or "")
        if not nxt or nxt in seen or not batch:
            break
        seen.add(nxt)
        cursor = nxt
    return programs


def series_for(program: dict) -> str:
    explicit = str(program.get("series_ticker") or "").upper()
    if explicit:
        return explicit
    ticker = str(program.get("market_ticker") or "").upper()
    for series in sorted(SERIES, key=len, reverse=True):
        if ticker.startswith(series + "-") or ticker == series:
            return series
    return ""


def parse_time(value: Any) -> pd.Timestamp | pd.NaT:
    if value in (None, ""):
        return pd.NaT
    return pd.to_datetime(value, utc=True, errors="coerce")


def parse_orderbook(body: dict) -> dict[str, list[tuple[float, float]]]:
    book = body.get("orderbook_fp") or body.get("orderbook") or {}
    output: dict[str, list[tuple[float, float]]] = {"yes": [], "no": []}
    aliases = {
        "yes": ["yes_dollars", "yes"],
        "no": ["no_dollars", "no"],
    }
    for side, keys in aliases.items():
        rows = None
        for key in keys:
            if key in book:
                rows = book.get(key)
                break
        for row in rows or []:
            try:
                if isinstance(row, dict):
                    price = float(row.get("price_dollars", row.get("price")))
                    size = float(row.get("count_fp", row.get("quantity", row.get("count"))))
                else:
                    price = float(row[0])
                    size = float(row[1])
            except (TypeError, ValueError, IndexError):
                continue
            if 0 < price < 1 and size > 0:
                output[side].append((price, size))
        output[side].sort(reverse=True)
    return output


def approximate_displayed_score(book: dict[str, list[tuple[float, float]]],
                                  target_size: float,
                                  discount_bps: float) -> tuple[float, float]:
    """Approximate current aggregate score using published scoring concepts.

    Exact exchange scoring may include details not exposed by the help article.
    We treat best bid on each side as full credit and apply an exponential
    per-cent distance penalty using the published discount factor. Orders only
    contribute until target size on each side. The result is explicitly a
    sensitivity estimate, never an authoritative payout calculation.
    """
    if target_size <= 0:
        return 0.0, 0.0
    discount = min(max(discount_bps / 10_000.0, 0.0), 1.0)
    total_score = 0.0
    raw_size = 0.0
    for side in ["yes", "no"]:
        levels = book.get(side) or []
        if not levels:
            continue
        best = levels[0][0]
        remaining = target_size
        for price, size in levels:
            if remaining <= 0:
                break
            qualifying = min(size, remaining)
            cents_away = max(round((best - price) * 100), 0)
            multiplier = (1.0 - discount) ** cents_away
            total_score += qualifying * multiplier
            raw_size += qualifying
            remaining -= qualifying
    return total_score, raw_size


@dataclass
class MarketProgram:
    status: str
    program_id: str
    incentive_type: str
    description: str
    series_ticker: str
    market_ticker: str
    start_date: str
    end_date: str
    period_hours: float
    period_reward: float
    target_size: float
    discount_factor_bps: float
    paid_out: bool
    market_status: str
    close_time: str
    yes_best_bid: float | None
    no_best_bid: float | None
    displayed_score_proxy: float
    displayed_target_size: float


def enrich(program: dict, status: str) -> MarketProgram:
    ticker = str(program.get("market_ticker") or "")
    market_body: dict = {}
    orderbook: dict[str, list[tuple[float, float]]] = {"yes": [], "no": []}
    if ticker:
        try:
            market_body = get_json(f"/markets/{ticker}").get("market") or {}
        except Exception:
            market_body = {}
        try:
            orderbook = parse_orderbook(get_json(f"/markets/{ticker}/orderbook", {"depth": 100}))
        except Exception:
            orderbook = {"yes": [], "no": []}
    start = parse_time(program.get("start_date"))
    end = parse_time(program.get("end_date"))
    hours = float((end - start).total_seconds() / 3600.0) if pd.notna(start) and pd.notna(end) else np.nan
    reward = float(program.get("period_reward") or 0.0)
    target = float(program.get("target_size_fp") or program.get("target_size") or 0.0)
    discount = float(program.get("discount_factor_bps") or 0.0)
    score, raw = approximate_displayed_score(orderbook, target, discount)
    return MarketProgram(
        status=status,
        program_id=str(program.get("id") or ""),
        incentive_type=str(program.get("incentive_type") or ""),
        description=str(program.get("incentive_description") or ""),
        series_ticker=series_for(program),
        market_ticker=ticker,
        start_date=str(program.get("start_date") or ""),
        end_date=str(program.get("end_date") or ""),
        period_hours=hours,
        period_reward=reward,
        target_size=target,
        discount_factor_bps=discount,
        paid_out=bool(program.get("paid_out", False)),
        market_status=str(market_body.get("status") or ""),
        close_time=str(market_body.get("close_time") or ""),
        yes_best_bid=(orderbook["yes"][0][0] if orderbook["yes"] else None),
        no_best_bid=(orderbook["no"][0][0] if orderbook["no"] else None),
        displayed_score_proxy=score,
        displayed_target_size=raw,
    )


def liquidity_bounds(row: MarketProgram, qty: float) -> dict:
    if row.period_reward <= 0 or not np.isfinite(row.period_hours) or row.period_hours <= 0:
        return {}
    # At best price, own score is qty. Lower share assumes both sides are fully
    # populated to target size. Current-book estimate uses the visible proxy.
    conservative_competition = max(2.0 * row.target_size, row.displayed_score_proxy, 0.0)
    current_competition = max(row.displayed_score_proxy, 0.0)
    lower_share = qty / (conservative_competition + qty) if conservative_competition + qty > 0 else 1.0
    current_share = qty / (current_competition + qty) if current_competition + qty > 0 else 1.0
    reward_per_hour_lower = row.period_reward / row.period_hours * lower_share
    reward_per_hour_current = row.period_reward / row.period_hours * current_share
    # Expected trading leak is per contract submitted. Convert it to break-even
    # resting time if every submitted contract ultimately realizes that leak.
    leak_total = -BASE_EV_SUBMITTED_DOLLARS * qty
    break_even_seconds_lower = (
        leak_total / reward_per_hour_lower * 3600.0 if reward_per_hour_lower > 0 else np.inf)
    break_even_seconds_current = (
        leak_total / reward_per_hour_current * 3600.0 if reward_per_hour_current > 0 else np.inf)
    return {
        "qty": qty,
        "lower_score_share": lower_share,
        "current_score_share_proxy": current_share,
        "reward_per_hour_lower": reward_per_hour_lower,
        "reward_per_hour_current_proxy": reward_per_hour_current,
        "trading_leak_if_filled": leak_total,
        "break_even_rest_seconds_lower": break_even_seconds_lower,
        "break_even_rest_seconds_current_proxy": break_even_seconds_current,
    }


def main() -> None:
    fetched_at = datetime.now(timezone.utc).isoformat()
    raw: dict[str, list[dict]] = {}
    for status in ["active", "upcoming"]:
        raw[status] = get_incentives(status)
    (OUT / "incentive_programs_raw.json").write_text(
        json.dumps({"fetched_at": fetched_at, **raw}, indent=2), encoding="utf-8")

    all_rows: list[MarketProgram] = []
    for status, programs in raw.items():
        for program in programs:
            if series_for(program) in SERIES:
                all_rows.append(enrich(program, status))
    frame = pd.DataFrame([asdict(row) for row in all_rows])
    if frame.empty:
        frame = pd.DataFrame(columns=[field.name for field in MarketProgram.__dataclass_fields__.values()])
    frame.to_csv(OUT / "crypto15m_incentive_programs.csv", index=False)

    bounds_rows = []
    for row in all_rows:
        if row.incentive_type.lower() != "liquidity":
            continue
        for qty in QTY_GRID:
            values = liquidity_bounds(row, qty)
            if values:
                bounds_rows.append({**asdict(row), **values})
    bounds = pd.DataFrame(bounds_rows)
    bounds.to_csv(OUT / "crypto15m_liquidity_reward_bounds.csv", index=False)

    active = [row for row in all_rows if row.status == "active"]
    active_liquidity = [row for row in active if row.incentive_type.lower() == "liquidity"]
    active_volume = [row for row in active if row.incentive_type.lower() == "volume"]
    max_volume_adjusted_ev = BASE_EV_SUBMITTED_DOLLARS + VOLUME_REWARD_CAP_DOLLARS
    volume_can_theoretically_offset = max_volume_adjusted_ev > 0

    summary = {
        "fetched_at": fetched_at,
        "program_counts_all_exchange": {key: len(value) for key, value in raw.items()},
        "crypto15m_programs": len(all_rows),
        "crypto15m_active": len(active),
        "active_liquidity": len(active_liquidity),
        "active_volume": len(active_volume),
        "series_with_active_program": sorted({row.series_ticker for row in active}),
        "base_fill_corrected_ev_per_submitted_contract_dollars": BASE_EV_SUBMITTED_DOLLARS,
        "official_volume_reward_cap_per_contract_dollars": VOLUME_REWARD_CAP_DOLLARS,
        "best_case_volume_adjusted_ev_dollars": max_volume_adjusted_ev,
        "volume_cap_can_theoretically_offset_base_leak": volume_can_theoretically_offset,
        "liquidity_bounds_generated": len(bounds_rows),
    }
    (OUT / "incentive_adjusted_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    if not active:
        verdict = "FAIL — no active CRYPTO15M incentive period"
    elif active_liquidity:
        verdict = "CANDIDATE — active liquidity rewards require prospective score-share capture"
    elif active_volume:
        verdict = "CANDIDATE — active volume rewards are capped and require realized payout confirmation"
    else:
        verdict = "FAIL — active program type does not match liquidity or volume"

    lines = [
        "# Incentive-adjusted CRYPTO15M audit", "",
        f"## Verdict: **{verdict}**", "",
        f"Snapshot UTC: `{fetched_at}`", "",
        "## Current program inventory", "",
        f"- Active programs exchange-wide: {len(raw['active'])}",
        f"- Upcoming programs exchange-wide: {len(raw['upcoming'])}",
        f"- Active/upcoming CRYPTO15M programs: {len(all_rows)}",
        f"- Active CRYPTO15M liquidity programs: {len(active_liquidity)}",
        f"- Active CRYPTO15M volume programs: {len(active_volume)}",
        f"- Active series: {', '.join(summary['series_with_active_program']) or 'none'}", "",
        "## Economic hurdle", "",
        "The current unsampled population estimate is approximately **−0.43¢ per",
        "submitted contract** after maker fill selection. The general volume",
        "program caps rewards at **0.50¢ per eligible contract**, so the absolute",
        "best case would be approximately **+0.07¢**. That upper bound is not an",
        "expected payout; proportional pool competition can make the actual reward",
        "much smaller.", "",
    ]
    if active:
        lines.extend([
            "## Active CRYPTO15M programs", "",
            "| Type | Series | Market | Reward | Target | Start | End |",
            "|---|---|---|---:|---:|---|---|",
        ])
        for row in active:
            lines.append(
                f"| {row.incentive_type} | {row.series_ticker} | {row.market_ticker} "
                f"| {row.period_reward:.2f} | {row.target_size:.2f} | "
                f"{row.start_date} | {row.end_date} |")
    else:
        lines.extend([
            "No active CRYPTO15M market was returned by the public incentive API.",
            "Rewards therefore cannot repair the present strategy economics at this",
            "snapshot. Upcoming schedules remain in the attached CSV/JSON evidence.",
        ])
    lines.extend(["", "## Limits", "",
        "Liquidity payout depends on the account's share of random one-second",
        "snapshots. Current aggregate order books can only bound that share; they",
        "cannot identify future participant scores. A live candidate requires a",
        "prospective recorder that captures program terms, full books, own resting",
        "orders, actual reward credits, and fills without self-trading or artificial",
        "volume.", "",
        "No result authorizes changing the repository's existing KILL state."])
    (OUT / "incentive_adjusted_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
