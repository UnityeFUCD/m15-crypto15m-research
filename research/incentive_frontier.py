"""Exchange-wide Kalshi liquidity-incentive frontier.

The current CRYPTO15M markets have no active incentive program. This audit asks
whether another active market offers a materially better reward-to-risk ratio.
It never places orders and does not require credentials.

`period_reward` is returned by the API in centi-cents, so:

    reward_dollars = period_reward / 10_000

The score-share calculations are bounds/proxies. Only Kalshi observes every
participant's random one-second snapshot score. A candidate must be confirmed
prospectively from actual reward credits before it can be called an edge.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "results"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://external-api.kalshi.com/trade-api/v2"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "m15-incentive-frontier/1.0"})
QTY_GRID = [1, 5, 10, 15, 20, 30, 50]
TOP_TERMS_TO_ENRICH = 300


def get_json(path: str, params: dict | None = None, retries: int = 4) -> dict:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            response = SESSION.get(BASE + path, params=params, timeout=25)
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict):
                raise RuntimeError("response is not a JSON object")
            return body
        except Exception as exc:
            last = exc
            if attempt + 1 < retries:
                time.sleep(0.7 * (attempt + 1))
    raise RuntimeError(f"GET {path} failed: {last}")


def all_programs(status: str, kind: str) -> list[dict]:
    result: list[dict] = []
    cursor = ""
    seen: set[str] = set()
    while True:
        params: dict[str, Any] = {
            "status": status, "type": kind, "limit": 10000,
        }
        if cursor:
            params["cursor"] = cursor
        body = get_json("/incentive_programs", params)
        batch = body.get("incentive_programs") or []
        if not isinstance(batch, list):
            raise RuntimeError("incentive_programs is not a list")
        result.extend(item for item in batch if isinstance(item, dict))
        nxt = str(body.get("next_cursor") or body.get("cursor") or "")
        if not nxt or nxt in seen or not batch:
            break
        seen.add(nxt)
        cursor = nxt
    return result


def orderbook(body: dict) -> dict[str, list[tuple[float, float]]]:
    raw = body.get("orderbook_fp") or body.get("orderbook") or {}
    output: dict[str, list[tuple[float, float]]] = {"yes": [], "no": []}
    aliases = {"yes": ["yes_dollars", "yes"], "no": ["no_dollars", "no"]}
    for side, keys in aliases.items():
        levels = []
        for key in keys:
            if key in raw:
                levels = raw.get(key) or []
                break
        for row in levels:
            try:
                if isinstance(row, dict):
                    price = float(row.get("price_dollars", row.get("price")))
                    size = float(row.get("count_fp", row.get("count", row.get("quantity"))))
                else:
                    price = float(row[0])
                    size = float(row[1])
            except (TypeError, ValueError, IndexError):
                continue
            if 0 < price < 1 and size > 0:
                output[side].append((price, size))
        output[side].sort(reverse=True)
    return output


def time_value(value: Any) -> pd.Timestamp | pd.NaT:
    return pd.to_datetime(value, utc=True, errors="coerce")


def normalize(program: dict) -> dict:
    start = time_value(program.get("start_date"))
    end = time_value(program.get("end_date"))
    hours = float((end - start).total_seconds() / 3600.0) \
        if pd.notna(start) and pd.notna(end) and end > start else np.nan
    reward_raw = float(program.get("period_reward") or 0.0)
    reward_dollars = reward_raw / 10_000.0
    target = float(program.get("target_size_fp") or program.get("target_size") or 0.0)
    reward_per_day = reward_dollars / hours * 24.0 \
        if hours > 0 else np.nan
    density = reward_per_day / target if target > 0 else np.nan
    return {
        "program_id": str(program.get("id") or ""),
        "market_id": str(program.get("market_id") or ""),
        "market_ticker": str(program.get("market_ticker") or ""),
        "series_ticker": str(program.get("series_ticker") or ""),
        "description": str(program.get("incentive_description") or ""),
        "start_date": str(program.get("start_date") or ""),
        "end_date": str(program.get("end_date") or ""),
        "period_hours": hours,
        "period_reward_raw_centi_cents": reward_raw,
        "period_reward_dollars": reward_dollars,
        "reward_per_day_dollars": reward_per_day,
        "target_size": target,
        "discount_factor_bps": float(program.get("discount_factor_bps") or 0.0),
        "reward_density_per_target_contract_day": density,
    }


def displayed_score_proxy(book: dict[str, list[tuple[float, float]]],
                          target: float,
                          discount_bps: float) -> tuple[float, float]:
    if target <= 0:
        return 0.0, 0.0
    discount = min(max(discount_bps / 10_000.0, 0.0), 1.0)
    score = 0.0
    raw_size = 0.0
    for side in ["yes", "no"]:
        levels = book.get(side) or []
        if not levels:
            continue
        best = levels[0][0]
        remaining = target
        for price, size in levels:
            if remaining <= 0:
                break
            qualifying = min(size, remaining)
            cents_away = max(int(round((best - price) * 100)), 0)
            multiplier = (1.0 - discount) ** cents_away
            score += qualifying * multiplier
            raw_size += qualifying
            remaining -= qualifying
    return score, raw_size


@dataclass
class Enriched:
    program_id: str
    market_ticker: str
    series_ticker: str
    title: str
    subtitle: str
    category: str
    market_status: str
    close_time: str
    reward_dollars: float
    reward_per_day: float
    target_size: float
    discount_factor_bps: float
    yes_best_bid: float | None
    no_best_bid: float | None
    implied_spread_c: float | None
    yes_size_best: float
    no_size_best: float
    displayed_score_proxy: float
    displayed_target_size: float
    open_interest: float
    volume_24h: float


def enrich(row: dict) -> Enriched:
    ticker = row["market_ticker"]
    market: dict = {}
    book: dict[str, list[tuple[float, float]]] = {"yes": [], "no": []}
    try:
        market = get_json(f"/markets/{ticker}").get("market") or {}
    except Exception:
        pass
    try:
        book = orderbook(get_json(f"/markets/{ticker}/orderbook", {"depth": 100}))
    except Exception:
        pass
    score, raw = displayed_score_proxy(
        book, float(row["target_size"]), float(row["discount_factor_bps"]))
    yes_best = book["yes"][0][0] if book["yes"] else None
    no_best = book["no"][0][0] if book["no"] else None
    spread = (1.0 - yes_best - no_best) * 100 \
        if yes_best is not None and no_best is not None else None
    return Enriched(
        program_id=row["program_id"], market_ticker=ticker,
        series_ticker=row["series_ticker"],
        title=str(market.get("title") or ""),
        subtitle=str(market.get("subtitle") or market.get("sub_title") or ""),
        category=str(market.get("category") or ""),
        market_status=str(market.get("status") or ""),
        close_time=str(market.get("close_time") or ""),
        reward_dollars=float(row["period_reward_dollars"]),
        reward_per_day=float(row["reward_per_day_dollars"]),
        target_size=float(row["target_size"]),
        discount_factor_bps=float(row["discount_factor_bps"]),
        yes_best_bid=yes_best, no_best_bid=no_best,
        implied_spread_c=spread,
        yes_size_best=float(book["yes"][0][1]) if book["yes"] else 0.0,
        no_size_best=float(book["no"][0][1]) if book["no"] else 0.0,
        displayed_score_proxy=score, displayed_target_size=raw,
        open_interest=float(market.get("open_interest_fp") or market.get("open_interest") or 0.0),
        volume_24h=float(market.get("volume_24h_fp") or market.get("volume_24h") or 0.0),
    )


def quote_bounds(market: Enriched, qty: float) -> dict:
    own_score = 2.0 * qty  # q at best bid on both YES and NO
    target_competition = max(2.0 * market.target_size, 0.0)
    current_competition = max(market.displayed_score_proxy, 0.0)
    share_lower = own_score / (target_competition + own_score) \
        if target_competition + own_score > 0 else 1.0
    share_current = own_score / (current_competition + own_score) \
        if current_competition + own_score > 0 else 1.0
    reward_day_lower = market.reward_per_day * share_lower
    reward_day_current = market.reward_per_day * share_current
    if market.yes_best_bid is None or market.no_best_bid is None:
        one_leg_risk = np.nan
        locked_gain = np.nan
    else:
        one_leg_risk = qty * max(market.yes_best_bid, market.no_best_bid)
        locked_gain = qty * max(1.0 - market.yes_best_bid - market.no_best_bid, 0.0)
    return {
        "qty_each_side": qty,
        "own_score": own_score,
        "share_lower_target_bound": share_lower,
        "share_current_book_proxy": share_current,
        "reward_per_day_lower": reward_day_lower,
        "reward_per_day_current_proxy": reward_day_current,
        "worst_one_leg_loss": one_leg_risk,
        "locked_spread_gain_if_both_fill": locked_gain,
        "lower_reward_to_one_leg_risk": reward_day_lower / one_leg_risk
            if one_leg_risk and one_leg_risk > 0 else np.nan,
        "current_reward_to_one_leg_risk": reward_day_current / one_leg_risk
            if one_leg_risk and one_leg_risk > 0 else np.nan,
        "hours_to_cover_one_leg_lower": one_leg_risk / reward_day_lower * 24.0
            if one_leg_risk and reward_day_lower > 0 else np.inf,
        "hours_to_cover_one_leg_current": one_leg_risk / reward_day_current * 24.0
            if one_leg_risk and reward_day_current > 0 else np.inf,
    }


def main() -> None:
    fetched_at = datetime.now(timezone.utc).isoformat()
    programs = all_programs("active", "liquidity")
    terms = pd.DataFrame([normalize(program) for program in programs])
    terms = terms[
        terms["market_ticker"].ne("")
        & terms["reward_per_day_dollars"].gt(0)
        & terms["target_size"].gt(0)
    ].sort_values("reward_density_per_target_contract_day", ascending=False)
    terms.to_csv(OUT / "incentive_frontier_all_terms.csv", index=False)

    top = terms.head(TOP_TERMS_TO_ENRICH)
    enriched_rows = []
    for index, row in enumerate(top.to_dict("records"), start=1):
        enriched_rows.append(enrich(row))
        if index % 50 == 0:
            print(f"enriched {index}/{len(top)}", flush=True)
    enriched = pd.DataFrame([asdict(row) for row in enriched_rows])
    enriched.to_csv(OUT / "incentive_frontier_enriched.csv", index=False)

    bounds_rows = []
    for market in enriched_rows:
        for qty in QTY_GRID:
            bounds_rows.append({**asdict(market), **quote_bounds(market, qty)})
    bounds = pd.DataFrame(bounds_rows)
    bounds.to_csv(OUT / "incentive_frontier_quote_bounds.csv", index=False)

    open_bounds = bounds[
        bounds["market_status"].isin(["active", "open"])
        & bounds["yes_best_bid"].notna()
        & bounds["no_best_bid"].notna()
    ].copy()
    if open_bounds.empty:
        candidates = open_bounds
    else:
        # Conservative research rank: reward relative to worst one-leg loss,
        # then prefer small q and wider locked spread.
        candidates = open_bounds.sort_values(
            ["lower_reward_to_one_leg_risk", "reward_per_day_lower",
             "qty_each_side", "implied_spread_c"],
            ascending=[False, False, True, False]).head(50)
    candidates.to_csv(OUT / "incentive_frontier_candidates.csv", index=False)

    # A hard candidate is rare: even under the target-size lower share bound,
    # one day's reward covers at least 25% of the worst possible one-leg loss,
    # with an open two-sided book and q <= 20.
    hard = candidates[
        candidates["lower_reward_to_one_leg_risk"].ge(0.25)
        & candidates["qty_each_side"].le(20)
    ].copy() if not candidates.empty else candidates

    summary = {
        "fetched_at": fetched_at,
        "active_liquidity_programs": int(len(programs)),
        "valid_terms": int(len(terms)),
        "enriched_top_terms": int(len(enriched)),
        "open_two_sided_quote_rows": int(len(open_bounds)),
        "hard_candidate_rows": int(len(hard)),
        "hard_candidate_markets": int(hard["market_ticker"].nunique()) if len(hard) else 0,
        "max_lower_reward_to_one_leg_risk": float(
            open_bounds["lower_reward_to_one_leg_risk"].max())
            if len(open_bounds) else None,
        "max_current_reward_to_one_leg_risk": float(
            open_bounds["current_reward_to_one_leg_risk"].max())
            if len(open_bounds) else None,
    }
    (OUT / "incentive_frontier_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    verdict = (
        "CANDIDATE — prospective reward capture justified"
        if len(hard) else
        "FAIL — no reward/risk candidate clears the predeclared hurdle"
    )
    lines = [
        "# Kalshi liquidity-incentive frontier", "",
        f"## Verdict: **{verdict}**", "",
        f"Snapshot UTC: `{fetched_at}`", "",
        "The audit ranks active liquidity programs using reward in dollars",
        "(`period_reward / 10,000`), period length, target size, current public",
        "books, and the worst possible loss when only one side of a two-sided",
        "maker quote fills.", "", "## Coverage", "",
        f"- Active liquidity programs: {len(programs):,}",
        f"- Programs with valid reward/target terms: {len(terms):,}",
        f"- Highest-density programs enriched with live market/book data: {len(enriched):,}",
        f"- Open two-sided quote configurations: {len(open_bounds):,}",
        f"- Predeclared hard-candidate markets: {summary['hard_candidate_markets']}", "",
        "## Hard hurdle", "",
        "A row qualifies only when the conservative target-size share estimate",
        "earns at least 25% of the worst one-leg loss per day at q20 or below.",
        "This does not assume that the reward is riskless; it identifies markets",
        "where a short prospective q1 quoting experiment is economically worth",
        "running.", "",
    ]
    if len(hard):
        lines.extend([
            "## Highest-ranked hard candidates", "",
            "| Market | q/side | Reward/day lower | One-leg risk | Ratio | Spread |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in hard.head(15).itertuples(index=False):
            lines.append(
                f"| {row.market_ticker} | {row.qty_each_side:.0f} | "
                f"${row.reward_per_day_lower:.2f} | ${row.worst_one_leg_loss:.2f} | "
                f"{row.lower_reward_to_one_leg_risk:.2f} | {row.implied_spread_c:.1f}¢ |")
    else:
        lines.extend([
            "No market cleared the conservative hurdle. Current-book proxy rankings",
            "are preserved in the CSV, but they are too dependent on transient",
            "competition to justify risking capital.",
        ])
    lines.extend(["", "## Required prospective double test", "",
        "1. Confirm account eligibility for the general liquidity program.",
        "2. Randomize q1 two-sided quotes versus no-order observation windows.",
        "3. Record official queue position, full books, fills, score terms, and",
        "   eventual reward credits.",
        "4. Never self-trade or manufacture volume.",
        "5. Promote only when reward plus trading P&L per assigned window has a",
        "   positive day/block-clustered lower bound.", "",
        "The account's existing KILL state remains binding."])
    (OUT / "incentive_frontier_report.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
