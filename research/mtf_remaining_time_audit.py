"""MTF-R: remaining-time correction to the target-frontier incentive screen.

Public endpoints only. No credentials and no order submission.
"""
from __future__ import annotations

import concurrent.futures
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "results" / "mtf_remaining"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://external-api.kalshi.com/trade-api/v2"
WORKERS = 24
MAX_ENRICH = 1500
MIN_PAYOUT = 1.0


def get_json(path: str, params: dict | None = None, retries: int = 4) -> dict:
    last = None
    for attempt in range(retries):
        try:
            response = requests.get(
                BASE + path, params=params, timeout=25,
                headers={"User-Agent": "m15-mtf-remaining/1.0"},
            )
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict):
                raise RuntimeError("non-object JSON")
            return body
        except Exception as exc:
            last = exc
            if attempt + 1 < retries:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"GET {path}: {last}")


def fetch_programs() -> list[dict]:
    rows: list[dict] = []
    cursor = ""
    seen: set[str] = set()
    while True:
        params: dict[str, Any] = {
            "status": "active", "type": "liquidity", "limit": 10000,
        }
        if cursor:
            params["cursor"] = cursor
        body = get_json("/incentive_programs", params)
        batch = body.get("incentive_programs") or []
        rows.extend(item for item in batch if isinstance(item, dict))
        nxt = str(body.get("next_cursor") or "")
        if not nxt or nxt in seen or not batch:
            break
        seen.add(nxt); cursor = nxt
    return rows


def normalize(program: dict, now: pd.Timestamp) -> dict | None:
    try:
        ticker = str(program.get("market_ticker") or "")
        start = pd.to_datetime(program.get("start_date"), utc=True)
        end = pd.to_datetime(program.get("end_date"), utc=True)
        pool = float(program.get("period_reward") or 0) / 10_000.0
        target = float(program.get("target_size_fp") or 0)
        discount = float(program.get("discount_factor_bps") or 0) / 10_000.0
    except Exception:
        return None
    total_seconds = (end - start).total_seconds()
    remaining_seconds = max((end - now).total_seconds(), 0.0)
    if not ticker or pool <= 0 or target <= 0 or total_seconds <= 0 or remaining_seconds <= 0:
        return None
    remaining_fraction = min(remaining_seconds / total_seconds, 1.0)
    return {
        "program_id": str(program.get("id") or ""),
        "ticker": ticker,
        "series": str(program.get("series_ticker") or ticker.split("-")[0]),
        "description": str(program.get("incentive_description") or ""),
        "start": start,
        "end": end,
        "pool": pool,
        "target": target,
        "discount": min(max(discount, 0.0), 1.0),
        "total_seconds": total_seconds,
        "remaining_seconds": remaining_seconds,
        "remaining_fraction": remaining_fraction,
        "density": pool / target / (total_seconds / 86400.0),
    }


def parse_book(body: dict) -> dict[str, list[tuple[float, float]]]:
    raw = body.get("orderbook_fp") or body.get("orderbook") or {}
    result = {"yes": [], "no": []}
    for side in ("yes", "no"):
        levels = raw.get(side + "_dollars") or raw.get(side) or []
        for level in levels:
            try:
                if isinstance(level, dict):
                    price = float(level.get("price_dollars", level.get("price")))
                    size = float(level.get("count_fp", level.get("count", level.get("quantity"))))
                else:
                    price, size = float(level[0]), float(level[1])
            except (TypeError, ValueError, IndexError):
                continue
            if 0 < price < 1 and size > 0:
                result[side].append((price, size))
        result[side].sort(key=lambda x: -x[0])
    return result


def frontier(levels: list[tuple[float, float]], target: float) -> dict | None:
    if not levels:
        return None
    best = levels[0][0]
    cumulative = 0.0
    candidate = None
    for price, size in levels:
        if cumulative + size + 1.0 <= target + 1e-9:
            candidate = {
                "price": price,
                "size_at_price": size,
                "cumulative_ahead": cumulative + size,
            }
        cumulative += size
        if cumulative >= target:
            break
    if candidate is None:
        return None
    candidate["best"] = best
    candidate["distance_c"] = max((best - candidate["price"]) * 100.0, 0.0)
    return candidate


def reward(pool: float, target: float, multiplier: float,
           remaining_fraction: float, qty: int) -> float:
    effective_own = qty * multiplier * remaining_fraction
    return pool * effective_own / (2.0 * target + effective_own)


def min_qty_for_payout(pool: float, target: float, multiplier: float,
                       remaining_fraction: float) -> int | None:
    effective = multiplier * remaining_fraction
    if pool <= MIN_PAYOUT or effective <= 0:
        return None
    raw = (MIN_PAYOUT * 2.0 * target) / (effective * (pool - MIN_PAYOUT))
    return max(int(math.ceil(raw - 1e-12)), 1)


def enrich(program: dict) -> list[dict]:
    try:
        market = get_json(f"/markets/{program['ticker']}").get("market") or {}
        book = parse_book(get_json(f"/markets/{program['ticker']}/orderbook", {"depth": 100}))
    except Exception:
        return []
    if str(market.get("status") or "").lower() not in {"active", "open"}:
        return []
    yes_best = book["yes"][0][0] if book["yes"] else None
    no_best = book["no"][0][0] if book["no"] else None
    if yes_best is None or no_best is None:
        return []
    opposing_ask = {"yes": 1.0 - no_best, "no": 1.0 - yes_best}
    rows = []
    for side in ("yes", "no"):
        f = frontier(book[side], program["target"])
        if f is None or f["price"] >= opposing_ask[side] - 1e-9:
            continue
        multiplier = (1.0 - program["discount"]) ** f["distance_c"]
        capacity = max(int(math.floor(program["target"] - f["cumulative_ahead"] + 1e-9)), 0)
        q1_reward = reward(
            program["pool"], program["target"], multiplier,
            program["remaining_fraction"], 1,
        )
        min_qty = min_qty_for_payout(
            program["pool"], program["target"], multiplier,
            program["remaining_fraction"],
        )
        if min_qty is None:
            min_qty = 10**9
            min_qty_reward = 0.0
        else:
            min_qty_reward = reward(
                program["pool"], program["target"], multiplier,
                program["remaining_fraction"], min_qty,
            )
        q1_cover = q1_reward >= f["price"] - 1e-12
        per_program_cover = (
            min_qty <= capacity
            and min_qty_reward >= min_qty * f["price"] - 1e-12
        )
        rows.append({
            "program_id": program["program_id"], "ticker": program["ticker"],
            "series": program["series"], "title": str(market.get("title") or ""),
            "side": side, "pool": program["pool"], "target": program["target"],
            "start": program["start"], "end": program["end"],
            "total_seconds": program["total_seconds"],
            "remaining_seconds": program["remaining_seconds"],
            "remaining_fraction": program["remaining_fraction"],
            "best_bid": f["best"], "frontier_bid": f["price"],
            "opposing_ask": opposing_ask[side], "distance_c": f["distance_c"],
            "multiplier": multiplier, "cumulative_ahead": f["cumulative_ahead"],
            "guaranteed_capacity": capacity,
            "q1_reward_to_end": q1_reward,
            "q1_worst_loss": f["price"],
            "q1_reward_to_loss": q1_reward / f["price"],
            "q1_cover": q1_cover,
            "min_qty_for_1usd": min_qty,
            "reward_at_min_qty": min_qty_reward,
            "loss_at_min_qty": min_qty * f["price"],
            "min_qty_reward_to_loss": min_qty_reward / (min_qty * f["price"]) if min_qty < 10**9 else 0.0,
            "fits_frontier": min_qty <= capacity,
            "per_program_cover": per_program_cover,
            "volume_24h": float(market.get("volume_24h_fp") or market.get("volume_24h") or 0),
            "open_interest": float(market.get("open_interest_fp") or market.get("open_interest") or 0),
        })
    return rows


def main() -> None:
    now = pd.Timestamp.now(tz="UTC")
    raw = fetch_programs()
    normalized = [x for x in (normalize(p, now) for p in raw) if x is not None]
    normalized.sort(key=lambda x: x["density"], reverse=True)
    selected = normalized[:MAX_ENRICH]
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(enrich, p) for p in selected]
        for idx, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                rows.extend(future.result())
            except Exception:
                pass
            if idx % 250 == 0:
                print(f"enriched {idx}/{len(selected)} rows={len(rows)}", flush=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no rows")
    frame.sort_values(
        ["per_program_cover", "q1_cover", "min_qty_reward_to_loss", "q1_reward_to_loss"],
        ascending=False, inplace=True,
    )
    frame.to_csv(OUT / "remaining_time_rows.csv", index=False)
    q1 = frame[frame["q1_cover"]].copy()
    per_program = frame[frame["per_program_cover"]].copy()
    q1.to_csv(OUT / "q1_cover.csv", index=False)
    per_program.to_csv(OUT / "per_program_cover.csv", index=False)
    summary = {
        "fetched_at": now.isoformat(), "active_programs": len(raw),
        "normalized_programs": len(normalized), "enriched_programs": len(selected),
        "rows": len(frame), "q1_cover_rows": len(q1),
        "q1_cover_markets": int(q1["ticker"].nunique()),
        "per_program_cover_rows": len(per_program),
        "per_program_cover_markets": int(per_program["ticker"].nunique()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    columns = [
        "ticker", "series", "side", "frontier_bid", "distance_c",
        "remaining_fraction", "guaranteed_capacity", "q1_reward_to_end",
        "q1_worst_loss", "q1_reward_to_loss", "q1_cover",
        "min_qty_for_1usd", "reward_at_min_qty", "loss_at_min_qty",
        "min_qty_reward_to_loss", "fits_frontier", "per_program_cover",
    ]
    lines = [
        "# MTF-R — Remaining-Time-Corrected Target-Frontier Screen", "",
        f"Snapshot: `{now.isoformat()}`", "",
        f"Per-program arithmetic-cover markets: **{summary['per_program_cover_markets']}**.",
        f"q1 arithmetic-cover markets under aggregate-minimum interpretation: **{summary['q1_cover_markets']}**.",
        "", "## Per-program minimum candidates", "",
        per_program[columns].head(50).to_markdown(index=False, floatfmt=".5f") if len(per_program) else "None.",
        "", "## q1 cover candidates", "",
        q1[columns].head(50).to_markdown(index=False, floatfmt=".5f") if len(q1) else "None.",
        "", "```json", json.dumps(summary, indent=2), "```", "",
        "This is a necessary arithmetic screen, not evidence of reward credit or execution performance.",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
