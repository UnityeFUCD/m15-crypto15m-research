"""MTF: marginal target-frontier liquidity incentive audit.

Public API only. No credentials and no order submission.
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
OUT = ROOT / "research" / "results" / "mtf"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://external-api.kalshi.com/trade-api/v2"
MAX_ENRICH = 1200
WORKERS = 24
QTY = 1.0


def get_json(path: str, params: dict | None = None, retries: int = 4) -> dict:
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(BASE + path, params=params, timeout=25,
                             headers={"User-Agent": "m15-mtf-audit/1.0"})
            r.raise_for_status()
            body = r.json()
            if not isinstance(body, dict):
                raise RuntimeError("non-object JSON")
            return body
        except Exception as exc:
            last = exc
            if attempt + 1 < retries:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"GET {path}: {last}")


def programs() -> list[dict]:
    rows = []
    cursor = ""
    seen = set()
    while True:
        params: dict[str, Any] = {"status": "active", "type": "liquidity", "limit": 10000}
        if cursor:
            params["cursor"] = cursor
        body = get_json("/incentive_programs", params)
        batch = body.get("incentive_programs") or []
        rows.extend(x for x in batch if isinstance(x, dict))
        nxt = str(body.get("next_cursor") or "")
        if not nxt or nxt in seen or not batch:
            break
        seen.add(nxt); cursor = nxt
    return rows


def parse_book(body: dict) -> dict[str, list[tuple[float, float]]]:
    raw = body.get("orderbook_fp") or body.get("orderbook") or {}
    result = {"yes": [], "no": []}
    for side in ("yes", "no"):
        levels = raw.get(side + "_dollars") or raw.get(side) or []
        for level in levels:
            try:
                if isinstance(level, dict):
                    p = float(level.get("price_dollars", level.get("price")))
                    q = float(level.get("count_fp", level.get("count", level.get("quantity"))))
                else:
                    p, q = float(level[0]), float(level[1])
            except (TypeError, ValueError, IndexError):
                continue
            if 0 < p < 1 and q > 0:
                result[side].append((p, q))
        result[side].sort(key=lambda x: -x[0])
    return result


def normalize(p: dict) -> dict | None:
    try:
        reward = float(p.get("period_reward") or 0) / 10_000.0
        target = float(p.get("target_size_fp") or p.get("target_size") or 0)
        discount = float(p.get("discount_factor_bps") or 0) / 10_000.0
        start = pd.to_datetime(p.get("start_date"), utc=True)
        end = pd.to_datetime(p.get("end_date"), utc=True)
        ticker = str(p.get("market_ticker") or "")
    except Exception:
        return None
    if not ticker or reward <= 0 or target <= 0 or end <= start:
        return None
    hours = (end - start).total_seconds() / 3600.0
    return {
        "program_id": str(p.get("id") or ""), "ticker": ticker,
        "series": str(p.get("series_ticker") or ticker.split("-")[0]),
        "description": str(p.get("incentive_description") or ""),
        "start": start, "end": end, "hours": hours,
        "reward": reward, "target": target,
        "discount": min(max(discount, 0.0), 1.0),
        "density": reward / target / hours if hours > 0 else 0.0,
    }


def frontier(levels: list[tuple[float, float]], target: float) -> dict | None:
    if not levels or target < QTY:
        return None
    best = levels[0][0]
    cumulative = 0.0
    candidate = None
    for price, size in levels:
        # Joining is guaranteed inside the raw target only if the entire
        # displayed level ahead of us plus our q1 remains within target.
        if cumulative + size + QTY <= target + 1e-9:
            candidate = {"price": price, "size_at_price": size,
                         "cumulative_ahead_including_level": cumulative + size}
        cumulative += size
        if cumulative >= target:
            break
    if candidate is None:
        return None
    candidate["best"] = best
    candidate["distance_c"] = max((best - candidate["price"]) * 100.0, 0.0)
    candidate["displayed_total"] = sum(q for _, q in levels)
    return candidate


def enrich(row: dict) -> list[dict]:
    ticker = row["ticker"]
    try:
        market = get_json(f"/markets/{ticker}").get("market") or {}
        book = parse_book(get_json(f"/markets/{ticker}/orderbook", {"depth": 100}))
    except Exception:
        return []
    if str(market.get("status") or "").lower() not in {"active", "open"}:
        return []
    yes_best = book["yes"][0][0] if book["yes"] else None
    no_best = book["no"][0][0] if book["no"] else None
    if yes_best is None or no_best is None:
        return []
    opposing_ask = {"yes": 1.0 - no_best, "no": 1.0 - yes_best}
    output = []
    for side in ("yes", "no"):
        f = frontier(book[side], row["target"])
        if f is None or not (f["price"] < opposing_ask[side] - 1e-9):
            continue
        multiplier = (1.0 - row["discount"]) ** f["distance_c"]
        own_score = QTY * multiplier
        denominator = 2.0 * row["target"] + own_score
        share = own_score / denominator
        reward_period = row["reward"] * share
        reward_day = reward_period * 24.0 / row["hours"]
        risk = QTY * f["price"]
        output.append({
            **{k: row[k] for k in ["program_id", "ticker", "series", "description", "hours", "reward", "target", "discount"]},
            "title": str(market.get("title") or ""),
            "side": side, "best_bid": f["best"], "frontier_bid": f["price"],
            "opposing_ask": opposing_ask[side], "distance_c": f["distance_c"],
            "size_at_price": f["size_at_price"],
            "cumulative_ahead": f["cumulative_ahead_including_level"],
            "displayed_total_side": f["displayed_total"],
            "multiplier": multiplier, "own_score": own_score,
            "share_lower": share, "reward_period_lower": reward_period,
            "reward_day_lower": reward_day, "worst_loss": risk,
            "full_reward_to_loss": reward_period / risk if risk > 0 else math.inf,
            "half_reward_to_loss": 0.5 * reward_period / risk if risk > 0 else math.inf,
            "arithmetic_cover": reward_period >= risk,
            "strong_candidate": (0.5 * reward_period >= 0.25 * risk and f["distance_c"] >= 2.0),
            "volume_24h": float(market.get("volume_24h_fp") or market.get("volume_24h") or 0),
            "open_interest": float(market.get("open_interest_fp") or market.get("open_interest") or 0),
            "close_time": str(market.get("close_time") or ""),
        })
    return output


def two_sided(one: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker, group in one.groupby("ticker"):
        sides = {r.side: r for r in group.itertuples(index=False)}
        if "yes" not in sides or "no" not in sides:
            continue
        y, n = sides["yes"], sides["no"]
        reward = y.reward_period_lower + n.reward_period_lower
        risk = max(y.worst_loss, n.worst_loss)
        rows.append({
            "ticker": ticker, "series": y.series, "title": y.title,
            "yes_bid": y.frontier_bid, "no_bid": n.frontier_bid,
            "yes_distance_c": y.distance_c, "no_distance_c": n.distance_c,
            "reward_period_lower": reward,
            "reward_day_lower": y.reward_day_lower + n.reward_day_lower,
            "worst_one_leg_loss": risk,
            "locked_gain_if_both_fill": max(1.0 - y.frontier_bid - n.frontier_bid, 0.0),
            "full_reward_to_loss": reward / risk if risk > 0 else math.inf,
            "half_reward_to_loss": 0.5 * reward / risk if risk > 0 else math.inf,
            "arithmetic_cover": reward >= risk,
            "strong_candidate": (0.5 * reward >= 0.25 * risk and min(y.distance_c, n.distance_c) >= 2.0),
        })
    return pd.DataFrame(rows)


def main() -> None:
    fetched = datetime.now(timezone.utc).isoformat()
    raw = programs()
    normalized = [x for x in (normalize(p) for p in raw) if x is not None]
    normalized.sort(key=lambda x: x["density"], reverse=True)
    selected = normalized[:MAX_ENRICH]
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(enrich, row) for row in selected]
        for idx, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                rows.extend(future.result())
            except Exception:
                pass
            if idx % 200 == 0:
                print(f"enriched {idx}/{len(selected)} sides={len(rows)}", flush=True)
    one = pd.DataFrame(rows)
    if one.empty:
        raise RuntimeError("no target-frontier rows")
    one.sort_values(["strong_candidate", "half_reward_to_loss", "reward_day_lower"], ascending=False, inplace=True)
    one.to_csv(OUT / "one_sided_frontiers.csv", index=False)
    pairs = two_sided(one)
    if not pairs.empty:
        pairs.sort_values(["strong_candidate", "half_reward_to_loss", "reward_day_lower"], ascending=False, inplace=True)
    pairs.to_csv(OUT / "two_sided_frontiers.csv", index=False)
    summary = {
        "fetched_at": fetched, "active_programs": len(raw),
        "normalized_programs": len(normalized), "enriched_programs": len(selected),
        "one_sided_rows": len(one), "two_sided_rows": len(pairs),
        "one_sided_arithmetic_cover": int(one["arithmetic_cover"].sum()),
        "one_sided_strong": int(one["strong_candidate"].sum()),
        "two_sided_arithmetic_cover": int(pairs["arithmetic_cover"].sum()) if len(pairs) else 0,
        "two_sided_strong": int(pairs["strong_candidate"].sum()) if len(pairs) else 0,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    one_top = one.head(30)[["ticker", "series", "side", "best_bid", "frontier_bid", "distance_c", "multiplier", "reward_period_lower", "reward_day_lower", "worst_loss", "full_reward_to_loss", "half_reward_to_loss", "arithmetic_cover", "strong_candidate"]]
    pair_top = pairs.head(20) if len(pairs) else pairs
    verdict = "ARITHMETIC-COVER CANDIDATE" if summary["one_sided_arithmetic_cover"] or summary["two_sided_arithmetic_cover"] else ("STRONG PROSPECTIVE CANDIDATE" if summary["one_sided_strong"] or summary["two_sided_strong"] else "NO STRONG CANDIDATE")
    lines = ["# MTF — Marginal Target-Frontier Liquidity Audit", "",
             f"## Verdict: **{verdict}**", "", f"Snapshot: `{fetched}`", "",
             "## One-sided frontiers", "", one_top.to_markdown(index=False, floatfmt=".5f"),
             "", "## Two-sided frontiers", "",
             pair_top.to_markdown(index=False, floatfmt=".5f") if len(pair_top) else "None.",
             "", "```json", json.dumps(summary, indent=2), "```", "",
             "This is a current-book economic screen, not proof of credited rewards or future fill risk.",
             "No result authorizes live orders."]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
