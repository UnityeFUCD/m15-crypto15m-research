from __future__ import annotations

import itertools
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

BASE = "https://external-api.kalshi.com/trade-api/v2"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "results"
OUT.mkdir(parents=True, exist_ok=True)

CRYPTO15_SERIES = {
    "KXBTC15M",
    "KXETH15M",
    "KXSOL15M",
    "KXXRP15M",
    "KXDOGE15M",
    "KXHYPE15M",
    "KXBNB15M",
}

START_DAY = os.getenv("HBCD_START", "2026-07-31")
END_DAY = os.getenv(
    "HBCD_END",
    (datetime.now(timezone.utc).date() + timedelta(days=1)).isoformat(),
)
MAX_PAGES_PER_DAY = int(os.getenv("HBCD_MAX_PAGES_PER_DAY", "75"))
HTTP_SLEEP = float(os.getenv("HBCD_HTTP_SLEEP", "0.06"))
MAX_CANDLE_AGE_SECONDS = float(os.getenv("HBCD_MAX_CANDLE_AGE_SECONDS", "90"))

# The primary cell is frozen before this audit runs.
PRIMARY_MARGIN = 0.01
PRIMARY_LATENCY_S = 1.0
PRIMARY_Q = 1
PRIMARY_STRICT = True

MARGIN_GRID = [0.005, 0.01, 0.02, 0.03]
LATENCY_GRID_S = [0.25, 0.50, 1.0, 2.0, 3.0, 5.0]
Q_GRID = [1, 5, 10, 20, 50]
STRICT_GRID = [True, False]

# Deliberately conservative: charge both legs as if they were ordinary taker
# quadratic trades, even though the combo quote is a maker/RFQ action.
FEE_RATE_STRESS = 0.07
MAX_ROUNDING_CHARGE_PER_ORDER = 0.0099
RNG_SEED = 20260813


def fnum(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def parse_ts(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        if v > 1e12:
            v /= 1000.0
        return v
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def iso(ts: float | None) -> str | None:
    if ts is None or not math.isfinite(ts):
        return None
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def daterange(start: str, end: str) -> list[date]:
    a = date.fromisoformat(start)
    b = date.fromisoformat(end)
    if b <= a:
        raise ValueError(f"HBCD_END must be after HBCD_START: {start=} {end=}")
    return [a + timedelta(days=i) for i in range((b - a).days)]


class PublicAPI:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.calls = 0
        self.errors: list[dict[str, Any]] = []

    def get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        attempts: int = 12,
    ) -> dict[str, Any] | None:
        url = BASE + path
        for k in range(attempts):
            self.calls += 1
            try:
                r = self.session.get(url, params=params, timeout=35)
            except requests.RequestException as exc:
                if k + 1 == attempts:
                    self.errors.append(
                        {"path": path, "params": params, "status": "exception", "body": repr(exc)}
                    )
                    return None
                time.sleep(min(10.0, 0.5 * (k + 1)))
                continue
            if r.status_code == 200:
                time.sleep(HTTP_SLEEP)
                return r.json()
            if r.status_code == 429:
                time.sleep(min(12.0, 0.75 * (k + 1)))
                continue
            if r.status_code in (404, 410):
                return None
            self.errors.append(
                {
                    "path": path,
                    "params": params,
                    "status": r.status_code,
                    "body": r.text[:500],
                }
            )
            return None
        return None


def series_of_leg(leg: dict[str, Any]) -> str:
    event = str(leg.get("event_ticker") or "")
    if event:
        return event.split("-", 1)[0]
    market = str(leg.get("market_ticker") or "")
    return market.split("-", 1)[0]


def is_crypto_combo(market: dict[str, Any]) -> bool:
    legs = market.get("mve_selected_legs") or []
    return bool(legs) and all(series_of_leg(leg) in CRYPTO15_SERIES for leg in legs)


def validate_dominance() -> dict[str, Any]:
    checks = 0
    min_slack = float("inf")
    argmin: dict[str, Any] | None = None
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    for nlegs in range(2, 7):
        for vals in itertools.product(grid, repeat=nlegs):
            combo = math.prod(vals)
            for j, selected in enumerate(vals):
                slack = selected + (1.0 - combo) - 1.0
                checks += 1
                if slack < min_slack:
                    min_slack = slack
                    argmin = {"nlegs": nlegs, "values": vals, "selected_index": j}
                if slack < -1e-12:
                    raise AssertionError(
                        f"dominance failed: {nlegs=} {vals=} {j=} {combo=} {slack=}"
                    )

    rng = np.random.default_rng(RNG_SEED)
    random_checks = 0
    for nlegs in range(2, 21):
        vals = rng.random((2000, nlegs))
        combo = vals.prod(axis=1)
        for j in range(nlegs):
            slack = vals[:, j] + 1.0 - combo - 1.0
            random_checks += len(slack)
            if float(slack.min()) < -1e-12:
                raise AssertionError(f"random dominance failed at nlegs={nlegs} j={j}")
            if float(slack.min()) < min_slack:
                min_slack = float(slack.min())
                argmin = {"nlegs": nlegs, "selected_index": j, "source": "random"}

    return {
        "status": "PASS",
        "grid_checks": checks,
        "random_checks": random_checks,
        "minimum_slack": min_slack,
        "argmin": argmin,
        "identity": "selected_leg + (1 - product(all selected legs)) >= 1",
    }


def fetch_mve_markets(api: PublicAPI) -> tuple[list[dict[str, Any]], list[str]]:
    out: dict[str, dict[str, Any]] = {}
    truncated_days: list[str] = []
    for day in daterange(START_DAY, END_DAY):
        lo = int(datetime.combine(day, datetime.min.time(), timezone.utc).timestamp())
        hi = lo + 86400
        cursor: str | None = None
        day_rows = 0
        exhausted = False
        for page in range(MAX_PAGES_PER_DAY):
            params: dict[str, Any] = {
                "limit": 1000,
                "mve_filter": "only",
                "min_created_ts": lo,
                "max_created_ts": hi,
            }
            if cursor:
                params["cursor"] = cursor
            body = api.get("/markets", params)
            if not body:
                exhausted = True
                break
            batch = body.get("markets") or []
            day_rows += len(batch)
            for market in batch:
                ticker = market.get("ticker")
                if ticker:
                    out[str(ticker)] = market
            cursor = body.get("cursor")
            if not cursor or not batch:
                exhausted = True
                break
        if not exhausted and cursor:
            truncated_days.append(day.isoformat())
        crypto_n = sum(
            1
            for m in out.values()
            if str(m.get("created_time") or "")[:10] == day.isoformat() and is_crypto_combo(m)
        )
        print(
            "MVE_DAY",
            day.isoformat(),
            "rows",
            day_rows,
            "crypto_unique",
            crypto_n,
            "truncated",
            day.isoformat() in truncated_days,
            flush=True,
        )
    return list(out.values()), truncated_days


def fetch_trades(api: PublicAPI, ticker: str) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    cursor: str | None = None
    for _ in range(10):
        params: dict[str, Any] = {"ticker": ticker, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        body = api.get("/markets/trades", params)
        if not body:
            break
        batch = body.get("trades") or []
        for row in batch:
            key = str(row.get("trade_id") or json.dumps(row, sort_keys=True, default=str))
            out[key] = row
        cursor = body.get("cursor")
        if not cursor or not batch:
            break
    return sorted(out.values(), key=lambda x: str(x.get("created_time") or ""))


def obj_number(obj: Any, names: tuple[str, ...]) -> float:
    if isinstance(obj, (int, float, str)):
        return fnum(obj)
    if not isinstance(obj, dict):
        return float("nan")
    for name in names:
        v = fnum(obj.get(name))
        if math.isfinite(v):
            return v
    return float("nan")


def candle_price(
    candle: dict[str, Any],
    selected_side: str,
) -> tuple[float, dict[str, Any]]:
    yes_ask = candle.get("yes_ask")
    yes_bid = candle.get("yes_bid")

    ask_close = obj_number(yes_ask, ("close_dollars", "close"))
    ask_high = obj_number(yes_ask, ("high_dollars", "high"))
    bid_close = obj_number(yes_bid, ("close_dollars", "close"))
    bid_low = obj_number(yes_bid, ("low_dollars", "low"))

    if selected_side == "yes":
        vals = [v for v in (ask_close, ask_high) if 0 < v < 1]
        ask = max(vals) if vals else float("nan")
        source = "max(yes_ask_close,yes_ask_high)"
    elif selected_side == "no":
        vals = [v for v in (bid_close, bid_low) if 0 < v < 1]
        ask = 1.0 - min(vals) if vals else float("nan")
        source = "1-min(yes_bid_close,yes_bid_low)"
    else:
        return float("nan"), {"error": f"unknown side {selected_side!r}"}

    return ask, {
        "source": source,
        "yes_ask_close": ask_close,
        "yes_ask_high": ask_high,
        "yes_bid_close": bid_close,
        "yes_bid_low": bid_low,
    }


def fetch_component_ask(
    api: PublicAPI,
    cache: dict[tuple[str, int, str], dict[str, Any] | None],
    ticker: str,
    side: str,
    decision_ts: float,
) -> dict[str, Any] | None:
    minute = int(decision_ts // 60)
    key = (ticker, minute, side)
    if key in cache:
        return cache[key]

    series = ticker.split("-", 1)[0]
    params = {
        "start_ts": max(0, int(decision_ts) - 900),
        "end_ts": int(decision_ts),
        "period_interval": 1,
        "include_latest_before_start": "true",
    }
    body = api.get(f"/series/{series}/markets/{ticker}/candlesticks", params)
    if not body:
        body = api.get(
            f"/historical/markets/{ticker}/candlesticks",
            {
                "start_ts": params["start_ts"],
                "end_ts": params["end_ts"],
                "period_interval": 1,
            },
        )
    candles = (body or {}).get("candlesticks") or []
    eligible: list[tuple[float, dict[str, Any]]] = []
    for candle in candles:
        end_ts = parse_ts(candle.get("end_period_ts"))
        if end_ts is not None and end_ts <= decision_ts + 1e-9:
            eligible.append((end_ts, candle))
    if not eligible:
        cache[key] = None
        return None

    end_ts, candle = max(eligible, key=lambda z: z[0])
    ask, detail = candle_price(candle, side)
    age = decision_ts - end_ts
    if not (0 < ask < 1) or age < -1e-9 or age > MAX_CANDLE_AGE_SECONDS:
        cache[key] = None
        return None

    result = {
        "component": ticker,
        "side": side,
        "ask": ask,
        "candle_end_ts": end_ts,
        "candle_age_s": age,
        "detail": detail,
    }
    cache[key] = result
    return result


def price_range_number(rng: dict[str, Any], base: str) -> float:
    candidates = (
        f"{base}_dollars",
        base,
        f"{base}_price_dollars",
        f"{base}_price",
    )
    for key in candidates:
        value = fnum(rng.get(key))
        if math.isfinite(value):
            if value > 1.0 and key == base:
                value /= 100.0
            return value
    return float("nan")


def tick_for_price(market: dict[str, Any], price: float) -> float:
    ranges = market.get("price_ranges") or []
    for rng in ranges:
        if not isinstance(rng, dict):
            continue
        start = price_range_number(rng, "start")
        end = price_range_number(rng, "end")
        step = price_range_number(rng, "step")
        if not math.isfinite(start):
            start = 0.0
        if not math.isfinite(end):
            end = 1.0
        if start - 1e-12 <= price <= end + 1e-12 and 0 < step <= 1:
            return step

    structure = str(market.get("price_level_structure") or "")
    if structure == "deci_cent":
        return 0.001
    # Tapered structures should provide price_ranges. Fail conservatively to
    # the coarser one-cent grid when those ranges are absent.
    return 0.01


def ceil_to_tick(market: dict[str, Any], value: float) -> tuple[float, float]:
    p = min(max(value, 0.0001), 0.9999)
    for _ in range(10):
        tick = tick_for_price(market, p)
        rounded = math.ceil((p - 1e-12) / tick) * tick
        rounded = round(rounded, 4)
        if abs(rounded - p) < 1e-10:
            return rounded, tick
        p = rounded
    return p, tick_for_price(market, p)


def stressed_order_fee(q: int, price: float) -> float:
    raw = FEE_RATE_STRESS * q * price * (1.0 - price)
    trade_fee = math.ceil(raw * 10000.0 - 1e-12) / 10000.0
    return trade_fee + MAX_ROUNDING_CHARGE_PER_ORDER


@dataclass(frozen=True)
class QuoteCell:
    margin: float
    latency_s: float
    q: int
    strict: bool


def construct_quote(
    market: dict[str, Any],
    component_ask: float,
    q: int,
    target_margin: float,
) -> dict[str, float] | None:
    if not (0 < component_ask < 1) or q <= 0:
        return None

    y = component_ask + target_margin
    for _ in range(20):
        y, tick = ceil_to_tick(market, y)
        if not (0 < y < 1):
            return None
        combo_no_cost = 1.0 - y
        fees = stressed_order_fee(q, component_ask) + stressed_order_fee(q, combo_no_cost)
        required = component_ask + target_margin + fees / q
        if y + 1e-12 >= required:
            net = q * (y - component_ask) - fees
            if net + 1e-10 < q * target_margin:
                raise AssertionError("quote solver returned insufficient margin")
            return {
                "yes_quote": y,
                "no_bid": combo_no_cost,
                "tick": tick,
                "fee_stress": fees,
                "guaranteed_net": net,
                "guaranteed_net_per_contract": net / q,
            }
        y = required
    return None


def trade_fields(trade: dict[str, Any]) -> tuple[float | None, float, float, str]:
    t = parse_ts(trade.get("created_time"))
    yes_price = fnum(trade.get("yes_price_dollars"))
    qty = fnum(trade.get("count_fp") or trade.get("count"))
    side = str(
        trade.get("taker_outcome_side")
        or ("yes" if trade.get("taker_book_side") == "bid" else "no")
    ).lower()
    return t, yes_price, qty, side


def first_contestable_trade(
    trades: list[dict[str, Any]],
    *,
    decision_ts: float,
    latency_s: float,
    yes_quote: float,
    tick: float,
    q: int,
    strict: bool,
) -> dict[str, Any] | None:
    threshold = yes_quote + (tick if strict else 0.0)
    for trade in trades:
        t, yes_price, qty, side = trade_fields(trade)
        if t is None or t + 1e-9 < decision_ts + latency_s:
            continue
        if side != "yes" or not (0 < yes_price < 1) or qty < q:
            continue
        if yes_price + 1e-12 < threshold:
            continue
        return {
            "trade_id": trade.get("trade_id"),
            "trade_ts": t,
            "trade_delay_s": t - decision_ts,
            "public_yes_price": yes_price,
            "public_qty": qty,
            "price_improvement_to_requester": yes_price - yes_quote,
        }
    return None


def bootstrap_mean(values: np.ndarray, seed: int = RNG_SEED) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def evaluate() -> None:
    mechanical = validate_dominance()
    (OUT / "hbcd_mechanical.json").write_text(json.dumps(mechanical, indent=2, default=str))
    print("MECHANICAL", json.dumps(mechanical, sort_keys=True), flush=True)

    api = PublicAPI()
    markets, truncated_days = fetch_mve_markets(api)
    crypto = [m for m in markets if is_crypto_combo(m)]
    traded = [m for m in crypto if fnum(m.get("volume_fp") or m.get("volume")) > 0]

    print(
        "UNIVERSE",
        "mve",
        len(markets),
        "crypto",
        len(crypto),
        "crypto_traded",
        len(traded),
        flush=True,
    )

    candle_cache: dict[tuple[str, int, str], dict[str, Any] | None] = {}
    funnel_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    cells = [
        QuoteCell(margin=m, latency_s=l, q=q, strict=s)
        for m in MARGIN_GRID
        for l in LATENCY_GRID_S
        for q in Q_GRID
        for s in STRICT_GRID
    ]

    for i, market in enumerate(sorted(traded, key=lambda x: str(x.get("created_time") or "")), 1):
        ticker = str(market.get("ticker") or "")
        decision_ts = parse_ts(market.get("created_time"))
        legs = market.get("mve_selected_legs") or []
        base = {
            "combo": ticker,
            "decision_ts": decision_ts,
            "decision_time": iso(decision_ts),
            "day": str(market.get("created_time") or "")[:10],
            "nlegs": len(legs),
            "volume": fnum(market.get("volume_fp") or market.get("volume")),
            "collection": market.get("mve_collection_ticker"),
        }
        if not ticker or decision_ts is None:
            funnel_rows.append({**base, "stage": "invalid_market_time"})
            continue

        trades = fetch_trades(api, ticker)
        yes_trades = [t for t in trades if trade_fields(t)[3] == "yes"]
        if not yes_trades:
            funnel_rows.append({**base, "stage": "no_public_taker_yes_trade"})
            continue

        leg_quotes: list[dict[str, Any]] = []
        for leg in legs:
            component = str(leg.get("market_ticker") or "")
            side = str(leg.get("side") or "").lower()
            if not component or side not in ("yes", "no"):
                continue
            quote = fetch_component_ask(api, candle_cache, component, side, decision_ts)
            if quote:
                leg_quotes.append({**quote, "event_ticker": leg.get("event_ticker")})

        if not leg_quotes:
            funnel_rows.append({**base, "stage": "no_fresh_component_ask"})
            continue

        # Frozen causal hedge choice: cheapest fully observed selected-leg ask.
        chosen = min(
            leg_quotes,
            key=lambda x: (float(x["ask"]), str(x["component"]), str(x["side"])),
        )
        funnel_rows.append(
            {
                **base,
                "stage": "eligible_component_price",
                "chosen_component": chosen["component"],
                "chosen_side": chosen["side"],
                "component_ask": chosen["ask"],
                "candle_end_ts": chosen["candle_end_ts"],
                "candle_age_s": chosen["candle_age_s"],
                "public_yes_trades": len(yes_trades),
            }
        )

        for cell in cells:
            qspec = construct_quote(market, float(chosen["ask"]), cell.q, cell.margin)
            if qspec is None:
                continue
            cert = first_contestable_trade(
                trades,
                decision_ts=decision_ts,
                latency_s=cell.latency_s,
                yes_quote=qspec["yes_quote"],
                tick=qspec["tick"],
                q=cell.q,
                strict=cell.strict,
            )
            row = {
                **base,
                "margin": cell.margin,
                "latency_s": cell.latency_s,
                "q": cell.q,
                "strict": cell.strict,
                "chosen_component": chosen["component"],
                "chosen_side": chosen["side"],
                "component_ask": chosen["ask"],
                "candle_end_ts": chosen["candle_end_ts"],
                "candle_age_s": chosen["candle_age_s"],
                **qspec,
                "contestable": cert is not None,
            }
            if cert:
                row.update(cert)
            candidate_rows.append(row)

        if i % 25 == 0:
            print(
                "PROGRESS",
                i,
                "/",
                len(traded),
                "calls",
                api.calls,
                "candidate_rows",
                len(candidate_rows),
                flush=True,
            )

    funnel = pd.DataFrame(funnel_rows)
    candidates = pd.DataFrame(candidate_rows)
    if funnel.empty:
        funnel = pd.DataFrame(columns=["combo", "stage"])
    if candidates.empty:
        candidates = pd.DataFrame(
            columns=[
                "combo",
                "day",
                "margin",
                "latency_s",
                "q",
                "strict",
                "chosen_component",
                "contestable",
                "guaranteed_net",
            ]
        )

    funnel.to_csv(OUT / "hbcd_market_funnel.csv", index=False)
    candidates.to_csv(OUT / "hbcd_all_candidate_rows.csv", index=False)

    # A component contract may not be reused to hedge multiple combo-NO
    # contracts. Select at most one combo per component ticker in each cell.
    allocated_parts: list[pd.DataFrame] = []
    contestable = candidates[candidates.contestable.fillna(False)].copy()
    if not contestable.empty:
        group_cols = ["margin", "latency_s", "q", "strict", "chosen_component"]
        contestable = contestable.sort_values(
            group_cols + ["guaranteed_net", "decision_ts", "combo"],
            ascending=[True, True, True, True, True, False, True, True],
        )
        allocated = contestable.groupby(group_cols, as_index=False, group_keys=False).head(1)
        allocated_parts.append(allocated)
    allocated = (
        pd.concat(allocated_parts, ignore_index=True)
        if allocated_parts
        else candidates.iloc[0:0].copy()
    )
    allocated.to_csv(OUT / "hbcd_contestability_rows.csv", index=False)

    all_days = pd.date_range(START_DAY, date.fromisoformat(END_DAY) - timedelta(days=1), freq="D")
    n_calendar_days = len(all_days)
    summaries: list[dict[str, Any]] = []
    for cell in cells:
        mask = (
            np.isclose(candidates.margin.astype(float), cell.margin)
            & np.isclose(candidates.latency_s.astype(float), cell.latency_s)
            & (candidates.q.astype(float) == cell.q)
            & (candidates.strict.astype(bool) == cell.strict)
        ) if len(candidates) else np.array([], dtype=bool)
        c = candidates[mask] if len(candidates) else candidates.iloc[0:0]

        amask = (
            np.isclose(allocated.margin.astype(float), cell.margin)
            & np.isclose(allocated.latency_s.astype(float), cell.latency_s)
            & (allocated.q.astype(float) == cell.q)
            & (allocated.strict.astype(bool) == cell.strict)
        ) if len(allocated) else np.array([], dtype=bool)
        a = allocated[amask].copy() if len(allocated) else allocated.iloc[0:0].copy()

        daily = pd.Series(0.0, index=all_days)
        if not a.empty:
            day_sum = a.assign(
                day_dt=pd.to_datetime(a.day, utc=True).dt.tz_convert(None).dt.normalize()
            ).groupby("day_dt").guaranteed_net.sum()
            for idx, value in day_sum.items():
                if idx in daily.index:
                    daily.loc[idx] = float(value)
        ci_lo, ci_hi = bootstrap_mean(daily.to_numpy(float))

        summary = {
            "margin": cell.margin,
            "latency_s": cell.latency_s,
            "q": cell.q,
            "strict": cell.strict,
            "calendar_days": n_calendar_days,
            "eligible_markets": int(len(c)),
            "contestable_markets_raw": int(c.contestable.fillna(False).sum()) if len(c) else 0,
            "allocated_unique_components": int(len(a)),
            "contestability_rate": float(c.contestable.fillna(False).mean()) if len(c) else 0.0,
            "allocated_contracts": float(cell.q * len(a)),
            "shadow_guaranteed_dollars": float(a.guaranteed_net.sum()) if len(a) else 0.0,
            "shadow_dollars_per_calendar_day": (
                float(a.guaranteed_net.sum()) / n_calendar_days if n_calendar_days else 0.0
            ),
            "mean_daily_bootstrap_ci_lo": ci_lo,
            "mean_daily_bootstrap_ci_hi": ci_hi,
            "contracts_per_calendar_day_upper_envelope": (
                float(cell.q * len(a)) / n_calendar_days if n_calendar_days else 0.0
            ),
            "median_public_trade_delay_s": (
                float(a.trade_delay_s.median()) if len(a) else float("nan")
            ),
            "median_public_qty": float(a.public_qty.median()) if len(a) else float("nan"),
            "median_component_candle_age_s": (
                float(a.candle_age_s.median()) if len(a) else float("nan")
            ),
            "evidence_tier": "Tier C contestability only",
        }
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT / "hbcd_cell_summary.csv", index=False)

    primary = summary_df[
        np.isclose(summary_df.margin, PRIMARY_MARGIN)
        & np.isclose(summary_df.latency_s, PRIMARY_LATENCY_S)
        & (summary_df.q == PRIMARY_Q)
        & (summary_df.strict == PRIMARY_STRICT)
    ]
    if len(primary) != 1:
        raise AssertionError(f"primary cell missing or duplicated: {len(primary)}")
    p = primary.iloc[0].to_dict()

    primary_rows = allocated[
        np.isclose(allocated.margin.astype(float), PRIMARY_MARGIN)
        & np.isclose(allocated.latency_s.astype(float), PRIMARY_LATENCY_S)
        & (allocated.q.astype(float) == PRIMARY_Q)
        & (allocated.strict.astype(bool) == PRIMARY_STRICT)
    ].copy() if len(allocated) else allocated.iloc[0:0].copy()

    concentration = {}
    if len(primary_rows):
        concentration = {
            "by_component_series": primary_rows.assign(
                component_series=primary_rows.chosen_component.str.split("-", n=1).str[0]
            ).groupby("component_series").guaranteed_net.sum().sort_values(ascending=False).to_dict(),
            "by_day": primary_rows.groupby("day").guaranteed_net.sum().sort_values(ascending=False).to_dict(),
            "largest_row_share": float(
                primary_rows.guaranteed_net.max() / primary_rows.guaranteed_net.sum()
            ) if primary_rows.guaranteed_net.sum() > 0 else float("nan"),
        }

    report = f"""# HBCD causal public-history audit

## Verdict

**Mechanical identity: {mechanical['status']}.**

**Public execution status: NOT CERTIFIED.** This audit measures whether a causal,
fee-stressed quote would have been strictly better than a later public trade. It
does not contain private RFQ acceptance, component depth, hedge FOK, quote
confirmation, or private fills.

## Frozen primary cell

- q = {PRIMARY_Q}
- quote latency from public combo `created_time` proxy = {PRIMARY_LATENCY_S:.2f}s
- minimum guaranteed net margin = {PRIMARY_MARGIN*100:.2f}c/contract
- strict certificate = our YES quote is at least one valid tick cheaper for the
  requester than a later public taker-YES trade
- component price = conservative worst observed ask in the last completed
  one-minute candle
- fee stress = taker quadratic rate plus maximum sub-cent rounding on **both**
  component and combo legs
- allocation = at most one combo per selected component market

## Primary result

```json
{json.dumps(p, indent=2, default=str)}
```

Concentration:

```json
{json.dumps(concentration, indent=2, default=str)}
```

## Data funnel

- Date range: {START_DAY} through {END_DAY} (end exclusive)
- Public MVE market records: {len(markets):,}
- Crypto-only combo records: {len(crypto):,}
- Crypto-only combos with reported volume: {len(traded):,}
- API calls: {api.calls:,}
- API errors: {len(api.errors)}
- Truncated pagination days: {truncated_days}

Funnel stages:

```json
{json.dumps(funnel.stage.value_counts(dropna=False).to_dict(), indent=2, default=str)}
```

## Mechanical validation

The script exhaustively checked {mechanical['grid_checks']:,} scalar-grid
states and {mechanical['random_checks']:,} random continuous states. Minimum
observed slack was {mechanical['minimum_slack']:.12g}.

## What would falsify the proposed engine

The idea is not promoted by this public result. It fails as an executable
strategy if any of these occur in Tier-A data:

1. accepted RFQs usually cannot be hedged before the three-second confirmation
   deadline;
2. component FOK failures or slippage consume the deterministic margin;
3. hedge fills followed by failed quote confirmations create orphan losses
   larger than completed-lock gains;
4. exact combo fee overrides exceed the stress model;
5. public `created_time` materially postdates the maker's actionable RFQ state;
6. accepted quantity is concentrated where component ask depth is absent;
7. the opportunity disappears after the quote policy is frozen.

## Required next proof

Run the q=1 authenticated state machine from `HBCD_FROZEN_SPEC.md`. Every
`rfq_created` must end in exactly one terminal reason: ineligible, quote
rejected, quote unaccepted, accepted-unhedged, accepted-hedged-unconfirmed,
confirmed-unexecuted, executed-settled, or unresolved incident. Only actual
cash PnL including orphan hedges can certify execution.

## Files

- `hbcd_mechanical.json`
- `hbcd_market_funnel.csv`
- `hbcd_all_candidate_rows.csv`
- `hbcd_contestability_rows.csv`
- `hbcd_cell_summary.csv`
"""

    (OUT / "hbcd_report.md").write_text(report)
    (OUT / "hbcd_run_metadata.json").write_text(
        json.dumps(
            {
                "start_day": START_DAY,
                "end_day": END_DAY,
                "max_pages_per_day": MAX_PAGES_PER_DAY,
                "max_candle_age_seconds": MAX_CANDLE_AGE_SECONDS,
                "primary": p,
                "mechanical": mechanical,
                "truncated_days": truncated_days,
                "api_calls": api.calls,
                "api_errors": api.errors,
                "funnel_counts": funnel.stage.value_counts(dropna=False).to_dict(),
            },
            indent=2,
            default=str,
        )
    )

    print("\nPRIMARY", json.dumps(p, indent=2, default=str), flush=True)
    print("\nREPORT\n", report, flush=True)


if __name__ == "__main__":
    evaluate()
