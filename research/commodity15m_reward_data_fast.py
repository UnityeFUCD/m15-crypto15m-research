"""Fast public-data loader for commodity15m incentive audits.

Filters the incentive API to `series_lip`, limits the historical window, batches
market metadata, and fetches archived one-minute paths concurrently. No orders
or credentials are used.
"""
from __future__ import annotations

import concurrent.futures
import time
from typing import Any

import pandas as pd

from research.reward_adjusted_commodity15m import (
    SERIES,
    get_json,
    parse_candle,
    reward_dollars,
    series_of,
)

LOOKBACK_DAYS = 21
MAX_WORKERS = 12


def all_series_lip(status: str) -> list[dict]:
    rows: list[dict] = []
    cursor = ""
    seen: set[str] = set()
    while True:
        params: dict[str, Any] = {
            "status": status,
            "type": "liquidity",
            "incentive_description": "series_lip",
            "limit": 10000,
        }
        if cursor:
            params["cursor"] = cursor
        body = get_json("/incentive_programs", params)
        batch = body.get("incentive_programs") or []
        rows.extend(item for item in batch if isinstance(item, dict))
        nxt = str(body.get("next_cursor") or body.get("cursor") or "")
        if not nxt or nxt in seen or not batch:
            break
        seen.add(nxt)
        cursor = nxt
    return rows


def load_programs() -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    cutoff = now - pd.Timedelta(days=LOOKBACK_DAYS)
    raw = all_series_lip("closed") + all_series_lip("paid_out")
    records = []
    seen: set[str] = set()
    for program in raw:
        pid = str(program.get("id") or "")
        if not pid or pid in seen:
            continue
        seen.add(pid)
        series = series_of(program)
        if series not in SERIES:
            continue
        start = pd.to_datetime(program.get("start_date"), utc=True, errors="coerce")
        end = pd.to_datetime(program.get("end_date"), utc=True, errors="coerce")
        target = float(program.get("target_size_fp") or 0.0)
        reward = reward_dollars(program.get("period_reward"))
        ticker = str(program.get("market_ticker") or "")
        if (pd.isna(start) or pd.isna(end) or end < cutoff or end > now
                or end <= start or target <= 0 or reward <= 0 or not ticker):
            continue
        records.append({
            "program_id": pid,
            "market_ticker": ticker,
            "series_ticker": series,
            "start": start,
            "end": end,
            "target_size": target,
            "period_reward_dollars": reward,
        })
    frame = pd.DataFrame(records).drop_duplicates("market_ticker")
    if frame.empty:
        raise RuntimeError("no recent commodity15m series_lip programs")
    return frame.sort_values("end").reset_index(drop=True)


def chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index:index + size] for index in range(0, len(values), size)]


def fetch_markets(tickers: list[str]) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for batch in chunks(tickers, 100):
        body = get_json("/historical/markets", {
            "tickers": ",".join(batch), "limit": 1000})
        for market in body.get("markets") or []:
            if isinstance(market, dict) and market.get("ticker"):
                output[str(market["ticker"])] = market
    missing = [ticker for ticker in tickers if ticker not in output]
    for batch in chunks(missing, 100):
        body = get_json("/markets", {
            "tickers": ",".join(batch), "limit": 1000})
        for market in body.get("markets") or []:
            if isinstance(market, dict) and market.get("ticker"):
                output[str(market["ticker"])] = market
    return output


def _historical_path(row: dict) -> tuple[str, list[dict]]:
    ticker = row["market_ticker"]
    params = {
        "start_ts": int(row["start"].timestamp()) - 120,
        "end_ts": int(row["end"].timestamp()) + 120,
        "period_interval": 1,
    }
    body: dict = {}
    for path in [
        f"/historical/markets/{ticker}/candlesticks",
        f"/series/{row['series_ticker']}/markets/{ticker}/candlesticks",
    ]:
        try:
            body = get_json(path, params)
            if body.get("candlesticks"):
                break
        except Exception:
            continue
    candles = [parse_candle(candle) for candle in body.get("candlesticks") or []]
    return ticker, sorted(
        [candle for candle in candles if candle is not None],
        key=lambda item: item["ts"])


def fetch_candles(programs: pd.DataFrame) -> dict[str, list[dict]]:
    rows = programs.to_dict("records")
    output: dict[str, list[dict]] = {}
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(_historical_path, row) for row in rows]
        for index, future in enumerate(
                concurrent.futures.as_completed(futures), start=1):
            try:
                ticker, candles = future.result()
            except Exception:
                continue
            if candles:
                output[ticker] = candles
            if index % 250 == 0:
                elapsed = time.time() - started
                print(
                    f"paths {index:,}/{len(rows):,}; kept {len(output):,}; "
                    f"{index/max(elapsed,1):.1f}/s",
                    flush=True,
                )
    return output
