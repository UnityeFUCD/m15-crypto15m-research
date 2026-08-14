#!/usr/bin/env python3
"""Prospective/post-cutoff evaluator for Anchor-Confirmed DRC (AC-DRC).

Research-only. This program uses only public Kalshi market data and never places an
order. It fails closed on missing fixed-point prices, missing floor strikes,
non-contiguous windows, malformed tickers, or incomplete one-minute quote bars.

Frozen discovery-independent trading rule evaluated here:
  * Core series: DOGE, ETH, SOL, XRP 15-minute markets.
  * Rebuild the original DRC-15 signal exactly:
      previous 15m benchmark return / rolling sample sigma of latest four >= 1;
      strict 15-minute continuity for all four observations;
      first 14..8 minute snapshot where NO bid is >= 0.65 and < 0.80;
      buy NO at the contemporaneous NO ask, one contract, hold to settlement.
  * Anchor confirmation: trade only the ETH DRC signal when at least one of
    DOGE/SOL/XRP has a DRC signal sharing the exact same close.

The historical packet ended on 2026-08-05. The default evaluation start is
2026-08-06 00:00:00 UTC, while 2026-08-02 onward is downloaded solely for warmup
and overlap diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
CORE_SERIES = {
    "DOGE": "KXDOGE15M",
    "ETH": "KXETH15M",
    "SOL": "KXSOL15M",
    "XRP": "KXXRP15M",
}
EASTERN = ZoneInfo("America/New_York")
TICKER_RE = re.compile(r"^[A-Z0-9]+-(\d{2}[A-Z]{3}\d{6})-[0-9]{2}$")


def utc_ts(text: str) -> datetime:
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_true_close(ticker: str) -> datetime:
    """Decode the event close embedded in Kalshi's ticker as US/Eastern time."""
    m = TICKER_RE.match(ticker)
    if not m:
        raise ValueError(f"Malformed 15m ticker: {ticker!r}")
    local_naive = datetime.strptime(m.group(1), "%y%b%d%H%M")
    return local_naive.replace(tzinfo=EASTERN).astimezone(timezone.utc)


def d(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def fixed_price(side_obj: Any, key: str = "close_dollars") -> float | None:
    if not isinstance(side_obj, dict):
        return None
    value = d(side_obj.get(key))
    if value is None or value < 0 or value > 1:
        return None
    return float(value)


def one_contract_fee(price: float) -> float:
    return math.ceil(0.07 * price * (1.0 - price) * 10_000 - 1e-12) / 10_000


def q1_cash_debit(price: float, fee: float) -> float:
    return math.ceil((price + fee) * 100 - 1e-12) / 100


class KalshiPublicClient:
    def __init__(self, timeout: float = 30.0, max_attempts: int = 8) -> None:
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "AC-DRC-research-audit/1.0"})
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.calls: list[dict[str, Any]] = []

    def get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = BASE_URL + path
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            started = time.time()
            try:
                r = self.s.get(url, params=params, timeout=self.timeout)
                elapsed = time.time() - started
                self.calls.append({
                    "url": r.url,
                    "status": r.status_code,
                    "elapsed_s": round(elapsed, 4),
                    "attempt": attempt,
                })
                if r.status_code == 429 or 500 <= r.status_code < 600:
                    raise requests.HTTPError(f"retryable HTTP {r.status_code}: {r.text[:300]}")
                r.raise_for_status()
                obj = r.json()
                if not isinstance(obj, dict):
                    raise ValueError("JSON response is not an object")
                return obj
            except Exception as exc:  # explicit retry log retained in calls
                last_error = exc
                if attempt == self.max_attempts:
                    break
                time.sleep(min(20.0, (2 ** (attempt - 1)) + random.random()))
        raise RuntimeError(f"GET failed after {self.max_attempts} attempts: {url}: {last_error}")


def fetch_markets(
    client: KalshiPublicClient,
    series: str,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cursor = ""
    seen: set[str] = set()
    while True:
        params: dict[str, Any] = {
            "series_ticker": series,
            "status": "settled",
            "min_settled_ts": int(start.timestamp()),
            "max_settled_ts": int(end.timestamp()),
            "limit": 1000,
            "mve_filter": "exclude",
        }
        if cursor:
            params["cursor"] = cursor
        obj = client.get("/markets", params)
        rows = obj.get("markets")
        if not isinstance(rows, list):
            raise RuntimeError(f"markets missing/not list for {series}")
        for row in rows:
            if not isinstance(row, dict):
                raise RuntimeError(f"non-object market row for {series}")
            ticker = str(row.get("ticker", ""))
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            out.append(row)
        nxt = obj.get("cursor")
        if not nxt:
            break
        if str(nxt) == cursor:
            raise RuntimeError("pagination cursor did not advance")
        cursor = str(nxt)
    return out


def chunks(items: list[Any], n: int) -> Iterable[list[Any]]:
    for i in range(0, len(items), n):
        yield items[i:i+n]


def fetch_candles(
    client: KalshiPublicClient,
    markets: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Batch by series/day so each response remains far below 10k candles."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for m in markets:
        groups[(str(m["series_ticker"]), m["true_close"].date().isoformat())].append(m)

    by_ticker: dict[str, list[dict[str, Any]]] = {}
    diagnostics: list[dict[str, Any]] = []
    for (series, day), rows in sorted(groups.items()):
        for batch in chunks(sorted(rows, key=lambda x: x["true_close"]), 100):
            tickers = [str(x["ticker"]) for x in batch]
            start = min(x["true_close"] for x in batch) - timedelta(minutes=20)
            end = max(x["true_close"] for x in batch) + timedelta(minutes=1)
            obj = client.get("/markets/candlesticks", {
                "market_tickers": ",".join(tickers),
                "start_ts": int(start.timestamp()),
                "end_ts": int(end.timestamp()),
                "period_interval": 1,
            })
            payload = obj.get("markets")
            if not isinstance(payload, list):
                raise RuntimeError(f"candlestick markets missing/not list for {series} {day}")
            returned: set[str] = set()
            total = 0
            for item in payload:
                if not isinstance(item, dict):
                    raise RuntimeError("non-object candlestick market")
                ticker = str(item.get("market_ticker") or item.get("ticker") or "")
                candles = item.get("candlesticks")
                if ticker not in tickers or not isinstance(candles, list):
                    continue
                by_ticker[ticker] = candles
                returned.add(ticker)
                total += len(candles)
            diagnostics.append({
                "series": series,
                "day": day,
                "requested_tickers": len(tickers),
                "returned_tickers": len(returned),
                "candles": total,
                "missing_tickers": sorted(set(tickers) - returned),
            })
    return by_ticker, diagnostics


def normalize_markets(raw_by_coin: dict[str, list[dict[str, Any]]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    for coin, markets in raw_by_coin.items():
        expected_series = CORE_SERIES[coin]
        for m in markets:
            ticker = str(m.get("ticker", ""))
            try:
                tc = parse_true_close(ticker)
            except Exception as exc:
                rejects.append({"coin": coin, "ticker": ticker, "reason": f"ticker:{exc}"})
                continue
            series = str(m.get("series_ticker") or expected_series)
            if series != expected_series or not ticker.startswith(expected_series + "-"):
                rejects.append({"coin": coin, "ticker": ticker, "reason": "series_mismatch"})
                continue
            floor = d(m.get("floor_strike"))
            result = str(m.get("result", "")).lower()
            if floor is None or floor <= 0:
                rejects.append({"coin": coin, "ticker": ticker, "reason": "missing_floor_strike"})
                continue
            if result not in {"yes", "no"}:
                rejects.append({"coin": coin, "ticker": ticker, "reason": f"bad_result:{result}"})
                continue
            rows.append({
                "ticker": ticker,
                "coin": coin,
                "series_ticker": series,
                "true_close": tc,
                "a0": float(floor),
                "result": result,
                "volume_fp": float(d(m.get("volume_fp")) or 0),
                "open_interest_fp": float(d(m.get("open_interest_fp")) or 0),
                "settlement_ts": m.get("settlement_ts"),
                "status": m.get("status"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No valid settled 15-minute markets returned")
    dup = df[df.duplicated(["coin", "true_close"], keep=False)]
    if not dup.empty:
        raise RuntimeError("Duplicate coin/true_close markets:\n" + dup.to_string(index=False))
    return df.sort_values(["coin", "true_close"]).reset_index(drop=True), rejects


def add_features(markets: pd.DataFrame) -> pd.DataFrame:
    x = markets.copy().sort_values(["coin", "true_close", "ticker"]).reset_index(drop=True)
    x["gap_minutes"] = x.groupby("coin")["true_close"].diff().dt.total_seconds() / 60.0
    x["previous_a0"] = x.groupby("coin")["a0"].shift(1)
    x["previous_return"] = x["a0"] / x["previous_a0"] - 1.0
    x.loc[x["gap_minutes"].ne(15.0), "previous_return"] = np.nan
    x["sigma4"] = x.groupby("coin", group_keys=False)["previous_return"].apply(
        lambda s: s.rolling(4, min_periods=4).std(ddof=1)
    ).reset_index(level=0, drop=True)
    x["z_up"] = x["previous_return"] / (x["sigma4"] + 1e-6)
    contiguous = x["gap_minutes"].eq(15.0)
    x["strict_four"] = contiguous.copy()
    for lag in (1, 2, 3):
        x["strict_four"] &= contiguous.groupby(x["coin"]).shift(lag).fillna(False).astype(bool)
    return x


def build_drc_entries(
    featured: pd.DataFrame,
    candle_map: dict[str, list[dict[str, Any]]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for row in featured.itertuples(index=False):
        candles = candle_map.get(row.ticker, [])
        points: list[dict[str, Any]] = []
        malformed = 0
        for c in candles:
            if not isinstance(c, dict):
                malformed += 1
                continue
            try:
                end_ts = int(c["end_period_ts"])
            except Exception:
                malformed += 1
                continue
            delta = row.true_close.timestamp() - end_ts
            ml_round = int(round(delta / 60.0))
            if abs(delta - ml_round * 60.0) > 2.0 or not (8 <= ml_round <= 14):
                continue
            yes_bid = fixed_price(c.get("yes_bid"))
            yes_ask = fixed_price(c.get("yes_ask"))
            if yes_bid is None or yes_ask is None:
                continue
            no_bid = 1.0 - yes_ask
            no_ask = 1.0 - yes_bid
            if not (0 <= no_bid <= no_ask <= 1):
                continue
            points.append({
                "minutes_left": ml_round,
                "end_period_ts": end_ts,
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "no_bid": no_bid,
                "no_ask": no_ask,
                "yes_bid_size": None,
                "no_bid_size": None,
            })
        points.sort(key=lambda p: p["minutes_left"], reverse=True)
        eligible = [p for p in points if 0.65 <= p["no_bid"] < 0.80]
        diagnostics.append({
            "ticker": row.ticker,
            "coin": row.coin,
            "true_close": row.true_close.isoformat(),
            "candles_total": len(candles),
            "decision_points": len(points),
            "malformed": malformed,
            "eligible_price_points": len(eligible),
        })
        if not eligible:
            continue
        p = eligible[0]
        if not bool(row.strict_four) or not np.isfinite(row.z_up) or row.z_up < 1.0:
            continue
        ask = float(p["no_ask"])
        fee = one_contract_fee(ask)
        debit = q1_cash_debit(ask, fee)
        won = int(row.result == "no")
        candidates.append({
            "ticker": row.ticker,
            "coin": row.coin,
            "true_close": row.true_close,
            "result": row.result,
            "won": won,
            "a0": row.a0,
            "previous_return": row.previous_return,
            "sigma4": row.sigma4,
            "z_up": row.z_up,
            "strict_four": bool(row.strict_four),
            "minutes_left": p["minutes_left"],
            "yes_bid": p["yes_bid"],
            "yes_ask": p["yes_ask"],
            "no_bid": p["no_bid"],
            "no_ask": ask,
            "fee_q1": fee,
            "debit_q1": debit,
            "pnl_q1_cash": won - debit,
            "volume_fp": row.volume_fp,
            "open_interest_fp": row.open_interest_fp,
        })
    return pd.DataFrame(candidates), diagnostics


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"n": 0}
    return {
        "n": int(len(df)),
        "wins": int(df["won"].sum()),
        "win_rate": float(df["won"].mean()),
        "mean_no_ask": float(df["no_ask"].mean()),
        "mean_q1_cash_pnl": float(df["pnl_q1_cash"].mean()),
        "total_q1_cash_pnl": float(df["pnl_q1_cash"].sum()),
        "days": int(df["true_close"].dt.date.nunique()),
        "first_close": df["true_close"].min().isoformat(),
        "last_close": df["true_close"].max().isoformat(),
        "median_volume_fp": float(df["volume_fp"].median()),
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json_gz(path: Path, obj: Any) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("ac_drc_postcutoff"))
    ap.add_argument("--fetch-start", default="2026-08-02T00:00:00Z")
    ap.add_argument("--evaluation-start", default="2026-08-06T00:00:00Z")
    ap.add_argument("--end", default=datetime.now(timezone.utc).isoformat())
    args = ap.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    fetch_start = utc_ts(args.fetch_start)
    eval_start = utc_ts(args.evaluation_start)
    end = utc_ts(args.end)
    if not fetch_start < eval_start < end:
        raise SystemExit("Require fetch_start < evaluation_start < end")

    client = KalshiPublicClient()
    raw_by_coin: dict[str, list[dict[str, Any]]] = {}
    for coin, series in CORE_SERIES.items():
        raw_by_coin[coin] = fetch_markets(client, series, fetch_start, end)
        print(f"{coin}: {len(raw_by_coin[coin])} settled markets", flush=True)

    markets, market_rejects = normalize_markets(raw_by_coin)
    markets = markets[(markets["true_close"] >= fetch_start) & (markets["true_close"] <= end)].copy()
    if markets.empty:
        raise RuntimeError("No normalized markets in requested true-close interval")

    raw_market_path = out / "raw_markets.json.gz"
    write_json_gz(raw_market_path, raw_by_coin)
    candles, candle_fetch = fetch_candles(client, markets.to_dict("records"))
    raw_candle_path = out / "raw_candles.json.gz"
    write_json_gz(raw_candle_path, candles)

    featured = add_features(markets)
    base, decision_diag = build_drc_entries(featured, candles)
    if base.empty:
        base = pd.DataFrame(columns=[
            "ticker", "coin", "true_close", "result", "won", "a0",
            "previous_return", "sigma4", "z_up", "strict_four", "minutes_left",
            "yes_bid", "yes_ask", "no_bid", "no_ask", "fee_q1", "debit_q1",
            "pnl_q1_cash", "volume_fp", "open_interest_fp",
        ])
    else:
        base = base.sort_values(["true_close", "coin"]).reset_index(drop=True)

    if base.empty:
        ac = base.copy()
    else:
        coins_by_close = base.groupby("true_close")["coin"].agg(lambda s: set(s))
        confirmed_closes = set(coins_by_close[
            coins_by_close.apply(lambda s: "ETH" in s and bool(s - {"ETH"}))
        ].index)
        ac = base[(base["coin"] == "ETH") & base["true_close"].isin(confirmed_closes)].copy()
        ac["confirming_coins"] = ac["true_close"].map(
            lambda tc: ",".join(sorted(coins_by_close.loc[tc] - {"ETH"}))
        )

    for frame in (featured, base, ac):
        if "true_close" in frame:
            frame["evaluation"] = frame["true_close"] >= eval_start

    featured.to_csv(out / "normalized_markets_with_features.csv", index=False)
    base.to_csv(out / "base_drc_candidates.csv", index=False)
    ac.to_csv(out / "ac_drc_candidates.csv", index=False)
    pd.DataFrame(decision_diag).to_csv(out / "decision_diagnostics.csv", index=False)
    pd.DataFrame(candle_fetch).to_csv(out / "candle_fetch_diagnostics.csv", index=False)
    pd.DataFrame(market_rejects).to_csv(out / "market_rejects.csv", index=False)
    pd.DataFrame(client.calls).to_csv(out / "http_calls.csv", index=False)

    base_eval = base[base.get("evaluation", False)].copy() if not base.empty else base
    ac_eval = ac[ac.get("evaluation", False)].copy() if not ac.empty else ac
    overlap_base = base[~base.get("evaluation", True)].copy() if not base.empty else base
    overlap_ac = ac[~ac.get("evaluation", True)].copy() if not ac.empty else ac

    report = {
        "frozen_rule": "ETH DRC candidate AND >=1 same-close DOGE/SOL/XRP DRC candidate; trade ETH NO q1 at first 14..8m eligible ask",
        "fetch_start": fetch_start.isoformat(),
        "evaluation_start": eval_start.isoformat(),
        "end": end.isoformat(),
        "series": CORE_SERIES,
        "normalized_markets": int(len(markets)),
        "market_counts": markets.groupby("coin").size().astype(int).to_dict(),
        "market_rejects": len(market_rejects),
        "candlestick_tickers": len(candles),
        "http_calls": len(client.calls),
        "base_overlap": summarize(overlap_base),
        "ac_overlap": summarize(overlap_ac),
        "base_postcutoff": summarize(base_eval),
        "ac_postcutoff": summarize(ac_eval),
        "coverage": {
            "markets_missing_candle_payload": int(sum(1 for t in markets["ticker"] if t not in candles)),
            "markets_with_7_decision_points": int(sum(1 for x in decision_diag if x["decision_points"] == 7)),
            "markets_with_lt7_decision_points": int(sum(1 for x in decision_diag if x["decision_points"] < 7)),
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    manifest = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest[path.name] = {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
