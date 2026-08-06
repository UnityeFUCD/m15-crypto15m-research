"""Clean seven-day Reward-Funded Hedged Liquidity (RFHL) discovery audit.

This is a self-contained public-data experiment. It never reads credentials
and never places an order.

Architecture
------------
1. Find recently incentivized KXGOLD15M, KXSILVER15M, and KXWTI15M markets.
2. At the first complete minute after the incentive period begins, quote only
   the cheaper outcome one tick inside the spread. This makes the order the
   best bid at submission and avoids assuming credit behind a full target.
3. Earn a conservative share of the liquidity reward while the maker order
   rests.
4. If the maker order fills, buy the complementary outcome using an IOC. A
   completed pair pays exactly $1, replacing settlement risk with the bounded
   maker-price + hedge-price + fee cost.
5. If the maker order never fills, cancel at the frozen horizon and keep only
   the modeled reward.

Conservative choices
--------------------
* Program `period_reward` is converted from centi-cents to dollars by /10,000,
  matching the live program schedules observed in the public API.
* Competitors are assumed to maintain target size on BOTH outcomes for the
  full reward period.
* Modeled reward below the official $1 minimum is zero. Eligible reward is
  rounded down to the nearest cent.
* Hedge execution uses the WORST complementary ask observed in the first full
  minute after the maker fill, plus 0-5 cents of additional stress.
* Two maker-fill definitions are required: strict minute-close crossing and
  optimistic intraminute touch.
* Parameters are selected on chronological validation and evaluated once on
  the final chronological test slice.

A discovery PASS is not permission to trade. Official score, queue priority,
subsecond hedge latency, IOC depth, and actual reward credits remain
prospective facts. The account's existing KILL state remains binding.
"""
from __future__ import annotations

import concurrent.futures
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
SESSION.headers.update({"User-Agent": "rfhl-clean-research/1.0"})

SERIES = ["KXGOLD15M", "KXSILVER15M", "KXWTI15M"]
LOOKBACK_DAYS = 7
MAX_WORKERS = 12
TICK = 0.01
RNG = np.random.default_rng(2026080641)

QTY_GRID = [16, 20, 32, 40, 50, 75, 100]
MAX_PRICE_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
CANCEL_MIN_GRID = [2, 3, 5, 8, 11, 14]
CAP_GRID = [1, 3]
SELECTION_GRID = ["reward_cost", "cheapest", "reward_density"]
SLIPPAGE_GRID_C = [0, 1, 2, 3, 5]
FILL_MODELS = ["strict", "touch"]


def get_json(path: str, params: dict | None = None, retries: int = 5) -> dict:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            response = SESSION.get(BASE + path, params=params, timeout=30)
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


def incentive_pages(status: str) -> list[dict]:
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
        if not isinstance(batch, list):
            raise RuntimeError("incentive_programs is not a list")
        rows.extend(item for item in batch if isinstance(item, dict))
        nxt = str(body.get("next_cursor") or body.get("cursor") or "")
        if not nxt or not batch or nxt in seen:
            break
        seen.add(nxt)
        cursor = nxt
    return rows


def series_of(program: dict) -> str:
    explicit = str(program.get("series_ticker") or "").upper()
    if explicit in SERIES:
        return explicit
    ticker = str(program.get("market_ticker") or "").upper()
    for series in SERIES:
        if ticker.startswith(series + "-"):
            return series
    return ""


def reward_dollars(raw: Any) -> float:
    return float(raw or 0.0) / 10_000.0


def load_programs() -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    cutoff = now - pd.Timedelta(days=LOOKBACK_DAYS)
    raw = incentive_pages("closed") + incentive_pages("paid_out")
    records: list[dict] = []
    seen: set[str] = set()
    for program in raw:
        program_id = str(program.get("id") or "")
        if not program_id or program_id in seen:
            continue
        seen.add(program_id)
        series = series_of(program)
        if not series:
            continue
        start = pd.to_datetime(
            program.get("start_date"), utc=True, errors="coerce")
        end = pd.to_datetime(
            program.get("end_date"), utc=True, errors="coerce")
        ticker = str(program.get("market_ticker") or "")
        target = float(program.get("target_size_fp") or 0.0)
        reward = reward_dollars(program.get("period_reward"))
        if (
            pd.isna(start) or pd.isna(end) or end <= start
            or end < cutoff or end > now or not ticker
            or target <= 0 or reward <= 0
        ):
            continue
        records.append({
            "program_id": program_id,
            "market_ticker": ticker,
            "series_ticker": series,
            "start": start,
            "end": end,
            "target_size": target,
            "period_reward_dollars": reward,
            "discount_factor_bps": float(
                program.get("discount_factor_bps") or 0.0),
        })
    frame = pd.DataFrame(records).drop_duplicates("market_ticker")
    if frame.empty:
        raise RuntimeError("no recent commodity15m series_lip programs")
    frame = frame.sort_values("end").reset_index(drop=True)
    frame.to_csv(OUT / "rfhl_clean_programs.csv", index=False)
    return frame


def parse_candle(candle: dict) -> dict | None:
    try:
        yes_bid = candle["yes_bid"]
        yes_ask = candle["yes_ask"]

        def value(block: dict, name: str) -> float:
            raw = block.get(f"{name}_dollars", block.get(name))
            return float(raw)

        row = {
            "ts": int(candle["end_period_ts"]),
            "yb_low": value(yes_bid, "low"),
            "yb_high": value(yes_bid, "high"),
            "yb_close": value(yes_bid, "close"),
            "ya_low": value(yes_ask, "low"),
            "ya_high": value(yes_ask, "high"),
            "ya_close": value(yes_ask, "close"),
            "volume": float(
                candle.get("volume_fp", candle.get("volume", 0.0)) or 0.0),
        }
    except (KeyError, TypeError, ValueError):
        return None
    if not (0 < row["yb_close"] < row["ya_close"] < 1):
        return None
    return row


def fetch_one_path(row: dict) -> tuple[str, list[dict]]:
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
    candles = [
        parsed for parsed in (
            parse_candle(candle)
            for candle in body.get("candlesticks") or []
        ) if parsed is not None
    ]
    return ticker, sorted(candles, key=lambda item: item["ts"])


def fetch_paths(programs: pd.DataFrame) -> dict[str, list[dict]]:
    records = programs.to_dict("records")
    output: dict[str, list[dict]] = {}
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(fetch_one_path, row) for row in records]
        for index, future in enumerate(
                concurrent.futures.as_completed(futures), start=1):
            try:
                ticker, candles = future.result()
            except Exception:
                continue
            if candles:
                output[ticker] = candles
            if index % 250 == 0:
                elapsed = max(time.time() - started, 1e-9)
                print(
                    f"paths {index:,}/{len(records):,}; "
                    f"kept {len(output):,}; {index/elapsed:.1f}/s",
                    flush=True,
                )
    return output


def first_entry(
    candles: list[dict], start: pd.Timestamp, end: pd.Timestamp
) -> dict | None:
    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())
    candidates = [
        candle for candle in candles
        if start_ts + 30 <= candle["ts"] <= min(start_ts + 150, end_ts)
    ]
    return min(candidates, key=lambda item: item["ts"]) \
        if candidates else None


def first_fill(
    candles: list[dict], *, entry_ts: int, end_ts: int,
    side: str, price: float, model: str,
) -> int | None:
    for candle in candles:
        if candle["ts"] <= entry_ts or candle["ts"] > end_ts:
            continue
        if candle["volume"] <= 0:
            continue
        if side == "yes":
            observed = candle["ya_close"] \
                if model == "strict" else candle["ya_low"]
            if observed <= price + 1e-9:
                return int(candle["ts"])
        else:
            trigger = 1.0 - price
            observed = candle["yb_close"] \
                if model == "strict" else candle["yb_high"]
            if observed >= trigger - 1e-9:
                return int(candle["ts"])
    return None


def next_candle(candles: list[dict], after_ts: int) -> dict | None:
    candidates = [candle for candle in candles if candle["ts"] > after_ts]
    return min(candidates, key=lambda item: item["ts"]) \
        if candidates else None


def taker_fee_total(quantity: int, price: float) -> float:
    raw = 0.07 * quantity * price * (1.0 - price)
    return math.ceil(raw * 10_000 - 1e-12) / 10_000


def build_states(
    programs: pd.DataFrame, paths: dict[str, list[dict]]
) -> pd.DataFrame:
    rows: list[dict] = []
    for program in programs.itertuples(index=False):
        candles = paths.get(program.market_ticker)
        if not candles:
            continue
        entry = first_entry(candles, program.start, program.end)
        if entry is None:
            continue
        yes_bid = float(entry["yb_close"])
        yes_ask = float(entry["ya_close"])
        no_bid = 1.0 - yes_ask
        no_ask = 1.0 - yes_bid
        if yes_bid <= no_bid:
            side, old_bid, own_ask = "yes", yes_bid, yes_ask
        else:
            side, old_bid, own_ask = "no", no_bid, no_ask
        maker_price = round(old_bid + TICK, 2)
        if not (0 < maker_price < own_ask - 1e-9):
            continue
        entry_ts = int(entry["ts"])
        end_ts = int(program.end.timestamp())
        period_seconds = max(
            (program.end - program.start).total_seconds(), 1.0)

        for fill_model in FILL_MODELS:
            possible_fill = first_fill(
                candles, entry_ts=entry_ts, end_ts=end_ts,
                side=side, price=maker_price, model=fill_model)
            for cancel_min in CANCEL_MIN_GRID:
                cancel_ts = min(entry_ts + cancel_min * 60, end_ts)
                fill_ts = possible_fill \
                    if possible_fill is not None and possible_fill <= cancel_ts \
                    else None
                rest_until = fill_ts if fill_ts is not None else cancel_ts
                hedge_candle = next_candle(candles, fill_ts) \
                    if fill_ts is not None else None
                if fill_ts is not None and hedge_candle is None:
                    continue
                if hedge_candle is None:
                    base_hedge_ask = np.nan
                elif side == "yes":
                    # Maker YES; hedge with NO. Worst NO ask during the next
                    # full candle is one minus the minimum YES bid.
                    base_hedge_ask = 1.0 - float(hedge_candle["yb_low"])
                else:
                    # Maker NO; hedge with YES at the maximum ask observed in
                    # the next full candle.
                    base_hedge_ask = float(hedge_candle["ya_high"])

                for extra_cost_c in SLIPPAGE_GRID_C:
                    if fill_ts is None:
                        hedge_price = np.nan
                    else:
                        hedge_price = min(
                            max(base_hedge_ask + extra_cost_c / 100.0, 0.0001),
                            0.9999,
                        )
                    rows.append({
                        "market_ticker": program.market_ticker,
                        "series_ticker": program.series_ticker,
                        "program_id": program.program_id,
                        "start": program.start,
                        "end": program.end,
                        "day": program.end.date().isoformat(),
                        "week": (
                            f"{program.end.isocalendar().year}-"
                            f"{program.end.isocalendar().week:02d}"
                        ),
                        "close_ts": end_ts,
                        "fill_model": fill_model,
                        "cancel_min": cancel_min,
                        "extra_cost_c": extra_cost_c,
                        "side": side,
                        "maker_price": maker_price,
                        "own_ask": own_ask,
                        "target_size": float(program.target_size),
                        "period_reward_dollars": float(
                            program.period_reward_dollars),
                        "period_seconds": period_seconds,
                        "rest_seconds": max(rest_until - entry_ts, 0),
                        "maker_filled": fill_ts is not None,
                        "hedge_price": hedge_price,
                    })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no RFHL states were constructed")
    frame.to_parquet(OUT / "rfhl_clean_states.parquet", index=False)
    return frame


def apply_quantity(frame: pd.DataFrame, quantity: int) -> pd.DataFrame:
    x = frame.copy()
    own_score_time = quantity * x["rest_seconds"]
    competitor_score_time = (
        2.0 * x["target_size"] * x["period_seconds"]
    )
    share = own_score_time / (competitor_score_time + own_score_time)
    raw_reward = x["period_reward_dollars"] * share
    x["reward_pnl_raw"] = raw_reward
    x["reward_pnl"] = np.where(
        raw_reward >= 1.0,
        np.floor(raw_reward * 100.0 + 1e-12) / 100.0,
        0.0,
    )

    fees = np.zeros(len(x), dtype=float)
    filled = x["maker_filled"] & x["hedge_price"].notna()
    if filled.any():
        fees[filled.to_numpy()] = [
            taker_fee_total(quantity, float(price))
            for price in x.loc[filled, "hedge_price"]
        ]
    x["hedge_fee"] = fees
    x["trading_pnl"] = np.where(
        x["maker_filled"],
        quantity * (
            1.0 - x["maker_price"] - x["hedge_price"]
        ) - x["hedge_fee"],
        0.0,
    )
    x["combined_pnl"] = x["reward_pnl"] + x["trading_pnl"]
    x["quantity"] = quantity
    x["reward_cost"] = x["reward_pnl"] / np.maximum(
        -x["trading_pnl"], 0.0001)
    x["reward_density"] = (
        x["period_reward_dollars"] / x["target_size"]
    )
    x["transient_one_leg_cost"] = (
        quantity * x["maker_price"]
    )
    return x


def chronological_splits(
    frame: pd.DataFrame,
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    days = sorted(pd.to_datetime(frame["end"], utc=True).dt.normalize().unique())
    if len(days) < 5:
        raise RuntimeError(f"only {len(days)} calendar days available")
    n = len(days)
    train_index = max(min(int(math.floor(n * 0.50)), n - 2), 1)
    valid_index = max(min(int(math.floor(n * 0.75)), n - 1), train_index + 1)
    return {
        "train": (pd.Timestamp(days[0]), pd.Timestamp(days[train_index])),
        "valid": (pd.Timestamp(days[train_index]), pd.Timestamp(days[valid_index])),
        "test": (
            pd.Timestamp(days[valid_index]),
            pd.Timestamp(days[-1]) + pd.Timedelta(days=1),
        ),
    }


def select_close(
    frame: pd.DataFrame, cap: int, method: str
) -> pd.DataFrame:
    if method == "reward_cost":
        columns = [
            "close_ts", "reward_cost", "transient_one_leg_cost",
            "series_ticker",
        ]
        ascending = [True, False, True, True]
    elif method == "cheapest":
        columns = [
            "close_ts", "transient_one_leg_cost", "reward_cost",
            "series_ticker",
        ]
        ascending = [True, True, False, True]
    else:
        columns = [
            "close_ts", "reward_density", "transient_one_leg_cost",
            "series_ticker",
        ]
        ascending = [True, False, True, True]
    return (
        frame.sort_values(columns, ascending=ascending)
        .groupby("close_ts", as_index=False)
        .head(cap)
        .sort_values(["close_ts", "series_ticker"])
    )


@dataclass
class Metrics:
    split: str
    fill_model: str
    quantity: int
    max_maker_price: float
    cancel_min: int
    cap: int
    selection: str
    extra_cost_c: int
    posted: int
    maker_fills: int
    fill_rate: float
    trading_pnl: float
    raw_reward_entitlement: float
    paid_reward_lower: float
    combined_pnl: float
    mean_day: float
    sd_day: float
    t_stat: float
    max_drawdown: float
    worst_window: float
    max_transient_one_leg_cost: float
    positive_day_fraction: float


def metrics(
    selected: pd.DataFrame, *, split: str,
    splits: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    fill_model: str, quantity: int, max_maker_price: float,
    cancel_min: int, cap: int, selection: str, extra_cost_c: int,
) -> Metrics:
    start, end = splits[split]
    days = pd.date_range(
        start.normalize(), end.normalize() - pd.Timedelta(days=1), freq="D"
    ).date
    if selected.empty:
        return Metrics(
            split, fill_model, quantity, max_maker_price, cancel_min,
            cap, selection, extra_cost_c, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        )
    daily = selected.groupby("day")["combined_pnl"].sum().reindex(
        [day.isoformat() for day in days], fill_value=0.0)
    window = selected.groupby("close_ts")["combined_pnl"].sum().sort_index()
    equity = window.cumsum()
    drawdown = equity.cummax() - equity
    sd_day = float(daily.std(ddof=1))
    mean_day = float(daily.mean())
    return Metrics(
        split=split,
        fill_model=fill_model,
        quantity=quantity,
        max_maker_price=max_maker_price,
        cancel_min=cancel_min,
        cap=cap,
        selection=selection,
        extra_cost_c=extra_cost_c,
        posted=int(len(selected)),
        maker_fills=int(selected["maker_filled"].sum()),
        fill_rate=float(selected["maker_filled"].mean()),
        trading_pnl=float(selected["trading_pnl"].sum()),
        raw_reward_entitlement=float(
            selected["reward_pnl_raw"].sum()),
        paid_reward_lower=float(selected["reward_pnl"].sum()),
        combined_pnl=float(selected["combined_pnl"].sum()),
        mean_day=mean_day,
        sd_day=sd_day,
        t_stat=(
            mean_day / (sd_day / math.sqrt(len(daily)))
            if sd_day > 0 else 0.0
        ),
        max_drawdown=float(drawdown.max()) if len(drawdown) else 0.0,
        worst_window=float(window.min()) if len(window) else 0.0,
        max_transient_one_leg_cost=float(
            selected["transient_one_leg_cost"].max()),
        positive_day_fraction=float((daily > 0).mean()),
    )


def select_validation_policy(
    states: pd.DataFrame,
    splits: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> tuple[dict, pd.DataFrame]:
    start, end = splits["valid"]
    validation = states[
        (states["end"] >= start) & (states["end"] < end)
        & states["extra_cost_c"].eq(2)
    ]
    rows: list[dict] = []
    for fill_model in FILL_MODELS:
        model_rows = validation[validation["fill_model"].eq(fill_model)]
        for quantity in QTY_GRID:
            sized = apply_quantity(model_rows, quantity)
            for max_price in MAX_PRICE_GRID:
                for cancel_min in CANCEL_MIN_GRID:
                    eligible = sized[
                        sized["cancel_min"].eq(cancel_min)
                        & (sized["maker_price"] <= max_price)
                    ]
                    for cap in CAP_GRID:
                        for selection in SELECTION_GRID:
                            chosen = select_close(eligible, cap, selection)
                            rows.append(asdict(metrics(
                                chosen, split="valid", splits=splits,
                                fill_model=fill_model, quantity=quantity,
                                max_maker_price=max_price,
                                cancel_min=cancel_min, cap=cap,
                                selection=selection, extra_cost_c=2,
                            )))
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "rfhl_clean_validation_grid.csv", index=False)
    keys = [
        "quantity", "max_maker_price", "cancel_min", "cap", "selection"
    ]
    worst = (
        grid.groupby(keys).agg(
            worst_pnl=("combined_pnl", "min"),
            worst_t=("t_stat", "min"),
            worst_drawdown=("max_drawdown", "max"),
            worst_transient=("max_transient_one_leg_cost", "max"),
            min_posted=("posted", "min"),
            min_fills=("maker_fills", "min"),
        ).reset_index()
    )
    viable = worst[
        (worst["min_posted"] >= 50)
        & (worst["worst_pnl"] > 0)
        & (worst["worst_transient"] <= 90.0)
    ].copy()
    if viable.empty:
        viable = worst[
            (worst["min_posted"] >= 25)
            & (worst["worst_transient"] <= 90.0)
        ].copy()
    if viable.empty:
        viable = worst.copy()
    viable["objective"] = (
        viable["worst_t"]
        + 0.10 * np.log1p(viable["min_posted"])
        - 0.015 * viable["worst_drawdown"]
        - 0.002 * viable["worst_transient"]
    )
    winner = viable.sort_values(
        ["objective", "worst_t", "worst_pnl"],
        ascending=False,
    ).iloc[0].to_dict()
    return winner, grid


def evaluate(
    states: pd.DataFrame,
    splits: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    policy: dict, fill_model: str, extra_cost_c: int, split: str,
) -> tuple[pd.DataFrame, Metrics]:
    start, end = splits[split]
    eligible = states[
        (states["end"] >= start) & (states["end"] < end)
        & states["fill_model"].eq(fill_model)
        & states["extra_cost_c"].eq(extra_cost_c)
        & states["cancel_min"].eq(int(policy["cancel_min"]))
        & (states["maker_price"] <= float(policy["max_maker_price"]))
    ]
    sized = apply_quantity(eligible, int(policy["quantity"]))
    selected = select_close(
        sized, int(policy["cap"]), str(policy["selection"]))
    return selected, metrics(
        selected, split=split, splits=splits,
        fill_model=fill_model, quantity=int(policy["quantity"]),
        max_maker_price=float(policy["max_maker_price"]),
        cancel_min=int(policy["cancel_min"]), cap=int(policy["cap"]),
        selection=str(policy["selection"]), extra_cost_c=extra_cost_c,
    )


def day_bootstrap(
    selected: pd.DataFrame, split: str,
    splits: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    repetitions: int = 12000,
) -> dict:
    start, end = splits[split]
    days = pd.date_range(
        start.normalize(), end.normalize() - pd.Timedelta(days=1), freq="D"
    ).date
    daily = selected.groupby("day")["combined_pnl"].sum().reindex(
        [day.isoformat() for day in days], fill_value=0.0).to_numpy()
    draws = RNG.integers(
        0, len(daily), size=(repetitions, len(daily)))
    means = daily[draws].mean(axis=1)
    return {
        "days": int(len(daily)),
        "observed_mean": float(daily.mean()),
        "ci_lo": float(np.quantile(means, 0.025)),
        "ci_hi": float(np.quantile(means, 0.975)),
        "p_nonpositive": float(np.mean(means <= 0)),
        "daily_values": daily.tolist(),
    }


def leave_one_series(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for series in sorted(selected["series_ticker"].unique()):
        subset = selected[~selected["series_ticker"].eq(series)]
        rows.append({
            "excluded_series": series,
            "posted": int(len(subset)),
            "trading_pnl": float(subset["trading_pnl"].sum()),
            "paid_reward_lower": float(subset["reward_pnl"].sum()),
            "combined_pnl": float(subset["combined_pnl"].sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    fetched_at = datetime.now(timezone.utc).isoformat()
    programs = load_programs()
    print(
        f"programs={len(programs):,}; days={programs.end.dt.date.nunique()}",
        flush=True,
    )
    paths = fetch_paths(programs)
    print(f"paths={len(paths):,}/{len(programs):,}", flush=True)
    states = build_states(programs, paths)
    print(
        f"states={len(states):,}; markets={states.market_ticker.nunique():,}",
        flush=True,
    )
    splits = chronological_splits(states)
    policy, _ = select_validation_policy(states, splits)

    sensitivity_rows: list[dict] = []
    selected_at_2c: dict[str, pd.DataFrame] = {}
    test_results: dict[str, dict[str, dict]] = {}
    for fill_model in FILL_MODELS:
        test_results[fill_model] = {}
        for extra_cost_c in SLIPPAGE_GRID_C:
            selected, result = evaluate(
                states, splits, policy, fill_model,
                extra_cost_c, "test")
            boot = day_bootstrap(selected, "test", splits)
            test_results[fill_model][str(extra_cost_c)] = {
                "metrics": asdict(result),
                "bootstrap": boot,
            }
            sensitivity_rows.append({
                **asdict(result),
                "ci_lo": boot["ci_lo"],
                "ci_hi": boot["ci_hi"],
                "p_nonpositive": boot["p_nonpositive"],
            })
            if extra_cost_c == 2:
                selected_at_2c[fill_model] = selected

    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(
        OUT / "rfhl_clean_test_sensitivity.csv", index=False)

    leave_rows = []
    for fill_model, selected in selected_at_2c.items():
        table = leave_one_series(selected)
        if len(table):
            table.insert(0, "fill_model", fill_model)
            leave_rows.append(table)
    leave = pd.concat(leave_rows, ignore_index=True) \
        if leave_rows else pd.DataFrame()
    leave.to_csv(OUT / "rfhl_clean_leave_one_series.csv", index=False)

    core_pass = True
    for fill_model in FILL_MODELS:
        item = test_results[fill_model]["2"]
        metric_row = item["metrics"]
        bootstrap_row = item["bootstrap"]
        model_leave = leave[
            leave["fill_model"].eq(fill_model)] if len(leave) else leave
        core_pass &= (
            metric_row["posted"] >= 50
            and metric_row["combined_pnl"] > 0
            and bootstrap_row["ci_lo"] > 0
            and metric_row["max_transient_one_leg_cost"] <= 90.0
            and len(model_leave) > 0
            and (model_leave["combined_pnl"] > 0).all()
        )
    five_cent_pass = all(
        test_results[model]["5"]["metrics"]["combined_pnl"] > 0
        for model in FILL_MODELS
    )
    discovery_pass = bool(core_pass and five_cent_pass)

    summary = {
        "fetched_at": fetched_at,
        "programs": int(len(programs)),
        "program_days": int(programs.end.dt.date.nunique()),
        "path_coverage": int(len(paths)),
        "state_rows": int(len(states)),
        "markets_in_states": int(states.market_ticker.nunique()),
        "splits": {
            key: [str(value[0]), str(value[1])]
            for key, value in splits.items()
        },
        "validation_policy": policy,
        "test": test_results,
        "core_pass_at_2c": bool(core_pass),
        "positive_at_5c_both_fill_models": bool(five_cent_pass),
        "discovery_pass": discovery_pass,
        "classification": (
            "DISCOVERY PASS — formal 21-day and prospective confirmation required"
            if discovery_pass else
            "FAIL / RESEARCH CANDIDATE ONLY"
        ),
    }
    (OUT / "rfhl_clean_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Reward-Funded Hedged Liquidity — clean seven-day audit",
        "",
        f"## Verdict: **{summary['classification']}**",
        "",
        "This audit uses exchange-funded liquidity rewards to subsidize a",
        "direction-neutral binary hedge. A maker fill is immediately paired with",
        "the complementary outcome; a no-fill earns only the modeled reward.",
        "",
        "## Data",
        "",
        f"- Recent programs: {len(programs):,}",
        f"- Program days: {programs.end.dt.date.nunique()}",
        f"- Full one-minute paths: {len(paths):,}",
        f"- Usable markets: {states.market_ticker.nunique():,}",
        "",
        "## Validation-selected policy",
        "",
        f"- Quantity: {int(policy['quantity'])}",
        f"- Maximum maker price: {float(policy['max_maker_price'])*100:.1f}¢",
        f"- Cancel no-fill after: {int(policy['cancel_min'])} minutes",
        f"- Maximum markets per close: {int(policy['cap'])}",
        f"- Selection: {policy['selection']}",
        "- Selection stress: 2¢ extra hedge cost and the worse fill model",
        "- Reward accounting: $1 minimum, rounded down to cents",
        "",
        "## Sealed test sensitivity",
        "",
        "| Fill model | Extra hedge cost | Posted | Fills | Trading | Raw reward | Paid reward | Combined | Mean/day | Day CI | Max DD | Max transient |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fill_model in FILL_MODELS:
        for extra_cost_c in SLIPPAGE_GRID_C:
            item = test_results[fill_model][str(extra_cost_c)]
            m = item["metrics"]
            b = item["bootstrap"]
            lines.append(
                f"| {fill_model} | {extra_cost_c}¢ | {m['posted']} | "
                f"{m['maker_fills']} | ${m['trading_pnl']:.2f} | "
                f"${m['raw_reward_entitlement']:.2f} | "
                f"${m['paid_reward_lower']:.2f} | "
                f"${m['combined_pnl']:.2f} | ${m['mean_day']:.2f} | "
                f"[${b['ci_lo']:.2f}, ${b['ci_hi']:.2f}] | "
                f"${m['max_drawdown']:.2f} | "
                f"${m['max_transient_one_leg_cost']:.2f} |"
            )
    lines.extend([
        "",
        "## What would make this a real breakthrough",
        "",
        "The seven-day discovery must first reproduce on a frozen 21-day sample.",
        "Then q1 prospective orders must confirm official score credit, actual",
        "reward payments, queue behavior, maker-to-IOC latency, and hedge depth.",
        "The prospective endpoint is reward plus trading P&L per assigned market.",
        "",
        "Even a discovery PASS is not authorization to trade while the existing",
        "account KILL state is active.",
    ])
    (OUT / "rfhl_clean_report.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
