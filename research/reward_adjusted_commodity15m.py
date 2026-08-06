"""Historical reward-adjusted audit for incentivized 15-minute commodities.

The live incentive frontier identified recurring KXGOLD15M, KXSILVER15M and
KXWTI15M liquidity programs. This script tests whether conservative liquidity
rewards can cover the directional loss created when only one side of a
complementary maker pair fills.

Public endpoints only. No credentials and no orders.

Important modeling choices
--------------------------
* Program reward is converted from centi-cents to dollars.
* Quote one YES bid and one NO bid at the first complete one-minute quote after
  program start.
* Reward competition is conservatively fixed at two full target-size sides for
  the entire period.
* Own score accrues only while each order remains resting.
* Two fill models are reported:
    strict: a complete-minute quote closes through our price;
    touch:  a minute high/low reaches our price.
* The production candidate must be positive under both models.
* The policy is selected on chronological validation and evaluated once on the
  final chronological test slice.

Historical reward share cannot be observed for a user who did not quote. This
is a conservative score-share counterfactual, not proof of an actual payout.
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
SESSION.headers.update({"User-Agent": "m15-reward-audit/1.0"})
SERIES = ["KXGOLD15M", "KXSILVER15M", "KXWTI15M"]
LOOKBACK_DAYS = 35
QTY_GRID = [1, 5, 10, 15, 20]
MIN_SPREAD_GRID = [0, 1, 2, 3, 4, 5]
NONE_CANCEL_MIN_GRID = [5, 8, 11, 14]
CAP_GRID = [1, 3]
SELECTION_GRID = ["reward_density", "widest_spread", "lowest_one_leg"]
RNG = np.random.default_rng(2026080627)


def get_json(path: str, params: dict | None = None, retries: int = 4) -> dict:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            response = SESSION.get(BASE + path, params=params, timeout=30)
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict):
                raise RuntimeError("non-object JSON")
            return body
        except Exception as exc:
            last = exc
            if attempt + 1 < retries:
                time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"GET {path} failed: {last}")


def all_incentives(status: str) -> list[dict]:
    rows: list[dict] = []
    cursor = ""
    seen: set[str] = set()
    while True:
        params: dict[str, Any] = {
            "status": status, "type": "liquidity", "limit": 10000,
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


def series_of(program: dict) -> str:
    explicit = str(program.get("series_ticker") or "").upper()
    if explicit in SERIES:
        return explicit
    ticker = str(program.get("market_ticker") or "").upper()
    for series in SERIES:
        if ticker.startswith(series + "-"):
            return series
    return ""


def chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index:index + size] for index in range(0, len(values), size)]


def fetch_markets(tickers: list[str]) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for batch in chunks(tickers, 100):
        params = {"tickers": ",".join(batch), "limit": 1000}
        body = get_json("/markets", params)
        for market in body.get("markets") or []:
            if isinstance(market, dict) and market.get("ticker"):
                output[str(market["ticker"])] = market
    missing = [ticker for ticker in tickers if ticker not in output]
    for batch in chunks(missing, 100):
        params = {"tickers": ",".join(batch), "limit": 1000}
        body = get_json("/historical/markets", params)
        for market in body.get("markets") or []:
            if isinstance(market, dict) and market.get("ticker"):
                output[str(market["ticker"])] = market
    return output


def numeric_quote(item: dict, field: str, default: float = np.nan) -> float:
    value = item.get(field)
    if value is None and field.endswith("_dollars"):
        value = item.get(field.replace("_dollars", ""))
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_candle(candle: dict) -> dict | None:
    try:
        ts = int(candle["end_period_ts"])
        yb = candle["yes_bid"]
        ya = candle["yes_ask"]
        def value(block: dict, name: str) -> float:
            raw = block.get(name + "_dollars", block.get(name))
            return float(raw)
        row = {
            "ts": ts,
            "yb_open": value(yb, "open"), "yb_low": value(yb, "low"),
            "yb_high": value(yb, "high"), "yb_close": value(yb, "close"),
            "ya_open": value(ya, "open"), "ya_low": value(ya, "low"),
            "ya_high": value(ya, "high"), "ya_close": value(ya, "close"),
            "volume": float(candle.get("volume_fp", candle.get("volume", 0)) or 0),
        }
    except (KeyError, TypeError, ValueError):
        return None
    if not (0 < row["yb_close"] < row["ya_close"] < 1):
        return None
    return row


def fetch_candles(programs: pd.DataFrame) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = {}
    # Batch recent markets by UTC calendar date. Each market only has a short
    # active path, keeping total candles under the 10,000 response cap.
    for day, group in programs.groupby(programs["end"].dt.date):
        tickers = group["market_ticker"].tolist()
        start_ts = int(group["start"].min().timestamp()) - 120
        end_ts = int(group["end"].max().timestamp()) + 120
        for batch in chunks(tickers, 100):
            try:
                body = get_json("/markets/candlesticks", {
                    "market_tickers": ",".join(batch),
                    "start_ts": start_ts, "end_ts": end_ts,
                    "period_interval": 1,
                })
            except Exception:
                body = {}
            for item in body.get("markets") or []:
                ticker = str(item.get("market_ticker") or item.get("ticker") or "")
                candles = [parse_candle(candle) for candle in item.get("candlesticks") or []]
                output[ticker] = sorted(
                    [candle for candle in candles if candle is not None],
                    key=lambda row: row["ts"])
        print(f"candles through {day}: {len(output):,}/{len(programs):,}", flush=True)
    # Recent endpoint should cover this lookback. Fall back individually for
    # any missing market, using the historical endpoint only if necessary.
    missing = programs[~programs["market_ticker"].isin(output)].copy()
    for index, row in enumerate(missing.itertuples(index=False), start=1):
        ticker = row.market_ticker
        params = {
            "start_ts": int(row.start.timestamp()) - 120,
            "end_ts": int(row.end.timestamp()) + 120,
            "period_interval": 1,
        }
        body: dict = {}
        for path in [
            f"/series/{row.series_ticker}/markets/{ticker}/candlesticks",
            f"/historical/markets/{ticker}/candlesticks",
        ]:
            try:
                body = get_json(path, params)
                if body.get("candlesticks"):
                    break
            except Exception:
                continue
        candles = [parse_candle(candle) for candle in body.get("candlesticks") or []]
        if candles:
            output[ticker] = sorted(
                [candle for candle in candles if candle is not None],
                key=lambda item: item["ts"])
        if index % 100 == 0:
            print(f"fallback {index}/{len(missing)}; paths={len(output)}", flush=True)
    return output


def first_entry(candles: list[dict], start: pd.Timestamp, end: pd.Timestamp) -> dict | None:
    # First complete one-minute observation after program start, no lookahead.
    lo = int(start.timestamp()) + 30
    hi = int(start.timestamp()) + 150
    candidates = [candle for candle in candles if lo <= candle["ts"] <= hi]
    return min(candidates, key=lambda row: row["ts"]) if candidates else None


def first_fill(candles: list[dict], entry_ts: int, price: float,
               side: str, model: str, end_ts: int) -> int | None:
    for candle in candles:
        if candle["ts"] <= entry_ts or candle["ts"] > end_ts:
            continue
        if candle["volume"] <= 0:
            continue
        if side == "yes":
            observed = candle["ya_close"] if model == "strict" else candle["ya_low"]
            if observed <= price + 1e-9:
                return int(candle["ts"])
        else:
            yes_trigger = 1.0 - price
            observed = candle["yb_close"] if model == "strict" else candle["yb_high"]
            if observed >= yes_trigger - 1e-9:
                return int(candle["ts"])
    return None


def reward_dollars(raw: Any) -> float:
    return float(raw or 0.0) / 10_000.0


def build_states(programs: pd.DataFrame, markets: dict[str, dict],
                 paths: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for program in programs.itertuples(index=False):
        market = markets.get(program.market_ticker)
        candles = paths.get(program.market_ticker)
        if market is None or not candles:
            continue
        result = str(market.get("result") or "").lower()
        if result not in {"yes", "no"}:
            continue
        entry = first_entry(candles, program.start, program.end)
        if entry is None:
            continue
        yes_bid = float(entry["yb_close"])
        yes_ask = float(entry["ya_close"])
        no_bid = 1.0 - yes_ask
        spread = max(1.0 - yes_bid - no_bid, 0.0)
        period_seconds = max((program.end - program.start).total_seconds(), 1.0)
        entry_ts = int(entry["ts"])
        end_ts = int(program.end.timestamp())
        for fill_model in ["strict", "touch"]:
            yes_fill = first_fill(candles, entry_ts, yes_bid, "yes", fill_model, end_ts)
            no_fill = first_fill(candles, entry_ts, no_bid, "no", fill_model, end_ts)
            for cancel_none_min in NONE_CANCEL_MIN_GRID:
                none_deadline = min(
                    entry_ts + cancel_none_min * 60,
                    end_ts)
                first_any = min([ts for ts in [yes_fill, no_fill] if ts is not None],
                                default=None)
                if first_any is None or first_any > none_deadline:
                    # Cancel both at the deadline if neither has filled.
                    effective_yes = None
                    effective_no = None
                    rest_yes_until = none_deadline
                    rest_no_until = none_deadline
                else:
                    # Once one leg fills, keep the complement alive through the
                    # period to maximize the chance of locking the spread.
                    effective_yes = yes_fill
                    effective_no = no_fill
                    rest_yes_until = yes_fill if yes_fill is not None else end_ts
                    rest_no_until = no_fill if no_fill is not None else end_ts
                yes_filled = effective_yes is not None
                no_filled = effective_no is not None
                both = yes_filled and no_filled
                only_yes = yes_filled and not no_filled
                only_no = no_filled and not yes_filled
                if both:
                    trading_per_contract = spread
                elif only_yes:
                    trading_per_contract = (1.0 if result == "yes" else 0.0) - yes_bid
                elif only_no:
                    trading_per_contract = (1.0 if result == "no" else 0.0) - no_bid
                else:
                    trading_per_contract = 0.0
                rows.append({
                    "market_ticker": program.market_ticker,
                    "series_ticker": program.series_ticker,
                    "program_id": program.program_id,
                    "start": program.start,
                    "end": program.end,
                    "day": program.end.date().isoformat(),
                    "week": f"{program.end.isocalendar().year}-{program.end.isocalendar().week:02d}",
                    "close_ts": end_ts,
                    "result": result,
                    "fill_model": fill_model,
                    "cancel_none_min": cancel_none_min,
                    "yes_bid": yes_bid,
                    "no_bid": no_bid,
                    "spread_c": spread * 100,
                    "target_size": program.target_size,
                    "period_reward_dollars": program.period_reward_dollars,
                    "period_seconds": period_seconds,
                    "entry_ts": entry_ts,
                    "rest_yes_seconds": max(rest_yes_until - entry_ts, 0),
                    "rest_no_seconds": max(rest_no_until - entry_ts, 0),
                    "yes_filled": yes_filled,
                    "no_filled": no_filled,
                    "both_filled": both,
                    "single_filled": only_yes or only_no,
                    "none_filled": not yes_filled and not no_filled,
                    "trading_per_contract": trading_per_contract,
                })
    states = pd.DataFrame(rows)
    if states.empty:
        raise RuntimeError("no historical reward states constructed")
    states.to_parquet(OUT / "commodity15m_reward_states.parquet", index=False)
    return states


def chronological_splits(states: pd.DataFrame) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    dates = sorted(pd.to_datetime(states["end"], utc=True).dt.normalize().unique())
    if len(dates) < 12:
        raise RuntimeError(f"only {len(dates)} calendar days")
    n = len(dates)
    train_stop = dates[max(int(n * 0.50), 1)]
    valid_stop = dates[max(int(n * 0.75), 2)]
    return {
        "train": (pd.Timestamp(dates[0]), pd.Timestamp(train_stop)),
        "valid": (pd.Timestamp(train_stop), pd.Timestamp(valid_stop)),
        "test": (pd.Timestamp(valid_stop), pd.Timestamp(dates[-1]) + pd.Timedelta(days=1)),
    }


def apply_qty(frame: pd.DataFrame, qty: int) -> pd.DataFrame:
    x = frame.copy()
    own_score_time = qty * (x["rest_yes_seconds"] + x["rest_no_seconds"])
    # Conservative maximum eligible competition: target size on both sides at
    # full score for the whole incentive period.
    competitor_score_time = 2.0 * x["target_size"] * x["period_seconds"]
    share = own_score_time / (competitor_score_time + own_score_time)
    x["reward_pnl"] = x["period_reward_dollars"] * share
    x["trading_pnl"] = qty * x["trading_per_contract"]
    x["combined_pnl"] = x["reward_pnl"] + x["trading_pnl"]
    x["reward_share_lower"] = share
    x["qty"] = qty
    return x


def select_close(frame: pd.DataFrame, cap: int, method: str) -> pd.DataFrame:
    if method == "reward_density":
        columns = ["close_ts", "period_reward_dollars", "target_size", "spread_c", "series_ticker"]
        ascending = [True, False, True, False, True]
    elif method == "widest_spread":
        columns = ["close_ts", "spread_c", "period_reward_dollars", "series_ticker"]
        ascending = [True, False, False, True]
    else:
        frame = frame.copy()
        frame["one_leg_price"] = frame[["yes_bid", "no_bid"]].max(axis=1)
        columns = ["close_ts", "one_leg_price", "period_reward_dollars", "series_ticker"]
        ascending = [True, True, False, True]
    return (frame.sort_values(columns, ascending=ascending)
            .groupby("close_ts", as_index=False).head(cap)
            .sort_values(["close_ts", "series_ticker"]))


@dataclass
class Metrics:
    split: str
    fill_model: str
    qty: int
    min_spread_c: int
    cancel_none_min: int
    cap: int
    selection: str
    n: int
    active: int
    both_rate: float
    single_rate: float
    none_rate: float
    trading_pnl: float
    reward_pnl_lower: float
    combined_pnl: float
    mean_day: float
    sd_day: float
    t_stat: float
    max_drawdown: float
    worst_window: float
    positive_day_fraction: float


def metric(selected: pd.DataFrame, split: str, splits: dict,
           fill_model: str, qty: int, min_spread_c: int,
           cancel_none_min: int, cap: int, selection: str) -> Metrics:
    start, end = splits[split]
    days = pd.date_range(start.normalize(), end.normalize() - pd.Timedelta(days=1), freq="D").date
    if selected.empty:
        return Metrics(split, fill_model, qty, min_spread_c,
                       cancel_none_min, cap, selection, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0)
    daily = selected.groupby("day")["combined_pnl"].sum().reindex(
        [day.isoformat() for day in days], fill_value=0.0)
    window = selected.groupby("close_ts")["combined_pnl"].sum().sort_index()
    equity = window.cumsum()
    drawdown = equity.cummax() - equity
    sd = float(daily.std(ddof=1))
    mean = float(daily.mean())
    active = selected[~selected["none_filled"]]
    return Metrics(
        split, fill_model, qty, min_spread_c, cancel_none_min, cap, selection,
        int(len(selected)), int(len(active)),
        float(selected["both_filled"].mean()),
        float(selected["single_filled"].mean()),
        float(selected["none_filled"].mean()),
        float(selected["trading_pnl"].sum()),
        float(selected["reward_pnl"].sum()),
        float(selected["combined_pnl"].sum()),
        mean, sd, mean / (sd / math.sqrt(len(daily))) if sd > 0 else 0.0,
        float(drawdown.max()) if len(drawdown) else 0.0,
        float(window.min()) if len(window) else 0.0,
        float((daily > 0).mean()))


def choose_policy(states: pd.DataFrame, splits: dict) -> tuple[dict, pd.DataFrame]:
    valid_start, valid_end = splits["valid"]
    valid = states[(states["end"] >= valid_start) & (states["end"] < valid_end)]
    rows = []
    for model in ["strict", "touch"]:
        for qty in QTY_GRID:
            sized = apply_qty(valid[valid["fill_model"].eq(model)], qty)
            for spread in MIN_SPREAD_GRID:
                for cancel in NONE_CANCEL_MIN_GRID:
                    subset = sized[
                        sized["cancel_none_min"].eq(cancel)
                        & (sized["spread_c"] >= spread)]
                    for cap in CAP_GRID:
                        for selection in SELECTION_GRID:
                            chosen = select_close(subset, cap, selection)
                            rows.append(asdict(metric(
                                chosen, "valid", splits, model, qty, spread,
                                cancel, cap, selection)))
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "commodity15m_reward_validation_grid.csv", index=False)
    # Select one policy using the WORSE of strict/touch for identical policy
    # parameters. This prevents choosing the fill model that flatters it.
    keys = ["qty", "min_spread_c", "cancel_none_min", "cap", "selection"]
    worst = (grid.groupby(keys).agg(
        worst_combined=("combined_pnl", "min"),
        worst_mean_day=("mean_day", "min"),
        worst_t=("t_stat", "min"),
        worst_dd=("max_drawdown", "max"),
        min_n=("n", "min"),
        min_active=("active", "min")).reset_index())
    viable = worst[(worst["min_n"] >= 100) & (worst["worst_combined"] > 0)].copy()
    if viable.empty:
        viable = worst[worst["min_n"] >= 50].copy()
    if viable.empty:
        viable = worst.copy()
    viable["objective"] = (
        viable["worst_t"] + 0.10 * np.log1p(viable["min_n"])
        - 0.01 * viable["worst_dd"])
    winner = viable.sort_values(
        ["objective", "worst_t", "worst_combined"], ascending=False).iloc[0].to_dict()
    return winner, grid


def evaluate(states: pd.DataFrame, splits: dict, policy: dict,
             fill_model: str, split: str) -> tuple[pd.DataFrame, Metrics]:
    start, end = splits[split]
    subset = states[
        (states["end"] >= start) & (states["end"] < end)
        & states["fill_model"].eq(fill_model)
        & states["cancel_none_min"].eq(int(policy["cancel_none_min"]))
        & (states["spread_c"] >= float(policy["min_spread_c"]))]
    sized = apply_qty(subset, int(policy["qty"]))
    selected = select_close(sized, int(policy["cap"]), str(policy["selection"]))
    return selected, metric(
        selected, split, splits, fill_model, int(policy["qty"]),
        int(policy["min_spread_c"]), int(policy["cancel_none_min"]),
        int(policy["cap"]), str(policy["selection"]))


def bootstrap(selected: pd.DataFrame, split: str, splits: dict,
              reps: int = 12000) -> dict:
    start, end = splits[split]
    days = pd.date_range(start.normalize(), end.normalize() - pd.Timedelta(days=1), freq="D").date
    daily = selected.groupby("day")["combined_pnl"].sum().reindex(
        [day.isoformat() for day in days], fill_value=0.0).to_numpy()
    draws = RNG.integers(0, len(daily), size=(reps, len(daily)))
    means = daily[draws].mean(axis=1)
    return {
        "mean_day": float(daily.mean()),
        "ci_lo": float(np.quantile(means, 0.025)),
        "ci_hi": float(np.quantile(means, 0.975)),
        "p_nonpositive": float(np.mean(means <= 0)),
    }


def leave_one(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for series in sorted(selected["series_ticker"].unique()):
        subset = selected[~selected["series_ticker"].eq(series)]
        rows.append({
            "excluded_series": series, "n": len(subset),
            "combined_pnl": float(subset["combined_pnl"].sum()),
            "trading_pnl": float(subset["trading_pnl"].sum()),
            "reward_pnl": float(subset["reward_pnl"].sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    now = pd.Timestamp.now(tz="UTC")
    cutoff = now - pd.Timedelta(days=LOOKBACK_DAYS)
    raw = all_incentives("closed") + all_incentives("paid_out")
    records = []
    seen: set[str] = set()
    for program in raw:
        program_id = str(program.get("id") or "")
        if not program_id or program_id in seen:
            continue
        seen.add(program_id)
        series = series_of(program)
        if not series:
            continue
        start = pd.to_datetime(program.get("start_date"), utc=True, errors="coerce")
        end = pd.to_datetime(program.get("end_date"), utc=True, errors="coerce")
        if pd.isna(start) or pd.isna(end) or end < cutoff or end > now or end <= start:
            continue
        target = float(program.get("target_size_fp") or 0.0)
        reward = reward_dollars(program.get("period_reward"))
        ticker = str(program.get("market_ticker") or "")
        if target <= 0 or reward <= 0 or not ticker:
            continue
        records.append({
            "program_id": program_id, "market_ticker": ticker,
            "series_ticker": series, "start": start, "end": end,
            "target_size": target, "period_reward_dollars": reward,
            "discount_factor_bps": float(program.get("discount_factor_bps") or 0.0),
        })
    programs = pd.DataFrame(records).drop_duplicates("market_ticker")
    if programs.empty:
        raise RuntimeError("no recent closed commodity15m incentives")
    programs = programs.sort_values("end").reset_index(drop=True)
    programs.to_csv(OUT / "commodity15m_incentive_programs.csv", index=False)
    print(f"programs: {len(programs):,} across {programs.end.dt.date.nunique()} days", flush=True)

    markets = fetch_markets(programs["market_ticker"].tolist())
    print(f"markets: {len(markets):,}/{len(programs):,}", flush=True)
    paths = fetch_candles(programs)
    print(f"paths: {len(paths):,}/{len(programs):,}", flush=True)
    states = build_states(programs, markets, paths)
    splits = chronological_splits(states)
    policy, _ = choose_policy(states, splits)

    results = {}
    selected_by_model = {}
    for model in ["strict", "touch"]:
        selected, metrics = evaluate(states, splits, policy, model, "test")
        selected_by_model[model] = selected
        results[model] = {
            "metrics": asdict(metrics),
            "bootstrap": bootstrap(selected, "test", splits),
        }
        selected.to_parquet(
            OUT / f"commodity15m_reward_test_{model}.parquet", index=False)
        leave_one(selected).to_csv(
            OUT / f"commodity15m_reward_leave_one_{model}.csv", index=False)

    hard_pass = True
    for model in ["strict", "touch"]:
        metrics = results[model]["metrics"]
        boot = results[model]["bootstrap"]
        loo = leave_one(selected_by_model[model])
        hard_pass &= (
            metrics["combined_pnl"] > 0
            and metrics["n"] >= 100
            and boot["ci_lo"] > 0
            and len(loo) > 0
            and (loo["combined_pnl"] > 0).all())

    summary = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "programs": int(len(programs)),
        "market_coverage": int(len(markets)),
        "path_coverage": int(len(paths)),
        "state_rows": int(len(states)),
        "splits": {key: [str(value[0]), str(value[1])] for key, value in splits.items()},
        "validation_policy_worst_case_selected": policy,
        "test": results,
        "hard_pass": bool(hard_pass),
    }
    (OUT / "commodity15m_reward_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    verdict = "PASS — reward-adjusted historical candidate" if hard_pass \
        else "FAIL / PROSPECTIVE CANDIDATE ONLY"
    lines = [
        "# Reward-adjusted commodity 15-minute maker audit", "",
        f"## Verdict: **{verdict}**", "",
        "This study combines trading P&L with a conservative lower-bound estimate",
        "of Kalshi liquidity rewards. It quotes complementary YES and NO maker bids",
        "and assumes maximum target-size competition on both sides throughout every",
        "program period.", "", "## Data", "",
        f"- Recent incentive programs: {len(programs):,}",
        f"- Markets resolved: {len(markets):,}",
        f"- Full price paths: {len(paths):,}",
        f"- State rows: {len(states):,}", "", "## Validation-selected policy", "",
        f"- Quantity per side: {int(policy['qty'])}",
        f"- Minimum initial spread: {float(policy['min_spread_c']):.1f}¢",
        f"- Cancel both if neither fills after: {int(policy['cancel_none_min'])} minutes",
        f"- Maximum markets per close: {int(policy['cap'])}",
        f"- Selection: {policy['selection']}", "", "## Sealed test", "",
        "| Fill model | n | Trading P&L | Reward lower bound | Combined | Mean/day | 95% day CI | Max DD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ["strict", "touch"]:
        m = results[model]["metrics"]
        b = results[model]["bootstrap"]
        lines.append(
            f"| {model} | {m['n']} | ${m['trading_pnl']:.2f} | "
            f"${m['reward_pnl_lower']:.2f} | ${m['combined_pnl']:.2f} | "
            f"${m['mean_day']:.2f} | [${b['ci_lo']:.2f}, ${b['ci_hi']:.2f}] | "
            f"${m['max_drawdown']:.2f} |")
    lines.extend(["", "## Interpretation", "",
        "A historical PASS means only that a maximum-competition reward bound",
        "covers the modeled legging loss under both quote-cross definitions.",
        "Actual reward eligibility, random snapshots, queue priority, and payout",
        "credits remain prospective facts. The first live test must use q1 per side",
        "and must never self-trade or manufacture volume.", "",
        "No result weakens the account's existing KILL state."])
    (OUT / "commodity15m_reward_report.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
