"""One-sided low-risk liquidity-reward audit for 15-minute commodities.

Quote only the cheaper outcome at its best bid. This cuts worst-case directional
loss while retaining a share of the liquidity reward. The audit uses the same
recent closed incentive programs, markets, and candlestick paths as
`reward_adjusted_commodity15m.py`.

A policy must survive both strict close-cross and optimistic touch fill models,
chronological validation/test, day bootstrap, and leave-one-series-out.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research.reward_adjusted_commodity15m import (
    LOOKBACK_DAYS,
    OUT,
    RNG,
    SERIES,
    all_incentives,
    chronological_splits,
    fetch_candles,
    fetch_markets,
    first_entry,
    first_fill,
    reward_dollars,
    series_of,
)

QTY_GRID = [1, 5, 10, 15, 20]
MAX_PRICE_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
CANCEL_MIN_GRID = [3, 5, 8, 11, 14]
CAP_GRID = [1, 3]
SELECTION_GRID = ["reward_risk", "cheapest", "reward_density"]
LOCAL_RNG = np.random.default_rng(2026080629)


def load_programs() -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    cutoff = now - pd.Timedelta(days=LOOKBACK_DAYS)
    raw = all_incentives("closed") + all_incentives("paid_out")
    records = []
    seen: set[str] = set()
    for program in raw:
        pid = str(program.get("id") or "")
        if not pid or pid in seen:
            continue
        seen.add(pid)
        series = series_of(program)
        if not series:
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
            "program_id": pid, "market_ticker": ticker,
            "series_ticker": series, "start": start, "end": end,
            "target_size": target, "period_reward_dollars": reward,
        })
    frame = pd.DataFrame(records).drop_duplicates("market_ticker")
    if frame.empty:
        raise RuntimeError("no recent commodity15m incentive programs")
    return frame.sort_values("end").reset_index(drop=True)


def build_states(programs: pd.DataFrame, markets: dict,
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
        no_bid = 1.0 - float(entry["ya_close"])
        if not (0 < yes_bid < 1 and 0 < no_bid < 1):
            continue
        if yes_bid <= no_bid:
            side, price = "yes", yes_bid
        else:
            side, price = "no", no_bid
        period_seconds = max((program.end - program.start).total_seconds(), 1.0)
        entry_ts = int(entry["ts"])
        end_ts = int(program.end.timestamp())
        for model in ["strict", "touch"]:
            full_fill = first_fill(
                candles, entry_ts, price, side, model, end_ts)
            for cancel_min in CANCEL_MIN_GRID:
                deadline = min(entry_ts + cancel_min * 60, end_ts)
                fill_ts = full_fill if full_fill is not None and full_fill <= deadline else None
                rest_until = fill_ts if fill_ts is not None else deadline
                filled = fill_ts is not None
                if filled:
                    payout = 1.0 if result == side else 0.0
                    trading_per_contract = payout - price
                else:
                    trading_per_contract = 0.0
                rows.append({
                    "market_ticker": program.market_ticker,
                    "series_ticker": program.series_ticker,
                    "program_id": program.program_id,
                    "start": program.start, "end": program.end,
                    "day": program.end.date().isoformat(),
                    "week": f"{program.end.isocalendar().year}-{program.end.isocalendar().week:02d}",
                    "close_ts": end_ts,
                    "fill_model": model,
                    "cancel_min": cancel_min,
                    "side": side, "price": price,
                    "target_size": program.target_size,
                    "period_reward_dollars": program.period_reward_dollars,
                    "period_seconds": period_seconds,
                    "rest_seconds": max(rest_until - entry_ts, 0),
                    "filled": filled,
                    "won": bool(filled and result == side),
                    "trading_per_contract": trading_per_contract,
                })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no one-sided states")
    frame.to_parquet(OUT / "commodity15m_onesided_states.parquet", index=False)
    return frame


def size_states(frame: pd.DataFrame, qty: int) -> pd.DataFrame:
    x = frame.copy()
    own_score_time = qty * x["rest_seconds"]
    competitor_score_time = 2.0 * x["target_size"] * x["period_seconds"]
    share = own_score_time / (competitor_score_time + own_score_time)
    x["reward_pnl"] = x["period_reward_dollars"] * share
    x["trading_pnl"] = qty * x["trading_per_contract"]
    x["combined_pnl"] = x["reward_pnl"] + x["trading_pnl"]
    x["reward_share_lower"] = share
    x["qty"] = qty
    x["reward_risk"] = x["reward_pnl"] / (qty * x["price"])
    x["reward_density"] = x["period_reward_dollars"] / x["target_size"]
    return x


def select_close(frame: pd.DataFrame, cap: int, selection: str) -> pd.DataFrame:
    if selection == "reward_risk":
        columns = ["close_ts", "reward_risk", "price", "series_ticker"]
        ascending = [True, False, True, True]
    elif selection == "cheapest":
        columns = ["close_ts", "price", "reward_risk", "series_ticker"]
        ascending = [True, True, False, True]
    else:
        columns = ["close_ts", "reward_density", "price", "series_ticker"]
        ascending = [True, False, True, True]
    return (frame.sort_values(columns, ascending=ascending)
            .groupby("close_ts", as_index=False).head(cap)
            .sort_values(["close_ts", "series_ticker"]))


@dataclass
class Metrics:
    split: str
    fill_model: str
    qty: int
    max_price: float
    cancel_min: int
    cap: int
    selection: str
    n: int
    fills: int
    fill_rate: float
    fill_win_rate: float
    trading_pnl: float
    reward_pnl: float
    combined_pnl: float
    mean_day: float
    sd_day: float
    t_stat: float
    max_drawdown: float
    worst_window: float
    positive_day_fraction: float


def metric(selected: pd.DataFrame, split: str, splits: dict,
           model: str, qty: int, max_price: float, cancel_min: int,
           cap: int, selection: str) -> Metrics:
    start, end = splits[split]
    days = pd.date_range(start.normalize(), end.normalize() - pd.Timedelta(days=1), freq="D").date
    if selected.empty:
        return Metrics(split, model, qty, max_price, cancel_min, cap, selection,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    daily = selected.groupby("day")["combined_pnl"].sum().reindex(
        [day.isoformat() for day in days], fill_value=0.0)
    window = selected.groupby("close_ts")["combined_pnl"].sum().sort_index()
    equity = window.cumsum()
    drawdown = equity.cummax() - equity
    filled = selected[selected["filled"]]
    sd = float(daily.std(ddof=1))
    mean = float(daily.mean())
    return Metrics(
        split, model, qty, max_price, cancel_min, cap, selection,
        int(len(selected)), int(len(filled)), float(selected["filled"].mean()),
        float(filled["won"].mean()) if len(filled) else np.nan,
        float(selected["trading_pnl"].sum()),
        float(selected["reward_pnl"].sum()),
        float(selected["combined_pnl"].sum()), mean, sd,
        mean / (sd / math.sqrt(len(daily))) if sd > 0 else 0.0,
        float(drawdown.max()) if len(drawdown) else 0.0,
        float(window.min()) if len(window) else 0.0,
        float((daily > 0).mean()))


def choose(states: pd.DataFrame, splits: dict) -> tuple[dict, pd.DataFrame]:
    start, end = splits["valid"]
    valid = states[(states["end"] >= start) & (states["end"] < end)]
    rows = []
    for model in ["strict", "touch"]:
        for qty in QTY_GRID:
            sized = size_states(valid[valid["fill_model"].eq(model)], qty)
            for max_price in MAX_PRICE_GRID:
                for cancel in CANCEL_MIN_GRID:
                    subset = sized[
                        sized["cancel_min"].eq(cancel)
                        & (sized["price"] <= max_price)]
                    for cap in CAP_GRID:
                        for selection in SELECTION_GRID:
                            selected = select_close(subset, cap, selection)
                            rows.append(asdict(metric(
                                selected, "valid", splits, model, qty,
                                max_price, cancel, cap, selection)))
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "commodity15m_onesided_validation_grid.csv", index=False)
    keys = ["qty", "max_price", "cancel_min", "cap", "selection"]
    worst = (grid.groupby(keys).agg(
        worst_pnl=("combined_pnl", "min"),
        worst_t=("t_stat", "min"),
        worst_dd=("max_drawdown", "max"),
        min_n=("n", "min"), min_fills=("fills", "min")).reset_index())
    viable = worst[(worst["min_n"] >= 100) & (worst["worst_pnl"] > 0)].copy()
    if viable.empty:
        viable = worst[worst["min_n"] >= 50].copy()
    if viable.empty:
        viable = worst.copy()
    viable["objective"] = (
        viable["worst_t"] + 0.1 * np.log1p(viable["min_n"])
        - 0.01 * viable["worst_dd"])
    winner = viable.sort_values(
        ["objective", "worst_t", "worst_pnl"], ascending=False).iloc[0].to_dict()
    return winner, grid


def evaluate(states: pd.DataFrame, splits: dict, policy: dict,
             model: str, split: str) -> tuple[pd.DataFrame, Metrics]:
    start, end = splits[split]
    subset = states[
        (states["end"] >= start) & (states["end"] < end)
        & states["fill_model"].eq(model)
        & states["cancel_min"].eq(int(policy["cancel_min"]))
        & (states["price"] <= float(policy["max_price"]))]
    sized = size_states(subset, int(policy["qty"]))
    selected = select_close(sized, int(policy["cap"]), str(policy["selection"]))
    return selected, metric(
        selected, split, splits, model, int(policy["qty"]),
        float(policy["max_price"]), int(policy["cancel_min"]),
        int(policy["cap"]), str(policy["selection"]))


def bootstrap(selected: pd.DataFrame, split: str, splits: dict,
              reps: int = 12000) -> dict:
    start, end = splits[split]
    days = pd.date_range(start.normalize(), end.normalize() - pd.Timedelta(days=1), freq="D").date
    daily = selected.groupby("day")["combined_pnl"].sum().reindex(
        [day.isoformat() for day in days], fill_value=0.0).to_numpy()
    draws = LOCAL_RNG.integers(0, len(daily), size=(reps, len(daily)))
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
    programs = load_programs()
    markets = fetch_markets(programs["market_ticker"].tolist())
    paths = fetch_candles(programs)
    states = build_states(programs, markets, paths)
    splits = chronological_splits(states)
    policy, _ = choose(states, splits)

    results = {}
    selected_models = {}
    hard_pass = True
    for model in ["strict", "touch"]:
        selected, metrics = evaluate(states, splits, policy, model, "test")
        selected_models[model] = selected
        boot = bootstrap(selected, "test", splits)
        loo = leave_one(selected)
        loo.to_csv(OUT / f"commodity15m_onesided_leave_one_{model}.csv", index=False)
        selected.to_parquet(OUT / f"commodity15m_onesided_test_{model}.parquet", index=False)
        results[model] = {"metrics": asdict(metrics), "bootstrap": boot}
        hard_pass &= (
            metrics.combined_pnl > 0 and metrics.n >= 100
            and boot["ci_lo"] > 0 and len(loo) > 0
            and (loo["combined_pnl"] > 0).all())

    summary = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "programs": len(programs), "markets": len(markets), "paths": len(paths),
        "state_rows": len(states),
        "splits": {key: [str(value[0]), str(value[1])] for key, value in splits.items()},
        "validation_policy": policy, "test": results,
        "hard_pass": bool(hard_pass),
    }
    (OUT / "commodity15m_onesided_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    verdict = "PASS — one-sided reward candidate" if hard_pass \
        else "FAIL / PROSPECTIVE CANDIDATE ONLY"
    lines = [
        "# One-sided reward-adjusted commodity15m audit", "",
        f"## Verdict: **{verdict}**", "",
        "Quote only the lower-priced outcome at the best bid. Reward share is",
        "computed against two fully populated target-size sides for the entire",
        "period, making the score competition assumption deliberately conservative.",
        "", "## Validation-selected policy", "",
        f"- Quantity: {int(policy['qty'])}",
        f"- Maximum quoted price: {float(policy['max_price'])*100:.1f}¢",
        f"- Cancel after: {int(policy['cancel_min'])} minutes",
        f"- Maximum markets/close: {int(policy['cap'])}",
        f"- Selection: {policy['selection']}", "", "## Sealed test", "",
        "| Fill model | n | fills | fill win | Trading | Reward lower | Combined | Mean/day | 95% CI | Max DD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ["strict", "touch"]:
        m = results[model]["metrics"]
        b = results[model]["bootstrap"]
        lines.append(
            f"| {model} | {m['n']} | {m['fills']} | {m['fill_win_rate']:.2%} | "
            f"${m['trading_pnl']:.2f} | ${m['reward_pnl']:.2f} | "
            f"${m['combined_pnl']:.2f} | ${m['mean_day']:.2f} | "
            f"[${b['ci_lo']:.2f}, ${b['ci_hi']:.2f}] | ${m['max_drawdown']:.2f} |")
    lines.extend(["", "## Limits", "",
        "The reward component is a conservative score-share counterfactual, not an",
        "observed user payout. Actual eligibility and snapshot credits require a q1",
        "prospective trial. Never self-trade or create artificial volume.", "",
        "The account KILL state remains binding."])
    (OUT / "commodity15m_onesided_report.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
