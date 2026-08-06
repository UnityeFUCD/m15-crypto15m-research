"""Priority-improved one-sided liquidity-reward audit.

A resting order only earns liquidity points when it helps reach the program's
Target Size. Joining a best bid that already contains the full target can earn
zero. This audit therefore posts one tick *inside* the spread on only the
cheaper outcome. At submission it becomes the new best bid, so its quantity is
unambiguously part of qualifying liquidity unless the market moves first.

The price-improved order carries more fill risk. The conservative reward bound
and the trading loss are evaluated jointly under strict and touch fill models,
with validation-only policy selection and a sealed chronological test.
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
    OUT,
    chronological_splits,
    fetch_candles,
    fetch_markets,
    first_entry,
    first_fill,
)
from research.reward_adjusted_commodity15m_onesided import load_programs

QTY_GRID = [1, 5, 10, 15, 20]
MAX_PRICE_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
CANCEL_MIN_GRID = [2, 3, 5, 8, 11, 14]
CAP_GRID = [1, 3]
SELECTION_GRID = ["reward_risk", "cheapest", "reward_density"]
TICK = 0.01
RNG = np.random.default_rng(2026080631)


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
        yes_ask = float(entry["ya_close"])
        no_bid = 1.0 - yes_ask
        no_ask = 1.0 - yes_bid
        if yes_bid <= no_bid:
            side, old_bid, ask = "yes", yes_bid, yes_ask
        else:
            side, old_bid, ask = "no", no_bid, no_ask
        price = round(old_bid + TICK, 2)
        # Strictly inside the spread, otherwise it crosses and is not a maker.
        if not (0 < price < ask - 1e-9):
            continue
        period_seconds = max((program.end - program.start).total_seconds(), 1.0)
        entry_ts = int(entry["ts"])
        end_ts = int(program.end.timestamp())
        for model in ["strict", "touch"]:
            possible_fill = first_fill(
                candles, entry_ts, price, side, model, end_ts)
            for cancel_min in CANCEL_MIN_GRID:
                deadline = min(entry_ts + cancel_min * 60, end_ts)
                fill_ts = possible_fill \
                    if possible_fill is not None and possible_fill <= deadline else None
                rest_until = fill_ts if fill_ts is not None else deadline
                filled = fill_ts is not None
                payout = 1.0 if result == side else 0.0
                trading_per_contract = payout - price if filled else 0.0
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
                    "side": side, "old_bid": old_bid, "price": price,
                    "ask": ask, "initial_spread_c": (ask - old_bid) * 100,
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
        raise RuntimeError("no priority-improved states")
    frame.to_parquet(OUT / "commodity15m_priority_states.parquet", index=False)
    return frame


def size_states(frame: pd.DataFrame, qty: int) -> pd.DataFrame:
    x = frame.copy()
    own_score_time = qty * x["rest_seconds"]
    # Maximum qualifying score in the program: target size on each outcome for
    # the full period. Adding own score to the denominator is conservative.
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


def select_close(frame: pd.DataFrame, cap: int, method: str) -> pd.DataFrame:
    if method == "reward_risk":
        cols, asc = ["close_ts", "reward_risk", "price", "series_ticker"], [True, False, True, True]
    elif method == "cheapest":
        cols, asc = ["close_ts", "price", "reward_risk", "series_ticker"], [True, True, False, True]
    else:
        cols, asc = ["close_ts", "reward_density", "price", "series_ticker"], [True, False, True, True]
    return (frame.sort_values(cols, ascending=asc)
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
    dd = equity.cummax() - equity
    filled = selected[selected["filled"]]
    sd = float(daily.std(ddof=1)); mean = float(daily.mean())
    return Metrics(
        split, model, qty, max_price, cancel_min, cap, selection,
        int(len(selected)), int(len(filled)), float(selected["filled"].mean()),
        float(filled["won"].mean()) if len(filled) else np.nan,
        float(selected["trading_pnl"].sum()), float(selected["reward_pnl"].sum()),
        float(selected["combined_pnl"].sum()), mean, sd,
        mean / (sd / math.sqrt(len(daily))) if sd > 0 else 0.0,
        float(dd.max()) if len(dd) else 0.0,
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
    grid.to_csv(OUT / "commodity15m_priority_validation_grid.csv", index=False)
    keys = ["qty", "max_price", "cancel_min", "cap", "selection"]
    worst = (grid.groupby(keys).agg(
        worst_pnl=("combined_pnl", "min"), worst_t=("t_stat", "min"),
        worst_dd=("max_drawdown", "max"), min_n=("n", "min"),
        min_fills=("fills", "min")).reset_index())
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
    selected = select_close(
        size_states(subset, int(policy["qty"])),
        int(policy["cap"]), str(policy["selection"]))
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
    draws = RNG.integers(0, len(daily), size=(reps, len(daily)))
    means = daily[draws].mean(axis=1)
    return {"mean_day": float(daily.mean()),
            "ci_lo": float(np.quantile(means, 0.025)),
            "ci_hi": float(np.quantile(means, 0.975)),
            "p_nonpositive": float(np.mean(means <= 0))}


def leave_one(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for series in sorted(selected["series_ticker"].unique()):
        subset = selected[~selected["series_ticker"].eq(series)]
        rows.append({"excluded_series": series, "n": len(subset),
                     "combined_pnl": float(subset["combined_pnl"].sum()),
                     "trading_pnl": float(subset["trading_pnl"].sum()),
                     "reward_pnl": float(subset["reward_pnl"].sum())})
    return pd.DataFrame(rows)


def main() -> None:
    programs = load_programs()
    markets = fetch_markets(programs["market_ticker"].tolist())
    paths = fetch_candles(programs)
    states = build_states(programs, markets, paths)
    splits = chronological_splits(states)
    policy, _ = choose(states, splits)
    results = {}; hard_pass = True
    for model in ["strict", "touch"]:
        selected, metrics = evaluate(states, splits, policy, model, "test")
        boot = bootstrap(selected, "test", splits)
        loo = leave_one(selected)
        selected.to_parquet(OUT / f"commodity15m_priority_test_{model}.parquet", index=False)
        loo.to_csv(OUT / f"commodity15m_priority_leave_one_{model}.csv", index=False)
        results[model] = {"metrics": asdict(metrics), "bootstrap": boot}
        hard_pass &= (
            metrics.combined_pnl > 0 and metrics.n >= 100
            and boot["ci_lo"] > 0 and len(loo) > 0
            and (loo["combined_pnl"] > 0).all())
    summary = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "programs": len(programs), "markets": len(markets), "paths": len(paths),
        "state_rows": len(states),
        "splits": {k: [str(v[0]), str(v[1])] for k, v in splits.items()},
        "validation_policy": policy, "test": results,
        "hard_pass": bool(hard_pass),
    }
    (OUT / "commodity15m_priority_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    verdict = "PASS — priority reward candidate" if hard_pass \
        else "FAIL / PROSPECTIVE CANDIDATE ONLY"
    lines = [
        "# Priority-improved one-sided commodity15m reward audit", "",
        f"## Verdict: **{verdict}**", "",
        "The order is one tick inside the spread on the cheaper outcome, making it",
        "the best bid at submission and therefore qualifying liquidity rather than",
        "an order hidden behind an already-complete target size.", "",
        "## Validation-selected policy", "",
        f"- Quantity: {int(policy['qty'])}",
        f"- Maximum quote price: {float(policy['max_price'])*100:.1f}¢",
        f"- Cancel after: {int(policy['cancel_min'])} minutes",
        f"- Maximum markets/close: {int(policy['cap'])}",
        f"- Selection: {policy['selection']}", "", "## Sealed test", "",
        "| Fill model | n | fills | fill win | Trading | Reward lower | Combined | Mean/day | 95% CI | Max DD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ["strict", "touch"]:
        m = results[model]["metrics"]; b = results[model]["bootstrap"]
        lines.append(
            f"| {model} | {m['n']} | {m['fills']} | {m['fill_win_rate']:.2%} | "
            f"${m['trading_pnl']:.2f} | ${m['reward_pnl']:.2f} | "
            f"${m['combined_pnl']:.2f} | ${m['mean_day']:.2f} | "
            f"[${b['ci_lo']:.2f}, ${b['ci_hi']:.2f}] | ${m['max_drawdown']:.2f} |")
    lines.extend(["", "## Limits", "",
        "At submission the order improves the best bid, but historical one-minute",
        "candles still cannot prove queue position or official score credit. Actual",
        "reward credits require a q1 prospective test. No self-trading or artificial",
        "volume is permitted.", "", "The account KILL state remains binding."])
    (OUT / "commodity15m_priority_report.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
