"""Reward-Funded Hedged Liquidity (RFHL) audit.

Architecture
------------
1. In an incentivized 15-minute Gold/Silver/WTI market, quote only the cheaper
   outcome one tick inside the spread so the order is qualifying best-price
   liquidity at submission.
2. Earn liquidity score while the maker order rests.
3. If it fills, immediately buy the complementary outcome with an IOC. The two
   contracts then pay exactly $1, eliminating settlement direction risk.
4. If it never fills, cancel and keep only the liquidity reward.

The remaining risks are reward-share uncertainty, maker-to-hedge latency,
slippage, and partial IOC execution. Historical one-minute candlesticks cannot
identify subsecond hedging, so the audit stresses 0-5 cents of extra hedge cost
and requires both strict and touch maker-fill models to pass.

Public data only. No credentials and no orders.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

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
SELECTION_GRID = ["reward_cost", "cheapest", "reward_density"]
SLIPPAGE_GRID_C = [0, 1, 2, 3, 5]
TICK = 0.01
RNG = np.random.default_rng(2026080637)


def taker_fee_total(qty: int, price: float) -> float:
    raw = 0.07 * qty * price * (1.0 - price)
    return math.ceil(raw * 10_000 - 1e-12) / 10_000


def next_candle(candles: list[dict], after_ts: int) -> dict | None:
    rows = [candle for candle in candles if candle["ts"] > after_ts]
    return min(rows, key=lambda row: row["ts"]) if rows else None


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
            side, old_bid, own_ask = "yes", yes_bid, yes_ask
        else:
            side, old_bid, own_ask = "no", no_bid, no_ask
        maker_price = round(old_bid + TICK, 2)
        if not (0 < maker_price < own_ask - 1e-9):
            continue
        period_seconds = max((program.end - program.start).total_seconds(), 1.0)
        entry_ts = int(entry["ts"])
        end_ts = int(program.end.timestamp())
        for fill_model in ["strict", "touch"]:
            maker_fill = first_fill(
                candles, entry_ts, maker_price, side, fill_model, end_ts)
            for cancel_min in CANCEL_MIN_GRID:
                deadline = min(entry_ts + cancel_min * 60, end_ts)
                fill_ts = maker_fill \
                    if maker_fill is not None and maker_fill <= deadline else None
                rest_until = fill_ts if fill_ts is not None else deadline
                hedge = next_candle(candles, fill_ts) if fill_ts is not None else None
                if fill_ts is not None and hedge is None:
                    continue
                if hedge is None:
                    base_hedge_ask = np.nan
                elif side == "yes":
                    # Complement is NO; NO ask = 1 - YES bid.
                    base_hedge_ask = 1.0 - float(hedge["yb_close"])
                else:
                    # Complement is YES.
                    base_hedge_ask = float(hedge["ya_close"])
                for slippage_c in SLIPPAGE_GRID_C:
                    if fill_ts is None:
                        hedge_price = np.nan
                        locked_per_contract = 0.0
                        hedge_fee_per_contract = 0.0
                    else:
                        hedge_price = min(base_hedge_ask + slippage_c / 100.0, 0.9999)
                        hedge_fee_per_contract = taker_fee_total(1, hedge_price)
                        locked_per_contract = (
                            1.0 - maker_price - hedge_price
                            - hedge_fee_per_contract)
                    rows.append({
                        "market_ticker": program.market_ticker,
                        "series_ticker": program.series_ticker,
                        "program_id": program.program_id,
                        "start": program.start, "end": program.end,
                        "day": program.end.date().isoformat(),
                        "week": f"{program.end.isocalendar().year}-{program.end.isocalendar().week:02d}",
                        "close_ts": end_ts,
                        "fill_model": fill_model,
                        "cancel_min": cancel_min,
                        "slippage_c": slippage_c,
                        "side": side, "old_bid": old_bid,
                        "maker_price": maker_price, "own_ask": own_ask,
                        "target_size": program.target_size,
                        "period_reward_dollars": program.period_reward_dollars,
                        "period_seconds": period_seconds,
                        "rest_seconds": max(rest_until - entry_ts, 0),
                        "maker_filled": fill_ts is not None,
                        "fill_ts": fill_ts,
                        "hedge_price": hedge_price,
                        "hedge_fee_per_contract": hedge_fee_per_contract,
                        "locked_per_contract": locked_per_contract,
                    })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no RFHL states")
    frame.to_parquet(OUT / "rfhl_states.parquet", index=False)
    return frame


def size_states(frame: pd.DataFrame, qty: int) -> pd.DataFrame:
    x = frame.copy()
    own_score_time = qty * x["rest_seconds"]
    max_competitor_score_time = 2.0 * x["target_size"] * x["period_seconds"]
    share = own_score_time / (max_competitor_score_time + own_score_time)
    x["reward_pnl"] = x["period_reward_dollars"] * share
    # Fee formula is block-rounded, not exactly linear from q1. Recompute.
    fees = np.zeros(len(x))
    mask = x["maker_filled"] & x["hedge_price"].notna()
    fees[mask] = [
        taker_fee_total(qty, float(price))
        for price in x.loc[mask, "hedge_price"]
    ]
    x["hedge_fee"] = fees
    x["trading_pnl"] = np.where(
        x["maker_filled"],
        qty * (1.0 - x["maker_price"] - x["hedge_price"]) - x["hedge_fee"],
        0.0)
    x["combined_pnl"] = x["reward_pnl"] + x["trading_pnl"]
    x["qty"] = qty
    x["reward_cost"] = x["reward_pnl"] / np.maximum(-x["trading_pnl"], 0.0001)
    x["reward_density"] = x["period_reward_dollars"] / x["target_size"]
    return x


def select_close(frame: pd.DataFrame, cap: int, selection: str) -> pd.DataFrame:
    if selection == "reward_cost":
        cols, asc = ["close_ts", "reward_cost", "maker_price", "series_ticker"], [True, False, True, True]
    elif selection == "cheapest":
        cols, asc = ["close_ts", "maker_price", "reward_cost", "series_ticker"], [True, True, False, True]
    else:
        cols, asc = ["close_ts", "reward_density", "maker_price", "series_ticker"], [True, False, True, True]
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
    slippage_c: int
    n: int
    maker_fills: int
    maker_fill_rate: float
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
           cap: int, selection: str, slippage_c: int) -> Metrics:
    start, end = splits[split]
    days = pd.date_range(start.normalize(), end.normalize() - pd.Timedelta(days=1), freq="D").date
    if selected.empty:
        return Metrics(split, model, qty, max_price, cancel_min, cap, selection,
                       slippage_c, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    daily = selected.groupby("day")["combined_pnl"].sum().reindex(
        [d.isoformat() for d in days], fill_value=0.0)
    window = selected.groupby("close_ts")["combined_pnl"].sum().sort_index()
    equity = window.cumsum(); dd = equity.cummax() - equity
    sd = float(daily.std(ddof=1)); mean = float(daily.mean())
    return Metrics(
        split, model, qty, max_price, cancel_min, cap, selection, slippage_c,
        int(len(selected)), int(selected["maker_filled"].sum()),
        float(selected["maker_filled"].mean()),
        float(selected["trading_pnl"].sum()),
        float(selected["reward_pnl"].sum()),
        float(selected["combined_pnl"].sum()), mean, sd,
        mean / (sd / math.sqrt(len(daily))) if sd > 0 else 0.0,
        float(dd.max()) if len(dd) else 0.0,
        float(window.min()) if len(window) else 0.0,
        float((daily > 0).mean()))


def choose(states: pd.DataFrame, splits: dict) -> tuple[dict, pd.DataFrame]:
    start, end = splits["valid"]
    valid = states[(states["end"] >= start) & (states["end"] < end)]
    rows = []
    # Policy selection uses a fixed 2c hedge-slippage stress and the worse of
    # strict/touch maker-fill definitions.
    for model in ["strict", "touch"]:
        for qty in QTY_GRID:
            sized = size_states(valid[valid["fill_model"].eq(model)
                                      & valid["slippage_c"].eq(2)], qty)
            for max_price in MAX_PRICE_GRID:
                for cancel in CANCEL_MIN_GRID:
                    subset = sized[
                        sized["cancel_min"].eq(cancel)
                        & (sized["maker_price"] <= max_price)]
                    for cap in CAP_GRID:
                        for selection in SELECTION_GRID:
                            selected = select_close(subset, cap, selection)
                            rows.append(asdict(metric(
                                selected, "valid", splits, model, qty,
                                max_price, cancel, cap, selection, 2)))
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "rfhl_validation_grid.csv", index=False)
    keys = ["qty", "max_price", "cancel_min", "cap", "selection"]
    worst = (grid.groupby(keys).agg(
        worst_pnl=("combined_pnl", "min"), worst_t=("t_stat", "min"),
        worst_dd=("max_drawdown", "max"), min_n=("n", "min"),
        min_fills=("maker_fills", "min")).reset_index())
    viable = worst[(worst["min_n"] >= 100) & (worst["worst_pnl"] > 0)].copy()
    if viable.empty:
        viable = worst[worst["min_n"] >= 50].copy()
    if viable.empty:
        viable = worst.copy()
    viable["objective"] = viable["worst_t"] + 0.1 * np.log1p(viable["min_n"]) - 0.01 * viable["worst_dd"]
    winner = viable.sort_values(
        ["objective", "worst_t", "worst_pnl"], ascending=False).iloc[0].to_dict()
    return winner, grid


def evaluate(states: pd.DataFrame, splits: dict, policy: dict,
             model: str, slippage_c: int, split: str) -> tuple[pd.DataFrame, Metrics]:
    start, end = splits[split]
    subset = states[
        (states["end"] >= start) & (states["end"] < end)
        & states["fill_model"].eq(model)
        & states["slippage_c"].eq(slippage_c)
        & states["cancel_min"].eq(int(policy["cancel_min"]))
        & (states["maker_price"] <= float(policy["max_price"]))]
    selected = select_close(
        size_states(subset, int(policy["qty"])),
        int(policy["cap"]), str(policy["selection"]))
    return selected, metric(
        selected, split, splits, model, int(policy["qty"]),
        float(policy["max_price"]), int(policy["cancel_min"]),
        int(policy["cap"]), str(policy["selection"]), slippage_c)


def bootstrap(selected: pd.DataFrame, split: str, splits: dict,
              reps: int = 12000) -> dict:
    start, end = splits[split]
    days = pd.date_range(start.normalize(), end.normalize() - pd.Timedelta(days=1), freq="D").date
    daily = selected.groupby("day")["combined_pnl"].sum().reindex(
        [d.isoformat() for d in days], fill_value=0.0).to_numpy()
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
        results[model] = {}
        for slippage_c in SLIPPAGE_GRID_C:
            selected, metrics = evaluate(
                states, splits, policy, model, slippage_c, "test")
            boot = bootstrap(selected, "test", splits)
            results[model][str(slippage_c)] = {
                "metrics": asdict(metrics), "bootstrap": boot}
            selected.to_parquet(
                OUT / f"rfhl_test_{model}_{slippage_c}c.parquet", index=False)
            if slippage_c == 2:
                loo = leave_one(selected)
                loo.to_csv(OUT / f"rfhl_leave_one_{model}.csv", index=False)
                hard_pass &= (
                    metrics.combined_pnl > 0 and metrics.n >= 100
                    and boot["ci_lo"] > 0 and len(loo) > 0
                    and (loo["combined_pnl"] > 0).all())
    summary = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "programs": len(programs), "markets": len(markets), "paths": len(paths),
        "state_rows": len(states),
        "splits": {k: [str(v[0]), str(v[1])] for k, v in splits.items()},
        "validation_policy_selected_at_2c_slippage": policy,
        "test": results, "hard_pass": bool(hard_pass),
    }
    (OUT / "rfhl_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    verdict = "PASS — reward-funded hedged liquidity candidate" if hard_pass \
        else "FAIL / PROSPECTIVE CANDIDATE ONLY"
    lines = [
        "# Reward-Funded Hedged Liquidity audit", "",
        f"## Verdict: **{verdict}**", "",
        "A one-tick-improved maker quote earns liquidity score. On fill, the",
        "complementary outcome is bought immediately, replacing settlement risk",
        "with a bounded spread/fee/slippage loss.", "", "## Validation-selected policy", "",
        f"- Quantity: {int(policy['qty'])}",
        f"- Maximum maker price: {float(policy['max_price'])*100:.1f}¢",
        f"- Cancel no-fill after: {int(policy['cancel_min'])} minutes",
        f"- Maximum markets/close: {int(policy['cap'])}",
        f"- Selection: {policy['selection']}",
        "- Policy selected under 2¢ hedge slippage and the worse fill model", "",
        "## Sealed test sensitivity", "",
        "| Fill model | Extra hedge cost | n | maker fills | Trading | Reward lower | Combined | Mean/day | 95% CI | Max DD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ["strict", "touch"]:
        for slippage_c in SLIPPAGE_GRID_C:
            item = results[model][str(slippage_c)]
            m = item["metrics"]; b = item["bootstrap"]
            lines.append(
                f"| {model} | {slippage_c}¢ | {m['n']} | {m['maker_fills']} | "
                f"${m['trading_pnl']:.2f} | ${m['reward_pnl']:.2f} | "
                f"${m['combined_pnl']:.2f} | ${m['mean_day']:.2f} | "
                f"[${b['ci_lo']:.2f}, ${b['ci_hi']:.2f}] | ${m['max_drawdown']:.2f} |")
    lines.extend(["", "## Limits", "",
        "One-minute candles cannot identify subsecond hedge latency, IOC depth, or",
        "official liquidity score. A historical PASS only authorizes a q1",
        "prospective trial with actual reward credits and immediate-hedge telemetry.",
        "No self-trading or artificial volume is permitted.", "",
        "The account KILL state remains binding."])
    (OUT / "rfhl_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
