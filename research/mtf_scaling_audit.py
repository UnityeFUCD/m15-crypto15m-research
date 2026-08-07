"""MTF-S: quantity, minimum-payout and arithmetic-cover audit."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INFILE = ROOT / "research" / "results" / "mtf" / "one_sided_frontiers.csv"
OUT = ROOT / "research" / "results" / "mtf_scaling"
OUT.mkdir(parents=True, exist_ok=True)
REST_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
MIN_PAYOUT = 1.0


def reward(pool: float, target: float, multiplier: float,
           qty: int, rest_fraction: float) -> float:
    own = qty * multiplier * rest_fraction
    return pool * own / (2.0 * target + own)


def min_qty_for_payout(pool: float, target: float, multiplier: float,
                       rest_fraction: float) -> int | None:
    if pool <= MIN_PAYOUT or multiplier <= 0 or rest_fraction <= 0:
        return None
    raw = (MIN_PAYOUT * 2.0 * target) / (
        multiplier * rest_fraction * (pool - MIN_PAYOUT))
    return max(int(math.ceil(raw - 1e-12)), 1)


def main() -> None:
    source = pd.read_csv(INFILE)
    rows = []
    for item in source.itertuples(index=False):
        capacity = max(int(math.floor(float(item.target) - float(item.cumulative_ahead) + 1e-9)), 0)
        for rest in REST_FRACTIONS:
            qty = min_qty_for_payout(
                float(item.reward), float(item.target),
                float(item.multiplier), rest)
            if qty is None:
                continue
            modeled_reward = reward(
                float(item.reward), float(item.target),
                float(item.multiplier), qty, rest)
            worst_loss = qty * float(item.frontier_bid)
            fits = qty <= capacity
            cover = modeled_reward >= worst_loss - 1e-12
            candidate = bool(
                fits and cover and rest <= 0.50
                and float(item.frontier_bid) <= 0.05)
            rows.append({
                "ticker": item.ticker, "series": item.series,
                "side": item.side, "frontier_bid": float(item.frontier_bid),
                "distance_c": float(item.distance_c),
                "multiplier": float(item.multiplier),
                "pool_reward": float(item.reward), "target": float(item.target),
                "cumulative_ahead": float(item.cumulative_ahead),
                "guaranteed_capacity": capacity,
                "rest_fraction": rest,
                "min_qty_for_1usd": qty,
                "modeled_reward": modeled_reward,
                "worst_one_fill_loss": worst_loss,
                "reward_to_loss": modeled_reward / worst_loss if worst_loss > 0 else math.inf,
                "fits_frontier": fits, "arithmetic_cover": cover,
                "scalable_candidate": candidate,
                "reward_day_if_repeated": float(item.reward_day_lower) * qty * rest / max(float(item.own_score), 1e-12) * float(item.multiplier),
            })
    frame = pd.DataFrame(rows)
    frame.sort_values(
        ["scalable_candidate", "rest_fraction", "reward_to_loss", "guaranteed_capacity"],
        ascending=[False, True, False, False], inplace=True)
    frame.to_csv(OUT / "scaling_rows.csv", index=False)
    candidates = frame[frame["scalable_candidate"]].copy()
    candidates.to_csv(OUT / "candidates.csv", index=False)
    summary = {
        "source_rows": int(len(source)), "scenario_rows": int(len(frame)),
        "candidate_rows": int(len(candidates)),
        "candidate_markets": int(candidates["ticker"].nunique()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    cols = ["ticker", "series", "side", "frontier_bid", "distance_c",
            "rest_fraction", "guaranteed_capacity", "min_qty_for_1usd",
            "modeled_reward", "worst_one_fill_loss", "reward_to_loss",
            "fits_frontier", "arithmetic_cover", "scalable_candidate"]
    lines = ["# MTF-S — Target-Frontier Scaling Audit", "",
             f"Scalable arithmetic-cover candidate markets: **{summary['candidate_markets']}**.",
             "", candidates[cols].head(50).to_markdown(index=False, floatfmt=".5f") if len(candidates) else "No candidates.",
             "", "```json", json.dumps(summary, indent=2), "```", "",
             "The calculation assumes the $1 minimum applies per program and",
             "does not prove actual credited reward, rest time, or future frontier stability."]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
