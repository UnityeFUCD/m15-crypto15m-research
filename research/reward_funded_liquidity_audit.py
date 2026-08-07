"""RFLA: reward-funded liquidity audit.

Research only. Public endpoints only. No credentials and no order submission.
See research/rfla_preregistration.md.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "results" / "rfla"
OUT.mkdir(parents=True, exist_ok=True)
VENDOR = Path(os.environ.get("RFLA_VENDOR_ROOT", ROOT / ".rfla_vendor"))
sys.path.insert(0, str(VENDOR))

from research import reward_adjusted_commodity15m as base  # type: ignore  # noqa:E402
from research import commodity15m_reward_data_fast as fast  # type: ignore  # noqa:E402
from research import reward_adjusted_commodity15m_priority as priority  # type: ignore  # noqa:E402

base.OUT = OUT
priority.OUT = OUT

QTY = 1
REWARD_STRESSES = (1.00, 0.50, 0.25)
BLOCK_MINUTES = (15, 30, 60, 90)
RISK_TIERS = (0.20, 0.30)
SEED = 2026080637
REPS = 20_000


def full_dates(programs: pd.DataFrame) -> list[str]:
    lo = programs["end"].dt.normalize().min()
    hi = programs["end"].dt.normalize().max()
    return [x.date().isoformat() for x in pd.date_range(lo, hi, freq="D")]


def finish_metrics(selected: pd.DataFrame, dates: list[str], stress: float) -> dict:
    x = selected.copy()
    x["reward_stressed"] = x["reward_pnl"] * stress
    x["combined_stressed"] = x["trading_pnl"] + x["reward_stressed"]
    daily = x.groupby("day")["combined_stressed"].sum().reindex(dates, fill_value=0.0)
    windows = x.groupby("close_ts")["combined_stressed"].sum().sort_index()
    equity = windows.cumsum()
    dd = equity.cummax() - equity
    active_col = "filled" if "filled" in x.columns else "none_filled"
    if active_col == "filled":
        active = x[x["filled"]]
        fill_rate = float(x["filled"].mean()) if len(x) else 0.0
        fill_win = float(active["won"].mean()) if len(active) else float("nan")
    else:
        active = x[~x["none_filled"]]
        fill_rate = float((~x["none_filled"]).mean()) if len(x) else 0.0
        if len(active):
            # A complementary pair is not directional when both fill, so only
            # report active fraction here.
            fill_win = float("nan")
        else:
            fill_win = float("nan")
    result = {
        "n": int(len(x)),
        "active": int(len(active)),
        "fill_rate": fill_rate,
        "fill_win_rate": fill_win,
        "trading_pnl": float(x["trading_pnl"].sum()),
        "reward_pnl_model": float(x["reward_pnl"].sum()),
        "reward_pnl_stressed": float(x["reward_stressed"].sum()),
        "combined_pnl": float(x["combined_stressed"].sum()),
        "mean_day": float(daily.mean()),
        "sd_day": float(daily.std(ddof=1)) if len(daily) > 1 else 0.0,
        "positive_day_fraction": float((daily > 0).mean()),
        "max_drawdown": float(dd.max()) if len(dd) else 0.0,
        "worst_window": float(windows.min()) if len(windows) else 0.0,
    }
    return result


def block_bootstrap(selected: pd.DataFrame, stress: float, minutes: int) -> dict:
    if selected.empty:
        return {"block_minutes": minutes, "ci_lo": 0.0, "ci_hi": 0.0,
                "p_nonpositive": 1.0, "n_blocks": 0}
    x = selected.copy()
    x["combined_stressed"] = x["trading_pnl"] + stress * x["reward_pnl"]
    dt = pd.to_datetime(x["end"], utc=True)
    x["block"] = dt.dt.floor(f"{minutes}min")
    observed = x.groupby("block")["combined_stressed"].sum()
    full_index = pd.date_range(observed.index.min(), observed.index.max(), freq=f"{minutes}min")
    blocks = observed.reindex(full_index, fill_value=0.0).to_numpy(float)
    rng = np.random.default_rng(SEED + minutes + int(stress * 100))
    totals = np.empty(REPS)
    chunk = 1000
    for start in range(0, REPS, chunk):
        stop = min(REPS, start + chunk)
        idx = rng.integers(0, len(blocks), size=(stop - start, len(blocks)))
        totals[start:stop] = blocks[idx].sum(axis=1)
    return {
        "block_minutes": minutes,
        "n_blocks": int(len(blocks)),
        "observed_total": float(blocks.sum()),
        "ci_lo": float(np.quantile(totals, 0.025)),
        "ci_hi": float(np.quantile(totals, 0.975)),
        "p_nonpositive": float(np.mean(totals <= 0.0)),
    }


def leave_one(selected: pd.DataFrame, stress: float) -> list[dict]:
    rows = []
    for series in sorted(selected["series_ticker"].unique()):
        kept = selected[~selected["series_ticker"].eq(series)].copy()
        combined = kept["trading_pnl"] + stress * kept["reward_pnl"]
        rows.append({
            "excluded_series": series,
            "n": int(len(kept)),
            "combined_pnl": float(combined.sum()),
            "trading_pnl": float(kept["trading_pnl"].sum()),
            "reward_pnl_stressed": float((stress * kept["reward_pnl"]).sum()),
        })
    return rows


def two_sided_rows(states: pd.DataFrame, dates: list[str]) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    loo_rows: list[dict] = []
    for model in ("strict", "touch"):
        model_states = base.apply_qty(states[states["fill_model"].eq(model)], QTY)
        for cancel in sorted(model_states["cancel_none_min"].unique()):
            subset = model_states[model_states["cancel_none_min"].eq(cancel)]
            selected = base.select_close(subset, 1, "reward_density")
            for stress in REWARD_STRESSES:
                metrics = finish_metrics(selected, dates, stress)
                block = [block_bootstrap(selected, stress, m) for m in BLOCK_MINUTES]
                row = {
                    "policy": "P1_two_sided_join",
                    "fill_model": model,
                    "cancel_min": int(cancel),
                    "max_price": np.nan,
                    "reward_stress": stress,
                    **metrics,
                    **{f"block{b['block_minutes']}_lo": b["ci_lo"] for b in block},
                    **{f"block{b['block_minutes']}_p": b["p_nonpositive"] for b in block},
                }
                rows.append(row)
                for loo in leave_one(selected, stress):
                    loo_rows.append({
                        "policy": "P1_two_sided_join", "fill_model": model,
                        "cancel_min": int(cancel), "max_price": np.nan,
                        "reward_stress": stress, **loo,
                    })
    return rows, loo_rows


def priority_rows(states: pd.DataFrame, dates: list[str]) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    loo_rows: list[dict] = []
    for model in ("strict", "touch"):
        model_states = priority.size_states(states[states["fill_model"].eq(model)], QTY)
        for tier in RISK_TIERS:
            for cancel in sorted(model_states["cancel_min"].unique()):
                subset = model_states[
                    model_states["cancel_min"].eq(cancel)
                    & model_states["price"].le(tier)
                ]
                selected = priority.select_close(subset, 1, "reward_risk")
                for stress in REWARD_STRESSES:
                    metrics = finish_metrics(selected, dates, stress)
                    block = [block_bootstrap(selected, stress, m) for m in BLOCK_MINUTES]
                    row = {
                        "policy": "P2_priority_cheap_side",
                        "fill_model": model,
                        "cancel_min": int(cancel),
                        "max_price": tier,
                        "reward_stress": stress,
                        **metrics,
                        **{f"block{b['block_minutes']}_lo": b["ci_lo"] for b in block},
                        **{f"block{b['block_minutes']}_p": b["p_nonpositive"] for b in block},
                    }
                    rows.append(row)
                    for loo in leave_one(selected, stress):
                        loo_rows.append({
                            "policy": "P2_priority_cheap_side", "fill_model": model,
                            "cancel_min": int(cancel), "max_price": tier,
                            "reward_stress": stress, **loo,
                        })
    return rows, loo_rows


def candidate_gate(summary: pd.DataFrame, loo: pd.DataFrame) -> tuple[bool, dict]:
    details = {}
    passed_any = False
    for tier in RISK_TIERS:
        cells = summary[
            summary["policy"].eq("P2_priority_cheap_side")
            & summary["max_price"].eq(tier)
            & summary["reward_stress"].eq(0.25)
        ]
        loo_cells = loo[
            loo["policy"].eq("P2_priority_cheap_side")
            & loo["max_price"].eq(tier)
            & loo["reward_stress"].eq(0.25)
        ]
        expected_cells = 2 * 6
        cell_positive = len(cells) == expected_cells and bool((cells["combined_pnl"] > 0).all())
        loo_positive = len(loo_cells) > 0 and bool((loo_cells["combined_pnl"] > 0).all())
        # Stronger descriptive check, not an extra post-hoc optimization:
        # every block lower bound positive.
        block_cols = [f"block{m}_lo" for m in BLOCK_MINUTES]
        blocks_positive = len(cells) == expected_cells and bool((cells[block_cols] > 0).all().all())
        tier_pass = cell_positive and loo_positive
        details[str(tier)] = {
            "cells": int(len(cells)),
            "cell_positive": cell_positive,
            "leave_one_positive": loo_positive,
            "all_block_lower_bounds_positive": blocks_positive,
            "pass": tier_pass,
        }
        passed_any |= tier_pass
    return passed_any, details


def main() -> None:
    programs = fast.load_programs()
    programs.to_csv(OUT / "programs.csv", index=False)
    markets = fast.fetch_markets(programs["market_ticker"].tolist())
    paths = fast.fetch_candles(programs)
    print(f"programs={len(programs)} days={programs.end.dt.date.nunique()} markets={len(markets)} paths={len(paths)}", flush=True)

    two_states = base.build_states(programs, markets, paths)
    priority_states = priority.build_states(programs, markets, paths)
    dates = full_dates(programs)

    p1, p1_loo = two_sided_rows(two_states, dates)
    p2, p2_loo = priority_rows(priority_states, dates)
    summary = pd.DataFrame(p1 + p2)
    loo = pd.DataFrame(p1_loo + p2_loo)
    summary.to_csv(OUT / "policy_summary.csv", index=False)
    loo.to_csv(OUT / "leave_one_series.csv", index=False)

    passed, gate = candidate_gate(summary, loo)
    result = {
        "programs": int(len(programs)),
        "days": int(programs.end.dt.date.nunique()),
        "market_coverage": int(len(markets)),
        "path_coverage": int(len(paths)),
        "two_sided_state_rows": int(len(two_states)),
        "priority_state_rows": int(len(priority_states)),
        "candidate_gate": bool(passed),
        "gate_details": gate,
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    stress25 = summary[summary["reward_stress"].eq(0.25)].copy()
    display_cols = [
        "policy", "fill_model", "max_price", "cancel_min", "n", "active",
        "fill_rate", "trading_pnl", "reward_pnl_stressed", "combined_pnl",
        "mean_day", "max_drawdown", "worst_window", "block60_lo",
        "block60_p",
    ]
    verdict = "STRONG HISTORICAL CANDIDATE" if passed else "FAIL / PROSPECTIVE ONLY"
    lines = [
        "# RFLA — Reward-Funded Liquidity Audit", "",
        f"## Verdict: **{verdict}**", "",
        f"Programs: {len(programs):,}; calendar days: {programs.end.dt.date.nunique()}; paths: {len(paths):,}.",
        "", "The table below uses only 25% of the conservative modeled reward credit.", "",
        stress25[display_cols].to_markdown(index=False, floatfmt=".4f"),
        "", "## Candidate gate", "", "```json",
        json.dumps(gate, indent=2), "```", "",
        "A historical candidate still requires an actual q1 reward-credit experiment.",
        "No result authorizes live trading or weakens any KILL state.",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
