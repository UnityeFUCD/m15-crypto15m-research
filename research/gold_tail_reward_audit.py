"""GTR: KXGOLD15M one-cent reward-funded tail audit.

Public historical data only. No credentials or order submission.
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
OUT = ROOT / "research" / "results" / "gtr"
OUT.mkdir(parents=True, exist_ok=True)
VENDOR = Path(os.environ.get("RFLA_VENDOR_ROOT", ROOT / ".rfla_vendor"))
sys.path.insert(0, str(VENDOR))

from research import reward_adjusted_commodity15m as base  # type: ignore  # noqa:E402
from research import commodity15m_reward_data_fast as fast  # type: ignore  # noqa:E402
from research import reward_adjusted_commodity15m_onesided as one  # type: ignore  # noqa:E402

base.OUT = OUT
one.OUT = OUT

QTY = 64
CANCEL_MIN = 8
PRICE = 0.01
SCORE_STRESSES = (1.00, 0.75)
MIN_PAYOUT = 1.00
REPS = 30_000
SEED = 2026080649


def payout(pool: float, target: float, period_seconds: float,
           rest_seconds: float, score_stress: float) -> float:
    own = QTY * max(rest_seconds, 0.0) * score_stress
    competitors = 2.0 * target * period_seconds
    raw = pool * own / (competitors + own) if competitors + own > 0 else 0.0
    if raw < MIN_PAYOUT:
        return 0.0
    return math.floor(raw * 100.0 + 1e-9) / 100.0


def build_eval(states: pd.DataFrame, model: str, stress: float) -> pd.DataFrame:
    x = states[
        states["fill_model"].eq(model)
        & states["cancel_min"].eq(CANCEL_MIN)
        & np.isclose(states["price"], PRICE, atol=1e-9)
    ].copy()
    x["trading_pnl"] = QTY * x["trading_per_contract"]
    x["reward_pnl"] = [
        payout(float(r.period_reward_dollars), float(r.target_size),
               float(r.period_seconds), float(r.rest_seconds), stress)
        for r in x.itertuples(index=False)
    ]
    x["combined_pnl"] = x["trading_pnl"] + x["reward_pnl"]
    return x.sort_values("end")


def bootstrap_blocks(x: pd.DataFrame, minutes: int) -> dict:
    if x.empty:
        return {"minutes": minutes, "ci_lo": 0.0, "ci_hi": 0.0,
                "p_nonpositive": 1.0, "n_blocks": 0}
    z = x.copy()
    z["block"] = pd.to_datetime(z["end"], utc=True).dt.floor(f"{minutes}min")
    observed = z.groupby("block")["combined_pnl"].sum()
    full = pd.date_range(observed.index.min(), observed.index.max(), freq=f"{minutes}min")
    arr = observed.reindex(full, fill_value=0.0).to_numpy(float)
    rng = np.random.default_rng(SEED + minutes)
    totals = np.empty(REPS)
    chunk = 1000
    for start in range(0, REPS, chunk):
        stop = min(REPS, start + chunk)
        idx = rng.integers(0, len(arr), size=(stop - start, len(arr)))
        totals[start:stop] = arr[idx].sum(axis=1)
    return {
        "minutes": minutes, "n_blocks": int(len(arr)),
        "observed_total": float(arr.sum()),
        "ci_lo": float(np.quantile(totals, 0.025)),
        "ci_hi": float(np.quantile(totals, 0.975)),
        "p_nonpositive": float(np.mean(totals <= 0.0)),
    }


def metrics(x: pd.DataFrame) -> dict:
    if x.empty:
        return {"n": 0}
    daily = x.groupby("day")["combined_pnl"].sum()
    windows = x.groupby("close_ts")["combined_pnl"].sum().sort_index()
    equity = windows.cumsum(); dd = equity.cummax() - equity
    filled = x[x["filled"]]
    leave_days = []
    for day in sorted(x["day"].unique()):
        kept = x[~x["day"].eq(day)]
        leave_days.append({"dropped": day, "pnl": float(kept["combined_pnl"].sum())})
    return {
        "n": int(len(x)), "days": int(x["day"].nunique()),
        "fills": int(x["filled"].sum()), "fill_rate": float(x["filled"].mean()),
        "fill_win_rate": float(filled["won"].mean()) if len(filled) else float("nan"),
        "trading_pnl": float(x["trading_pnl"].sum()),
        "reward_pnl": float(x["reward_pnl"].sum()),
        "combined_pnl": float(x["combined_pnl"].sum()),
        "paid_program_fraction": float((x["reward_pnl"] >= MIN_PAYOUT).mean()),
        "mean_paid_reward": float(x.loc[x["reward_pnl"] > 0, "reward_pnl"].mean()) if (x["reward_pnl"] > 0).any() else 0.0,
        "mean_day": float(daily.mean()), "sd_day": float(daily.std(ddof=1)) if len(daily)>1 else 0.0,
        "max_drawdown": float(dd.max()), "worst_window": float(windows.min()),
        "leave_one_day_min": float(min(r["pnl"] for r in leave_days)),
        "leave_one_day": leave_days,
    }


def main() -> None:
    programs = fast.load_programs()
    programs = programs[programs["series_ticker"].eq("KXGOLD15M")].copy()
    if programs.empty:
        raise RuntimeError("no KXGOLD15M reward programs")
    markets = fast.fetch_markets(programs["market_ticker"].tolist())
    paths = fast.fetch_candles(programs)
    states = one.build_states(programs, markets, paths)

    results = {}
    hard = True
    rows = []
    for model in ("strict", "touch"):
        for stress in SCORE_STRESSES:
            selected = build_eval(states, model, stress)
            selected.to_parquet(OUT / f"selected_{model}_{int(stress*100)}.parquet", index=False)
            m = metrics(selected)
            blocks = [bootstrap_blocks(selected, minutes) for minutes in (15,30,60,90)]
            m["blocks"] = blocks
            results[f"{model}_{stress:.2f}"] = m
            rows.append({
                "fill_model": model, "score_stress": stress,
                **{k:v for k,v in m.items() if k not in {"leave_one_day","blocks"}},
                **{f"block{b['minutes']}_lo": b["ci_lo"] for b in blocks},
                **{f"block{b['minutes']}_p": b["p_nonpositive"] for b in blocks},
            })
            hard &= (
                m.get("n",0) >= 50
                and m.get("combined_pnl",-1) > 0
                and m.get("leave_one_day_min",-1) > 0
                and m.get("max_drawdown",1e9) < 10.0
                and all(b["ci_lo"] > 0 for b in blocks)
            )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "summary_table.csv", index=False)
    summary = {
        "programs": int(len(programs)), "markets": int(len(markets)),
        "paths": int(len(paths)), "state_rows": int(len(states)),
        "hard_pass": bool(hard), "results": results,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    verdict = "STRONG PROSPECTIVE CANDIDATE" if hard else "FAIL"
    cols = ["fill_model","score_stress","n","fills","fill_rate","fill_win_rate",
            "trading_pnl","reward_pnl","combined_pnl","paid_program_fraction",
            "mean_day","max_drawdown","worst_window","leave_one_day_min",
            "block60_lo","block60_p"]
    lines = ["# GTR — Gold Tail-Reward Audit", "",
             f"## Verdict: **{verdict}**", "",
             table[cols].to_markdown(index=False, floatfmt=".5f"), "",
             "The key unresolved fact is whether a live order actually receives",
             "the modeled qualifying score and reward credit. No order is authorized."]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
