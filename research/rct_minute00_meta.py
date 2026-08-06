
"""Convergent audit for the frozen minute-:00 delayed-taker/RCT candidate.

This script is deliberately narrower than the discovery audits.

Questions:
1. Does the minute-:00 delayed-taker population have positive full-period value
   after a four-minute max-stat selection correction?
2. Does the pre-frozen RCT bid-strengthening condition add value beyond the
   same delayed-taker population?
3. Does a second same-close RCT candidate add robust dollars, or only
   correlated tail risk?
4. What quantity is compatible with the $300 reference bankroll under the
   observed day sequence?

The primary RCT configuration and minute :00 were fixed independently before
this script. Rank-2 is explicitly exploratory.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.runaway_confirmation import (
    COINS,
    DELAYED_ASK_FLOOR,
    PRIMARY,
    QTY,
    Config,
    SPLITS,
    build_panel,
    fee_total,
    qualifying,
    split_mask,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "results"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 2026080627
RNG = np.random.default_rng(SEED)
MINUTES = [0, 15, 30, 45]

PRIMARY_CONFIG = Config(
    delay=int(PRIMARY["delay"]),
    bid_move_c=float(PRIMARY["bid_move_c"]),
    ask_ceiling=float(PRIMARY["ask_ceiling"]),
    spread_widen_limit_c=PRIMARY["spread_widen_limit_c"],
    volume_filter=bool(PRIMARY["volume_filter"]),
)
CONTROL_CONFIG = Config(
    delay=int(PRIMARY["delay"]),
    bid_move_c=-999.0,
    ask_ceiling=float(PRIMARY["ask_ceiling"]),
    spread_widen_limit_c=PRIMARY["spread_widen_limit_c"],
    volume_filter=bool(PRIMARY["volume_filter"]),
)


def all_calendar() -> list[str]:
    start = min(value[0] for value in SPLITS.values())
    stop = max(value[1] for value in SPLITS.values())
    return [
        day.isoformat()
        for day in pd.date_range(
            start.normalize(),
            stop.normalize() - pd.Timedelta(days=1),
            freq="D",
        ).date
    ]


def ranked(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.assign(rank=np.nan)
    return (
        frame.sort_values(
            [
                "close_ts",
                "bid_move_c",
                "spread_change_c",
                "delayed_ask",
                "volume",
                "coin",
            ],
            ascending=[True, False, True, True, False, True],
        )
        .assign(rank=lambda x: x.groupby("close_ts").cumcount() + 1)
        .sort_values(["close_ts", "rank"])
    )


def select(
    panel: pd.DataFrame,
    config: Config,
    minute: int,
    split: str | None = None,
    cap: int = 1,
    exact_rank: int | None = None,
) -> pd.DataFrame:
    data = panel[panel["close_minute"].eq(minute)]
    if split is not None:
        data = data[split_mask(data, split)]
    candidates = ranked(qualifying(data, config))
    if exact_rank is not None:
        return candidates[candidates["rank"].eq(exact_rank)].copy()
    return candidates[candidates["rank"] <= cap].copy()


def pnl_for_qty(frame: pd.DataFrame, quantity: int) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    fees = np.asarray(
        [fee_total(quantity, price) for price in frame["delayed_ask"]],
        dtype=float,
    )
    return quantity * (frame["won"].to_numpy() - frame["delayed_ask"].to_numpy()) - fees


def daily_series(
    frame: pd.DataFrame,
    calendar: list[str],
    quantity: int = QTY,
) -> pd.Series:
    if frame.empty:
        return pd.Series(0.0, index=calendar)
    values = frame.assign(_pnl=pnl_for_qty(frame, quantity))
    return (
        values.groupby("day")["_pnl"]
        .sum()
        .reindex(calendar, fill_value=0.0)
    )


def metrics(
    frame: pd.DataFrame,
    calendar: list[str],
    quantity: int = QTY,
) -> dict[str, float]:
    daily = daily_series(frame, calendar, quantity)
    values = frame.assign(_pnl=pnl_for_qty(frame, quantity))
    windows = values.groupby("close_ts")["_pnl"].sum().sort_index()
    cumulative = windows.cumsum()
    dd = cumulative.cummax() - cumulative
    total = float(values["_pnl"].sum())
    return {
        "n": int(len(frame)),
        "days": int(len(calendar)),
        "total_pnl": total,
        "mean_day": float(daily.mean()),
        "sd_day": float(daily.std(ddof=1)),
        "edge_per_contract": (
            total / (len(frame) * quantity) if len(frame) else np.nan
        ),
        "win_rate": float(frame["won"].mean()) if len(frame) else np.nan,
        "mean_ask": float(frame["delayed_ask"].mean()) if len(frame) else np.nan,
        "max_drawdown": float(dd.max()) if len(dd) else 0.0,
        "worst_window": float(windows.min()) if len(windows) else 0.0,
        "positive_day_fraction": float((daily > 0).mean()),
    }


def bootstrap_mean(
    values: np.ndarray,
    repetitions: int = 40000,
) -> dict[str, float]:
    draws = RNG.integers(0, len(values), size=(repetitions, len(values)))
    means = values[draws].mean(axis=1)
    return {
        "observed": float(values.mean()),
        "ci_lo": float(np.quantile(means, 0.025)),
        "ci_hi": float(np.quantile(means, 0.975)),
        "p_nonpositive": float(np.mean(means <= 0)),
    }


def fixed_effect_meta(rows: list[dict[str, float]]) -> dict[str, float]:
    effects = np.asarray([row["mean_day"] for row in rows], dtype=float)
    ses = np.asarray(
        [
            row["sd_day"] / math.sqrt(max(int(row["days"]), 1))
            for row in rows
        ],
        dtype=float,
    )
    variances = np.maximum(ses**2, 1e-12)
    weights = 1.0 / variances
    estimate = float(np.sum(weights * effects) / np.sum(weights))
    se = float(math.sqrt(1.0 / np.sum(weights)))
    q = float(np.sum(weights * (effects - estimate) ** 2))
    return {
        "estimate_mean_day": estimate,
        "se": se,
        "ci_lo": estimate - 1.96 * se,
        "ci_hi": estimate + 1.96 * se,
        "z": estimate / se if se > 0 else np.nan,
        "cochran_q": q,
        "df": max(len(rows) - 1, 0),
    }


def permutation_max_stat(
    daily_by_minute: pd.DataFrame,
    observed_minute: int = 0,
    repetitions: int = 30000,
) -> dict[str, float]:
    """Shuffle minute labels within day and record the best of four.

    This corrects for the fact that four close-minute labels are available.
    Every day's four values stay together; only their labels are permuted.
    """
    matrix = daily_by_minute[MINUTES].to_numpy(dtype=float)
    observed_means = matrix.mean(axis=0)
    observed = float(observed_means[MINUTES.index(observed_minute)])
    maxima = np.empty(repetitions, dtype=float)
    direct = np.empty(repetitions, dtype=float)
    for iteration in range(repetitions):
        permuted = np.empty_like(matrix)
        for row in range(len(matrix)):
            permuted[row] = matrix[row, RNG.permutation(len(MINUTES))]
        means = permuted.mean(axis=0)
        maxima[iteration] = means.max()
        direct[iteration] = means[MINUTES.index(observed_minute)]
    return {
        "observed_minute00_mean_day": observed,
        "observed_best_mean_day": float(observed_means.max()),
        "best_minute": int(MINUTES[int(np.argmax(observed_means))]),
        "p_direct_label": float(np.mean(direct >= observed)),
        "p_best_of_four": float(np.mean(maxima >= observed)),
        "permuted_best_p95": float(np.quantile(maxima, 0.95)),
    }


def moving_block_bootstrap_days(
    daily: np.ndarray,
    horizon_days: int = 30,
    block_days: int = 3,
    repetitions: int = 100000,
    start_equity: float = 300.0,
    floor: float = 211.0,
) -> dict[str, float]:
    n = len(daily)
    blocks = np.asarray(
        [
            np.asarray([daily[(start + offset) % n] for offset in range(block_days)])
            for start in range(n)
        ]
    )
    ruin = 0
    drawdown_30 = 0
    terminals = np.empty(repetitions, dtype=float)
    max_drawdowns = np.empty(repetitions, dtype=float)
    for begin in range(0, repetitions, 2000):
        count = min(2000, repetitions - begin)
        needed = math.ceil(horizon_days / block_days)
        choices = RNG.integers(0, len(blocks), size=(count, needed))
        sampled = blocks[choices].reshape(count, -1)[:, :horizon_days]
        equity = start_equity + np.cumsum(sampled, axis=1)
        peaks = np.maximum.accumulate(
            np.concatenate(
                [np.full((count, 1), start_equity), equity],
                axis=1,
            ),
            axis=1,
        )[:, 1:]
        drawdowns = peaks - equity
        hit = np.any(equity <= floor, axis=1)
        max_dd = drawdowns.max(axis=1)
        ruin += int(hit.sum())
        drawdown_30 += int((max_dd >= 0.30 * start_equity).sum())
        terminals[begin : begin + count] = equity[:, -1]
        max_drawdowns[begin : begin + count] = max_dd
    return {
        "quantity": None,
        "mean_day": float(np.mean(daily)),
        "p_hit_211_30d": ruin / repetitions,
        "p_drawdown_ge_30pct_30d": drawdown_30 / repetitions,
        "median_terminal_30d": float(np.median(terminals)),
        "terminal_p05": float(np.quantile(terminals, 0.05)),
        "terminal_p95": float(np.quantile(terminals, 0.95)),
        "median_max_drawdown": float(np.median(max_drawdowns)),
        "max_drawdown_p95": float(np.quantile(max_drawdowns, 0.95)),
    }


def leave_one(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for value in sorted(frame[column].dropna().unique()):
        subset = frame[~frame[column].eq(value)]
        result = metrics(subset, all_calendar())
        rows.append(
            {
                f"excluded_{column}": value,
                "n": result["n"],
                "total_pnl": result["total_pnl"],
                "edge_per_contract": result["edge_per_contract"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    panel = build_panel()
    calendar = all_calendar()

    split_rows = []
    for split in ["train", "valid", "test"]:
        split_calendar = [
            day.isoformat()
            for day in pd.date_range(
                SPLITS[split][0].normalize(),
                SPLITS[split][1].normalize() - pd.Timedelta(days=1),
                freq="D",
            ).date
        ]
        for label, config in [("rct", PRIMARY_CONFIG), ("control", CONTROL_CONFIG)]:
            frame = select(panel, config, minute=0, split=split, cap=1)
            result = metrics(frame, split_calendar)
            split_rows.append({"split": split, "policy": label, **result})
    split_table = pd.DataFrame(split_rows)

    rct = select(panel, PRIMARY_CONFIG, minute=0, cap=1)
    control = select(panel, CONTROL_CONFIG, minute=0, cap=1)
    rct_all = metrics(rct, calendar)
    control_all = metrics(control, calendar)
    rct_daily = daily_series(rct, calendar)
    control_daily = daily_series(control, calendar)
    increment_daily = rct_daily - control_daily

    absolute_bootstrap = bootstrap_mean(rct_daily.to_numpy())
    control_bootstrap = bootstrap_mean(control_daily.to_numpy())
    increment_bootstrap = bootstrap_mean(increment_daily.to_numpy())

    meta_rct = fixed_effect_meta(
        [
            split_table[
                split_table["split"].eq(split)
                & split_table["policy"].eq("rct")
            ].iloc[0].to_dict()
            for split in ["train", "valid", "test"]
        ]
    )
    meta_control = fixed_effect_meta(
        [
            split_table[
                split_table["split"].eq(split)
                & split_table["policy"].eq("control")
            ].iloc[0].to_dict()
            for split in ["train", "valid", "test"]
        ]
    )

    # All-minute daily matrices for conservative best-of-four correction.
    rct_daily_by_minute = pd.DataFrame(index=calendar)
    control_daily_by_minute = pd.DataFrame(index=calendar)
    for minute in MINUTES:
        rct_m = select(panel, PRIMARY_CONFIG, minute=minute, cap=1)
        control_m = select(panel, CONTROL_CONFIG, minute=minute, cap=1)
        rct_daily_by_minute[minute] = daily_series(rct_m, calendar)
        control_daily_by_minute[minute] = daily_series(control_m, calendar)
    selection_absolute = permutation_max_stat(rct_daily_by_minute)
    selection_incremental = permutation_max_stat(
        rct_daily_by_minute - control_daily_by_minute
    )

    # Marginal rank and cap audit. Rank 2/3 are exploratory.
    rank_rows = []
    cap_rows = []
    for split in ["train", "valid", "test"]:
        split_calendar = [
            day.isoformat()
            for day in pd.date_range(
                SPLITS[split][0].normalize(),
                SPLITS[split][1].normalize() - pd.Timedelta(days=1),
                freq="D",
            ).date
        ]
        for rank in [1, 2, 3]:
            frame = select(
                panel,
                PRIMARY_CONFIG,
                minute=0,
                split=split,
                exact_rank=rank,
            )
            rank_rows.append(
                {"split": split, "rank": rank, **metrics(frame, split_calendar)}
            )
        for cap in [1, 2, 3]:
            frame = select(
                panel, PRIMARY_CONFIG, minute=0, split=split, cap=cap
            )
            cap_rows.append(
                {"split": split, "cap": cap, **metrics(frame, split_calendar)}
            )
    rank_table = pd.DataFrame(rank_rows)
    cap_table = pd.DataFrame(cap_rows)

    # Risk under q10/q15/q20 for the frozen cap-1 RCT candidate.
    risk_rows = []
    for quantity in [10, 15, 20]:
        daily = daily_series(rct, calendar, quantity).to_numpy()
        risk = moving_block_bootstrap_days(daily)
        risk["quantity"] = quantity
        risk_rows.append(risk)
    risk_table = pd.DataFrame(risk_rows)

    coin_loo = leave_one(rct, "coin")
    week_loo = leave_one(rct, "week")
    hour_loo = leave_one(rct, "hour")

    # Decision logic:
    # - absolute candidate passes convergence only if full-period clustered CI
    #   is positive, all chronological splits are positive, and conservative
    #   best-of-four permutation is below 5%.
    # - confirmation increment is evaluated separately.
    chronological_rct_positive = bool(
        (
            split_table[
                split_table["policy"].eq("rct")
            ]["total_pnl"]
            > 0
        ).all()
    )
    chronological_control_positive = bool(
        (
            split_table[
                split_table["policy"].eq("control")
            ]["total_pnl"]
            > 0
        ).all()
    )
    rct_convergence_pass = bool(
        chronological_rct_positive
        and absolute_bootstrap["ci_lo"] > 0
        and selection_absolute["p_best_of_four"] < 0.05
        and (coin_loo["total_pnl"] > 0).all()
        and (week_loo["total_pnl"] > 0).all()
        and (hour_loo["total_pnl"] > 0).all()
    )
    incremental_pass = bool(
        increment_bootstrap["ci_lo"] > 0
        and selection_incremental["p_best_of_four"] < 0.05
    )

    rank2 = rank_table[rank_table["rank"].eq(2)]
    rank2_pass = bool(
        len(rank2) == 3
        and (rank2["total_pnl"] > 0).all()
        and rank2[rank2["split"].eq("test")]["n"].iloc[0] >= 25
    )

    summary = {
        "primary_config": {
            "delay": PRIMARY_CONFIG.delay,
            "bid_move_c": PRIMARY_CONFIG.bid_move_c,
            "ask_ceiling": PRIMARY_CONFIG.ask_ceiling,
            "spread_widen_limit_c": PRIMARY_CONFIG.spread_widen_limit_c,
            "volume_filter": PRIMARY_CONFIG.volume_filter,
            "minute": 0,
            "cap": 1,
        },
        "full_period": {
            "rct": rct_all,
            "delayed_control": control_all,
            "incremental_total_pnl": float(rct_all["total_pnl"] - control_all["total_pnl"]),
        },
        "split_metrics": split_table.to_dict("records"),
        "absolute_day_bootstrap": absolute_bootstrap,
        "control_day_bootstrap": control_bootstrap,
        "incremental_day_bootstrap": increment_bootstrap,
        "meta_rct": meta_rct,
        "meta_control": meta_control,
        "selection_corrected_absolute": selection_absolute,
        "selection_corrected_incremental": selection_incremental,
        "rank_metrics": rank_table.to_dict("records"),
        "cap_metrics": cap_table.to_dict("records"),
        "risk": risk_table.to_dict("records"),
        "pass_components": {
            "chronological_rct_positive": chronological_rct_positive,
            "chronological_delayed_control_positive": chronological_control_positive,
            "rct_full_period_convergence_pass": rct_convergence_pass,
            "confirmation_increment_pass": incremental_pass,
            "rank2_exploratory_pass": rank2_pass,
        },
    }

    split_table.to_csv(OUT / "rct_minute00_meta_splits.csv", index=False)
    rank_table.to_csv(OUT / "rct_minute00_meta_ranks.csv", index=False)
    cap_table.to_csv(OUT / "rct_minute00_meta_caps.csv", index=False)
    risk_table.to_csv(OUT / "rct_minute00_meta_risk.csv", index=False)
    rct_daily_by_minute.to_csv(
        OUT / "rct_minute00_meta_rct_daily_by_minute.csv"
    )
    control_daily_by_minute.to_csv(
        OUT / "rct_minute00_meta_control_daily_by_minute.csv"
    )
    coin_loo.to_csv(OUT / "rct_minute00_meta_leave_one_coin.csv", index=False)
    week_loo.to_csv(OUT / "rct_minute00_meta_leave_one_week.csv", index=False)
    hour_loo.to_csv(OUT / "rct_minute00_meta_leave_one_hour.csv", index=False)
    (OUT / "rct_minute00_meta_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    if rct_convergence_pass and incremental_pass:
        verdict = "CONVERGENT SIGNAL + CONFIRMATION INCREMENT PASS"
    elif rct_convergence_pass:
        verdict = "MINUTE-00 DELAYED EDGE PASS; RCT INCREMENT UNPROVEN"
    else:
        verdict = "CANDIDATE ONLY"

    lines = [
        "# Minute-:00 delayed-taker / RCT convergent audit",
        "",
        f"## Verdict: **{verdict}**",
        "",
        "The full 73-day population is used only after both components were fixed.",
        "A within-day best-of-four permutation corrects for the availability of",
        "four close-minute labels.",
        "",
        "## Full-period q15 economics",
        "",
        "| Policy | n | P&L | edge/contract | mean/day | max DD |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| RCT minute :00 | {rct_all['n']} | ${rct_all['total_pnl']:.2f} | "
            f"{rct_all['edge_per_contract']*100:+.2f}¢ | "
            f"${rct_all['mean_day']:.2f} | ${rct_all['max_drawdown']:.2f} |"
        ),
        (
            f"| Delayed control minute :00 | {control_all['n']} | "
            f"${control_all['total_pnl']:.2f} | "
            f"{control_all['edge_per_contract']*100:+.2f}¢ | "
            f"${control_all['mean_day']:.2f} | ${control_all['max_drawdown']:.2f} |"
        ),
        "",
        "## Full-period uncertainty",
        "",
        (
            f"- RCT mean/day 95% CI: [${absolute_bootstrap['ci_lo']:.2f}, "
            f"${absolute_bootstrap['ci_hi']:.2f}], "
            f"P(≤0)={absolute_bootstrap['p_nonpositive']:.4f}"
        ),
        (
            f"- Delayed control mean/day 95% CI: [${control_bootstrap['ci_lo']:.2f}, "
            f"${control_bootstrap['ci_hi']:.2f}], "
            f"P(≤0)={control_bootstrap['p_nonpositive']:.4f}"
        ),
        (
            f"- RCT minus control mean/day 95% CI: "
            f"[${increment_bootstrap['ci_lo']:.2f}, "
            f"${increment_bootstrap['ci_hi']:.2f}], "
            f"P(≤0)={increment_bootstrap['p_nonpositive']:.4f}"
        ),
        "",
        "## Four-minute selection correction",
        "",
        (
            f"- RCT absolute best-of-four p: "
            f"{selection_absolute['p_best_of_four']:.4f}"
        ),
        (
            f"- RCT incremental best-of-four p: "
            f"{selection_incremental['p_best_of_four']:.4f}"
        ),
        "",
        "## Rank capacity",
        "",
        rank_table.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Portfolio caps",
        "",
        cap_table.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 30-day bankroll stress from $300",
        "",
        risk_table.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Decision",
        "",
        (
            "- The minute-:00 delayed/RCT population clears the convergent "
            f"gate: **{rct_convergence_pass}**."
        ),
        (
            "- The 2¢ quote-strengthening condition adds independently proven "
            f"value over the same delayed population: **{incremental_pass}**."
        ),
        (
            "- Exploratory rank 2 is positive in all three periods: "
            f"**{rank2_pass}**."
        ),
        "",
        "A full-period pass is not a prospective deployment pass. The rule must",
        "still be logged forward without retuning, and actual IOC depth/latency",
        "must be measured. No result bypasses the account KILL state.",
    ]
    (OUT / "rct_minute00_meta_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
