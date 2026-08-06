
"""Independent interaction audit: frozen RCT primary rule at minute :00.

Minute :00 was identified independently in the cross-finding audit, while the
Runaway Confirmation Taker primary rule was fixed before its grid was viewed.
This script tests their intersection without re-optimizing either component.

The load-bearing comparison is not merely whether the intersection makes
money. It must also add value over a delayed-taker policy with the same ask,
spread, volume, and close-minute constraints but no quote-strengthening gate.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from research.runaway_confirmation import (
    OUT,
    PRIMARY,
    QTY,
    Config,
    block_bootstrap,
    build_panel,
    choose_one_per_close,
    day_bootstrap,
    leave_one_out,
    matched_confirmation_lift,
    metrics,
    qualifying,
    split_calendar,
    split_mask,
)

RNG = np.random.default_rng(2026080622)
PRIMARY_MINUTE = 0


def control_config(config: Config) -> Config:
    return Config(
        delay=config.delay,
        bid_move_c=-999.0,
        ask_ceiling=config.ask_ceiling,
        spread_widen_limit_c=config.spread_widen_limit_c,
        volume_filter=config.volume_filter,
    )


def select_policy(
    panel: pd.DataFrame, config: Config, split: str, minute: int
) -> pd.DataFrame:
    data = panel[
        split_mask(panel, split)
        & panel["close_minute"].eq(minute)
    ]
    return choose_one_per_close(qualifying(data, config))


def daily_series(selected: pd.DataFrame, split: str) -> pd.Series:
    return (
        selected.groupby("day")["pnl"]
        .sum()
        .reindex(split_calendar(split), fill_value=0.0)
    )


def daily_difference_bootstrap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    split: str,
    repetitions: int = 30_000,
) -> dict[str, float]:
    difference = (
        daily_series(left, split) - daily_series(right, split)
    ).to_numpy(dtype=float)
    draws = RNG.integers(
        0, len(difference), size=(repetitions, len(difference))
    )
    means = difference[draws].mean(axis=1)
    return {
        "observed_mean_day_difference": float(difference.mean()),
        "ci_lo": float(np.quantile(means, 0.025)),
        "ci_hi": float(np.quantile(means, 0.975)),
        "p_nonpositive": float(np.mean(means <= 0)),
    }


def block_difference_bootstrap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    repetitions: int = 30_000,
) -> list[dict[str, float]]:
    if left.empty and right.empty:
        return []
    left_window = left.groupby("close_ts")["pnl"].sum()
    right_window = right.groupby("close_ts")["pnl"].sum()
    all_windows = sorted(set(left_window.index) | set(right_window.index))
    frame = pd.DataFrame(
        {
            "close_ts": all_windows,
            "difference": [
                float(left_window.get(window, 0.0))
                - float(right_window.get(window, 0.0))
                for window in all_windows
            ],
        }
    )
    origin = int(frame["close_ts"].min())
    output = []
    for minutes in [15, 30, 60, 90]:
        values = (
            frame.assign(
                block=(
                    (frame["close_ts"] - origin) // (minutes * 60)
                ).astype(int)
            )
            .groupby("block")["difference"]
            .sum()
            .to_numpy(dtype=float)
        )
        if len(values) < 2:
            continue
        draws = RNG.integers(
            0, len(values), size=(repetitions, len(values))
        )
        totals = values[draws].sum(axis=1)
        output.append(
            {
                "block_minutes": minutes,
                "n_blocks": int(len(values)),
                "observed_total_difference": float(values.sum()),
                "ci_lo": float(np.quantile(totals, 0.025)),
                "ci_hi": float(np.quantile(totals, 0.975)),
                "p_nonpositive": float(np.mean(totals <= 0)),
            }
        )
    return output


def per_minute_table(
    panel: pd.DataFrame, config: Config
) -> pd.DataFrame:
    rows = []
    for minute in [0, 15, 30, 45]:
        for split in ["train", "valid", "test"]:
            selected = select_policy(panel, config, split, minute)
            result = asdict(metrics(selected, config, split))
            result["close_minute"] = minute
            rows.append(result)
    return pd.DataFrame(rows)


def contribution_table(
    selected: pd.DataFrame, column: str
) -> pd.DataFrame:
    rows = []
    for value, group in selected.groupby(column):
        rows.append(
            {
                column: value,
                "n": int(len(group)),
                "total_pnl": float(group["pnl"].sum()),
                "edge_per_contract": float(group["pnl"].mean() / QTY),
                "win_rate": float(group["won"].mean()),
                "mean_ask": float(group["delayed_ask"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    panel = build_panel()
    primary = Config(**PRIMARY)
    delayed_control = control_config(primary)

    minute_table = per_minute_table(panel, primary)
    minute_table.to_csv(
        OUT / "rct_minute00_all_minutes.csv", index=False
    )

    summary = {
        "primary_config": asdict(primary),
        "minute": PRIMARY_MINUTE,
        "splits": {},
    }

    selected_by_split = {}
    control_by_split = {}
    for split in ["train", "valid", "test"]:
        selected = select_policy(
            panel, primary, split, PRIMARY_MINUTE
        )
        control = select_policy(
            panel, delayed_control, split, PRIMARY_MINUTE
        )
        selected_by_split[split] = selected
        control_by_split[split] = control

        selected_result = asdict(metrics(selected, primary, split))
        control_result = asdict(
            metrics(control, delayed_control, split)
        )
        difference = daily_difference_bootstrap(
            selected, control, split
        )
        summary["splits"][split] = {
            "rct": selected_result,
            "delayed_control": control_result,
            "daily_increment": difference,
            "incremental_total_pnl": float(
                selected["pnl"].sum() - control["pnl"].sum()
            ),
        }

    test_selected = selected_by_split["test"]
    test_control = control_by_split["test"]

    day_uncertainty = day_bootstrap(test_selected, "test")
    block_uncertainty = block_bootstrap(test_selected)
    incremental_blocks = block_difference_bootstrap(
        test_selected, test_control
    )
    coins, weeks, minutes, hours = leave_one_out(test_selected)
    matched = {
        split: matched_confirmation_lift(
            panel[
                panel["close_minute"].eq(PRIMARY_MINUTE)
            ],
            primary,
            split,
        )
        for split in ["train", "valid", "test"]
    }

    contribution_table(test_selected, "coin").to_csv(
        OUT / "rct_minute00_test_by_coin.csv", index=False
    )
    contribution_table(test_selected, "week").to_csv(
        OUT / "rct_minute00_test_by_week.csv", index=False
    )
    contribution_table(test_selected, "hour").to_csv(
        OUT / "rct_minute00_test_by_hour.csv", index=False
    )
    coins.to_csv(
        OUT / "rct_minute00_leave_one_coin.csv", index=False
    )
    weeks.to_csv(
        OUT / "rct_minute00_leave_one_week.csv", index=False
    )
    hours.to_csv(
        OUT / "rct_minute00_leave_one_hour.csv", index=False
    )
    test_selected.to_parquet(
        OUT / "rct_minute00_test_trades.parquet", index=False
    )

    positive_splits = all(
        summary["splits"][split]["rct"]["total_pnl"] > 0
        for split in ["train", "valid", "test"]
    )
    positive_increment_splits = all(
        summary["splits"][split]["incremental_total_pnl"] > 0
        for split in ["train", "valid", "test"]
    )
    day_pass = day_uncertainty["ci_lo"] > 0
    blocks_pass = bool(
        block_uncertainty
        and min(row["ci_lo"] for row in block_uncertainty) > 0
    )
    incremental_day_pass = (
        summary["splits"]["test"]["daily_increment"]["ci_lo"] > 0
    )
    incremental_block_pass = bool(
        incremental_blocks
        and min(row["ci_lo"] for row in incremental_blocks) > 0
    )
    loo_pass = all(
        len(table) > 0 and (table["total_pnl"] > 0).all()
        for table in [coins, weeks, hours]
    )
    matched_pass = all(
        np.isfinite(matched[split]["edge_lift"])
        and matched[split]["edge_lift"] > 0
        for split in ["train", "valid", "test"]
    )
    n_pass = len(test_selected) >= 40

    pass_components = {
        "rct_positive_train_valid_test": positive_splits,
        "incremental_over_delayed_control_positive_all_splits": (
            positive_increment_splits
        ),
        "test_day_ci_positive": day_pass,
        "test_all_block_cis_positive": blocks_pass,
        "test_incremental_day_ci_positive": incremental_day_pass,
        "test_incremental_all_block_cis_positive": incremental_block_pass,
        "leave_one_coin_week_hour_positive": loo_pass,
        "matched_edge_lift_positive_all_splits": matched_pass,
        "test_n_ge_40": n_pass,
    }
    hard_pass = all(pass_components.values())

    summary.update(
        {
            "test_day_uncertainty": day_uncertainty,
            "test_block_uncertainty": block_uncertainty,
            "test_incremental_block_uncertainty": incremental_blocks,
            "matched_confirmation_lift": matched,
            "pass_components": pass_components,
            "hard_pass": hard_pass,
        }
    )

    lines = [
        "# Runaway Confirmation × minute :00 interaction audit",
        "",
        f"## Verdict: **{'PASS' if hard_pass else 'FAIL / CANDIDATE ONLY'}**",
        "",
        "Both components were fixed independently before this interaction test:",
        "",
        "- minute :00 came from the cross-finding pricing audit;",
        "- the RCT primary rule was frozen before the RCT configuration grid.",
        "",
        f"Frozen rule: `{primary.name}` restricted to close minute `:00`.",
        "",
        "| Split | RCT n | RCT P&L | edge/ct | delayed-control P&L | incremental |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ["train", "valid", "test"]:
        item = summary["splits"][split]
        rct = item["rct"]
        control = item["delayed_control"]
        lines.append(
            f"| {split} | {rct['n']} | ${rct['total_pnl']:.2f} | "
            f"{rct['edge_per_contract']*100:+.2f}¢ | "
            f"${control['total_pnl']:.2f} | "
            f"${item['incremental_total_pnl']:.2f} |"
        )

    lines.extend(
        [
            "",
            f"Test day-bootstrap RCT mean/day: "
            f"[${day_uncertainty['ci_lo']:.2f}, "
            f"${day_uncertainty['ci_hi']:.2f}], "
            f"P(≤0)={day_uncertainty['p_nonpositive']:.4f}",
            "",
            f"Test incremental mean/day over delayed control: "
            f"[${summary['splits']['test']['daily_increment']['ci_lo']:.2f}, "
            f"${summary['splits']['test']['daily_increment']['ci_hi']:.2f}], "
            f"P(≤0)="
            f"{summary['splits']['test']['daily_increment']['p_nonpositive']:.4f}",
            "",
            "Matched edge lift after delayed-ask and state controls:",
        ]
    )
    for split in ["train", "valid", "test"]:
        item = matched[split]
        lines.append(
            f"- {split}: {item['strata']} strata, "
            f"win lift {item['win_rate_lift']*100:+.2f}pp, "
            f"edge lift {item['edge_lift']*100:+.2f}¢"
        )
    lines.extend(["", "## Hard gates", ""])
    for key, passed in pass_components.items():
        lines.append(f"- {key}: **{'PASS' if passed else 'FAIL'}**")
    lines.extend(
        [
            "",
            "A positive absolute result is not enough. The interaction must beat",
            "the same delayed-taker population without the quote-strengthening",
            "condition; otherwise minute :00 or the new ask price explains the",
            "result rather than runaway confirmation.",
            "",
            "No result authorizes live orders while the account KILL state is active.",
        ]
    )

    (OUT / "rct_minute00_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    (OUT / "rct_minute00_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
