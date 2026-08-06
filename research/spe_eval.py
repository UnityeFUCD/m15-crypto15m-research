"""Economic policy selection and sealed-test hostile evaluation for SPE."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from spe_data import COINS, HORIZONS, OUT, QTY, SEED, SPLITS, fee_total
from spe_models import MODEL_NAMES

RNG = np.random.default_rng(SEED)
EDGE_THRESHOLDS = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08]


def executable_values(predictions: pd.DataFrame) -> pd.DataFrame:
    x = predictions.copy()
    yes_fee = np.array([fee_total(QTY, p) / QTY for p in x.yes_ask])
    no_fee = np.array([fee_total(QTY, p) / QTY for p in x.no_ask])
    x["yes_all_in"] = x.yes_ask + yes_fee
    x["no_all_in"] = x.no_ask + no_fee
    x["edge_yes"] = x.p_yes - x.yes_all_in
    x["edge_no"] = 1.0 - x.p_yes - x.no_all_in
    x["trade_side"] = np.where(x.edge_yes >= x.edge_no, "yes", "no")
    x["predicted_edge"] = np.maximum(x.edge_yes, x.edge_no)
    x["trade_price"] = np.where(x.trade_side.eq("yes"), x.yes_ask, x.no_ask)
    x["fee_per_contract"] = np.where(x.trade_side.eq("yes"), yes_fee, no_fee)
    x["trade_won"] = np.where(x.trade_side.eq("yes"), x.y, 1 - x.y)
    x["pnl"] = QTY * (x.trade_won - x.trade_price - x.fee_per_contract)
    return x


def one_per_close(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    eligible = frame[frame.predicted_edge >= threshold].copy()
    if eligible.empty:
        return eligible
    return (
        eligible.sort_values(
            ["close_ts", "predicted_edge", "trade_price", "coin"],
            ascending=[True, False, True, True],
        )
        .groupby("close_ts", as_index=False).head(1)
        .sort_values("close_ts")
    )


@dataclass
class PolicyResult:
    model: str
    horizon: int
    threshold: float
    split: str
    n: int
    days: int
    total_pnl: float
    mean_day: float
    sd_day: float
    t_stat: float
    edge_per_contract: float
    win_rate: float
    average_price: float
    max_drawdown: float
    worst_window: float
    positive_day_fraction: float


def result(selected: pd.DataFrame, model: str, horizon: int,
           threshold: float, split: str) -> PolicyResult:
    start, stop = SPLITS[split]
    days = [
        day.isoformat() for day in pd.date_range(
            start.normalize(), stop.normalize() - pd.Timedelta(days=1), freq="D"
        ).date
    ]
    if selected.empty:
        return PolicyResult(
            model, horizon, threshold, split, 0, len(days), 0, 0, 0, 0,
            0, np.nan, np.nan, 0, 0, 0
        )
    daily = selected.groupby("day").pnl.sum().reindex(days, fill_value=0.0)
    window = selected.groupby("close_ts").pnl.sum().sort_index()
    equity = window.cumsum()
    drawdown = equity.cummax() - equity
    sd, mean = float(daily.std(ddof=1)), float(daily.mean())
    return PolicyResult(
        model, horizon, threshold, split, int(len(selected)), len(days),
        float(selected.pnl.sum()), mean, sd,
        mean / (sd / math.sqrt(len(days))) if sd > 0 else 0.0,
        float(selected.pnl.sum() / (len(selected) * QTY)),
        float(selected.trade_won.mean()), float(selected.trade_price.mean()),
        float(drawdown.max()), float(window.min()), float((daily > 0).mean()),
    )


def select_validation_policy(predictions: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    x = executable_values(predictions)
    valid = x[x.split.eq("valid") & x.model.isin(MODEL_NAMES)]
    rows = []
    for (model, horizon), group in valid.groupby(["model", "horizon"]):
        for threshold in EDGE_THRESHOLDS:
            rows.append(asdict(result(
                one_per_close(group, threshold), str(model), int(horizon),
                float(threshold), "valid",
            )))
    grid = pd.DataFrame(rows)
    viable = grid[(grid.n >= 50) & (grid.total_pnl > 0)].copy()
    if viable.empty:
        viable = grid[grid.n >= 25].copy()
    if viable.empty:
        viable = grid.copy()
    winner = (
        viable.sort_values(["t_stat", "n", "total_pnl"], ascending=False)
        .iloc[0].to_dict()
    )
    grid.to_csv(OUT / "spe_validation_policy_grid.csv", index=False)
    x.to_parquet(OUT / "spe_executable_values.parquet", index=False)
    return winner, x


def day_bootstrap(selected: pd.DataFrame, split: str, reps: int = 12000) -> dict:
    start, stop = SPLITS[split]
    days = [
        day.isoformat() for day in pd.date_range(
            start.normalize(), stop.normalize() - pd.Timedelta(days=1), freq="D"
        ).date
    ]
    daily = selected.groupby("day").pnl.sum().reindex(days, fill_value=0.0).to_numpy()
    draw = RNG.integers(0, len(daily), size=(reps, len(daily)))
    means = daily[draw].mean(axis=1)
    return {
        "observed_mean_day": float(daily.mean()),
        "ci_lo": float(np.quantile(means, 0.025)),
        "ci_hi": float(np.quantile(means, 0.975)),
        "p_nonpositive": float(np.mean(means <= 0)),
    }


def block_bootstrap(selected: pd.DataFrame, reps: int = 12000) -> list[dict]:
    if selected.empty:
        return []
    origin = int(selected.close_ts.min())
    output = []
    for minutes in [15, 30, 60, 90]:
        seconds = minutes * 60
        values = (
            selected.assign(block=((selected.close_ts - origin) // seconds).astype(int))
            .groupby("block").pnl.sum().to_numpy()
        )
        if len(values) < 2:
            continue
        draw = RNG.integers(0, len(values), size=(reps, len(values)))
        totals = values[draw].sum(axis=1)
        output.append({
            "block_minutes": minutes, "n_blocks": int(len(values)),
            "observed_total": float(values.sum()),
            "ci_lo": float(np.quantile(totals, 0.025)),
            "ci_hi": float(np.quantile(totals, 0.975)),
            "p_nonpositive": float(np.mean(totals <= 0)),
        })
    return output


def leave_one_out(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coin_rows, week_rows, hour_rows = [], [], []
    for coin in COINS:
        subset = selected[~selected.coin.eq(coin)]
        coin_rows.append({
            "excluded_coin": coin,
            "n": int(len(subset)),
            "total_pnl": float(subset.pnl.sum()),
            "edge_per_contract": float(subset.pnl.mean() / QTY) if len(subset) else np.nan,
        })
    for week in sorted(selected.week.unique()):
        subset = selected[~selected.week.eq(week)]
        week_rows.append({
            "excluded_week": week,
            "n": int(len(subset)),
            "total_pnl": float(subset.pnl.sum()),
            "edge_per_contract": float(subset.pnl.mean() / QTY) if len(subset) else np.nan,
        })
    for hour in sorted(selected.hour.unique()):
        subset = selected[~selected.hour.eq(hour)]
        hour_rows.append({
            "excluded_hour": int(hour),
            "n": int(len(subset)),
            "total_pnl": float(subset.pnl.sum()),
            "edge_per_contract": float(subset.pnl.mean() / QTY) if len(subset) else np.nan,
        })
    return pd.DataFrame(coin_rows), pd.DataFrame(week_rows), pd.DataFrame(hour_rows)


def deciles(rows: pd.DataFrame) -> pd.DataFrame:
    if len(rows) < 20:
        return pd.DataFrame()
    x = rows.copy()
    x["decile"] = pd.qcut(x.residual_vs_mid.rank(method="first"), 10, labels=False)
    return x.groupby("decile").agg(
        n=("y", "size"), mean_mid=("mid_yes", "mean"),
        mean_model=("p_yes", "mean"), actual_yes=("y", "mean"),
        residual=("residual_vs_mid", "mean"),
    ).reset_index()


def make_report(panel: pd.DataFrame, probability: pd.DataFrame, chosen_c: dict,
                winner: dict[str, Any], executable: pd.DataFrame) -> dict:
    model, horizon, threshold = (
        str(winner["model"]), int(winner["horizon"]), float(winner["threshold"])
    )
    test_rows = executable[
        executable.split.eq("test")
        & executable.model.eq(model)
        & executable.horizon.eq(horizon)
    ].copy()
    selected = one_per_close(test_rows, threshold)
    test_result = result(selected, model, horizon, threshold, "test")
    day = day_bootstrap(selected, "test")
    blocks = block_bootstrap(selected)
    coins, weeks, hours = leave_one_out(selected)
    residual_bins = deciles(test_rows)

    ptab = probability[
        probability.split.eq("test") & probability.horizon.eq(horizon)
    ]
    chosen = ptab[ptab.model.eq(model)]
    market = ptab[ptab.model.eq("market")]
    raw = ptab[ptab.model.eq("raw_mid")]
    increment = {}
    if len(chosen) and len(raw):
        increment["log_loss_vs_raw_mid"] = float(chosen.iloc[0].log_loss - raw.iloc[0].log_loss)
        increment["brier_vs_raw_mid"] = float(chosen.iloc[0].brier - raw.iloc[0].brier)
    if len(chosen) and len(market):
        increment["log_loss_vs_market_calibrator"] = float(
            chosen.iloc[0].log_loss - market.iloc[0].log_loss
        )
        increment["brier_vs_market_calibrator"] = float(
            chosen.iloc[0].brier - market.iloc[0].brier
        )

    gates = {
        "physical_information_beyond_market": (
            model in {"blend", "full"}
            and increment.get("log_loss_vs_market_calibrator", 1) < 0
            and increment.get("brier_vs_market_calibrator", 1) < 0
        ),
        "positive_test_economics_and_n_ge_50": (
            test_result.n >= 50 and test_result.total_pnl > 0
        ),
        "positive_day_ci": day["ci_lo"] > 0,
        "positive_all_block_cis": bool(blocks and min(b["ci_lo"] for b in blocks) > 0),
        "leave_one_coin_all_positive": bool(len(coins) and (coins.total_pnl > 0).all()),
        "leave_one_week_all_positive": bool(len(weeks) and (weeks.total_pnl > 0).all()),
        "leave_one_hour_all_positive": bool(len(hours) and (hours.total_pnl > 0).all()),
    }
    hard_pass = all(gates.values())

    coins.to_csv(OUT / "spe_test_leave_one_coin.csv", index=False)
    weeks.to_csv(OUT / "spe_test_leave_one_week.csv", index=False)
    hours.to_csv(OUT / "spe_test_leave_one_hour.csv", index=False)
    residual_bins.to_csv(OUT / "spe_test_residual_deciles.csv", index=False)
    selected.to_parquet(OUT / "spe_test_selected_trades.parquet", index=False)

    summary = {
        "data": {
            "panel_rows": int(len(panel)),
            "unique_markets": int(panel.ticker.nunique()),
            "calendar_days": int(panel.day.nunique()),
            "horizons": HORIZONS,
        },
        "validation_winner": winner,
        "chosen_regularization": chosen_c,
        "test_probability_metrics": ptab.to_dict("records"),
        "probability_increment": increment,
        "test_policy": asdict(test_result),
        "day_bootstrap": day,
        "block_bootstrap": blocks,
        "pass_components": gates,
        "hard_pass": hard_pass,
    }

    lines = [
        "# Settlement Probability Engine — clean-room audit", "",
        f"## Verdict: **{'PASS' if hard_pass else 'FAIL / CANDIDATE ONLY'}**", "",
        "All model and policy choices were made on TRAIN/VALID. TEST was evaluated once.", "",
        "## Validation-selected policy", "",
        f"- Model: `{model}`",
        f"- Horizon: {horizon} minutes remaining",
        f"- Minimum predicted all-in edge: {threshold*100:.1f}¢",
        f"- Validation trades: {int(winner['n'])}",
        f"- Validation P&L: ${float(winner['total_pnl']):.2f}", "",
        "## Sealed test result", "",
        f"- Trades: {test_result.n}",
        f"- Total P&L at q{QTY}: ${test_result.total_pnl:.2f}",
        f"- Mean/day: ${test_result.mean_day:.2f}",
        f"- Daily SD: ${test_result.sd_day:.2f}",
        f"- Edge/contract: {test_result.edge_per_contract*100:+.2f}¢",
        f"- Win rate: {test_result.win_rate:.4f}",
        f"- Average entry: {test_result.average_price*100:.2f}¢",
        f"- Maximum drawdown: ${test_result.max_drawdown:.2f}",
        f"- Worst close window: ${test_result.worst_window:.2f}", "",
        "## Probability increment on sealed test", "",
        f"- Δ log loss vs raw midpoint: {increment.get('log_loss_vs_raw_mid', np.nan):+.6f}",
        f"- Δ Brier vs raw midpoint: {increment.get('brier_vs_raw_mid', np.nan):+.6f}",
        f"- Δ log loss vs market-only calibrator: "
        f"{increment.get('log_loss_vs_market_calibrator', np.nan):+.6f}",
        f"- Δ Brier vs market-only calibrator: "
        f"{increment.get('brier_vs_market_calibrator', np.nan):+.6f}", "",
        "Negative differences are improvements.", "",
        "## Uncertainty", "",
        f"- Day-bootstrap mean/day 95% CI: [${day['ci_lo']:.2f}, ${day['ci_hi']:.2f}]",
        f"- P(mean day ≤ 0): {day['p_nonpositive']:.4f}", "",
        "| Block | CI on total P&L | P(nonpositive) |",
        "|---:|---:|---:|",
    ]
    lines += [
        f"| {b['block_minutes']} min | [${b['ci_lo']:.2f}, ${b['ci_hi']:.2f}] | "
        f"{b['p_nonpositive']:.4f} |" for b in blocks
    ]
    lines += ["", "## Hard gates", ""]
    lines += [f"- {name}: **{'PASS' if value else 'FAIL'}**" for name, value in gates.items()]
    lines += [
        "", "## Interpretation", "",
        "A profitable threshold backtest is not enough. Causal settlement state must",
        "improve sealed-test probability quality beyond a market-only calibrator,",
        "and the policy must survive clustered uncertainty and leave-one-group tests.",
        "", "Nothing here authorizes live orders while the account KILL state is active.",
    ]
    (OUT / "spe_report.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "spe_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary
