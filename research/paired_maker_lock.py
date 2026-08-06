"""Paired Maker Lock (PML): two-sided binary spread-capture audit.

For one Kalshi binary market, buy one YES at the YES bid and one NO at the NO
bid. If both maker orders fill, the pair pays $1 and locks the displayed spread
with no directional risk. The danger is legging: exactly one side fills.

This audit reconstructs that state from the trusted full one-minute quote paths.
A side is treated as filled only when a later complete minute closes through
the resting price and that candle has positive reported volume. This is a
historical quote-path model, not proof of queue execution.

The policy is chosen on chronological validation and touched once on test.
Nothing places orders or reads credentials.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "research" / "results"
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(2026080621)
QTY = 15
EPS = 1e-9
MIN_SPREADS_C = [1, 2, 3, 4, 5]
CANCEL_IF_NONE_AFTER_MIN = [1, 2, 3, 5, 7, 10]
CAPS = [1, 2, 6]
SELECTIONS = ["widest", "cheapest_favorite", "earliest"]
SPLITS = {
    "train": (pd.Timestamp("2026-05-25", tz="UTC"),
              pd.Timestamp("2026-06-30", tz="UTC")),
    "valid": (pd.Timestamp("2026-06-30", tz="UTC"),
              pd.Timestamp("2026-07-18", tz="UTC")),
    "test": (pd.Timestamp("2026-07-18", tz="UTC"),
             pd.Timestamp("2026-08-07", tz="UTC")),
}
COIN_ORDER = {"BTC": 0, "ETH": 1, "SOL": 2, "XRP": 3, "DOGE": 4, "HYPE": 5}


def split_mask(frame: pd.DataFrame, name: str) -> pd.Series:
    start, stop = SPLITS[name]
    return (frame["close_dt"] >= start) & (frame["close_dt"] < stop)


def parse_path(value: Any) -> list[dict]:
    try:
        raw = json.loads(value) if isinstance(value, str) else value
    except Exception:
        return []
    out = []
    for point in raw or []:
        try:
            ml = float(point["ml"])
            yb = float(point.get("yb", point.get("bc")))
            ya = float(point.get("ya", point.get("ac")))
            volume = float(point.get("v", 0.0) or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= ml <= 15 and 0 < yb < ya < 1:
            out.append({"ml": ml, "yb": yb, "ya": ya, "v": volume})
    return sorted(out, key=lambda row: -row["ml"])


def first_entry(points: list[dict]) -> dict | None:
    return next((point for point in points if 8 <= point["ml"] <= 14), None)


def first_cross(
    later: list[dict],
    *,
    yes_bid: float,
    yes_ask: float,
    side: str,
    deadline_ml: float,
) -> dict | None:
    """First complete-minute quote that consumes the resting side."""
    for point in later:
        if point["ml"] < deadline_ml - EPS:
            continue
        if point["v"] <= 0:
            continue
        if side == "yes" and point["ya"] <= yes_bid + EPS:
            return point
        if side == "no" and point["yb"] >= yes_ask - EPS:
            return point
    return None


def build_markets() -> pd.DataFrame:
    paths = pd.read_parquet(DATA / "paths_full.parquet")
    rows = []
    for market in paths.itertuples(index=False):
        points = parse_path(market.path)
        entry = first_entry(points)
        if entry is None:
            continue
        close_ts = int(market.close_ts)
        close_dt = pd.to_datetime(close_ts, unit="s", utc=True)
        yes_bid = float(entry["yb"])
        yes_ask = float(entry["ya"])
        spread = yes_ask - yes_bid
        if spread <= 0:
            continue
        no_bid = 1.0 - yes_ask
        true_yes = int(market.won) if str(market.side) == "yes" else 1 - int(market.won)
        later = [point for point in points if point["ml"] < float(entry["ml"]) - EPS]

        for cancel_after in CANCEL_IF_NONE_AFTER_MIN:
            deadline_ml = max(float(entry["ml"]) - cancel_after, 0.0)
            yes_fill = first_cross(
                later, yes_bid=yes_bid, yes_ask=yes_ask, side="yes",
                deadline_ml=deadline_ml)
            no_fill = first_cross(
                later, yes_bid=yes_bid, yes_ask=yes_ask, side="no",
                deadline_ml=deadline_ml)

            if yes_fill is None and no_fill is None:
                yes_final = None
                no_final = None
            else:
                # After the first leg fills, leave the complementary maker
                # order alive because a later second fill can only reduce risk.
                yes_final = yes_fill
                no_final = no_fill
                if yes_fill is not None and no_fill is None:
                    no_final = first_cross(
                        later, yes_bid=yes_bid, yes_ask=yes_ask, side="no",
                        deadline_ml=0.0)
                if no_fill is not None and yes_fill is None:
                    yes_final = first_cross(
                        later, yes_bid=yes_bid, yes_ask=yes_ask, side="yes",
                        deadline_ml=0.0)

            filled_yes = yes_final is not None
            filled_no = no_final is not None
            both = filled_yes and filled_no
            only_yes = filled_yes and not filled_no
            only_no = filled_no and not filled_yes
            none = not filled_yes and not filled_no
            pnl_per_pair = (
                (1.0 - yes_bid - no_bid) if both
                else (true_yes - yes_bid) if only_yes
                else ((1 - true_yes) - no_bid) if only_no
                else 0.0
            )
            rows.append({
                "ticker": market.ticker,
                "coin": market.coin,
                "coin_order": COIN_ORDER.get(str(market.coin), 99),
                "close_ts": close_ts,
                "close_dt": close_dt,
                "day": close_dt.date().isoformat(),
                "week": f"{close_dt.isocalendar().year}-{close_dt.isocalendar().week:02d}",
                "entry_ml": float(entry["ml"]),
                "yes_bid": yes_bid,
                "no_bid": no_bid,
                "favorite_price": max(yes_bid, no_bid),
                "spread_c": spread * 100,
                "cancel_after_min": cancel_after,
                "true_yes": true_yes,
                "yes_filled": filled_yes,
                "no_filled": filled_no,
                "both_filled": both,
                "only_yes": only_yes,
                "only_no": only_no,
                "none_filled": none,
                "first_fill_elapsed_min": (
                    min([float(entry["ml"]) - point["ml"]
                         for point in [yes_final, no_final] if point is not None])
                    if (filled_yes or filled_no) else np.nan),
                "pnl_per_pair": pnl_per_pair,
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no PML rows")
    frame.to_parquet(OUT / "pml_market_states.parquet", index=False)
    return frame


def select_windows(
    data: pd.DataFrame,
    *,
    min_spread_c: float,
    cancel_after_min: int,
    cap: int,
    selection: str,
) -> pd.DataFrame:
    eligible = data[
        data["cancel_after_min"].eq(cancel_after_min)
        & (data["spread_c"] >= min_spread_c)
    ].copy()
    if selection == "widest":
        columns = ["close_ts", "spread_c", "favorite_price", "coin_order"]
        ascending = [True, False, True, True]
    elif selection == "cheapest_favorite":
        columns = ["close_ts", "favorite_price", "spread_c", "coin_order"]
        ascending = [True, True, False, True]
    else:
        columns = ["close_ts", "coin_order", "spread_c"]
        ascending = [True, True, False]
    return (eligible.sort_values(columns, ascending=ascending)
            .groupby("close_ts", as_index=False).head(cap)
            .sort_values(["close_ts", "coin_order"]))


@dataclass
class Metrics:
    split: str
    min_spread_c: float
    cancel_after_min: int
    cap: int
    selection: str
    n_markets: int
    active_markets: int
    both_rate: float
    single_rate: float
    none_rate: float
    total_pnl_q15: float
    mean_day_q15: float
    sd_day_q15: float
    t_stat: float
    edge_per_posted_pair_c: float
    edge_per_active_pair_c: float
    max_drawdown_q15: float
    worst_window_q15: float
    positive_day_fraction: float


def metric(
    selected: pd.DataFrame,
    *,
    split: str,
    min_spread_c: float,
    cancel_after_min: int,
    cap: int,
    selection: str,
) -> Metrics:
    days = pd.date_range(
        SPLITS[split][0].normalize(),
        SPLITS[split][1].normalize() - pd.Timedelta(days=1), freq="D").date
    if selected.empty:
        return Metrics(split, min_spread_c, cancel_after_min, cap, selection,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    frame = selected.copy()
    frame["pnl_q15"] = QTY * frame["pnl_per_pair"]
    daily = frame.groupby("day")["pnl_q15"].sum().reindex(
        [day.isoformat() for day in days], fill_value=0.0)
    window = frame.groupby("close_ts")["pnl_q15"].sum().sort_index()
    equity = window.cumsum()
    drawdown = equity.cummax() - equity
    active = frame[~frame["none_filled"]]
    sd = float(daily.std(ddof=1))
    mean = float(daily.mean())
    return Metrics(
        split=split, min_spread_c=min_spread_c,
        cancel_after_min=cancel_after_min, cap=cap, selection=selection,
        n_markets=int(len(frame)), active_markets=int(len(active)),
        both_rate=float(frame["both_filled"].mean()),
        single_rate=float((frame["only_yes"] | frame["only_no"]).mean()),
        none_rate=float(frame["none_filled"].mean()),
        total_pnl_q15=float(frame["pnl_q15"].sum()), mean_day_q15=mean,
        sd_day_q15=sd,
        t_stat=mean / (sd / math.sqrt(len(daily))) if sd > 0 else 0.0,
        edge_per_posted_pair_c=float(frame["pnl_per_pair"].mean() * 100),
        edge_per_active_pair_c=float(active["pnl_per_pair"].mean() * 100)
        if len(active) else 0.0,
        max_drawdown_q15=float(drawdown.max()) if len(drawdown) else 0.0,
        worst_window_q15=float(window.min()) if len(window) else 0.0,
        positive_day_fraction=float((daily > 0).mean()))


def validation_grid(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    rows = []
    valid = frame[split_mask(frame, "valid")]
    for spread in MIN_SPREADS_C:
        for cutoff in CANCEL_IF_NONE_AFTER_MIN:
            for cap in CAPS:
                for selection in SELECTIONS:
                    chosen = select_windows(
                        valid, min_spread_c=spread,
                        cancel_after_min=cutoff, cap=cap, selection=selection)
                    rows.append(asdict(metric(
                        chosen, split="valid", min_spread_c=spread,
                        cancel_after_min=cutoff, cap=cap, selection=selection)))
    grid = pd.DataFrame(rows)
    viable = grid[(grid["active_markets"] >= 50)
                  & (grid["total_pnl_q15"] > 0)].copy()
    if viable.empty:
        viable = grid[grid["active_markets"] >= 25].copy()
    if viable.empty:
        viable = grid.copy()
    viable["objective"] = (
        viable["t_stat"] + 0.15 * np.log1p(viable["active_markets"])
        - 0.01 * viable["max_drawdown_q15"])
    winner = viable.sort_values(
        ["objective", "t_stat", "total_pnl_q15"],
        ascending=False).iloc[0].to_dict()
    grid.to_csv(OUT / "pml_validation_grid.csv", index=False)
    return winner, grid


def bootstrap(selected: pd.DataFrame, split: str,
              repetitions: int = 12000) -> dict:
    days = pd.date_range(
        SPLITS[split][0].normalize(),
        SPLITS[split][1].normalize() - pd.Timedelta(days=1), freq="D").date
    daily = (selected.assign(pnl_q15=QTY * selected["pnl_per_pair"])
             .groupby("day")["pnl_q15"].sum()
             .reindex([day.isoformat() for day in days], fill_value=0.0)
             .to_numpy())
    draws = RNG.integers(0, len(daily), size=(repetitions, len(daily)))
    means = daily[draws].mean(axis=1)
    return {
        "mean_day": float(daily.mean()),
        "ci_lo": float(np.quantile(means, 0.025)),
        "ci_hi": float(np.quantile(means, 0.975)),
        "p_nonpositive": float(np.mean(means <= 0)),
    }


def block_bootstrap(selected: pd.DataFrame,
                    repetitions: int = 12000) -> list[dict]:
    if selected.empty:
        return []
    frame = selected.assign(pnl_q15=QTY * selected["pnl_per_pair"])
    start = int(frame["close_ts"].min())
    output = []
    for minutes in [15, 30, 60, 90]:
        block_ids = ((frame["close_ts"] - start) // (minutes * 60)).astype(int)
        values = frame.assign(block=block_ids).groupby("block")["pnl_q15"].sum().to_numpy()
        draws = RNG.integers(0, len(values), size=(repetitions, len(values)))
        totals = values[draws].sum(axis=1)
        output.append({
            "minutes": minutes, "n_blocks": int(len(values)),
            "observed": float(values.sum()),
            "ci_lo": float(np.quantile(totals, 0.025)),
            "ci_hi": float(np.quantile(totals, 0.975)),
            "p_nonpositive": float(np.mean(totals <= 0)),
        })
    return output


def leave_one(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    coins = []
    for coin in sorted(selected["coin"].unique()):
        subset = selected[~selected["coin"].eq(coin)]
        coins.append({
            "excluded_coin": coin, "n": len(subset),
            "pnl_q15": float(QTY * subset["pnl_per_pair"].sum()),
            "edge_c": float(subset["pnl_per_pair"].mean() * 100)
            if len(subset) else np.nan,
        })
    weeks = []
    for week in sorted(selected["week"].unique()):
        subset = selected[~selected["week"].eq(week)]
        weeks.append({
            "excluded_week": week, "n": len(subset),
            "pnl_q15": float(QTY * subset["pnl_per_pair"].sum()),
            "edge_c": float(subset["pnl_per_pair"].mean() * 100)
            if len(subset) else np.nan,
        })
    return pd.DataFrame(coins), pd.DataFrame(weeks)


def branch_economics(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    masks = {
        "both": selected["both_filled"],
        "only_yes": selected["only_yes"],
        "only_no": selected["only_no"],
        "none": selected["none_filled"],
    }
    for label, mask in masks.items():
        subset = selected[mask]
        rows.append({
            "branch": label, "n": int(len(subset)),
            "share": float(len(subset) / len(selected)) if len(selected) else 0.0,
            "mean_pnl_c": float(subset["pnl_per_pair"].mean() * 100)
            if len(subset) else 0.0,
            "total_pnl_q15": float(QTY * subset["pnl_per_pair"].sum()),
            "yes_win_rate": float(subset["true_yes"].mean())
            if len(subset) else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    frame = build_markets()
    winner, _ = validation_grid(frame)
    params = {
        "min_spread_c": float(winner["min_spread_c"]),
        "cancel_after_min": int(winner["cancel_after_min"]),
        "cap": int(winner["cap"]),
        "selection": str(winner["selection"]),
    }
    test_frame = frame[split_mask(frame, "test")]
    selected = select_windows(test_frame, **params)
    test_metrics = metric(selected, split="test", **params)
    day = bootstrap(selected, "test")
    blocks = block_bootstrap(selected)
    coins, weeks = leave_one(selected)
    branches = branch_economics(selected)
    selected.to_parquet(OUT / "pml_test_selected.parquet", index=False)
    coins.to_csv(OUT / "pml_leave_one_coin.csv", index=False)
    weeks.to_csv(OUT / "pml_leave_one_week.csv", index=False)
    branches.to_csv(OUT / "pml_branch_economics.csv", index=False)

    hard_pass = bool(
        test_metrics.total_pnl_q15 > 0
        and test_metrics.active_markets >= 50
        and day["ci_lo"] > 0
        and blocks and min(block["ci_lo"] for block in blocks) > 0
        and len(coins) and (coins["pnl_q15"] > 0).all()
        and len(weeks) and (weeks["pnl_q15"] > 0).all())
    summary = {
        "data": {"markets": int(frame["ticker"].nunique()),
                 "state_rows": int(len(frame)),
                 "days": int(frame["day"].nunique())},
        "validation_winner": winner,
        "test": asdict(test_metrics),
        "day_bootstrap": day,
        "block_bootstrap": blocks,
        "hard_pass": hard_pass,
    }
    (OUT / "pml_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    verdict = "PASS" if hard_pass else "FAIL / RESEARCH ONLY"
    lines = [
        "# Paired Maker Lock — full-path audit", "",
        f"## Verdict: **{verdict}**", "",
        "Two maker bids are posted on complementary YES and NO outcomes. If both",
        "fill, the displayed spread is locked. Exactly one fill leaves directional",
        "legging risk.", "",
        "The policy was selected on validation and evaluated once on the sealed",
        "chronological test period.", "", "## Validation-selected policy", "",
        f"- Minimum spread: {params['min_spread_c']:.1f}¢",
        f"- Cancel both if neither fills after: {params['cancel_after_min']} minutes",
        f"- Maximum paired markets per close: {params['cap']}",
        f"- Selection: {params['selection']}",
        f"- Validation active markets: {int(winner['active_markets'])}",
        f"- Validation P&L q{QTY}: ${float(winner['total_pnl_q15']):.2f}", "",
        "## Sealed test", "",
        f"- Posted pairs: {test_metrics.n_markets}",
        f"- Active pairs: {test_metrics.active_markets}",
        f"- Both-fill rate: {test_metrics.both_rate:.2%}",
        f"- Single-leg rate: {test_metrics.single_rate:.2%}",
        f"- No-fill rate: {test_metrics.none_rate:.2%}",
        f"- Edge per posted pair: {test_metrics.edge_per_posted_pair_c:+.3f}¢",
        f"- Edge per active pair: {test_metrics.edge_per_active_pair_c:+.3f}¢",
        f"- Total P&L q{QTY}: ${test_metrics.total_pnl_q15:.2f}",
        f"- Mean/day q{QTY}: ${test_metrics.mean_day_q15:.2f}",
        f"- Daily SD: ${test_metrics.sd_day_q15:.2f}",
        f"- Maximum drawdown: ${test_metrics.max_drawdown_q15:.2f}",
        f"- Worst close window: ${test_metrics.worst_window_q15:.2f}", "",
        "## Uncertainty", "",
        f"- Day-bootstrap mean/day CI: [${day['ci_lo']:.2f}, ${day['ci_hi']:.2f}]",
        f"- P(mean day ≤ 0): {day['p_nonpositive']:.4f}", "",
        "| Block | Total-P&L CI | P(nonpositive) |", "|---:|---:|---:|",
    ]
    for block in blocks:
        lines.append(
            f"| {block['minutes']} min | [${block['ci_lo']:.2f}, "
            f"${block['ci_hi']:.2f}] | {block['p_nonpositive']:.4f} |")
    lines.extend(["", "## Interpretation", "",
        "A positive both-fill branch is guaranteed arithmetically, but the strategy",
        "passes only if locked-spread income exceeds one-leg losses after",
        "chronological selection and concentration controls.", "",
        "The fill reconstruction uses complete-minute quote crossings and positive",
        "candle volume. It does not identify queue priority or q15 capacity, so even",
        "a historical PASS requires q1 prospective execution before scaling."])
    (OUT / "pml_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
