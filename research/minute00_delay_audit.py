"""Clean-room audit of the proxy-free top-of-hour delayed-taker candidate.

Frozen candidate
----------------
- Start with the original favourite observed at the first valid 8-14 minute
  observation in paths_full.
- Initial favourite bid in [0.65, 0.80).
- Keep that original side fixed.
- Wait exactly 120 seconds.
- Use the first complete one-minute quote at or after the wait.
- Require the original side's ask in [0.60, 0.80].
- Close minute == :00 UTC.
- Buy at the displayed ask as a q15 taker, pay the exact historical fee, hold
  to settlement.

No fill proxy, probe, HCR, DRC, or sampled dataset enters this audit.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "research" / "results"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(2026080623)
QTY = 15
WAIT_SECONDS = 120
ASK_FLOOR = 0.60
ASK_CEILING = 0.80
TRAIN_END = pd.Timestamp("2026-06-30", tz="UTC")
VALID_END = pd.Timestamp("2026-07-18", tz="UTC")
TEST_END = pd.Timestamp("2026-08-07", tz="UTC")


def fee_total(qty: int, price: float) -> float:
    raw = 0.07 * qty * price * (1 - price)
    return math.ceil(raw * 10_000 - 1e-12) / 10_000


def held_quote(side: str, yb: float, ya: float) -> tuple[float, float]:
    return (yb, ya) if side == "yes" else (1 - ya, 1 - yb)


def points(value: object, close_ts: int, side: str) -> list[dict]:
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
            volume = float(point.get("v", 0) or 0)
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= ml <= 15 and 0 < yb < ya < 1):
            continue
        bid, ask = held_quote(side, yb, ya)
        out.append({
            "ts": float(close_ts) - 60 * ml,
            "ml": ml, "bid": bid, "ask": ask, "volume": volume,
        })
    return sorted(out, key=lambda row: row["ts"])


def delayed_point(path: list[dict], entry_ts: float, wait: int) -> dict | None:
    target = entry_ts + wait
    later = [point for point in path if point["ts"] >= target - 1e-9]
    return later[0] if later else None


def build() -> pd.DataFrame:
    source = pd.read_parquet(DATA / "paths_full.parquet")
    rows = []
    for market in source.itertuples(index=False):
        try:
            initial_bid = float(market.bid)
            entry_ml = float(market.entry_ml)
        except (TypeError, ValueError):
            continue
        if not (0.65 <= initial_bid < 0.80 and 8 <= entry_ml <= 14):
            continue
        close_ts = int(market.close_ts)
        close_dt = pd.to_datetime(close_ts, unit="s", utc=True)
        path = points(market.path, close_ts, str(market.side))
        if not path:
            continue
        entry_ts = close_ts - 60 * entry_ml
        delayed = delayed_point(path, entry_ts, WAIT_SECONDS)
        if delayed is None:
            continue
        effective_wait = delayed["ts"] - entry_ts
        if not (WAIT_SECONDS - 1e-9 <= effective_wait <= WAIT_SECONDS + 90):
            raise RuntimeError(f"invalid effective wait {effective_wait}")
        ask = float(delayed["ask"])
        bid = float(delayed["bid"])
        if not (0 < bid < ask < 1):
            continue
        fee = fee_total(QTY, ask)
        won = int(market.won)
        pnl = QTY * (won - ask) - fee
        iso = close_dt.isocalendar()
        rows.append({
            "ticker": market.ticker,
            "coin": market.coin,
            "close_ts": close_ts,
            "close_dt": close_dt,
            "day": close_dt.date().isoformat(),
            "week": f"{iso.year}-{iso.week:02d}",
            "hour": close_dt.hour,
            "minute": close_dt.minute,
            "side": market.side,
            "entry_ml": entry_ml,
            "initial_bid": initial_bid,
            "initial_ask": float(market.ask),
            "delayed_ml": float(delayed["ml"]),
            "effective_wait": effective_wait,
            "bid": bid,
            "ask": ask,
            "spread_c": (ask - bid) * 100,
            "volume_log": math.log1p(max(float(delayed["volume"]), 0)),
            "won": won,
            "fee": fee,
            "pnl": pnl,
            "edge_per_contract": pnl / QTY,
            "eligible": ASK_FLOOR <= ask <= ASK_CEILING,
        })
    data = pd.DataFrame(rows)
    data.to_parquet(OUT / "minute00_delay_population.parquet", index=False)
    return data


def day_bootstrap(frame: pd.DataFrame, column: str = "edge_per_contract",
                  reps: int = 12000) -> dict:
    grouped = {day: group for day, group in frame.groupby("day")}
    days = sorted(grouped)
    if len(days) < 3 or frame.empty:
        return {"mean": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "p_nonpositive": np.nan, "days": len(days)}
    values = np.empty(reps)
    for i in range(reps):
        sample = pd.concat([grouped[days[j]] for j in RNG.integers(0, len(days), len(days))])
        values[i] = sample[column].mean()
    return {
        "mean": float(frame[column].mean()),
        "ci_lo": float(np.quantile(values, 0.025)),
        "ci_hi": float(np.quantile(values, 0.975)),
        "p_nonpositive": float(np.mean(values <= 0)),
        "days": len(days),
    }


def cap_one(frame: pd.DataFrame) -> pd.DataFrame:
    """Risk-aware fixed rule: lowest all-in ask, fixed coin tie-break."""
    order = {coin: idx for idx, coin in enumerate(["BTC", "ETH", "SOL", "XRP", "DOGE", "HYPE"])}
    x = frame.copy()
    x["coin_order"] = x.coin.map(order).fillna(99)
    return (
        x.sort_values(["close_ts", "ask", "coin_order", "ticker"])
        .groupby("close_ts", as_index=False).head(1)
        .sort_values("close_ts")
    )


def policy_metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {}
    daily = frame.groupby("day").pnl.sum()
    window = frame.groupby("close_ts").pnl.sum().sort_index()
    equity = window.cumsum()
    return {
        "n": int(len(frame)),
        "days": int(frame.day.nunique()),
        "mean_ask": float(frame.ask.mean()),
        "win_rate": float(frame.won.mean()),
        "edge_per_contract": float(frame.edge_per_contract.mean()),
        "total_pnl": float(frame.pnl.sum()),
        "mean_pnl_per_active_day": float(daily.mean()),
        "sd_active_day": float(daily.std(ddof=1)),
        "max_drawdown": float((equity.cummax() - equity).max()),
        "worst_window": float(window.min()),
    }


def chronological(frame: pd.DataFrame) -> list[dict]:
    output = []
    ranges = {
        "train": (pd.Timestamp("2026-05-25", tz="UTC"), TRAIN_END),
        "valid": (TRAIN_END, VALID_END),
        "test": (VALID_END, TEST_END),
        "heldout": (TRAIN_END, TEST_END),
    }
    for name, (start, stop) in ranges.items():
        part = frame[(frame.close_dt >= start) & (frame.close_dt < stop)]
        output.append({"split": name, **policy_metrics(part),
                       "bootstrap": day_bootstrap(part)})
    return output


def leave_one(frame: pd.DataFrame, column: str) -> list[dict]:
    rows = []
    for value in sorted(frame[column].unique()):
        part = frame[frame[column] != value]
        rows.append({
            f"excluded_{column}": str(value), "n": int(len(part)),
            "edge_per_contract": float(part.edge_per_contract.mean()),
            "total_pnl": float(part.pnl.sum()),
        })
    return rows


def matched_day_difference(data: pd.DataFrame, reps: int = 12000) -> dict:
    pairs = []
    for day, group in data.groupby("day"):
        top = group[group.minute == 0]
        other = group[group.minute != 0]
        if len(top) and len(other):
            pairs.append(top.edge_per_contract.mean() - other.edge_per_contract.mean())
    pairs = np.asarray(pairs, float)
    draw = RNG.integers(0, len(pairs), size=(reps, len(pairs)))
    means = pairs[draw].mean(axis=1)
    return {
        "n_days": int(len(pairs)), "mean_difference": float(pairs.mean()),
        "ci_lo": float(np.quantile(means, 0.025)),
        "ci_hi": float(np.quantile(means, 0.975)),
        "p_nonpositive": float(np.mean(means <= 0)),
    }


def selection_permutation(data: pd.DataFrame, reps: int = 6000) -> dict:
    observed = data[data.minute == 0].edge_per_contract.mean()
    best = []
    for _ in range(reps):
        q = data.copy()
        q["permuted"] = q.groupby("day").edge_per_contract.transform(
            lambda values: RNG.permutation(values.to_numpy())
        )
        best.append(max(
            q[q.minute == minute].permuted.mean()
            for minute in [0, 15, 30, 45]
        ))
    best = np.asarray(best)
    return {
        "observed": float(observed), "permuted_best_mean": float(best.mean()),
        "permuted_best_p95": float(np.quantile(best, 0.95)),
        "p_value": float(np.mean(best >= observed)),
    }


def calibration_test(data: pd.DataFrame) -> dict:
    """Does a predeclared :00 indicator improve held-out calibration?"""
    x = data.copy()
    x["mid"] = (x.bid + x.ask) / 2
    x["market_logit"] = logit(np.clip(x.mid, 1e-6, 1 - 1e-6))
    x["is_minute00"] = (x.minute == 0).astype(int)
    features0 = ["market_logit", "spread_c", "entry_ml", "volume_log"]
    features1 = features0 + ["is_minute00"]
    categorical = ["coin", "side"]

    def fit(features: list[str]) -> Pipeline:
        prep = ColumnTransformer([
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ])
        return Pipeline([
            ("prep", prep),
            ("model", LogisticRegression(C=0.1, solver="liblinear", max_iter=1000)),
        ])

    train = x[x.close_dt < TRAIN_END]
    held = x[(x.close_dt >= TRAIN_END) & (x.close_dt < TEST_END)]
    base, minute = fit(features0), fit(features1)
    base.fit(train[features0 + categorical], train.won)
    minute.fit(train[features1 + categorical], train.won)
    p0 = base.predict_proba(held[features0 + categorical])[:, 1]
    p1 = minute.predict_proba(held[features1 + categorical])[:, 1]
    return {
        "n_train": int(len(train)), "n_heldout": int(len(held)),
        "base_log_loss": float(log_loss(held.won, p0)),
        "minute_log_loss": float(log_loss(held.won, p1)),
        "delta_log_loss": float(log_loss(held.won, p1) - log_loss(held.won, p0)),
        "base_brier": float(brier_score_loss(held.won, p0)),
        "minute_brier": float(brier_score_loss(held.won, p1)),
        "delta_brier": float(brier_score_loss(held.won, p1) - brier_score_loss(held.won, p0)),
    }


def wait_sensitivity(source: pd.DataFrame) -> pd.DataFrame:
    paths = pd.read_parquet(DATA / "paths_full.parquet").set_index("ticker")
    rows = []
    for wait in [60, 120, 180, 300]:
        temporary = []
        for original in source.itertuples(index=False):
            market = paths.loc[original.ticker]
            path = points(market.path, int(market.close_ts), str(market.side))
            delayed = delayed_point(path, int(market.close_ts) - 60 * float(market.entry_ml), wait)
            if delayed is None:
                continue
            ask = float(delayed["ask"])
            if not ASK_FLOOR <= ask <= ASK_CEILING:
                continue
            fee = fee_total(QTY, ask)
            temporary.append({
                "ticker": original.ticker, "close_ts": original.close_ts,
                "day": original.day, "coin": original.coin,
                "won": int(original.won), "ask": ask,
                "pnl": QTY * (int(original.won) - ask) - fee,
            })
        frame = pd.DataFrame(temporary)
        frame["edge_per_contract"] = frame.pnl / QTY if len(frame) else np.nan
        frame = cap_one(frame) if len(frame) else frame
        rows.append({"wait_seconds": wait, **policy_metrics(frame)})
    return pd.DataFrame(rows)


def main() -> None:
    all_rows = build()
    eligible = all_rows[all_rows.eligible].copy()
    top = eligible[eligible.minute == 0].copy()
    top_cap = cap_one(top)

    per_minute = []
    for minute, group in eligible.groupby("minute"):
        per_minute.append({
            "minute": int(minute), **policy_metrics(group),
            "bootstrap": day_bootstrap(group),
        })

    chrono_all = chronological(top)
    chrono_cap = chronological(top_cap)
    heldout = top[top.close_dt >= TRAIN_END]
    heldout_cap = top_cap[top_cap.close_dt >= TRAIN_END]
    wait_grid = wait_sensitivity(top)

    summary = {
        "definition": {
            "wait_seconds": WAIT_SECONDS, "ask_floor": ASK_FLOOR,
            "ask_ceiling": ASK_CEILING, "quantity": QTY,
            "close_minute": 0,
        },
        "population": {
            "initial_in_band": int(len(all_rows)),
            "delayed_ask_in_band": int(len(eligible)),
            "top_of_hour": int(len(top)),
            "top_of_hour_cap_one": int(len(top_cap)),
            "days": int(all_rows.day.nunique()),
        },
        "per_minute": per_minute,
        "top_of_hour_all": policy_metrics(top),
        "top_of_hour_cap_one": policy_metrics(top_cap),
        "chronological_all": chrono_all,
        "chronological_cap_one": chrono_cap,
        "heldout_all_bootstrap": day_bootstrap(heldout),
        "heldout_cap_one_bootstrap": day_bootstrap(heldout_cap),
        "day_matched_difference": matched_day_difference(eligible),
        "best_of_four_permutation": selection_permutation(eligible),
        "leave_one_coin": leave_one(top, "coin"),
        "leave_one_week": leave_one(top, "week"),
        "leave_one_hour": leave_one(top, "hour"),
        "calibration": calibration_test(eligible),
        "wait_sensitivity": wait_grid.to_dict("records"),
    }

    gates = {
        "heldout_all_lower_bound_positive": summary["heldout_all_bootstrap"]["ci_lo"] > 0,
        "heldout_cap_one_lower_bound_positive": summary["heldout_cap_one_bootstrap"]["ci_lo"] > 0,
        "train_valid_test_all_positive": all(
            row.get("edge_per_contract", -1) > 0
            for row in chrono_all if row["split"] in {"train", "valid", "test"}
        ),
        "best_of_four_p_below_005": summary["best_of_four_permutation"]["p_value"] < 0.05,
        "minute_improves_heldout_logloss_and_brier": (
            summary["calibration"]["delta_log_loss"] < 0
            and summary["calibration"]["delta_brier"] < 0
        ),
        "leave_one_coin_positive": min(row["edge_per_contract"] for row in summary["leave_one_coin"]) > 0,
        "leave_one_week_positive": min(row["edge_per_contract"] for row in summary["leave_one_week"]) > 0,
    }
    summary["gates"] = gates
    summary["hard_pass"] = all(gates.values())

    wait_grid.to_csv(OUT / "minute00_wait_sensitivity.csv", index=False)
    pd.DataFrame(summary["leave_one_coin"]).to_csv(OUT / "minute00_leave_one_coin.csv", index=False)
    pd.DataFrame(summary["leave_one_week"]).to_csv(OUT / "minute00_leave_one_week.csv", index=False)
    top_cap.to_parquet(OUT / "minute00_cap_one_trades.parquet", index=False)
    (OUT / "minute00_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = [
        "# Top-of-hour delayed-taker — independent proxy-free audit", "",
        f"## Verdict: **{'PASS' if summary['hard_pass'] else 'FAIL / CANDIDATE ONLY'}**", "",
        "Frozen policy: initial favourite bid 65-80¢, wait 120 seconds, original",
        "side ask 60-80¢, close minute :00, exact q15 taker fee.", "",
        "## Population", "",
        f"- Initial in-band markets: {len(all_rows):,}",
        f"- Delayed-ask eligible: {len(eligible):,}",
        f"- :00 candidates: {len(top):,}",
        f"- :00 cap-one trades: {len(top_cap):,}", "",
        "## Uncapped :00", "",
    ]
    for key, value in policy_metrics(top).items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## Cap one per close window", ""]
    for key, value in policy_metrics(top_cap).items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## Held-out uncertainty", ""]
    for label, block in [
        ("all", summary["heldout_all_bootstrap"]),
        ("cap one", summary["heldout_cap_one_bootstrap"]),
    ]:
        lines.append(
            f"- {label}: edge {block['mean']*100:+.2f}¢, 95% CI "
            f"[{block['ci_lo']*100:+.2f}, {block['ci_hi']*100:+.2f}]¢, "
            f"P(≤0) {block['p_nonpositive']:.4f}"
        )
    lines += ["", "## Calibration", ""]
    for key, value in summary["calibration"].items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## Hard gates", ""]
    for name, value in gates.items():
        lines.append(f"- {name}: **{'PASS' if value else 'FAIL'}**")
    lines += [
        "", "The audit uses no fill proxy. A historical pass would still require a",
        "frozen prospective test because :00 has been examined repeatedly.",
        "Nothing authorizes live orders while the account KILL state is active.",
    ]
    (OUT / "minute00_report.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
