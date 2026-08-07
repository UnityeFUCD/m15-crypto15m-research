"""FOP: first-observation versus drift-in provenance audit.

Research-only; no exchange connection or order submission.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "results" / "fop"
OUT.mkdir(parents=True, exist_ok=True)
QTY = 15
SEED = 2026080641
REPS = 20_000


def fee_total(qty: int, p: float) -> float:
    return math.ceil(0.07 * qty * p * (1 - p) * 10_000 - 1e-12) / 10_000


def decode_close(values: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, utc=True)
    x = pd.to_numeric(values, errors="coerce")
    med = float(x.dropna().abs().median())
    unit = "ns" if med > 1e17 else "us" if med > 1e14 else "ms" if med > 1e11 else "s"
    out = pd.to_datetime(x, unit=unit, utc=True)
    if out.dt.year.min() < 2020 or out.dt.year.max() > 2035:
        raise RuntimeError("implausible close timestamps")
    return out


def quote(point: dict) -> tuple[float, float, str, float, float, float] | None:
    try:
        ml_raw = float(point["ml"])
        yb = float(point["yb"])
        ya = float(point["ya"])
        vol = float(point.get("v", 0.0) or 0.0)
    except (KeyError, TypeError, ValueError):
        return None
    if not (0.0 <= yb < ya <= 1.0):
        return None
    yes_bid, yes_ask = yb, ya
    no_bid, no_ask = 1.0 - ya, 1.0 - yb
    if yes_bid >= no_bid:
        side, bid, ask = "yes", yes_bid, yes_ask
    else:
        side, bid, ask = "no", no_bid, no_ask
    return ml_raw, bid, side, ask, yb, vol


def maker_filled(path: list[dict], entry_ml: float, side: str, bid: float) -> bool:
    for point in path:
        q = quote(point)
        if q is None:
            continue
        ml, _, _, _, yb, vol = q
        if ml >= entry_ml - 1e-9 or vol <= 0:
            continue
        try:
            ya = float(point["ya"])
        except (KeyError, TypeError, ValueError):
            continue
        if side == "yes" and ya <= bid + 1e-9:
            return True
        if side == "no" and yb >= 1.0 - bid - 1e-9:
            return True
    return False


def build() -> tuple[pd.DataFrame, dict]:
    src = pd.read_parquet(ROOT / "data" / "paths_full.parquet").copy()
    src["close_dt"] = decode_close(src["close_ts"])
    rows = []
    side_disagreements = 0
    no_entry = 0
    for market in src.itertuples(index=False):
        try:
            path = json.loads(market.path) if isinstance(market.path, str) else market.path
        except Exception:
            continue
        observations = []
        for point in path:
            q = quote(point)
            if q is None:
                continue
            ml = q[0]
            if 8.0 - 1e-9 <= ml <= 14.0 + 1e-9:
                observations.append((ml, point, q))
        observations.sort(key=lambda item: -item[0])
        if not observations:
            continue
        initial = observations[0]
        entry = None
        entry_index = None
        for idx, item in enumerate(observations):
            if 0.65 <= item[2][1] < 0.80:
                entry = item
                entry_index = idx
                break
        if entry is None:
            no_entry += 1
            continue
        entry_ml, _, q = entry
        _, bid, side, ask, _, volume = q
        provenance = "first_observation" if entry_index == 0 else "drift_in"
        stored_side = str(market.side).lower()
        stored_won = int(market.won)
        if stored_side not in {"yes", "no"} or stored_won not in {0, 1}:
            continue
        yes_won = stored_won if stored_side == "yes" else 1 - stored_won
        trade_won = yes_won if side == "yes" else 1 - yes_won
        if side != stored_side:
            side_disagreements += 1
        fee = fee_total(QTY, ask)
        taker_pnl = QTY * (trade_won - ask) - fee
        filled = maker_filled(path, entry_ml, side, bid)
        maker_pnl = QTY * (trade_won - bid) if filled else 0.0
        rows.append({
            "ticker": str(market.ticker), "coin": str(market.coin),
            "close_dt": market.close_dt, "date": market.close_dt.date(),
            "week": market.close_dt.strftime("%G-W%V"),
            "minute": int(market.minute), "entry_ml": float(entry_ml),
            "provenance": provenance, "side": side, "bid": bid, "ask": ask,
            "spread": ask - bid, "volume": volume, "won": int(trade_won),
            "premium": trade_won - bid, "taker_pnl": taker_pnl,
            "taker_edge": taker_pnl / QTY, "maker_filled": int(filled),
            "maker_pnl": maker_pnl, "maker_submitted_edge": maker_pnl / QTY,
        })
    frame = pd.DataFrame(rows).sort_values(["close_dt", "coin"]).reset_index(drop=True)
    audit = {
        "source_rows": int(len(src)), "eligible_rows": int(len(frame)),
        "no_entry": int(no_entry), "derived_vs_stored_side_disagreements": int(side_disagreements),
    }
    return frame, audit


def add_split(frame: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(frame["date"].unique())
    i1, i2 = int(len(dates) * 0.50), int(len(dates) * 0.75)
    mapping = {d: "train" for d in dates[:i1]}
    mapping.update({d: "valid" for d in dates[i1:i2]})
    mapping.update({d: "test" for d in dates[i2:]})
    out = frame.copy()
    out["split"] = out["date"].map(mapping)
    return out


def metrics(group: pd.DataFrame) -> dict:
    return {
        "n": int(len(group)), "days": int(group["date"].nunique()),
        "win_rate": float(group["won"].mean()), "mean_bid": float(group["bid"].mean()),
        "mean_ask": float(group["ask"].mean()), "premium": float(group["premium"].mean()),
        "taker_edge": float(group["taker_edge"].mean()),
        "maker_fill_rate": float(group["maker_filled"].mean()),
        "maker_submitted_edge": float(group["maker_submitted_edge"].mean()),
        "taker_total_q15": float(group["taker_pnl"].sum()),
        "maker_total_q15": float(group["maker_pnl"].sum()),
    }


def difference_bootstrap(frame: pd.DataFrame, minute: int, split: str) -> dict:
    x = frame[(frame["minute"] == minute) & (frame["split"] == split)].copy()
    dates = sorted(x["date"].unique())
    if not dates or set(x["provenance"].unique()) != {"first_observation", "drift_in"}:
        return {"minute": minute, "split": split, "observed": np.nan}

    def stat(sample_dates: list) -> float:
        parts = [x[x["date"] == d] for d in sample_dates]
        z = pd.concat(parts, ignore_index=True)
        a = z[z["provenance"] == "first_observation"]["taker_edge"]
        b = z[z["provenance"] == "drift_in"]["taker_edge"]
        return float(a.mean() - b.mean()) if len(a) and len(b) else np.nan

    observed = stat(dates)
    rng = np.random.default_rng(SEED + minute + len(split))
    draws = []
    for _ in range(REPS):
        sample = list(rng.choice(dates, size=len(dates), replace=True))
        value = stat(sample)
        if math.isfinite(value):
            draws.append(value)
    arr = np.asarray(draws)
    return {
        "minute": minute, "split": split, "observed": observed,
        "ci_lo": float(np.quantile(arr, 0.025)), "ci_hi": float(np.quantile(arr, 0.975)),
        "p_nonpositive": float(np.mean(arr <= 0)), "bootstrap_n": int(len(arr)),
    }


def cap_one(frame: pd.DataFrame) -> pd.DataFrame:
    # Frozen ranking: first-observation before drift-in, then lower all-in ask,
    # narrower spread, deterministic coin/ticker.
    x = frame.copy()
    x["prov_rank"] = x["provenance"].map({"first_observation": 0, "drift_in": 1})
    x.sort_values(["close_dt", "prov_rank", "ask", "spread", "coin", "ticker"], inplace=True)
    return x.groupby("close_dt", as_index=False, group_keys=False).head(1).copy()


def portfolio_metrics(selected: pd.DataFrame) -> dict:
    windows = selected.groupby("close_dt")["taker_pnl"].sum().sort_index()
    equity = windows.cumsum(); dd = equity.cummax() - equity
    daily = selected.groupby("date")["taker_pnl"].sum()
    return {
        "n": int(len(selected)), "days": int(selected["date"].nunique()),
        "total_pnl": float(selected["taker_pnl"].sum()),
        "mean_day": float(daily.mean()), "sd_day": float(daily.std(ddof=1)),
        "edge_per_contract": float(selected["taker_pnl"].sum() / (QTY * len(selected))),
        "max_drawdown": float(dd.max()), "worst_window": float(windows.min()),
    }


def main() -> None:
    frame, audit = build()
    frame = add_split(frame)
    frame.to_parquet(OUT / "entries.parquet", index=False)

    rows = []
    for split in ["train", "valid", "test", "all"]:
        base_frame = frame if split == "all" else frame[frame["split"] == split]
        for minute in [0, 15, 30, 45]:
            for provenance in ["first_observation", "drift_in"]:
                group = base_frame[(base_frame["minute"] == minute) & (base_frame["provenance"] == provenance)]
                if len(group):
                    rows.append({"split": split, "minute": minute, "provenance": provenance, **metrics(group)})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "group_metrics.csv", index=False)

    diffs = pd.DataFrame([
        difference_bootstrap(frame, minute, split)
        for split in ["train", "valid", "test", "all"]
        for minute in [0, 15, 30, 45]
    ])
    diffs.to_csv(OUT / "provenance_differences.csv", index=False)

    portfolios = []
    for split in ["train", "valid", "test", "all"]:
        base_frame = frame if split == "all" else frame[frame["split"] == split]
        for minute_rule in ["all_minutes", "minute00"]:
            eligible = base_frame if minute_rule == "all_minutes" else base_frame[base_frame["minute"] == 0]
            selected = cap_one(eligible)
            portfolios.append({"split": split, "minute_rule": minute_rule, **portfolio_metrics(selected)})
    portfolios_df = pd.DataFrame(portfolios)
    portfolios_df.to_csv(OUT / "cap_one_portfolios.csv", index=False)

    primary = summary[(summary["minute"] == 0) & summary["split"].isin(["train", "valid", "test"])]
    first = primary[primary["provenance"] == "first_observation"].set_index("split")
    diff_primary = diffs[(diffs["minute"] == 0) & diffs["split"].isin(["train", "valid", "test"])].set_index("split")
    pass_first = all(s in first.index and first.loc[s, "taker_edge"] > 0 for s in ["train", "valid", "test"])
    pass_diff = all(s in diff_primary.index and diff_primary.loc[s, "observed"] > 0 for s in ["train", "valid", "test"])
    result = {"audit": audit, "first_observation_positive_all_splits": pass_first,
              "first_minus_drift_positive_all_splits": pass_diff,
              "historical_candidate": bool(pass_first and pass_diff)}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    verdict = "HISTORICAL CANDIDATE" if result["historical_candidate"] else "FAIL"
    lines = ["# FOP — First-Observation Provenance Audit", "",
             f"## Verdict: **{verdict}**", "", "## Data audit", "", "```json",
             json.dumps(audit, indent=2), "```", "", "## Group metrics", "",
             summary.to_markdown(index=False, floatfmt=".5f"), "", "## Provenance differences", "",
             diffs.to_markdown(index=False, floatfmt=".5f"), "", "## Cap-one portfolios", "",
             portfolios_df.to_markdown(index=False, floatfmt=".5f"), "",
             "This overlapping historical population cannot provide independent confirmation.",
             "No result authorizes live trading."]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
