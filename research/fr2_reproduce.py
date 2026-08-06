"""Clean-room reproduction of FR2: two-minute favorite repricing.

FR2 is a candidate, not a proven live strategy.

Input: data_csv/ladder_paths.csv

Frozen rule
-----------
1. At the first complete observation with 8-14 minutes remaining, identify the
   favorite.
2. Require favorite bid in [0.65, 0.80).
3. Wait two complete minutes.
4. Keep the original side.
5. Buy only when that side's current ask is in [0.60, 0.80].
6. Use one candidate per close window, following the existing runner series
   order: BTC, ETH, SOL, XRP, DOGE, HYPE.
7. Model a 20-contract IOC at the displayed ask, charge the historical taker
   fee formula once for the block, and hold to settlement.

This script does not place orders or use credentials.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

COIN_ORDER = {"BTC": 0, "ETH": 1, "SOL": 2, "XRP": 3, "DOGE": 4, "HYPE": 5}
QTY = 20
DELAY_MINUTES = 2
ASK_MIN = 0.60
ASK_MAX = 0.80

SPLITS = {
    "train": ("2026-05-25", "2026-06-30"),
    "valid": ("2026-06-30", "2026-07-18"),
    "test": ("2026-07-18", "2026-08-07"),
}


def taker_fee(quantity: int, price: float) -> float:
    raw = 0.07 * quantity * price * (1.0 - price)
    return math.ceil(raw * 10_000 - 1e-12) / 10_000


def build_candidates(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path)
    rows: list[dict] = []

    for market in source.itertuples(index=False):
        try:
            candles = json.loads(market.path)
        except Exception:
            continue

        by_minute = {float(c["ml"]): c for c in candles}
        eligible = sorted(
            [c for c in candles if 8 <= float(c["ml"]) <= 14],
            key=lambda c: -float(c["ml"]),
        )
        if not eligible:
            continue

        entry = eligible[0]
        try:
            yes_bid = float(entry["bc"])
            yes_ask = float(entry["ac"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 < yes_bid < yes_ask < 1):
            continue

        favorite_yes = yes_bid >= 0.5
        side = "yes" if favorite_yes else "no"
        initial_bid = yes_bid if favorite_yes else 1.0 - yes_ask
        initial_ask = yes_ask if favorite_yes else 1.0 - yes_bid
        if not (0.65 <= initial_bid < 0.80):
            continue

        entry_ml = float(entry["ml"])
        target_ml = entry_ml - DELAY_MINUTES
        if target_ml not in by_minute:
            continue

        delayed = by_minute[target_ml]
        try:
            delayed_ask = (
                float(delayed["ac"])
                if favorite_yes
                else 1.0 - float(delayed["bc"])
            )
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 < delayed_ask < 1):
            continue

        try:
            close = (
                pd.to_datetime(
                    market.ticker.split("-")[1],
                    format="%y%b%d%H%M",
                    utc=True,
                )
                + pd.Timedelta(hours=4)
            )
        except Exception:
            continue

        won = int(side == market.result)
        qualifies = ASK_MIN <= delayed_ask <= ASK_MAX
        fee = taker_fee(QTY, delayed_ask) if qualifies else 0.0
        pnl = QTY * (won - delayed_ask) - fee if qualifies else 0.0

        rows.append(
            {
                "ticker": market.ticker,
                "coin": market.coin,
                "close_dt": close,
                "side": side,
                "entry_ml": entry_ml,
                "initial_bid": initial_bid,
                "initial_ask": initial_ask,
                "delayed_ask": delayed_ask,
                "won": won,
                "qualifies_pre_cap": qualifies,
                "fee": fee,
                "pnl_pre_cap": pnl,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("no eligible rows reconstructed")

    result["coin_order"] = result["coin"].map(COIN_ORDER).fillna(99)
    result = result.sort_values(["close_dt", "coin_order", "ticker"])
    result["candidate_rank"] = result.groupby("close_dt")[
        "qualifies_pre_cap"
    ].cumsum()
    result["selected"] = result["qualifies_pre_cap"] & (
        result["candidate_rank"] <= 1
    )
    result["pnl"] = np.where(result["selected"], result["pnl_pre_cap"], 0.0)
    result["contracts"] = np.where(result["selected"], QTY, 0)
    result["net_edge_per_contract"] = np.where(
        result["selected"], result["pnl"] / QTY, np.nan
    )
    return result.sort_values("close_dt").reset_index(drop=True)


def drawdown_metrics(data: pd.DataFrame) -> dict:
    windows = data.groupby("close_dt", as_index=False)["pnl"].sum()
    windows = windows.sort_values("close_dt")
    equity = windows["pnl"].cumsum()
    drawdown = equity.cummax() - equity
    return {
        "total_pnl": float(windows["pnl"].sum()),
        "max_drawdown": float(drawdown.max()),
        "worst_window": float(windows["pnl"].min()),
        "best_window": float(windows["pnl"].max()),
    }


def bootstrap_days(data: pd.DataFrame, repetitions: int = 20_000) -> dict:
    daily = data.groupby(data["close_dt"].dt.date)["pnl"].sum().to_numpy(float)
    rng = np.random.default_rng(20260815)
    estimates = np.empty(repetitions)
    for start in range(0, repetitions, 2_000):
        stop = min(repetitions, start + 2_000)
        indices = rng.integers(0, len(daily), size=(stop - start, len(daily)))
        estimates[start:stop] = daily[indices].mean(axis=1)
    return {
        "daily_mean": float(daily.mean()),
        "daily_sd": float(daily.std(ddof=1)),
        "daily_mean_ci95": [
            float(np.quantile(estimates, 0.025)),
            float(np.quantile(estimates, 0.975)),
        ],
        "p_daily_mean_nonpositive": float((estimates <= 0).mean()),
        "positive_day_fraction": float((daily > 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    source = args.root / "data_csv" / "ladder_paths.csv"
    data = build_candidates(source)
    selected = data[data["selected"]].copy()

    summary = {
        "source_markets": int(pd.read_csv(source, usecols=["ticker"]).shape[0]),
        "initial_in_band_markets": int(len(data)),
        "unique_close_windows": int(data["close_dt"].nunique()),
        "selected_markets": int(len(selected)),
        "calendar_days": int(data["close_dt"].dt.date.nunique()),
        "win_rate": float(selected["won"].mean()),
        "average_delayed_ask": float(selected["delayed_ask"].mean()),
        "average_fee_per_contract": float((selected["fee"] / QTY).mean()),
        "net_edge_per_contract": float(
            selected["net_edge_per_contract"].mean()
        ),
        **drawdown_metrics(data),
        **bootstrap_days(data),
        "splits": {},
    }

    for name, (start, stop) in SPLITS.items():
        part = data[(data["close_dt"] >= start) & (data["close_dt"] < stop)]
        chosen = part[part["selected"]]
        summary["splits"][name] = {
            "selected": int(len(chosen)),
            "pnl": float(part["pnl"].sum()),
            "net_edge_per_contract": float(
                chosen["net_edge_per_contract"].mean()
            ),
            "win_rate": float(chosen["won"].mean()),
        }

    output = args.output or (
        args.root / "data" / "results" / "fr2_candidates.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    print(json.dumps(summary, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
