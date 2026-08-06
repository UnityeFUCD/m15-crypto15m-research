"""Data engineering and causal physical nowcast for SPE."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "research" / "results"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 2026080617
HORIZONS = [13, 12, 10, 8, 6, 4, 2]
COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "HYPE"]
QTY = 15
EPS = 1e-6
SPLITS = {
    "train": (pd.Timestamp("2026-05-25", tz="UTC"),
              pd.Timestamp("2026-06-30", tz="UTC")),
    "valid": (pd.Timestamp("2026-06-30", tz="UTC"),
              pd.Timestamp("2026-07-18", tz="UTC")),
    "test": (pd.Timestamp("2026-07-18", tz="UTC"),
             pd.Timestamp("2026-08-07", tz="UTC")),
}


def clip_probability(values: Any) -> Any:
    return np.clip(values, EPS, 1.0 - EPS)


def fee_total(quantity: int, price: float) -> float:
    raw = 0.07 * quantity * price * (1.0 - price)
    return math.ceil(raw * 10_000 - 1e-12) / 10_000


def split_mask(frame: pd.DataFrame, name: str) -> pd.Series:
    start, stop = SPLITS[name]
    return (frame.close_dt >= start) & (frame.close_dt < stop)


def parse_path(value: Any) -> list[dict]:
    try:
        raw = json.loads(value) if isinstance(value, str) else value
    except Exception:
        return []
    points = []
    for point in raw or []:
        try:
            ml = float(point["ml"])
            yb = float(point.get("yb", point.get("bc")))
            ya = float(point.get("ya", point.get("ac")))
            volume = float(point.get("v", point.get("volume", 0.0)) or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= ml <= 15 and 0 < yb < ya < 1:
            points.append({"ml": ml, "yb": yb, "ya": ya, "v": volume})
    return sorted(points, key=lambda row: -row["ml"])


def exact_point(points: list[dict], horizon: int) -> dict | None:
    candidates = [
        (abs(float(point["ml"]) - horizon), point)
        for point in points
        if abs(float(point["ml"]) - horizon) <= 0.11
    ]
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def load_spot_maps() -> dict[str, dict[str, dict[int, float]]]:
    spot = pd.read_parquet(DATA / "spot_1m.parquet")
    required = {"coin", "ts", "open", "high", "low", "close", "vol"}
    missing = required - set(spot.columns)
    if missing:
        raise RuntimeError(f"spot_1m missing columns: {sorted(missing)}")
    return {
        str(coin): {
            column: dict(zip(group.ts.astype(int), group[column].astype(float)))
            for column in ["open", "high", "low", "close", "vol"]
        }
        for coin, group in spot.groupby("coin")
    }


def momentum(close_map: dict[int, float], key: int, price: float, minutes: int) -> float:
    old = close_map.get(key - minutes * 60)
    if old is None or not np.isfinite(old) or old <= 0:
        return np.nan
    return math.log(price / float(old))


def build_feature_panel() -> pd.DataFrame:
    paths = pd.read_parquet(DATA / "paths_full.parquet")
    truth = pd.read_parquet(
        DATA / "underlying.parquet",
        columns=["ticker", "coin", "result", "a0", "a1"],
    )
    truth["a0"] = pd.to_numeric(truth.a0, errors="coerce")
    truth["a1"] = pd.to_numeric(truth.a1, errors="coerce")
    truth = truth[
        truth.result.isin(["yes", "no"])
        & truth.a0.notna() & truth.a1.notna()
        & (truth.a0 > 0) & (truth.a1 > 0)
    ].drop_duplicates("ticker")
    truth_map = truth.set_index("ticker").to_dict("index")
    spots = load_spot_maps()
    rows = []

    for market in paths.itertuples(index=False):
        ticker = str(market.ticker)
        meta = truth_map.get(ticker)
        coin = str(market.coin)
        if meta is None or coin not in spots:
            continue

        close_ts = int(market.close_ts)
        close_dt = pd.to_datetime(close_ts, unit="s", utc=True)
        open_ts = close_ts - 900
        spot = spots[coin]
        open_proxy = spot["close"].get(open_ts - 60)
        if open_proxy is None or not np.isfinite(open_proxy) or open_proxy <= 0:
            continue

        points = parse_path(market.path)
        hp = {h: exact_point(points, h) for h in HORIZONS}
        mids = {
            h: ((p["yb"] + p["ya"]) / 2 if p else np.nan)
            for h, p in hp.items()
        }
        y = int(str(meta["result"]) == "yes")
        true_return_bp = math.log(float(meta["a1"]) / float(meta["a0"])) * 1e4

        for horizon in HORIZONS:
            point = hp[horizon]
            if point is None:
                continue
            decision_ts = close_ts - horizon * 60
            current_key = decision_ts - 60
            current = spot["close"].get(current_key)
            if current is None or not np.isfinite(current) or current <= 0:
                continue
            current = float(current)

            keys = list(range(open_ts - 60, current_key + 1, 60))
            closes = np.asarray([spot["close"].get(k, np.nan) for k in keys], float)
            if len(closes) < 2 or np.isnan(closes).any():
                continue
            rets = np.diff(np.log(closes))
            rv = float(np.std(rets, ddof=1 if len(rets) >= 2 else 0))
            rv = max(rv, 1e-8)

            highs = np.asarray([spot["high"].get(k, np.nan) for k in keys[1:]], float)
            lows = np.asarray([spot["low"].get(k, np.nan) for k in keys[1:]], float)
            vols = np.asarray([spot["vol"].get(k, np.nan) for k in keys[1:]], float)
            high = float(np.nanmax(highs)) if np.isfinite(highs).any() else current
            low = float(np.nanmin(lows)) if np.isfinite(lows).any() else current
            range_bp = math.log(high / low) * 1e4 if high > 0 and low > 0 else 0.0

            r_now = math.log(current / float(open_proxy))
            yb, ya = float(point["yb"]), float(point["ya"])
            mid = (yb + ya) / 2
            p1, p2 = mids.get(horizon + 1, np.nan), mids.get(horizon + 2, np.nan)
            iso = close_dt.isocalendar()
            rows.append({
                "ticker": ticker,
                "coin": coin,
                "close_ts": close_ts,
                "close_dt": close_dt,
                "day": close_dt.date().isoformat(),
                "week": f"{iso.year}-{iso.week:02d}",
                "hour": close_dt.hour,
                "close_minute": close_dt.minute,
                "horizon": horizon,
                "y": y,
                "true_return_bp": true_return_bp,
                "yes_bid": yb,
                "yes_ask": ya,
                "no_bid": 1.0 - ya,
                "no_ask": 1.0 - yb,
                "mid_yes": mid,
                "market_logit": float(logit(clip_probability(mid))),
                "spread_c": (ya - yb) * 100,
                "spot_return_bp": r_now * 1e4,
                "distance_z": r_now / max(rv * math.sqrt(horizon), 1e-6),
                "realized_vol_bp": rv * 1e4,
                "range_bp": range_bp,
                "mom1_bp": momentum(spot["close"], current_key, current, 1) * 1e4,
                "mom3_bp": momentum(spot["close"], current_key, current, 3) * 1e4,
                "mom5_bp": momentum(spot["close"], current_key, current, 5) * 1e4,
                "quote_mom1_c": (mid - p1) * 100 if np.isfinite(p1) else np.nan,
                "quote_mom2_c": (mid - p2) * 100 if np.isfinite(p2) else np.nan,
                "path_volume_log": math.log1p(max(float(point["v"]), 0.0)),
                "spot_volume_log": math.log1p(max(float(np.nansum(vols)), 0.0)),
                "hour_sin": math.sin(2 * math.pi * close_dt.hour / 24),
                "hour_cos": math.cos(2 * math.pi * close_dt.hour / 24),
            })

    panel = pd.DataFrame(rows)
    if panel.empty:
        raise RuntimeError("feature panel is empty")

    group = panel.groupby(["close_ts", "horizon"], sort=False)
    panel["common_spot_bp"] = group.spot_return_bp.transform("mean")
    panel["cross_dispersion_bp"] = group.spot_return_bp.transform("std").fillna(0)
    panel["coin_residual_bp"] = panel.spot_return_bp - panel.common_spot_bp
    panel["common_mid"] = group.mid_yes.transform("mean")
    panel["market_residual"] = panel.mid_yes - panel.common_mid
    panel["same_sign_share"] = group.spot_return_bp.transform(
        lambda x: float(np.mean(np.sign(x) == np.sign(float(np.mean(x)))))
    )
    panel["coin_return_rank"] = group.spot_return_bp.rank(pct=True)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel.to_parquet(OUT / "spe_feature_panel.parquet", index=False)
    return panel


def empirical_physical_probability(panel: pd.DataFrame, horizon: int) -> pd.Series:
    rows = panel[panel.horizon.eq(horizon)].copy()
    train = rows[split_mask(rows, "train")].copy()
    train["residual_bp"] = train.true_return_bp - train.spot_return_bp
    cuts = np.unique(
        train.realized_vol_bp.quantile([0.25, 0.5, 0.75])
        .dropna().to_numpy()
    )

    def bins(series: pd.Series) -> np.ndarray:
        return np.searchsorted(cuts, series.to_numpy(), side="right")

    train["vol_bin"] = bins(train.realized_vol_bp)
    rows["vol_bin"] = bins(rows.realized_vol_bp)
    pooled = {
        int(b): np.sort(g.residual_bp.dropna().to_numpy(float))
        for b, g in train.groupby("vol_bin")
    }
    by_coin = {
        (str(c), int(b)): np.sort(g.residual_bp.dropna().to_numpy(float))
        for (c, b), g in train.groupby(["coin", "vol_bin"])
    }
    all_resid = np.sort(train.residual_bp.dropna().to_numpy(float))
    preds = []
    for row in rows.itertuples(index=False):
        sample = by_coin.get((str(row.coin), int(row.vol_bin)))
        if sample is None or len(sample) < 120:
            sample = pooled.get(int(row.vol_bin), all_resid)
        if sample is None or len(sample) == 0:
            preds.append(0.5)
            continue
        threshold = -float(row.spot_return_bp)
        successes = len(sample) - np.searchsorted(sample, threshold, side="left")
        preds.append((successes + 0.5) / (len(sample) + 1.0))
    return pd.Series(preds, index=rows.index)
