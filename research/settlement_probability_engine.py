"""Settlement Probability Engine (SPE): clean-room causal nowcast audit.

Trusted population inputs:
- data/paths_full.parquet
- data/spot_1m.parquet
- data/underlying.parquet

The audit asks whether a causal physical settlement nowcast adds information
beyond the contemporaneous Kalshi quote, and whether that incremental
probability supports one taker position per close window after exact fees.

Hyperparameters and the economic policy are selected on validation only.  The
chronological test slice is evaluated once.  Nothing here places orders or
reads credentials.
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "research" / "results"
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(2026080617)
HORIZONS = [13, 12, 10, 8, 6, 4, 2]
QTY = 15
EPS = 1e-6
C_GRID = [0.01, 0.1, 1.0, 10.0]
EDGE_THRESHOLDS = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
MODELS = ["market", "blend", "struct"]
COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "HYPE"]

SPLITS = {
    "train": (pd.Timestamp("2026-05-25", tz="UTC"),
              pd.Timestamp("2026-06-30", tz="UTC")),
    "valid": (pd.Timestamp("2026-06-30", tz="UTC"),
              pd.Timestamp("2026-07-18", tz="UTC")),
    "test": (pd.Timestamp("2026-07-18", tz="UTC"),
             pd.Timestamp("2026-08-07", tz="UTC")),
}


def clip_prob(x: Any) -> Any:
    return np.clip(x, EPS, 1.0 - EPS)


def fee_total(qty: int, price: float) -> float:
    raw = 0.07 * qty * price * (1.0 - price)
    return math.ceil(raw * 10_000 - 1e-12) / 10_000


def split_mask(frame: pd.DataFrame, name: str) -> pd.Series:
    start, stop = SPLITS[name]
    return (frame["close_dt"] >= start) & (frame["close_dt"] < stop)


def nearest_integer_point(points: list[dict], horizon: int) -> dict | None:
    candidates = []
    for point in points:
        try:
            ml = float(point["ml"])
        except (KeyError, TypeError, ValueError):
            continue
        if abs(ml - horizon) <= 0.11:
            candidates.append((abs(ml - horizon), point))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def parse_path(value: Any) -> list[dict]:
    try:
        points = json.loads(value) if isinstance(value, str) else value
    except Exception:
        return []
    cleaned = []
    for point in points or []:
        try:
            ml = float(point["ml"])
            yb = float(point.get("yb", point.get("bc")))
            ya = float(point.get("ya", point.get("ac")))
            volume = float(point.get("v", point.get("volume", 0.0)) or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= ml <= 15 and 0 < yb < ya < 1:
            cleaned.append({"ml": ml, "yb": yb, "ya": ya, "v": volume})
    return sorted(cleaned, key=lambda point: -point["ml"])


def spot_maps() -> dict[str, dict[str, dict[int, float]]]:
    spot = pd.read_parquet(DATA / "spot_1m.parquet")
    required = {"coin", "ts", "open", "high", "low", "close", "vol"}
    missing = required - set(spot.columns)
    if missing:
        raise RuntimeError(f"spot_1m missing columns: {sorted(missing)}")
    maps: dict[str, dict[str, dict[int, float]]] = {}
    for coin, group in spot.groupby("coin"):
        maps[str(coin)] = {
            column: dict(zip(group["ts"].astype(int), group[column].astype(float)))
            for column in ["open", "high", "low", "close", "vol"]
        }
    return maps


def build_feature_panel() -> pd.DataFrame:
    paths = pd.read_parquet(DATA / "paths_full.parquet")
    underlying = pd.read_parquet(
        DATA / "underlying.parquet",
        columns=["ticker", "coin", "wkey", "result", "a0", "a1"],
    )
    underlying = underlying[
        underlying["result"].isin(["yes", "no"])
        & underlying["a0"].notna()
        & underlying["a1"].notna()
        & (underlying["a0"] > 0)
        & (underlying["a1"] > 0)
    ].drop_duplicates("ticker")
    truth = underlying.set_index("ticker").to_dict("index")
    spots = spot_maps()

    rows: list[dict] = []
    for market in paths.itertuples(index=False):
        ticker = str(market.ticker)
        meta = truth.get(ticker)
        if meta is None:
            continue
        coin = str(market.coin)
        if coin not in spots:
            continue
        close_ts = int(market.close_ts)
        close_dt = pd.to_datetime(close_ts, unit="s", utc=True)
        open_ts = close_ts - 15 * 60
        spot = spots[coin]
        p0 = spot["close"].get(open_ts - 60)
        if p0 is None or not np.isfinite(p0) or p0 <= 0:
            continue

        points = parse_path(market.path)
        if not points:
            continue
        by_integer = {h: nearest_integer_point(points, h) for h in range(16)}
        by_h = {h: by_integer.get(h) for h in HORIZONS}
        mids = {
            h: ((point["yb"] + point["ya"]) / 2.0
                if point is not None else np.nan)
            for h, point in by_integer.items()
        }

        true_yes = int(str(meta["result"]) == "yes")
        true_return = math.log(float(meta["a1"]) / float(meta["a0"]))

        for horizon in HORIZONS:
            point = by_h[horizon]
            if point is None:
                continue
            decision_ts = close_ts - horizon * 60
            current_key = decision_ts - 60
            pt = spot["close"].get(current_key)
            if pt is None or not np.isfinite(pt) or pt <= 0:
                continue

            bar_keys = list(range(open_ts - 60, current_key + 1, 60))
            closes = np.array([spot["close"].get(key, np.nan) for key in bar_keys], dtype=float)
            highs = np.array([spot["high"].get(key, np.nan) for key in bar_keys[1:]], dtype=float)
            lows = np.array([spot["low"].get(key, np.nan) for key in bar_keys[1:]], dtype=float)
            volumes = np.array([spot["vol"].get(key, np.nan) for key in bar_keys[1:]], dtype=float)
            if np.isnan(closes).any() or len(closes) < 2:
                continue
            log_returns = np.diff(np.log(closes))
            rv = float(np.std(log_returns, ddof=1)) if len(log_returns) >= 2 else np.nan
            if not np.isfinite(rv) or rv <= 0:
                rv = max(float(np.std(log_returns, ddof=0)), 1e-8)
            current_return = math.log(float(pt) / float(p0))
            forecast_scale = max(rv * math.sqrt(max(horizon, 1)), 1e-6)
            z_distance = current_return / forecast_scale

            def momentum(minutes: int) -> float:
                key = current_key - minutes * 60
                old = spot["close"].get(key)
                if old is None or not np.isfinite(old) or old <= 0:
                    return np.nan
                return math.log(float(pt) / float(old))

            mom1 = momentum(1)
            mom3 = momentum(3)
            mom5 = momentum(5)
            yb = float(point["yb"])
            ya = float(point["ya"])
            mid = (yb + ya) / 2.0
            spread = ya - yb
            prev1 = mids.get(horizon + 1, np.nan)
            prev2 = mids.get(horizon + 2, np.nan)
            quote_mom1 = mid - prev1 if np.isfinite(prev1) else np.nan
            quote_mom2 = mid - prev2 if np.isfinite(prev2) else np.nan
            high_value = float(np.nanmax(highs)) if np.isfinite(highs).any() else float(pt)
            low_value = float(np.nanmin(lows)) if np.isfinite(lows).any() else float(pt)
            range_log = math.log(high_value / low_value) if high_value > 0 and low_value > 0 else 0.0
            volume_sum = float(np.nansum(volumes)) if len(volumes) else 0.0

            rows.append({
                "ticker": ticker,
                "coin": coin,
                "close_ts": close_ts,
                "close_dt": close_dt,
                "day": close_dt.date().isoformat(),
                "week": f"{close_dt.isocalendar().year}-{close_dt.isocalendar().week:02d}",
                "hour": close_dt.hour,
                "close_minute": close_dt.minute,
                "horizon": horizon,
                "y": true_yes,
                "true_return_bp": true_return * 1e4,
                "yes_bid": yb,
                "yes_ask": ya,
                "no_bid": 1.0 - ya,
                "no_ask": 1.0 - yb,
                "mid_yes": mid,
                "market_logit": float(logit(clip_prob(mid))),
                "spread_c": spread * 100.0,
                "spot_return_bp": current_return * 1e4,
                "z_distance": z_distance,
                "rv_bp": rv * 1e4,
                "range_bp": range_log * 1e4,
                "mom1_bp": mom1 * 1e4 if np.isfinite(mom1) else np.nan,
                "mom3_bp": mom3 * 1e4 if np.isfinite(mom3) else np.nan,
                "mom5_bp": mom5 * 1e4 if np.isfinite(mom5) else np.nan,
                "quote_mom1_c": quote_mom1 * 100.0 if np.isfinite(quote_mom1) else np.nan,
                "quote_mom2_c": quote_mom2 * 100.0 if np.isfinite(quote_mom2) else np.nan,
                "path_volume_log": math.log1p(max(float(point["v"]), 0.0)),
                "spot_volume_log": math.log1p(max(volume_sum, 0.0)),
                "hour_sin": math.sin(2.0 * math.pi * close_dt.hour / 24.0),
                "hour_cos": math.cos(2.0 * math.pi * close_dt.hour / 24.0),
            })

    panel = pd.DataFrame(rows)
    if panel.empty:
        raise RuntimeError("feature panel is empty")
    group = panel.groupby(["close_ts", "horizon"], sort=False)
    panel["common_spot_bp"] = group["spot_return_bp"].transform("mean")
    panel["cross_dispersion_bp"] = group["spot_return_bp"].transform("std").fillna(0.0)
    panel["coin_residual_bp"] = panel["spot_return_bp"] - panel["common_spot_bp"]
    panel["common_mid"] = group["mid_yes"].transform("mean")
    panel["market_residual"] = panel["mid_yes"] - panel["common_mid"]
    panel["same_sign_share"] = group["spot_return_bp"].transform(
        lambda x: np.mean(np.sign(x) == np.sign(np.mean(x))))
    panel["coin_rank_return"] = group["spot_return_bp"].rank(pct=True)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel.to_parquet(OUT / "spe_feature_panel.parquet", index=False)
    return panel


def physical_probabilities(frame: pd.DataFrame, horizon: int) -> pd.Series:
    """Gaussian nowcast estimated on TRAIN only.

    Residual = exact CF final return minus causal Coinbase return known at the
    decision time. Coin moments shrink toward pooled moments.
    """
    train = frame[split_mask(frame, "train") & frame["horizon"].eq(horizon)].copy()
    residual = train["true_return_bp"] - train["spot_return_bp"]
    pooled_mu = float(residual.mean())
    pooled_sd = float(residual.std(ddof=1))
    if not np.isfinite(pooled_sd) or pooled_sd <= 1e-6:
        pooled_sd = 1.0
    params: dict[str, tuple[float, float]] = {}
    for coin in COINS:
        values = residual[train["coin"].eq(coin)].dropna()
        n = len(values)
        weight = n / (n + 200.0)
        mu = weight * float(values.mean()) + (1.0 - weight) * pooled_mu if n else pooled_mu
        sd_raw = float(values.std(ddof=1)) if n >= 2 else pooled_sd
        sd = math.sqrt(max(weight * sd_raw**2 + (1.0 - weight) * pooled_sd**2, 1e-6))
        params[coin] = (mu, sd)
    subset = frame[frame["horizon"].eq(horizon)]
    output = []
    for row in subset.itertuples(index=False):
        mu, sd = params.get(str(row.coin), (pooled_mu, pooled_sd))
        output.append(float(norm.cdf((float(row.spot_return_bp) + mu) / sd)))
    return pd.Series(output, index=subset.index)


class SimpleMedianImputer(BaseEstimator, TransformerMixin):
    def fit(self, x: Any, y: Any = None) -> "SimpleMedianImputer":
        array = np.asarray(x, dtype=float)
        medians = np.nanmedian(array, axis=0)
        medians[~np.isfinite(medians)] = 0.0
        self.medians_ = medians
        return self

    def transform(self, x: Any) -> np.ndarray:
        array = np.asarray(x, dtype=float).copy()
        bad = ~np.isfinite(array)
        if bad.any():
            array[bad] = np.take(self.medians_, np.where(bad)[1])
        return array


MARKET_NUMERIC = ["market_logit"]
BLEND_NUMERIC = ["market_logit", "physical_logit"]
STRUCT_NUMERIC = [
    "market_logit", "physical_logit", "spot_return_bp", "z_distance",
    "rv_bp", "range_bp", "mom1_bp", "mom3_bp", "mom5_bp", "spread_c",
    "quote_mom1_c", "quote_mom2_c", "path_volume_log", "spot_volume_log",
    "common_spot_bp", "cross_dispersion_bp", "coin_residual_bp", "common_mid",
    "market_residual", "same_sign_share", "coin_rank_return", "hour_sin",
    "hour_cos",
]
CATEGORICAL = ["coin", "close_minute"]


def make_pipeline(numeric: list[str], c_value: float) -> Pipeline:
    preprocess = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleMedianImputer()),
            ("scale", StandardScaler()),
        ]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ], remainder="drop")
    model = LogisticRegression(
        C=c_value, penalty="l2", solver="liblinear", max_iter=1000,
        random_state=2026080617)
    return Pipeline([("prep", preprocess), ("model", model)])


def calibration_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    lp = logit(clip_prob(p)).reshape(-1, 1)
    try:
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        model.fit(lp, y)
        return float(model.intercept_[0]), float(model.coef_[0, 0])
    except Exception:
        return np.nan, np.nan


def probability_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    intercept, slope = calibration_slope(y, p)
    return {
        "n": int(len(y)),
        "log_loss": float(log_loss(y, clip_prob(p))),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def train_models(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    predictions = []
    metrics = []
    chosen_c: dict[str, dict[int, float]] = {name: {} for name in MODELS}
    for horizon in HORIZONS:
        idx = panel["horizon"].eq(horizon)
        panel.loc[idx, "p_physical"] = physical_probabilities(panel, horizon)
    panel["physical_logit"] = logit(clip_prob(panel["p_physical"].astype(float)))
    feature_sets = {"market": MARKET_NUMERIC, "blend": BLEND_NUMERIC, "struct": STRUCT_NUMERIC}

    for horizon in HORIZONS:
        data = panel[panel["horizon"].eq(horizon)].copy()
        train = data[split_mask(data, "train")]
        valid = data[split_mask(data, "valid")]
        test = data[split_mask(data, "test")]
        if min(len(train), len(valid), len(test)) < 100:
            continue
        for model_name, features in feature_sets.items():
            best_pipe = None
            best_c = None
            best_loss = np.inf
            for c_value in C_GRID:
                pipe = make_pipeline(features, c_value)
                pipe.fit(train[features + CATEGORICAL], train["y"])
                p_valid = pipe.predict_proba(valid[features + CATEGORICAL])[:, 1]
                score = log_loss(valid["y"], clip_prob(p_valid))
                if score < best_loss:
                    best_loss, best_pipe, best_c = score, pipe, c_value
            assert best_pipe is not None and best_c is not None
            chosen_c[model_name][horizon] = float(best_c)
            for split_name, frame in [("train", train), ("valid", valid), ("test", test)]:
                p = best_pipe.predict_proba(frame[features + CATEGORICAL])[:, 1]
                block = frame[[
                    "ticker", "coin", "close_ts", "close_dt", "day", "week",
                    "hour", "close_minute", "horizon", "y", "yes_bid", "yes_ask",
                    "no_bid", "no_ask", "mid_yes", "p_physical",
                ]].copy()
                block["model"] = model_name
                block["split"] = split_name
                block["p_yes"] = p
                block["residual_vs_mid"] = p - block["mid_yes"]
                predictions.append(block)
                metrics.append({
                    "horizon": horizon, "model": model_name, "split": split_name,
                    "c": float(best_c), **probability_metrics(frame["y"].to_numpy(), p),
                })
        for split_name, frame in [("train", train), ("valid", valid), ("test", test)]:
            for model_name, p in [("raw_mid", frame["mid_yes"].to_numpy()),
                                  ("physical", frame["p_physical"].to_numpy())]:
                metrics.append({
                    "horizon": horizon, "model": model_name, "split": split_name,
                    "c": np.nan, **probability_metrics(frame["y"].to_numpy(), p),
                })
    pred = pd.concat(predictions, ignore_index=True)
    met = pd.DataFrame(metrics)
    pred.to_parquet(OUT / "spe_predictions.parquet", index=False)
    met.to_csv(OUT / "spe_probability_metrics.csv", index=False)
    return pred, met, chosen_c


def add_trade_values(pred: pd.DataFrame, qty: int = QTY) -> pd.DataFrame:
    x = pred.copy()
    yes_fee = np.array([fee_total(qty, p) / qty for p in x["yes_ask"]])
    no_fee = np.array([fee_total(qty, p) / qty for p in x["no_ask"]])
    x["yes_allin"] = x["yes_ask"] + yes_fee
    x["no_allin"] = x["no_ask"] + no_fee
    x["pred_edge_yes"] = x["p_yes"] - x["yes_allin"]
    x["pred_edge_no"] = (1.0 - x["p_yes"]) - x["no_allin"]
    x["trade_side"] = np.where(x["pred_edge_yes"] >= x["pred_edge_no"], "yes", "no")
    x["pred_edge"] = np.maximum(x["pred_edge_yes"], x["pred_edge_no"])
    x["trade_price"] = np.where(x["trade_side"].eq("yes"), x["yes_ask"], x["no_ask"])
    x["trade_fee"] = np.where(x["trade_side"].eq("yes"), yes_fee, no_fee)
    x["trade_won"] = np.where(x["trade_side"].eq("yes"), x["y"], 1 - x["y"])
    x["pnl"] = qty * (x["trade_won"] - x["trade_price"]) - qty * x["trade_fee"]
    return x


@dataclass
class PolicyMetrics:
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
    avg_price: float
    max_drawdown: float
    worst_window: float
    positive_day_fraction: float


def choose_one_per_window(data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    eligible = data[data["pred_edge"] >= threshold].copy()
    if eligible.empty:
        return eligible
    return (eligible.sort_values(
        ["close_ts", "pred_edge", "trade_price", "coin"],
        ascending=[True, False, True, True])
        .groupby("close_ts", as_index=False).head(1).sort_values("close_ts"))


def policy_metrics(selected: pd.DataFrame, model: str, horizon: int,
                   threshold: float, split: str) -> PolicyMetrics:
    if selected.empty:
        return PolicyMetrics(model, horizon, threshold, split, 0, 0, 0, 0, 0,
                             0, 0, np.nan, np.nan, 0, 0, 0)
    daily = selected.groupby("day")["pnl"].sum()
    all_days = pd.date_range(
        SPLITS[split][0].normalize(),
        SPLITS[split][1].normalize() - pd.Timedelta(days=1), freq="D").date
    daily = daily.reindex([d.isoformat() for d in all_days], fill_value=0.0)
    window = selected.groupby("close_ts")["pnl"].sum().sort_index()
    cumulative = window.cumsum()
    max_dd = float((cumulative.cummax() - cumulative).max())
    sd = float(daily.std(ddof=1))
    mean = float(daily.mean())
    return PolicyMetrics(
        model=model, horizon=horizon, threshold=threshold, split=split,
        n=int(len(selected)), days=int(len(daily)),
        total_pnl=float(selected["pnl"].sum()), mean_day=mean, sd_day=sd,
        t_stat=mean / (sd / math.sqrt(len(daily))) if sd > 0 else 0.0,
        edge_per_contract=float(selected["pnl"].sum() / (len(selected) * QTY)),
        win_rate=float(selected["trade_won"].mean()),
        avg_price=float(selected["trade_price"].mean()), max_drawdown=max_dd,
        worst_window=float(window.min()),
        positive_day_fraction=float((daily > 0).mean()))


def select_validation_policy(pred: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    trade = add_trade_values(pred)
    rows = []
    candidates = trade[trade["split"].eq("valid") & trade["model"].isin(MODELS)]
    for (model, horizon), group in candidates.groupby(["model", "horizon"]):
        for threshold in EDGE_THRESHOLDS:
            rows.append(asdict(policy_metrics(
                choose_one_per_window(group, threshold), model, int(horizon),
                threshold, "valid")))
    table = pd.DataFrame(rows)
    viable = table[(table["n"] >= 50) & (table["total_pnl"] > 0)].copy()
    if viable.empty:
        viable = table[table["n"] >= 25].copy()
    if viable.empty:
        viable = table.copy()
    viable["objective"] = (
        viable["t_stat"] + 0.25 * np.log1p(viable["n"])
        - 0.02 * viable["max_drawdown"])
    winner = viable.sort_values(
        ["objective", "t_stat", "total_pnl"], ascending=False).iloc[0].to_dict()
    table.to_csv(OUT / "spe_validation_policy_grid.csv", index=False)
    return winner, trade


def block_bootstrap(selected: pd.DataFrame, repetitions: int = 12000) -> list[dict]:
    if selected.empty:
        return []
    start = int(selected["close_ts"].min())
    results = []
    for minutes in [15, 30, 60, 90]:
        block_id = ((selected["close_ts"] - start) // (minutes * 60)).astype(int)
        values = selected.assign(block=block_id).groupby("block")["pnl"].sum().to_numpy()
        if len(values) < 2:
            continue
        draws = RNG.integers(0, len(values), size=(repetitions, len(values)))
        totals = values[draws].sum(axis=1)
        results.append({
            "block_minutes": minutes, "n_blocks": int(len(values)),
            "observed_total": float(values.sum()),
            "ci_lo": float(np.quantile(totals, 0.025)),
            "ci_hi": float(np.quantile(totals, 0.975)),
            "p_nonpositive": float(np.mean(totals <= 0)),
        })
    return results


def day_bootstrap(selected: pd.DataFrame, split: str,
                  repetitions: int = 12000) -> dict:
    all_days = pd.date_range(
        SPLITS[split][0].normalize(),
        SPLITS[split][1].normalize() - pd.Timedelta(days=1), freq="D").date
    daily = selected.groupby("day")["pnl"].sum().reindex(
        [d.isoformat() for d in all_days], fill_value=0.0).to_numpy()
    draws = RNG.integers(0, len(daily), size=(repetitions, len(daily)))
    means = daily[draws].mean(axis=1)
    return {
        "observed_mean_day": float(daily.mean()),
        "ci_lo": float(np.quantile(means, 0.025)),
        "ci_hi": float(np.quantile(means, 0.975)),
        "p_nonpositive": float(np.mean(means <= 0)),
    }


def robustness(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    coin_rows = []
    for coin in COINS:
        subset = selected[~selected["coin"].eq(coin)]
        coin_rows.append({
            "excluded_coin": coin, "n": int(len(subset)),
            "total_pnl": float(subset["pnl"].sum()),
            "edge_per_contract": float(subset["pnl"].mean() / QTY) if len(subset) else np.nan,
        })
    week_rows = []
    for week in sorted(selected["week"].unique()):
        subset = selected[~selected["week"].eq(week)]
        week_rows.append({
            "excluded_week": week, "n": int(len(subset)),
            "total_pnl": float(subset["pnl"].sum()),
            "edge_per_contract": float(subset["pnl"].mean() / QTY) if len(subset) else np.nan,
        })
    return pd.DataFrame(coin_rows), pd.DataFrame(week_rows)


def residual_deciles(test_rows: pd.DataFrame) -> pd.DataFrame:
    x = test_rows.copy()
    if len(x) < 20:
        return pd.DataFrame()
    x["residual_decile"] = pd.qcut(
        x["residual_vs_mid"].rank(method="first"), 10, labels=False)
    return (x.groupby("residual_decile").agg(
        n=("y", "size"), mean_mid=("mid_yes", "mean"),
        mean_model=("p_yes", "mean"), actual_yes=("y", "mean"),
        residual=("residual_vs_mid", "mean")).reset_index())


def report(panel: pd.DataFrame, prob_metrics: pd.DataFrame, chosen_c: dict,
           winner: dict, trade: pd.DataFrame) -> dict:
    model = str(winner["model"])
    horizon = int(winner["horizon"])
    threshold = float(winner["threshold"])
    final_rows = trade[
        trade["split"].eq("test") & trade["model"].eq(model)
        & trade["horizon"].eq(horizon)].copy()
    selected = choose_one_per_window(final_rows, threshold)
    final_metric = policy_metrics(selected, model, horizon, threshold, "test")
    block = block_bootstrap(selected)
    day = day_bootstrap(selected, "test")
    coins, weeks = robustness(selected)
    deciles = residual_deciles(final_rows)
    test_probability = prob_metrics[
        prob_metrics["split"].eq("test")
        & prob_metrics["horizon"].eq(horizon)].copy()
    market_row = test_probability[test_probability["model"].eq("market")]
    chosen_row = test_probability[test_probability["model"].eq(model)]
    raw_row = test_probability[test_probability["model"].eq("raw_mid")]
    model_delta: dict[str, float] = {}
    if len(chosen_row) and len(raw_row):
        model_delta["log_loss_vs_raw_mid"] = float(
            chosen_row.iloc[0]["log_loss"] - raw_row.iloc[0]["log_loss"])
        model_delta["brier_vs_raw_mid"] = float(
            chosen_row.iloc[0]["brier"] - raw_row.iloc[0]["brier"])
    if len(chosen_row) and len(market_row):
        model_delta["log_loss_vs_market_calibrator"] = float(
            chosen_row.iloc[0]["log_loss"] - market_row.iloc[0]["log_loss"])
        model_delta["brier_vs_market_calibrator"] = float(
            chosen_row.iloc[0]["brier"] - market_row.iloc[0]["brier"])

    coin_pass = bool(len(coins) and (coins["total_pnl"] > 0).all())
    week_pass = bool(len(weeks) and (weeks["total_pnl"] > 0).all())
    day_ci_pass = bool(day["ci_lo"] > 0)
    block_pass = bool(block and min(item["ci_lo"] for item in block) > 0)
    model_pass = bool(model_delta.get("log_loss_vs_raw_mid", 1.0) < 0)
    economic_pass = bool(final_metric.total_pnl > 0 and final_metric.n >= 50)
    hard_pass = all([model_pass, economic_pass, day_ci_pass, block_pass,
                     coin_pass, week_pass])

    coins.to_csv(OUT / "spe_test_leave_one_coin.csv", index=False)
    weeks.to_csv(OUT / "spe_test_leave_one_week.csv", index=False)
    deciles.to_csv(OUT / "spe_test_residual_deciles.csv", index=False)
    selected.to_parquet(OUT / "spe_test_selected_trades.parquet", index=False)

    summary = {
        "data": {"panel_rows": int(len(panel)),
                 "markets": int(panel["ticker"].nunique()),
                 "days": int(panel["day"].nunique()), "horizons": HORIZONS},
        "validation_winner": winner,
        "chosen_regularization": chosen_c,
        "test_probability_metrics": test_probability.to_dict("records"),
        "model_increment": model_delta,
        "test_policy": asdict(final_metric),
        "day_bootstrap": day,
        "block_bootstrap": block,
        "leave_one_coin_all_positive": coin_pass,
        "leave_one_week_all_positive": week_pass,
        "hard_pass": hard_pass,
        "pass_components": {
            "probability_increment": model_pass,
            "positive_test_economics_and_n_ge_50": economic_pass,
            "positive_day_ci": day_ci_pass,
            "positive_all_block_cis": block_pass,
            "leave_one_coin": coin_pass,
            "leave_one_week": week_pass,
        },
    }

    verdict = "PASS" if hard_pass else "FAIL / CANDIDATE ONLY"
    lines = [
        "# Settlement Probability Engine — clean-room result", "",
        f"## Verdict: **{verdict}**", "",
        "The model and economic policy were selected on the chronological validation",
        "slice. The final test slice was evaluated once.", "", "## Data", "",
        f"- Feature rows: {len(panel):,}",
        f"- Unique markets: {panel['ticker'].nunique():,}",
        f"- Calendar days: {panel['day'].nunique()}",
        f"- Fixed horizons: {HORIZONS}", "", "## Validation-selected policy", "",
        f"- Model: `{model}`", f"- Horizon: {horizon} minutes remaining",
        f"- Minimum predicted all-in edge: {threshold*100:.1f}¢",
        f"- Validation opportunities: {int(winner['n'])}",
        f"- Validation P&L: ${float(winner['total_pnl']):.2f}",
        f"- Validation mean/day: ${float(winner['mean_day']):.2f}", "",
        "## Sealed test policy", "", f"- Trades: {final_metric.n}",
        f"- Total P&L at q{QTY}: ${final_metric.total_pnl:.2f}",
        f"- Mean/day: ${final_metric.mean_day:.2f}",
        f"- Daily SD: ${final_metric.sd_day:.2f}",
        f"- Edge/contract: {final_metric.edge_per_contract*100:+.2f}¢",
        f"- Win rate: {final_metric.win_rate:.4f}",
        f"- Average entry: {final_metric.avg_price*100:.2f}¢",
        f"- Maximum drawdown: ${final_metric.max_drawdown:.2f}",
        f"- Worst close window: ${final_metric.worst_window:.2f}", "",
        "## Probability increment on sealed test", "",
        f"- Δ log loss vs raw midpoint: {model_delta.get('log_loss_vs_raw_mid', np.nan):+.6f}",
        f"- Δ Brier vs raw midpoint: {model_delta.get('brier_vs_raw_mid', np.nan):+.6f}",
        f"- Δ log loss vs market-only calibrator: {model_delta.get('log_loss_vs_market_calibrator', np.nan):+.6f}",
        "", "Negative deltas are improvements.", "", "## Uncertainty", "",
        f"- Day-bootstrap mean/day 95% interval: [${day['ci_lo']:.2f}, ${day['ci_hi']:.2f}]",
        f"- P(mean day ≤ 0): {day['p_nonpositive']:.4f}", "",
        "| Block | CI on total P&L | P(nonpositive) |", "|---:|---:|---:|",
    ]
    for item in block:
        lines.append(
            f"| {item['block_minutes']} min | [${item['ci_lo']:.2f}, "
            f"${item['ci_hi']:.2f}] | {item['p_nonpositive']:.4f} |")
    lines.extend(["", "## Hard-gate components", ""])
    for key, value in summary["pass_components"].items():
        lines.append(f"- {key}: **{'PASS' if value else 'FAIL'}**")
    lines.extend(["", "## Interpretation", "",
        "A probability model can be useful even when the final trading gate fails.",
        "The key result is whether causal spot/settlement state improves sealed-test",
        "calibration beyond the contemporaneous quote. A positive validation",
        "backtest without that calibration increment is threshold search, not a",
        "scientific breakthrough.", "",
        "No result authorizes live orders while the repository's existing account",
        "KILL state remains active."])
    (OUT / "spe_report.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "spe_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def main() -> None:
    panel = build_feature_panel()
    predictions, metrics, chosen_c = train_models(panel)
    winner, trade = select_validation_policy(predictions)
    trade.to_parquet(OUT / "spe_trade_values.parquet", index=False)
    summary = report(panel, metrics, chosen_c, winner, trade)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
