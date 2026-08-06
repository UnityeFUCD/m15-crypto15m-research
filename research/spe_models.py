"""Chronological probability models for SPE."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from spe_data import (
    HORIZONS, OUT, SEED, clip_probability,
    empirical_physical_probability, split_mask,
)

C_GRID = [0.01, 0.1, 1.0, 10.0]
MODEL_NAMES = ["market", "blend", "full"]
CATEGORICAL = ["coin", "close_minute"]
FEATURES = {
    "market": ["market_logit"],
    "blend": [
        "market_logit", "physical_logit", "market_physical_gap", "spread_c"
    ],
    "full": [
        "market_logit", "physical_logit", "market_physical_gap",
        "spot_return_bp", "distance_z", "realized_vol_bp", "range_bp",
        "mom1_bp", "mom3_bp", "mom5_bp", "spread_c", "quote_mom1_c",
        "quote_mom2_c", "path_volume_log", "spot_volume_log",
        "common_spot_bp", "cross_dispersion_bp", "coin_residual_bp",
        "common_mid", "market_residual", "same_sign_share",
        "coin_return_rank", "hour_sin", "hour_cos",
    ],
}


def pipeline(features: list[str], c_value: float) -> Pipeline:
    transform = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    model = LogisticRegression(
        C=c_value, penalty="l2", solver="liblinear",
        max_iter=1500, random_state=SEED,
    )
    return Pipeline([("transform", transform), ("model", model)])


def calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    try:
        m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        m.fit(logit(clip_probability(p)).reshape(-1, 1), y)
        return float(m.intercept_[0]), float(m.coef_[0, 0])
    except Exception:
        return np.nan, np.nan


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    intercept, slope = calibration(y, p)
    return {
        "n": int(len(y)),
        "log_loss": float(log_loss(y, clip_probability(p))),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def train_models(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    for horizon in HORIZONS:
        idx = panel.horizon.eq(horizon)
        panel.loc[idx, "p_physical"] = empirical_physical_probability(panel, horizon)
    panel["physical_logit"] = logit(clip_probability(panel.p_physical.astype(float)))
    panel["market_physical_gap"] = panel.p_physical - panel.mid_yes

    predictions, metric_rows = [], []
    chosen_c = {name: {} for name in MODEL_NAMES}

    for horizon in HORIZONS:
        data = panel[panel.horizon.eq(horizon)].copy()
        split = {name: data[split_mask(data, name)] for name in ["train", "valid", "test"]}
        if min(map(len, split.values())) < 100:
            continue

        for name, features in FEATURES.items():
            best, best_c, best_loss = None, None, np.inf
            for c_value in C_GRID:
                candidate = pipeline(features, c_value)
                candidate.fit(
                    split["train"][features + CATEGORICAL],
                    split["train"].y,
                )
                p = candidate.predict_proba(
                    split["valid"][features + CATEGORICAL]
                )[:, 1]
                score = log_loss(split["valid"].y, clip_probability(p))
                if score < best_loss:
                    best, best_c, best_loss = candidate, c_value, score
            if best is None:
                raise RuntimeError("model selection failed")
            chosen_c[name][horizon] = float(best_c)

            for split_name, frame in split.items():
                p = best.predict_proba(frame[features + CATEGORICAL])[:, 1]
                block = frame[[
                    "ticker", "coin", "close_ts", "close_dt", "day", "week",
                    "hour", "close_minute", "horizon", "y", "yes_bid",
                    "yes_ask", "no_bid", "no_ask", "mid_yes", "p_physical",
                ]].copy()
                block["model"] = name
                block["split"] = split_name
                block["p_yes"] = p
                block["residual_vs_mid"] = p - block.mid_yes
                predictions.append(block)
                metric_rows.append({
                    "horizon": horizon, "model": name, "split": split_name,
                    "c": float(best_c), **metrics(frame.y.to_numpy(), p),
                })

        for split_name, frame in split.items():
            for name, p in [
                ("raw_mid", frame.mid_yes.to_numpy()),
                ("physical", frame.p_physical.to_numpy()),
            ]:
                metric_rows.append({
                    "horizon": horizon, "model": name, "split": split_name,
                    "c": np.nan, **metrics(frame.y.to_numpy(), p),
                })

    predictions = pd.concat(predictions, ignore_index=True)
    metric_table = pd.DataFrame(metric_rows)
    predictions.to_parquet(OUT / "spe_predictions.parquet", index=False)
    metric_table.to_csv(OUT / "spe_probability_metrics.csv", index=False)
    return predictions, metric_table, chosen_c
