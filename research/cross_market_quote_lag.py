"""Cross-market quote-lag and relative-value audit.

Frozen design: see research/cmql_preregistration.md.

This script is research-only. It never connects to the exchange and never
places orders.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "research" / "results" / "cmql"
OUT.mkdir(parents=True, exist_ok=True)

QTY = 15
DECISION_MLS = (12, 11, 10)
EDGE_THRESHOLDS = (0.02, 0.04, 0.06, 0.08)
SEED = 20260806
EPS = 1e-6

BASE_NUM = ["cur_logit", "spread", "log_volume", "ml"]
OWN_NUM = BASE_NUM + ["d1", "d2", "abs_d1", "accel"]
PEER_NUM = OWN_NUM + [
    "peer_d1",
    "peer_d2",
    "gap_d1",
    "peer_level",
    "level_gap",
    "peer_dispersion",
    "peer_breadth",
]
CAT = ["coin", "minute"]


def fee_total(qty: int, p: float) -> float:
    raw = 0.07 * qty * p * (1.0 - p)
    return math.ceil(raw * 10_000.0 - 1e-12) / 10_000.0


def clamp_prob(x: float) -> float:
    return float(np.clip(x, EPS, 1.0 - EPS))


def logit(x: float) -> float:
    p = clamp_prob(x)
    return math.log(p / (1.0 - p))


def normalize_close(values: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values):
        out = pd.to_datetime(values, utc=True)
    else:
        numeric = pd.to_numeric(values, errors="coerce")
        med = float(numeric.dropna().abs().median())
        if med > 1e17:
            unit = "ns"
        elif med > 1e14:
            unit = "us"
        elif med > 1e11:
            unit = "ms"
        else:
            unit = "s"
        out = pd.to_datetime(numeric, unit=unit, utc=True)
    years = out.dt.year.dropna()
    if years.empty or years.min() < 2020 or years.max() > 2035:
        raise RuntimeError("implausible close timestamps")
    return out


def build_panel() -> pd.DataFrame:
    source = pd.read_parquet(DATA / "paths_full.parquet")
    required = {
        "ticker",
        "coin",
        "close_ts",
        "minute",
        "side",
        "won",
        "path",
    }
    missing = required - set(source.columns)
    if missing:
        raise RuntimeError(f"paths_full missing columns: {sorted(missing)}")

    source = source.copy()
    source["close_dt"] = normalize_close(source["close_ts"])
    rows: list[dict] = []
    needed_mls = set(range(8, 15))

    for market in source.itertuples(index=False):
        try:
            path = json.loads(market.path) if isinstance(market.path, str) else market.path
        except Exception:
            continue
        by_ml: dict[int, dict] = {}
        for point in path:
            try:
                ml_raw = float(point["ml"])
                ml = int(round(ml_raw))
                if ml not in needed_mls or abs(ml_raw - ml) > 0.08:
                    continue
                yb = float(point["yb"])
                ya = float(point["ya"])
                if not (0.0 <= yb < ya <= 1.0):
                    continue
                volume = float(point.get("v", 0.0) or 0.0)
            except (KeyError, TypeError, ValueError):
                continue
            # One complete quote close per minute-left. If duplicate, retain the
            # last one as stored in the path.
            by_ml[ml] = {"yb": yb, "ya": ya, "volume": max(volume, 0.0)}

        side = str(market.side).lower()
        won_fav = int(market.won)
        if side not in {"yes", "no"} or won_fav not in {0, 1}:
            continue
        yes_won = won_fav if side == "yes" else 1 - won_fav

        for ml in DECISION_MLS:
            required_mls = (ml + 2, ml + 1, ml, ml - 1)
            if any(x not in by_ml for x in required_mls):
                continue
            pts = [by_ml[x] for x in required_mls]
            mids = [clamp_prob((p["yb"] + p["ya"]) / 2.0) for p in pts]
            logits = [logit(x) for x in mids]
            old2, old1, cur, nxt = logits
            pcur = by_ml[ml]
            d1 = cur - old1
            d2 = old1 - old2
            rows.append(
                {
                    "ticker": str(market.ticker),
                    "coin": str(market.coin),
                    "close_dt": market.close_dt,
                    "close_ts": int(pd.Timestamp(market.close_dt).timestamp()),
                    "date": market.close_dt.date(),
                    "week": market.close_dt.strftime("%G-W%V"),
                    "minute": int(market.minute),
                    "ml": int(ml),
                    "yb": float(pcur["yb"]),
                    "ya": float(pcur["ya"]),
                    "ask_yes": float(pcur["ya"]),
                    "ask_no": float(1.0 - pcur["yb"]),
                    "mid": mids[2],
                    "cur_logit": cur,
                    "spread": float(pcur["ya"] - pcur["yb"]),
                    "volume": float(pcur["volume"]),
                    "log_volume": math.log1p(float(pcur["volume"])),
                    "d1": d1,
                    "d2": d2,
                    "abs_d1": abs(d1),
                    "accel": d1 - d2,
                    "next_dlogit": nxt - cur,
                    "yes_won": int(yes_won),
                }
            )

    panel = pd.DataFrame(rows)
    if panel.empty:
        raise RuntimeError("no synchronized path observations")

    # Require all six original coins at the same close and decision minute.
    group_cols = ["close_ts", "ml"]
    counts = panel.groupby(group_cols)["coin"].transform("nunique")
    panel = panel[counts == 6].copy()

    # Peer quantities, leave-one-coin-out within the synchronous window.
    for col in ["d1", "d2", "cur_logit"]:
        gsum = panel.groupby(group_cols)[col].transform("sum")
        panel[f"peer_{col}"] = (gsum - panel[col]) / 5.0
    panel["peer_d1"] = panel.pop("peer_d1")
    panel["peer_d2"] = panel.pop("peer_d2")
    panel["peer_level"] = panel.pop("peer_cur_logit")
    panel["gap_d1"] = panel["peer_d1"] - panel["d1"]
    panel["level_gap"] = panel["peer_level"] - panel["cur_logit"]

    def peer_dispersion(group: pd.DataFrame) -> pd.Series:
        vals = group["d1"].to_numpy(float)
        out = []
        for idx in range(len(vals)):
            out.append(float(np.std(np.delete(vals, idx), ddof=0)))
        return pd.Series(out, index=group.index)

    def peer_breadth(group: pd.DataFrame) -> pd.Series:
        vals = group["d1"].to_numpy(float)
        out = []
        for idx in range(len(vals)):
            peers = np.delete(vals, idx)
            out.append(float(np.mean(peers > 0.0)))
        return pd.Series(out, index=group.index)

    panel["peer_dispersion"] = (
        panel.groupby(group_cols, group_keys=False).apply(peer_dispersion)
    )
    panel["peer_breadth"] = (
        panel.groupby(group_cols, group_keys=False).apply(peer_breadth)
    )

    panel.replace([np.inf, -np.inf], np.nan, inplace=True)
    panel.dropna(subset=list(dict.fromkeys(PEER_NUM + ["next_dlogit", "yes_won"])), inplace=True)
    panel.sort_values(["close_dt", "ml", "coin"], inplace=True)
    panel.reset_index(drop=True, inplace=True)
    return panel


def split_dates(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    dates = sorted(panel["date"].unique())
    n = len(dates)
    if n < 20:
        raise RuntimeError(f"only {n} dates")
    i1 = int(math.floor(n * 0.50))
    i2 = int(math.floor(n * 0.75))
    train_dates = set(dates[:i1])
    valid_dates = set(dates[i1:i2])
    test_dates = set(dates[i2:])
    train = panel[panel.date.isin(train_dates)].copy()
    valid = panel[panel.date.isin(valid_dates)].copy()
    test = panel[panel.date.isin(test_dates)].copy()
    meta = {
        "dates": n,
        "train": [str(min(train_dates)), str(max(train_dates)), len(train_dates)],
        "valid": [str(min(valid_dates)), str(max(valid_dates)), len(valid_dates)],
        "test": [str(min(test_dates)), str(max(test_dates)), len(test_dates)],
    }
    return train, valid, test, meta


def design_matrix(
    train: pd.DataFrame,
    other: pd.DataFrame,
    numeric: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    both = pd.concat(
        [train[numeric + CAT], other[numeric + CAT]],
        axis=0,
        ignore_index=True,
    )
    both = pd.get_dummies(both, columns=CAT, dtype=float)
    x_train = both.iloc[: len(train)].copy()
    x_other = both.iloc[len(train) :].copy()
    for col in numeric:
        mean = float(x_train[col].mean())
        sd = float(x_train[col].std(ddof=0))
        if not math.isfinite(sd) or sd < 1e-12:
            sd = 1.0
        x_train[col] = (x_train[col] - mean) / sd
        x_other[col] = (x_other[col] - mean) / sd
    return x_train.to_numpy(float), x_other.to_numpy(float), list(x_train.columns)


def fit_models(train: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> tuple[dict, dict]:
    models: dict[str, dict] = {}
    predictions: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    specs = {"market": BASE_NUM, "own": OWN_NUM, "peer": PEER_NUM}
    for name, numeric in specs.items():
        # Build a single feature basis against concatenated OOS frames so all
        # categories line up. No outcome information enters the design.
        other = pd.concat(list(frames.values()), axis=0, ignore_index=True)
        xtr, xoos, columns = design_matrix(train, other, numeric)
        ridge = Ridge(alpha=10.0, solver="lsqr")
        ridge.fit(xtr, train["next_dlogit"].to_numpy(float))
        logistic = LogisticRegression(
            C=0.20,
            penalty="l2",
            solver="lbfgs",
            max_iter=1500,
            random_state=SEED,
        )
        logistic.fit(xtr, train["yes_won"].to_numpy(int))
        models[name] = {"ridge": ridge, "logistic": logistic, "columns": columns}
        predictions[name] = {}
        offset = 0
        for frame_name, frame in frames.items():
            n = len(frame)
            x = xoos[offset : offset + n]
            offset += n
            predictions[name][frame_name] = {
                "next": ridge.predict(x),
                "p_yes": logistic.predict_proba(x)[:, 1],
            }
    return models, predictions


def info_metrics(frame: pd.DataFrame, pred: dict[str, np.ndarray]) -> dict:
    actual_next = frame["next_dlogit"].to_numpy(float)
    actual_y = frame["yes_won"].to_numpy(int)
    p = np.clip(pred["p_yes"], EPS, 1.0 - EPS)
    return {
        "n": int(len(frame)),
        "next_mse": float(mean_squared_error(actual_next, pred["next"])),
        "next_direction_accuracy": float(np.mean(np.sign(pred["next"]) == np.sign(actual_next))),
        "log_loss": float(log_loss(actual_y, p, labels=[0, 1])),
        "brier": float(np.mean((actual_y - p) ** 2)),
        "auc": float(roc_auc_score(actual_y, p)) if len(np.unique(actual_y)) == 2 else float("nan"),
    }


def add_economics(frame: pd.DataFrame, p_yes: np.ndarray) -> pd.DataFrame:
    out = frame.copy()
    out["pred_p_yes"] = np.clip(p_yes, EPS, 1.0 - EPS)
    fee_yes = np.array([fee_total(QTY, p) / QTY for p in out["ask_yes"]], dtype=float)
    fee_no = np.array([fee_total(QTY, p) / QTY for p in out["ask_no"]], dtype=float)
    edge_yes = out["pred_p_yes"].to_numpy(float) - out["ask_yes"].to_numpy(float) - fee_yes
    edge_no = (1.0 - out["pred_p_yes"].to_numpy(float)) - out["ask_no"].to_numpy(float) - fee_no
    choose_yes = edge_yes >= edge_no
    out["trade_side"] = np.where(choose_yes, "yes", "no")
    out["pred_edge"] = np.where(choose_yes, edge_yes, edge_no)
    out["exec_ask"] = np.where(choose_yes, out["ask_yes"], out["ask_no"])
    out["fee_total"] = [fee_total(QTY, p) for p in out["exec_ask"]]
    out["trade_won"] = np.where(choose_yes, out["yes_won"], 1 - out["yes_won"]).astype(int)
    out["pnl"] = QTY * (out["trade_won"] - out["exec_ask"]) - out["fee_total"]
    out["edge_actual"] = out["pnl"] / QTY
    return out


def select_policy(data: pd.DataFrame, ml: int, threshold: float) -> pd.DataFrame:
    eligible = data[(data["ml"] == ml) & (data["pred_edge"] >= threshold)].copy()
    if eligible.empty:
        return eligible
    eligible.sort_values(
        ["close_ts", "pred_edge", "spread", "ticker"],
        ascending=[True, False, True, True],
        inplace=True,
    )
    selected = eligible.groupby("close_ts", as_index=False, group_keys=False).head(1).copy()
    selected.sort_values("close_dt", inplace=True)
    return selected


def daily_stats(selected: pd.DataFrame, all_dates: Iterable, reps: int = 20_000) -> dict:
    dates = list(all_dates)
    daily = selected.groupby("date")["pnl"].sum().reindex(dates, fill_value=0.0)
    arr = daily.to_numpy(float)
    rng = np.random.default_rng(SEED)
    boots = np.empty(reps, dtype=float)
    chunk = 2_000
    for start in range(0, reps, chunk):
        stop = min(reps, start + chunk)
        idx = rng.integers(0, len(arr), size=(stop - start, len(arr)))
        boots[start:stop] = arr[idx].mean(axis=1)
    return {
        "n": int(len(selected)),
        "days": int(len(arr)),
        "total_pnl": float(selected["pnl"].sum()),
        "mean_day": float(arr.mean()),
        "sd_day": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "edge_per_contract": float(selected["pnl"].sum() / (QTY * len(selected))) if len(selected) else float("nan"),
        "win_rate": float(selected["trade_won"].mean()) if len(selected) else float("nan"),
        "mean_ask": float(selected["exec_ask"].mean()) if len(selected) else float("nan"),
        "ci_lo": float(np.quantile(boots, 0.025)),
        "ci_hi": float(np.quantile(boots, 0.975)),
        "p_nonpositive": float(np.mean(boots <= 0.0)),
    }


def drawdown_stats(selected: pd.DataFrame) -> dict:
    if selected.empty:
        return {"max_drawdown": 0.0, "worst_window": 0.0}
    windows = selected.groupby("close_dt", as_index=False)["pnl"].sum().sort_values("close_dt")
    equity = windows["pnl"].cumsum()
    drawdown = equity.cummax() - equity
    return {
        "max_drawdown": float(drawdown.max()),
        "worst_window": float(windows["pnl"].min()),
    }


def block_stats(selected: pd.DataFrame, minutes: int, reps: int = 20_000) -> dict:
    if selected.empty:
        return {"minutes": minutes, "n_blocks": 0, "ci_lo": 0.0, "ci_hi": 0.0, "p_nonpositive": 1.0}
    frame = selected.copy()
    freq = f"{minutes}min"
    frame["block"] = frame["close_dt"].dt.floor(freq)
    blocks = frame.groupby("block")["pnl"].sum().to_numpy(float)
    rng = np.random.default_rng(SEED + minutes)
    boots = np.empty(reps, dtype=float)
    chunk = 2_000
    for start in range(0, reps, chunk):
        stop = min(reps, start + chunk)
        idx = rng.integers(0, len(blocks), size=(stop - start, len(blocks)))
        boots[start:stop] = blocks[idx].sum(axis=1)
    return {
        "minutes": minutes,
        "n_blocks": int(len(blocks)),
        "observed_total": float(blocks.sum()),
        "ci_lo": float(np.quantile(boots, 0.025)),
        "ci_hi": float(np.quantile(boots, 0.975)),
        "p_nonpositive": float(np.mean(boots <= 0.0)),
    }


def choose_config(econ: pd.DataFrame, valid_dates: list) -> tuple[dict, pd.DataFrame]:
    rows = []
    best = None
    for ml in DECISION_MLS:
        for threshold in EDGE_THRESHOLDS:
            selected = select_policy(econ, ml, threshold)
            stats = daily_stats(selected, valid_dates, reps=5_000)
            row = {"ml": ml, "threshold": threshold, **stats, **drawdown_stats(selected)}
            rows.append(row)
            if stats["n"] < 40:
                continue
            score = (stats["mean_day"], stats["edge_per_contract"], -threshold)
            if best is None or score > best[0]:
                best = (score, {"ml": ml, "threshold": threshold})
    grid = pd.DataFrame(rows)
    if best is None:
        raise RuntimeError("no validation configuration with >=40 trades")
    return best[1], grid


def leave_one(selected: pd.DataFrame, col: str) -> list[dict]:
    rows = []
    for value in sorted(selected[col].dropna().unique(), key=str):
        kept = selected[selected[col] != value]
        rows.append(
            {
                "dimension": col,
                "dropped": str(value),
                "n": int(len(kept)),
                "total_pnl": float(kept["pnl"].sum()),
                "edge_per_contract": float(kept["pnl"].sum() / (QTY * len(kept))) if len(kept) else float("nan"),
            }
        )
    return rows


def decile_table(selected: pd.DataFrame) -> pd.DataFrame:
    if len(selected) < 20:
        return pd.DataFrame()
    out = selected.copy()
    out["decile"] = pd.qcut(out["pred_edge"], 10, labels=False, duplicates="drop")
    return (
        out.groupby("decile", as_index=False)
        .agg(
            n=("pnl", "size"),
            predicted_edge=("pred_edge", "mean"),
            actual_edge=("edge_actual", "mean"),
            win_rate=("trade_won", "mean"),
            mean_ask=("exec_ask", "mean"),
        )
    )


def pair_candidates(econ: pd.DataFrame, ml: int) -> pd.DataFrame:
    frame = econ[econ["ml"] == ml].copy()
    rows: list[dict] = []
    for close_ts, group in frame.groupby("close_ts"):
        records = list(group.itertuples(index=False))
        for left in records:
            for right in records:
                if left.coin == right.coin:
                    continue
                ask_yes = float(left.ask_yes)
                ask_no = float(right.ask_no)
                fee = fee_total(QTY, ask_yes) + fee_total(QTY, ask_no)
                predicted_payoff = float(left.pred_p_yes) + 1.0 - float(right.pred_p_yes)
                predicted_edge = predicted_payoff - ask_yes - ask_no - fee / QTY
                actual_payoff = int(left.yes_won) + 1 - int(right.yes_won)
                pnl = QTY * (actual_payoff - ask_yes - ask_no) - fee
                rows.append(
                    {
                        "close_ts": int(close_ts),
                        "close_dt": left.close_dt,
                        "date": left.date,
                        "week": left.week,
                        "minute": int(left.minute),
                        "long_yes_coin": str(left.coin),
                        "long_no_coin": str(right.coin),
                        "pred_edge": predicted_edge,
                        "pnl": pnl,
                        "edge_actual": pnl / (2 * QTY),
                        "pair_cost": ask_yes + ask_no + fee / QTY,
                        "bad_state": int((int(left.yes_won) == 0) and (int(right.yes_won) == 1)),
                    }
                )
    return pd.DataFrame(rows)


def choose_pair_config(valid_econ: pd.DataFrame, valid_dates: list) -> tuple[dict, pd.DataFrame]:
    rows = []
    best = None
    for ml in DECISION_MLS:
        pairs = pair_candidates(valid_econ, ml)
        for threshold in EDGE_THRESHOLDS:
            eligible = pairs[pairs["pred_edge"] >= threshold].copy()
            if not eligible.empty:
                eligible.sort_values(["close_ts", "pred_edge"], ascending=[True, False], inplace=True)
                selected = eligible.groupby("close_ts", as_index=False, group_keys=False).head(1)
            else:
                selected = eligible
            # Pair pnl already includes both legs; use QTY * 2 denominator below.
            daily = selected.groupby("date")["pnl"].sum().reindex(valid_dates, fill_value=0.0)
            row = {
                "ml": ml,
                "threshold": threshold,
                "n": int(len(selected)),
                "total_pnl": float(selected["pnl"].sum()),
                "mean_day": float(daily.mean()),
                "edge_per_contract": float(selected["pnl"].sum() / (2 * QTY * len(selected))) if len(selected) else float("nan"),
                "bad_state_rate": float(selected["bad_state"].mean()) if len(selected) else float("nan"),
            }
            rows.append(row)
            if len(selected) < 40:
                continue
            score = (row["mean_day"], row["edge_per_contract"], -threshold)
            if best is None or score > best[0]:
                best = (score, {"ml": ml, "threshold": threshold})
    if best is None:
        return {"ml": 12, "threshold": 0.08}, pd.DataFrame(rows)
    return best[1], pd.DataFrame(rows)


def select_pairs(econ: pd.DataFrame, config: dict) -> pd.DataFrame:
    pairs = pair_candidates(econ, int(config["ml"]))
    eligible = pairs[pairs["pred_edge"] >= float(config["threshold"])].copy()
    if eligible.empty:
        return eligible
    eligible.sort_values(["close_ts", "pred_edge"], ascending=[True, False], inplace=True)
    return eligible.groupby("close_ts", as_index=False, group_keys=False).head(1).copy()


def main() -> None:
    panel = build_panel()
    train, valid, test, split_meta = split_dates(panel)
    frames = {"valid": valid, "test": test}
    _, predictions = fit_models(train, frames)

    info_rows = []
    for model_name in ["market", "own", "peer"]:
        for split_name, frame in frames.items():
            metrics = info_metrics(frame, predictions[model_name][split_name])
            info_rows.append({"model": model_name, "split": split_name, **metrics})
    info = pd.DataFrame(info_rows)
    info.to_csv(OUT / "information_metrics.csv", index=False)

    valid_econ: dict[str, pd.DataFrame] = {}
    test_econ: dict[str, pd.DataFrame] = {}
    configs: dict[str, dict] = {}
    grids = []
    test_summaries = []
    selected_by_model: dict[str, pd.DataFrame] = {}
    valid_dates = sorted(valid["date"].unique())
    test_dates = sorted(test["date"].unique())

    for model_name in ["market", "own", "peer"]:
        valid_econ[model_name] = add_economics(valid, predictions[model_name]["valid"])
        test_econ[model_name] = add_economics(test, predictions[model_name]["test"])
        config, grid = choose_config(valid_econ[model_name], valid_dates)
        config["model"] = model_name
        configs[model_name] = config
        grid.insert(0, "model", model_name)
        grids.append(grid)
        selected = select_policy(test_econ[model_name], int(config["ml"]), float(config["threshold"]))
        selected_by_model[model_name] = selected
        summary = {
            "model": model_name,
            **config,
            **daily_stats(selected, test_dates),
            **drawdown_stats(selected),
        }
        test_summaries.append(summary)
        selected.to_parquet(OUT / f"test_selected_{model_name}.parquet", index=False)

    pd.concat(grids, ignore_index=True).to_csv(OUT / "validation_policy_grid.csv", index=False)
    test_summary_df = pd.DataFrame(test_summaries)
    test_summary_df.to_csv(OUT / "test_policy_summary.csv", index=False)

    peer_selected = selected_by_model["peer"]
    peer_blocks = [block_stats(peer_selected, m) for m in (30, 60, 90)]
    pd.DataFrame(peer_blocks).to_csv(OUT / "peer_block_uncertainty.csv", index=False)
    loo = []
    if not peer_selected.empty:
        loo.extend(leave_one(peer_selected, "coin"))
        loo.extend(leave_one(peer_selected, "week"))
        loo.extend(leave_one(peer_selected, "minute"))
    pd.DataFrame(loo).to_csv(OUT / "peer_leave_one.csv", index=False)
    deciles = decile_table(peer_selected)
    deciles.to_csv(OUT / "peer_edge_deciles.csv", index=False)

    # Pair policy, selected only on validation using peer-model probabilities.
    pair_config, pair_grid = choose_pair_config(valid_econ["peer"], valid_dates)
    pair_grid.to_csv(OUT / "validation_pair_grid.csv", index=False)
    pair_test = select_pairs(test_econ["peer"], pair_config)
    pair_test.to_parquet(OUT / "test_selected_pairs.parquet", index=False)
    if not pair_test.empty:
        pair_daily = pair_test.groupby("date")["pnl"].sum().reindex(test_dates, fill_value=0.0)
        pair_stats = {
            "config": pair_config,
            "n": int(len(pair_test)),
            "total_pnl": float(pair_test["pnl"].sum()),
            "mean_day": float(pair_daily.mean()),
            "sd_day": float(pair_daily.std(ddof=1)),
            "edge_per_contract": float(pair_test["pnl"].sum() / (2 * QTY * len(pair_test))),
            "bad_state_rate": float(pair_test["bad_state"].mean()),
        }
        # Reuse bootstrap with a temporary trade_won/exec_ask-compatible frame
        rng = np.random.default_rng(SEED + 99)
        arr = pair_daily.to_numpy(float)
        boots = np.empty(20_000)
        for start in range(0, 20_000, 2_000):
            stop = min(20_000, start + 2_000)
            idx = rng.integers(0, len(arr), size=(stop - start, len(arr)))
            boots[start:stop] = arr[idx].mean(axis=1)
        pair_stats.update(
            {
                "ci_lo": float(np.quantile(boots, 0.025)),
                "ci_hi": float(np.quantile(boots, 0.975)),
                "p_nonpositive": float(np.mean(boots <= 0.0)),
                **drawdown_stats(pair_test),
            }
        )
        pair_blocks = [block_stats(pair_test, m) for m in (30, 60, 90)]
    else:
        pair_stats = {"config": pair_config, "n": 0}
        pair_blocks = []
    pd.DataFrame(pair_blocks).to_csv(OUT / "pair_block_uncertainty.csv", index=False)

    # Hard-pass logic.
    info_pivot = info.set_index(["model", "split"])
    info_pass = all(
        [
            info_pivot.loc[("peer", "valid"), "next_mse"] < info_pivot.loc[("own", "valid"), "next_mse"],
            info_pivot.loc[("peer", "test"), "next_mse"] < info_pivot.loc[("own", "test"), "next_mse"],
            info_pivot.loc[("peer", "valid"), "log_loss"] < info_pivot.loc[("own", "valid"), "log_loss"],
            info_pivot.loc[("peer", "test"), "log_loss"] < info_pivot.loc[("own", "test"), "log_loss"],
        ]
    )
    summaries = {r["model"]: r for r in test_summaries}
    peer_summary = summaries["peer"]
    peer_baseline_pass = (
        peer_summary.get("total_pnl", -1e9) > summaries["own"].get("total_pnl", 1e9)
        and peer_summary.get("total_pnl", -1e9) > summaries["market"].get("total_pnl", 1e9)
    )
    block_pass = bool(peer_blocks) and all(x["ci_lo"] > 0.0 for x in peer_blocks)
    loo_pass = bool(loo) and all((not math.isfinite(x["edge_per_contract"])) or x["edge_per_contract"] > 0.0 for x in loo)
    if len(deciles) >= 3:
        monotone_corr = float(deciles["predicted_edge"].corr(deciles["actual_edge"], method="spearman"))
        decile_pass = monotone_corr > 0.0 and float(deciles.iloc[-1]["actual_edge"]) > float(deciles.iloc[0]["actual_edge"])
    else:
        monotone_corr = float("nan")
        decile_pass = False
    economic_pass = (
        peer_summary.get("edge_per_contract", -1.0) > 0.0
        and peer_summary.get("ci_lo", -1.0) > 0.0
        and peer_baseline_pass
        and block_pass
        and loo_pass
        and decile_pass
    )
    pair_pass = (
        pair_stats.get("n", 0) >= 40
        and pair_stats.get("ci_lo", -1.0) > 0.0
        and pair_stats.get("max_drawdown", 1e9) < peer_summary.get("max_drawdown", -1.0)
        and bool(pair_blocks)
        and all(x["ci_lo"] > 0.0 for x in pair_blocks)
    )

    result = {
        "panel_rows": int(len(panel)),
        "synchronized_windows": int(panel[["close_ts", "ml"]].drop_duplicates().shape[0]),
        "split_meta": split_meta,
        "configs": configs,
        "information_pass": bool(info_pass),
        "economic_pass": bool(economic_pass),
        "pair_pass": bool(pair_pass),
        "peer_vs_baseline_pass": bool(peer_baseline_pass),
        "peer_block_pass": bool(block_pass),
        "peer_leave_one_pass": bool(loo_pass),
        "peer_decile_pass": bool(decile_pass),
        "peer_decile_spearman": monotone_corr,
        "test_summaries": test_summaries,
        "pair_summary": pair_stats,
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    verdict = "PASS" if (info_pass and economic_pass) else "FAIL"
    pair_verdict = "PASS" if pair_pass else "FAIL"
    lines = [
        "# CMQL — Cross-Market Quote-Lag / Relative-Value Audit",
        "",
        f"## Primary verdict: **{verdict}**",
        f"## Dominance-pair verdict: **{pair_verdict}**",
        "",
        f"Panel rows: {len(panel):,}; synchronized window/minute states: {result['synchronized_windows']:,}.",
        "",
        "## Information metrics",
        "",
        info.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Validation-selected single-leg policies and sealed TEST",
        "",
        test_summary_df.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Peer policy block uncertainty",
        "",
        pd.DataFrame(peer_blocks).to_markdown(index=False, floatfmt=".6f") if peer_blocks else "No peer trades.",
        "",
        "## Predicted-edge deciles",
        "",
        deciles.to_markdown(index=False, floatfmt=".6f") if not deciles.empty else "Insufficient observations.",
        "",
        "## Dominance pair",
        "",
        "```json",
        json.dumps(pair_stats, indent=2, default=str),
        "```",
        "",
        "## Gate summary",
        "",
        f"- peer information improves own-path on VALID and TEST: **{info_pass}**",
        f"- peer sealed economics positive with positive day lower bound: **{peer_summary.get('edge_per_contract', float('nan')) > 0 and peer_summary.get('ci_lo', -1) > 0}**",
        f"- peer beats simpler baselines: **{peer_baseline_pass}**",
        f"- all 30/60/90-minute lower bounds positive: **{block_pass}**",
        f"- leave-one robustness: **{loo_pass}**",
        f"- residual-to-P&L monotonicity: **{decile_pass}** (Spearman {monotone_corr:.4f})",
        "",
        "No result authorizes live exchange action.",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
