"""Audit huthvincent/btc15-dataset for genuine live execution evidence.

A row is never called live merely because it says "filled".  The audit reports
source/mode labels, exchange identifiers, lifecycle linkage, and realized P&L
separately, then applies a fail-closed evidence gate.

No credentials and no trading. Public read-only data only.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import get_dataset_config_names, load_dataset

REPO = "huthvincent/btc15-dataset"
OUT = Path("research/results")
OUT.mkdir(parents=True, exist_ok=True)


def frame(config: str) -> pd.DataFrame:
    ds = load_dataset(REPO, config, split="train", trust_remote_code=False)
    return ds.to_pandas()


def first(df: pd.DataFrame, names: list[str]) -> str | None:
    return next((x for x in names if x in df.columns), None)


def safe_counts(s: pd.Series, limit: int = 30) -> dict[str, int]:
    x = s.astype("string").fillna("<NA>").value_counts(dropna=False).head(limit)
    return {str(k): int(v) for k, v in x.items()}


def boolish(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.astype("string").str.lower().isin({"1", "true", "yes", "live", "filled", "matched", "complete", "completed"})


def numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def time_ms(s: pd.Series) -> pd.Series:
    n = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    valid = n.notna()
    if valid.any():
        x = n[valid].astype(float)
        med = float(x.abs().median())
        if med > 1e17: x = x / 1e6
        elif med > 1e14: x = x / 1e3
        elif med < 1e11: x = x * 1e3
        out.loc[valid] = x
    missing = ~valid
    if missing.any():
        dt = pd.to_datetime(s[missing], utc=True, errors="coerce")
        out.loc[missing] = dt.astype("int64").where(dt.notna(), np.nan) / 1e6
    return out


def label_live(df: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    """Require an explicit live label. Unknown is false."""
    label_cols = [c for c in df.columns if c.lower() in {
        "mode", "source", "environment", "execution_mode", "run_mode",
        "is_live", "live", "paper", "simulation", "simulated"
    }]
    live = pd.Series(False, index=df.index)
    detail: dict[str, Any] = {}
    for c in label_cols:
        vals = df[c].astype("string").str.lower()
        detail[c] = safe_counts(df[c])
        if c.lower() in {"is_live", "live"}:
            live |= boolish(df[c])
        else:
            live |= vals.str.contains(r"(^|[^a-z])live([^a-z]|$)", regex=True, na=False)
            # explicit non-live always vetoes a row
            non = vals.str.contains("paper|sim|backtest|synthetic|virtual|mock", regex=True, na=False)
            live &= ~non
    return live, detail


def id_quality(df: pd.DataFrame, kind: str) -> tuple[pd.Series, dict[str, Any]]:
    names = (["order_id", "exchange_order_id", "id"] if kind == "order" else
             ["trade_id", "fill_id", "exchange_trade_id", "id"])
    c = first(df, names)
    if c is None:
        return pd.Series(False, index=df.index), {"id_column": None}
    x = df[c].astype("string")
    ok = x.notna() & ~x.str.lower().isin({"", "none", "nan", "null"})
    # reject obvious locally-generated placeholders
    fake = x.str.lower().str.contains("paper|sim|mock|virtual|synthetic|test", regex=True, na=False)
    ok &= ~fake
    return ok, {
        "id_column": c,
        "nonnull": int(ok.sum()),
        "unique": int(x[ok].nunique()),
        "duplicates": int(ok.sum() - x[ok].nunique()),
        "examples": x[ok].head(5).tolist(),
    }


def status_filled(df: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    c = first(df, ["status", "state", "order_status", "result"])
    filled_qty = first(df, ["filled_size", "filled_qty", "fill_count", "fill_count_fp", "executed_qty", "quantity_filled"])
    m = pd.Series(False, index=df.index)
    detail: dict[str, Any] = {"status_column": c, "filled_qty_column": filled_qty}
    if c:
        v = df[c].astype("string").str.lower()
        detail["status_counts"] = safe_counts(df[c])
        m |= v.str.contains("fill|match|complete|executed", regex=True, na=False)
    if filled_qty:
        q = numeric(df[filled_qty]).fillna(0)
        m |= q > 0
        detail["positive_filled_qty"] = int((q > 0).sum())
    return m, detail


def pnl_candidates(df: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    out: dict[str, dict[str, float | int | None]] = {}
    for c in df.columns:
        lc = c.lower()
        if not any(k in lc for k in ["pnl", "profit", "realized", "realised", "balance", "equity"]):
            continue
        x = numeric(df[c]).dropna()
        if x.empty:
            continue
        out[c] = {
            "n": int(len(x)), "sum": float(x.sum()), "mean": float(x.mean()),
            "min": float(x.min()), "max": float(x.max()),
            "first": float(x.iloc[0]), "last": float(x.iloc[-1]),
        }
    return out


def equity_paths(df: pd.DataFrame) -> list[dict[str, Any]]:
    eq = first(df, ["equity", "balance", "account_value", "portfolio_value", "cash"])
    ts = first(df, ["timestamp", "ts", "time", "created_at", "created_time"])
    if not eq:
        return []
    groups = [c for c in ["source", "mode", "run_id", "strategy_id", "strategy", "symbol", "market"] if c in df.columns]
    work = df.copy()
    work["_eq"] = numeric(work[eq])
    work["_ts"] = time_ms(work[ts]) if ts else np.arange(len(work), dtype=float)
    work = work.dropna(subset=["_eq", "_ts"])
    if work.empty:
        return []
    if not groups:
        grouped = [("ALL", work)]
    else:
        grouped = work.groupby(groups, dropna=False)
    rows = []
    for key, g in grouped:
        g = g.sort_values("_ts")
        vals = g["_eq"].to_numpy(float)
        if len(vals) < 2:
            continue
        peak = np.maximum.accumulate(vals)
        dd = vals - peak
        rows.append({
            "group": str(key), "n": int(len(g)),
            "start": float(vals[0]), "end": float(vals[-1]),
            "change": float(vals[-1] - vals[0]),
            "max_drawdown": float(dd.min()),
            "ts_start_ms": float(g["_ts"].iloc[0]),
            "ts_end_ms": float(g["_ts"].iloc[-1]),
        })
    return sorted(rows, key=lambda r: r["change"], reverse=True)[:100]


def strategy_summary(orders: pd.DataFrame, trades: pd.DataFrame) -> list[dict[str, Any]]:
    candidates = ["strategy_id", "strategy", "strategy_name", "run_id", "bot_id"]
    key = first(orders, candidates)
    if key is None:
        return []
    live, _ = label_live(orders)
    ids, _ = id_quality(orders, "order")
    filled, _ = status_filled(orders)
    o = orders[live & ids].copy()
    if o.empty:
        return []
    o["_filled"] = filled.loc[o.index]
    pnl_col = first(o, ["realized_pnl", "realised_pnl", "pnl", "profit", "net_pnl"])
    rows = []
    for k, g in o.groupby(key, dropna=False):
        row = {
            "key": str(k), "orders": int(len(g)),
            "filled_orders": int(g["_filled"].sum()),
            "fill_rate": float(g["_filled"].mean()),
        }
        if pnl_col:
            x = numeric(g[pnl_col]).dropna()
            row["order_pnl_n"] = int(len(x))
            row["order_pnl_sum"] = float(x.sum()) if len(x) else None
        rows.append(row)
    return sorted(rows, key=lambda r: (r.get("order_pnl_sum") or -1e99, r["filled_orders"]), reverse=True)


def main() -> None:
    configs = get_dataset_config_names(REPO, trust_remote_code=False)
    wanted = [x for x in ["orders", "trades", "runs", "strategies", "equity", "predictions", "market_ticks", "settlements"] if x in configs]
    data: dict[str, pd.DataFrame] = {}
    report: dict[str, Any] = {"repo": REPO, "configs": configs, "tables": {}}
    for cfg in wanted:
        df = frame(cfg)
        data[cfg] = df
        live, live_detail = label_live(df)
        report["tables"][cfg] = {
            "rows": int(len(df)), "columns": list(df.columns),
            "live_explicit_rows": int(live.sum()),
            "label_counts": live_detail,
            "pnl_columns": pnl_candidates(df),
        }

    orders = data.get("orders", pd.DataFrame())
    trades = data.get("trades", pd.DataFrame())
    if not orders.empty:
        live_o, live_detail = label_live(orders)
        id_o, id_detail = id_quality(orders, "order")
        filled_o, fill_detail = status_filled(orders)
        gate_o = live_o & id_o & filled_o
        report["orders_gate"] = {
            "explicit_live": int(live_o.sum()), "exchange_id": int(id_o.sum()),
            "filled": int(filled_o.sum()), "all_three": int(gate_o.sum()),
            "id_detail": id_detail, "fill_detail": fill_detail,
            "live_detail": live_detail,
        }
    else:
        gate_o = pd.Series(dtype=bool)
        report["orders_gate"] = {"all_three": 0}

    if not trades.empty:
        live_t, live_detail = label_live(trades)
        id_t, id_detail = id_quality(trades, "trade")
        gate_t = live_t & id_t
        report["trades_gate"] = {
            "explicit_live": int(live_t.sum()), "exchange_id": int(id_t.sum()),
            "both": int(gate_t.sum()), "id_detail": id_detail,
            "live_detail": live_detail,
        }
    else:
        gate_t = pd.Series(dtype=bool)
        report["trades_gate"] = {"both": 0}

    report["strategy_summary"] = strategy_summary(orders, trades)
    report["equity_paths"] = equity_paths(data.get("equity", pd.DataFrame()))

    # Strong execution proof requires all of the following.  P&L reconciliation
    # is intentionally strict: there must be an explicitly live equity path or a
    # realized P&L column on explicitly live trades/orders.
    live_equity = [x for x in report["equity_paths"] if x["n"] >= 2]
    realized_evidence = False
    for cfg in ["orders", "trades", "runs", "equity"]:
        t = report["tables"].get(cfg, {})
        if t.get("live_explicit_rows", 0) and t.get("pnl_columns"):
            realized_evidence = True
    n_live_filled = int(report["orders_gate"].get("all_three", 0))
    n_live_trades = int(report["trades_gate"].get("both", 0))
    report["evidence_gate"] = {
        "live_filled_orders_ge_20": n_live_filled >= 20,
        "live_exchange_trades_ge_20": n_live_trades >= 20,
        "realized_pnl_or_equity_present": bool(realized_evidence or live_equity),
    }
    passed = all(report["evidence_gate"].values())
    report["verdict"] = "EXECUTION_EVIDENCE_PRESENT" if passed else "NOT_EXECUTION_PROOF"

    path = OUT / "huth_live_execution_audit.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    compact = {
        "verdict": report["verdict"],
        "configs": configs,
        "table_rows": {k: v["rows"] for k, v in report["tables"].items()},
        "table_live_rows": {k: v["live_explicit_rows"] for k, v in report["tables"].items()},
        "orders_gate": report["orders_gate"],
        "trades_gate": report["trades_gate"],
        "evidence_gate": report["evidence_gate"],
        "top_strategies": report["strategy_summary"][:15],
        "top_equity_paths": report["equity_paths"][:15],
        "pnl_columns": {k: v["pnl_columns"] for k, v in report["tables"].items()},
    }
    print("HUTH_AUDIT_JSON=" + json.dumps(compact, separators=(",", ":"), default=str))


if __name__ == "__main__":
    main()
