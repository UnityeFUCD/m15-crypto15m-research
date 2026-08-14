"""Fail-closed inventory for the Dynamic Pair Completion experiment.

This script does not place orders, call external APIs, or read credentials. It
inspects the repository's immutable historical artifacts and capture ledger to
answer one question before any strategy test is written:

    Do we have timestamp-aligned private fills and public books/tape at enough
    resolution to measure the cost and probability of completing the opposite
    leg after a real first fill?

The output is evidence, not a trading verdict.
"""
from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "results"
OUT.mkdir(parents=True, exist_ok=True)
DB = ROOT / "capture" / "capture.db"

RELEVANT_TABLES = [
    "opportunities", "opportunity_snapshots", "orders", "order_events",
    "queue_snapshots", "book_snapshots", "public_trades", "fills",
    "positions", "settlements", "virtual_orders", "virtual_queue",
]
PARQUETS = [
    ROOT / "data" / "orders_history.parquet",
    ROOT / "data" / "fills_history.parquet",
    ROOT / "data" / "trades_lsm.parquet",
    ROOT / "data" / "queue_ahead.parquet",
    ROOT / "data" / "book_full.parquet",
    ROOT / "data" / "paths_full.parquet",
    ROOT / "data" / "underlying.parquet",
]


def jload(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    try:
        out = json.loads(value)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def epoch_ms(value: Any) -> float:
    """Parse a numeric or datetime-like timestamp to epoch milliseconds."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return math.nan
    try:
        x = float(value)
        if np.isfinite(x):
            ax = abs(x)
            if ax > 1e17:       # ns
                return x / 1e6
            if ax > 1e14:       # us
                return x / 1e3
            if ax > 1e11:       # ms
                return x
            if ax > 1e8:        # s
                return x * 1e3
    except Exception:
        pass
    try:
        ts = pd.to_datetime(value, utc=True)
        if pd.isna(ts):
            return math.nan
        return float(ts.value / 1e6)
    except Exception:
        return math.nan


def first_present(frame: pd.DataFrame, names: list[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def parquet_meta(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path.relative_to(ROOT)), "exists": False}
    pf = pq.ParquetFile(path)
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": True,
        "rows": int(pf.metadata.num_rows),
        "row_groups": int(pf.metadata.num_row_groups),
        "size_bytes": int(path.stat().st_size),
        "columns": list(pf.schema_arrow.names),
    }


def sqlite_inventory() -> tuple[dict, dict[str, list[dict]]]:
    result: dict[str, Any] = {"exists": DB.exists(), "path": str(DB.relative_to(ROOT))}
    samples: dict[str, list[dict]] = {}
    if not DB.exists():
        return result, samples
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    tables = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    result["size_bytes"] = int(DB.stat().st_size)
    result["tables"] = {}
    for table in tables:
        if table.startswith("sqlite_"):
            continue
        try:
            n = int(con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        except Exception as exc:
            result["tables"][table] = {"error": repr(exc)}
            continue
        info: dict[str, Any] = {"rows": n}
        cols = [r[1] for r in con.execute(f'PRAGMA table_info("{table}")')]
        info["columns"] = cols
        if "ts_ingest_ms" in cols and n:
            lo, hi = con.execute(
                f'SELECT MIN(ts_ingest_ms), MAX(ts_ingest_ms) FROM "{table}"').fetchone()
            info["ingest_min_ms"] = lo
            info["ingest_max_ms"] = hi
        rows: list[dict] = []
        if "payload" in cols and n:
            raw = con.execute(
                f'SELECT payload FROM "{table}" ORDER BY seq DESC LIMIT 5').fetchall()
            rows = [jload(r[0]) for r in raw]
            keys = Counter(k for row in rows for k in row)
            info["sample_payload_keys"] = sorted(keys)
            info["sample_payload_nonnull"] = {
                k: sum(row.get(k) is not None for row in rows) for k in sorted(keys)
            }
        result["tables"][table] = info
        samples[table] = rows
    con.close()
    return result, samples


def read_payload_table(table: str) -> pd.DataFrame:
    if not DB.exists():
        return pd.DataFrame()
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        names = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if table not in names:
            return pd.DataFrame()
        raw = con.execute(f'SELECT payload FROM "{table}" ORDER BY seq').fetchall()
    finally:
        con.close()
    return pd.DataFrame([jload(row[0]) for row in raw])


def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def normalize_ticker(frame: pd.DataFrame) -> pd.Series:
    col = first_present(frame, ["ticker", "market_ticker", "market"])
    if not col:
        return pd.Series([None] * len(frame), index=frame.index, dtype="object")
    return frame[col].astype(str)


def normalize_fill_times(frame: pd.DataFrame) -> pd.Series:
    candidates = [
        "ts_ms", "created_time", "created_at", "timestamp", "ts",
        "fill_time", "time", "trade_time",
    ]
    col = first_present(frame, candidates)
    if not col:
        return pd.Series(np.nan, index=frame.index)
    return frame[col].map(epoch_ms)


def actual_lsm_alignment() -> dict:
    orders = load_parquet(ROOT / "data" / "orders_history.parquet")
    fills = load_parquet(ROOT / "data" / "fills_history.parquet")
    books = read_payload_table("book_snapshots")
    tape = read_payload_table("public_trades")
    out: dict[str, Any] = {
        "orders_rows": int(len(orders)), "fills_rows": int(len(fills)),
        "capture_book_rows": int(len(books)), "capture_trade_rows": int(len(tape)),
        "orders_columns": list(orders.columns), "fills_columns": list(fills.columns),
        "book_columns": list(books.columns), "tape_columns": list(tape.columns),
    }
    if orders.empty or fills.empty:
        out["verdict"] = "NO_PRIVATE_ORDER_FILL_HISTORY"
        return out

    client_col = first_present(orders, ["client_order_id", "client_id"])
    order_id_col = first_present(orders, ["order_id", "id"])
    if client_col:
        lsm_orders = orders[orders[client_col].astype(str).str.startswith("lsm", na=False)].copy()
    else:
        lsm_orders = pd.DataFrame(columns=orders.columns)
    out["lsm_orders"] = int(len(lsm_orders))

    fill_order_col = first_present(fills, ["order_id", "orderId"])
    if order_id_col and fill_order_col and not lsm_orders.empty:
        ids = set(lsm_orders[order_id_col].astype(str))
        lsm_fills = fills[fills[fill_order_col].astype(str).isin(ids)].copy()
    else:
        fill_client_col = first_present(fills, ["client_order_id", "client_id"])
        if fill_client_col:
            lsm_fills = fills[fills[fill_client_col].astype(str).str.startswith("lsm", na=False)].copy()
        else:
            lsm_fills = pd.DataFrame(columns=fills.columns)
    out["lsm_fill_rows"] = int(len(lsm_fills))
    if lsm_fills.empty:
        out["verdict"] = "NO_LSM_FILLS"
        return out

    lsm_fills = lsm_fills.copy()
    lsm_fills["_ticker"] = normalize_ticker(lsm_fills)
    lsm_fills["_ts_ms"] = normalize_fill_times(lsm_fills)
    if lsm_fills["_ticker"].eq("None").all() and fill_order_col and order_id_col:
        map_ticker_col = first_present(lsm_orders, ["ticker", "market_ticker", "market"])
        if map_ticker_col:
            mapping = dict(zip(lsm_orders[order_id_col].astype(str), lsm_orders[map_ticker_col].astype(str)))
            lsm_fills["_ticker"] = lsm_fills[fill_order_col].astype(str).map(mapping)
    out["lsm_fill_with_time"] = int(lsm_fills["_ts_ms"].notna().sum())
    out["lsm_fill_with_ticker"] = int(lsm_fills["_ticker"].notna().sum())
    out["lsm_fill_time_min_ms"] = float(lsm_fills["_ts_ms"].min()) if lsm_fills["_ts_ms"].notna().any() else None
    out["lsm_fill_time_max_ms"] = float(lsm_fills["_ts_ms"].max()) if lsm_fills["_ts_ms"].notna().any() else None

    def align(reference: pd.DataFrame, label: str) -> None:
        if reference.empty:
            out[f"{label}_overlap"] = {"available": False}
            return
        reference = reference.copy()
        reference["_ticker"] = normalize_ticker(reference)
        reference["_ts_ms"] = normalize_fill_times(reference)
        reference = reference.dropna(subset=["_ticker", "_ts_ms"])
        if reference.empty:
            out[f"{label}_overlap"] = {"available": False, "reason": "no parseable ticker/time"}
            return
        nearest: list[float] = []
        by_ticker = {
            t: np.sort(g["_ts_ms"].to_numpy(dtype=float))
            for t, g in reference.groupby("_ticker")
        }
        for row in lsm_fills.dropna(subset=["_ticker", "_ts_ms"]).itertuples(index=False):
            ticker = getattr(row, "_ticker")
            ts = float(getattr(row, "_ts_ms"))
            arr = by_ticker.get(ticker)
            if arr is None or len(arr) == 0:
                nearest.append(math.inf)
                continue
            i = int(np.searchsorted(arr, ts))
            cand = []
            if i < len(arr): cand.append(abs(arr[i] - ts))
            if i: cand.append(abs(arr[i - 1] - ts))
            nearest.append(min(cand) if cand else math.inf)
        finite = np.array([x for x in nearest if np.isfinite(x)], dtype=float)
        out[f"{label}_overlap"] = {
            "available": True,
            "reference_rows": int(len(reference)),
            "fills_with_same_ticker": int(len(finite)),
            "nearest_ms_median": float(np.median(finite)) if len(finite) else None,
            "nearest_ms_p90": float(np.quantile(finite, .9)) if len(finite) else None,
            "within_250ms": int(np.sum(finite <= 250)),
            "within_1s": int(np.sum(finite <= 1000)),
            "within_5s": int(np.sum(finite <= 5000)),
            "within_10s": int(np.sum(finite <= 10000)),
            "within_60s": int(np.sum(finite <= 60000)),
        }

    align(books, "book")
    align(tape, "trade")
    book_good = out.get("book_overlap", {}).get("within_5s", 0)
    if book_good >= 100:
        out["verdict"] = "HIGH_RES_PRIVATE_FILL_BOOK_OVERLAP"
    elif book_good > 0:
        out["verdict"] = "LIMITED_PRIVATE_FILL_BOOK_OVERLAP"
    else:
        out["verdict"] = "NO_PRIVATE_FILL_BOOK_OVERLAP"
    return out


def markdown(report: dict) -> str:
    lines = [
        "# Dynamic Pair Completion — data audit",
        "",
        "This is a fail-closed inventory. It makes no P&L claim and places no orders.",
        "",
        f"## Resolution verdict: **{report['alignment'].get('verdict')}**",
        "",
        "## Capture ledger",
        "",
        f"- DB exists: `{report['sqlite'].get('exists')}`",
        f"- DB size: `{report['sqlite'].get('size_bytes', 0):,}` bytes",
        "",
        "| Table | Rows | Sample payload keys |",
        "|---|---:|---|",
    ]
    tables = report["sqlite"].get("tables", {})
    for table in RELEVANT_TABLES:
        info = tables.get(table, {})
        keys = ", ".join(info.get("sample_payload_keys", [])[:16])
        lines.append(f"| `{table}` | {info.get('rows', 0):,} | {keys} |")
    lines += ["", "## Repository data", "", "| File | Rows | Size | Columns |", "|---|---:|---:|---|"]
    for item in report["parquets"]:
        cols = ", ".join(item.get("columns", [])[:16])
        lines.append(
            f"| `{item['path']}` | {item.get('rows', 0):,} | {item.get('size_bytes', 0):,} | {cols} |")
    a = report["alignment"]
    lines += [
        "", "## Actual execution overlap", "",
        f"- Orders: **{a.get('orders_rows', 0):,}**; LSM orders: **{a.get('lsm_orders', 0):,}**",
        f"- Fills: **{a.get('fills_rows', 0):,}**; LSM fill rows: **{a.get('lsm_fill_rows', 0):,}**",
        f"- Captured books: **{a.get('capture_book_rows', 0):,}**; captured trades: **{a.get('capture_trade_rows', 0):,}**",
    ]
    for label in ("book", "trade"):
        x = a.get(f"{label}_overlap", {})
        lines += [
            "",
            f"### Nearest {label} event to each real LSM fill",
            f"- Same-ticker matches: **{x.get('fills_with_same_ticker', 0):,}**",
            f"- Median nearest distance: **{x.get('nearest_ms_median')} ms**",
            f"- Within 250 ms / 1 s / 5 s / 10 s / 60 s: "
            f"**{x.get('within_250ms', 0)} / {x.get('within_1s', 0)} / "
            f"{x.get('within_5s', 0)} / {x.get('within_10s', 0)} / {x.get('within_60s', 0)}**",
        ]
    lines += [
        "", "## Decision rule", "",
        "A historical dynamic-completion test is permitted only if at least 100 real first fills have a same-ticker book snapshot within 5 seconds. Otherwise the result would be interpolation, not execution evidence.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    sqlite_report, _ = sqlite_inventory()
    report = {
        "sqlite": sqlite_report,
        "parquets": [parquet_meta(p) for p in PARQUETS],
        "alignment": actual_lsm_alignment(),
    }
    (OUT / "dpc_data_audit.json").write_text(json.dumps(report, indent=2, default=str))
    (OUT / "dpc_data_audit.md").write_text(markdown(report))
    print(markdown(report))


if __name__ == "__main__":
    main()
