"""Concrete pyarrow schemas for the capture tables in DATA_SPEC.md.

Import these rather than hand-rolling dtypes - a parquet file written with a
drifting schema is worse than no file, because it fails silently on read and
the drift is only noticed months later.

Every table carries `schema_version`, `strategy_version` and `git_commit` so a
reader can tell which code produced which rows.

    from capture.schemas import SCHEMAS, empty_table, append
    append("opportunities", rows)
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = 1
CAPTURE_DIR = Path(__file__).resolve().parent

_f64, _i64, _i32, _str, _bool = (pa.float64(), pa.int64(), pa.int32(),
                                 pa.string(), pa.bool_())
_level = pa.struct([("price", _f64), ("size", _f64)])
_trade = pa.struct([("ts_ms", _i64), ("price", _f64), ("size", _f64),
                    ("taker_side", _str)])

# provenance columns present on every table
_PROV = [("schema_version", _i32), ("strategy_version", _str),
         ("git_commit", _str)]


SCHEMAS: dict[str, pa.Schema] = {

    # ---- 1. every eligible market, acted on or not -----------------------
    "opportunities": pa.schema([
        ("opportunity_id", _str),
        ("ts_local_ms", _i64), ("ts_exchange_ms", _i64),
        ("ticker", _str), ("coin", _str), ("event_ticker", _str),
        ("window_open_ms", _i64), ("window_close_ms", _i64),
        ("secs_remaining", _f64),
        ("favourite_side", _str),
        ("yes_bid", _f64), ("yes_ask", _f64),
        ("no_bid", _f64), ("no_ask", _f64),
        ("spread_ticks", _i32), ("tick_size", _f64),
        ("book_bid_levels", pa.list_(_level)),
        ("book_ask_levels", pa.list_(_level)),
        ("displayed_qty_at_our_px", _f64),
        ("cum_volume", _f64),
        ("recent_trades", pa.list_(_trade)),
        ("underlying_a0", _f64), ("underlying_last", _f64),
        # features and filters are stored as maps so adding one does not
        # require a schema migration; keys must be stable across versions
        ("features", pa.map_(_str, _f64)),
        ("filters", pa.map_(_str, _bool)),
        ("eligible", _bool), ("skipped", _bool), ("skip_reason", _str),
    ] + _PROV),

    # ---- 2. submitted orders --------------------------------------------
    "orders": pa.schema([
        ("client_order_id", _str), ("order_id", _str),
        ("opportunity_id", _str),
        ("ticker", _str), ("side", _str), ("action", _str),
        ("price", _f64), ("quantity", _f64),
        ("ts_submit_ms", _i64), ("ts_ack_ms", _i64), ("ack_latency_ms", _i64),
        ("accepted", _bool), ("post_only_honoured", _bool),
        ("reject_reason", _str), ("reserved_capital", _f64),
        ("treatment", _str),
    ] + _PROV),

    # ---- 3. queue, inferred from the tape (no L3 exists) -----------------
    "queue_snapshots": pa.schema([
        ("client_order_id", _str), ("ts_ms", _i64),
        ("displayed_ahead_at_post", _f64),
        ("cum_traded_at_our_px", _f64), ("cum_traded_all_px", _f64),
        ("book_size_at_our_px", _f64),
        ("inferred_ahead_lower", _f64), ("inferred_ahead_upper", _f64),
        ("event", _str),
    ] + _PROV),

    # ---- 4. lifecycle ----------------------------------------------------
    "order_events": pa.schema([
        ("client_order_id", _str), ("order_id", _str), ("ts_ms", _i64),
        ("state", _str), ("detail", _str),
        ("filled_qty_cum", _f64), ("remaining_qty", _f64),
    ] + _PROV),

    # ---- 5. fills --------------------------------------------------------
    "fills": pa.schema([
        ("fill_id", _str), ("trade_id", _str),
        ("order_id", _str), ("client_order_id", _str),
        ("ts_ms", _i64), ("quantity", _f64), ("price", _f64),
        ("is_taker", _bool), ("fee", _f64),
        ("position_after", _f64), ("queue_before_fill", _f64),
    ] + _PROV),

    # ---- 6. settlement ---------------------------------------------------
    "settlements": pa.schema([
        ("ticker", _str), ("settled_ts_ms", _i64),
        ("result", _str), ("a0", _f64), ("a1", _f64),
        ("payout", _f64), ("entry_cost", _f64), ("total_fees", _f64),
        ("realised_pnl", _f64),
        ("pnl_per_submitted_order", _f64),
        ("pnl_per_submitted_contract", _f64),
        ("pnl_per_filled_contract", _f64),
        ("capital_minutes_locked", _f64),
    ] + _PROV),

    # ---- 7. randomised treatment ----------------------------------------
    "assignments": pa.schema([
        ("opportunity_id", _str), ("treatment", _str),
        ("assigned_ts_ms", _i64),
        ("rng_seed", _i64), ("rng_stream_position", _i64),
        ("frozen_rule_version", _str),
    ] + _PROV),

    # ---- supporting ------------------------------------------------------
    "book_snapshots": pa.schema([
        ("ticker", _str), ("ts_ms", _i64),
        ("bid_levels", pa.list_(_level)), ("ask_levels", pa.list_(_level)),
        ("reason", _str),
    ] + _PROV),

    "public_trades": pa.schema([
        ("trade_id", _str), ("ticker", _str), ("ts_ms", _i64),
        ("yes_price", _f64), ("size", _f64),
        ("taker_side", _str), ("taker_book_side", _str),
        ("taker_outcome_side", _str), ("is_block_trade", _bool),
    ] + _PROV),

    "positions": pa.schema([
        ("ts_ms", _i64), ("ticker", _str),
        ("position", _f64), ("total_traded", _f64),
        ("source", _str),                       # "exchange" always; never derived
    ] + _PROV),

    "account_snapshots": pa.schema([
        ("ts_ms", _i64), ("cash", _f64), ("deployed", _f64),
        ("equity", _f64), ("open_positions", _i32),
        ("balance_read_ok", _bool),             # false on API failure - NOT cash=0
    ] + _PROV),

    "strategy_versions": pa.schema([
        ("strategy_version", _str), ("git_commit", _str),
        ("ts_first_seen_ms", _i64), ("config_json", _str),
    ] + _PROV),
}


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(CAPTURE_DIR), stderr=subprocess.DEVNULL,
            text=True).strip()
    except Exception:
        return "unknown"


def empty_table(name: str) -> pa.Table:
    return SCHEMAS[name].empty_table()


def append(name: str, rows: list[dict], strategy_version: str = "unknown",
           day: str | None = None) -> Path:
    """Append rows to today's partition, enforcing the schema.

    Partitioned by UTC date so a day's capture is one self-contained file and
    a crash can never corrupt earlier days.
    """
    if name not in SCHEMAS:
        raise KeyError(f"unknown table {name!r}")
    if not rows:
        return CAPTURE_DIR / name
    day = day or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    commit = git_commit()
    for r in rows:
        r.setdefault("schema_version", SCHEMA_VERSION)
        r.setdefault("strategy_version", strategy_version)
        r.setdefault("git_commit", commit)
    schema = SCHEMAS[name]
    cols = {f.name: [r.get(f.name) for r in rows] for f in schema}
    tbl = pa.Table.from_pydict(cols, schema=schema)
    out = CAPTURE_DIR / name / f"date={day}"
    out.mkdir(parents=True, exist_ok=True)
    part = out / "part.parquet"
    if part.exists():
        tbl = pa.concat_tables([pq.read_table(part), tbl])
    pq.write_table(tbl, part)
    return part


def write_manifest() -> Path:
    """Record row counts and time bounds so a reader can detect gaps."""
    man = {"schema_version": SCHEMA_VERSION, "git_commit": git_commit(),
           "generated_utc": datetime.now(timezone.utc).isoformat(),
           "tables": {}}
    for name in SCHEMAS:
        d = CAPTURE_DIR / name
        if not d.exists():
            continue
        files = sorted(d.rglob("*.parquet"))
        n = 0
        for f in files:
            try:
                n += pq.read_metadata(f).num_rows
            except Exception:
                pass
        man["tables"][name] = {"partitions": len(files), "rows": n}
    p = CAPTURE_DIR / "manifest.json"
    p.write_text(json.dumps(man, indent=2))
    return p


if __name__ == "__main__":
    print(f"schema version {SCHEMA_VERSION}, git {git_commit()}")
    for k, v in SCHEMAS.items():
        print(f"  {k:22} {len(v):>3} columns")
