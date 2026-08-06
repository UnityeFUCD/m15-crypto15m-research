"""Transactional ledger. SQLite in WAL mode, idempotent, crash-safe.

WHY NOT PARQUET AS THE WRITE PATH
  The previous implementation read the whole daily parquet, concatenated, and
  rewrote it on every append. That is O(n^2) across a day and a crash during
  the rewrite destroys the entire partition - including rows written hours
  earlier. Parquet is an export format here, never the write path.

GUARANTEES
  - every write is a transaction; a crash leaves the last committed state
  - UNIQUE constraints make ingestion idempotent: replaying the same fill,
    order event or queue read is a no-op, not a duplicate
  - a monotonic sequence number per table gives total ordering for replay
  - a clean-shutdown marker distinguishes "stopped" from "crashed"
  - gaps are detectable: sequence numbers are dense per table

WAL mode matters: readers (the integrity report, the exporter) never block the
writer, so capture is never paused to run a report.
"""
from __future__ import annotations

import json
import os
import socket
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 2
DEFAULT_DB = Path(__file__).resolve().parent / "capture.db"

# Tables whose natural key makes replay idempotent. The UNIQUE index is the
# mechanism - INSERT OR IGNORE then becomes safe to call repeatedly.
_UNIQUE = {
    "opportunities": "(opportunity_id)",
    "opportunity_snapshots": "(opportunity_id, ts_local_ms)",
    "assignments": "(decision_episode_id)",
    "orders": "(client_order_id)",
    "order_events": "(client_order_id, state, ts_ms, seq_hint)",
    "queue_snapshots": "(client_order_id, ts_request_ms)",
    "book_snapshots": "(ticker, ts_ms, reason)",
    "public_trades": "(trade_id)",
    "fills": "(fill_id)",
    "positions": "(ts_ms, ticker)",
    "account_snapshots": "(ts_ms)",
    "settlements": "(ticker)",
    "strategy_versions": "(strategy_version, config_hash)",
    "capture_heartbeats": "(ts_ms, component)",
    "api_errors": "(ts_ms, path, seq_hint)",
    "reconciliation_runs": "(run_id)",
}

_DDL = """
CREATE TABLE IF NOT EXISTS %(t)s (
    seq            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ingest_ms   INTEGER NOT NULL,
    payload        TEXT    NOT NULL
%(cols)s
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_%(t)s ON %(t)s %(uniq)s;
"""


class Store:
    """Append-only ledger with idempotent ingestion."""

    def __init__(self, path: Path | str = DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), timeout=30.0,
                                  isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")   # durability over speed
        self.db.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        cur = self.db.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY, value TEXT NOT NULL)""")
        for t, uniq in _UNIQUE.items():
            keycols = [c.strip() for c in uniq.strip("()").split(",")]
            coldefs = "".join(
                f",\n    {c} {'INTEGER' if c.endswith('_ms') or c=='seq_hint' else 'TEXT'}"
                for c in keycols)
            cur.executescript(_DDL % {"t": t, "cols": coldefs, "uniq": uniq})
        cur.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)",
                    ("schema_version", str(SCHEMA_VERSION)))
        self.db.commit()

    # ---------- writes ----------
    @contextmanager
    def txn(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield self.db
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def put(self, table: str, row: dict) -> bool:
        """Insert one row. Returns True if newly inserted, False if duplicate.

        A False return is not an error - it means the event was already
        recorded, which is exactly what should happen on replay after a crash.
        """
        return self.put_many(table, [row]) == 1

    def put_many(self, table: str, rows: list[dict]) -> int:
        if table not in _UNIQUE:
            raise KeyError(f"unknown table {table!r}")
        if not rows:
            return 0
        keycols = [c.strip() for c in _UNIQUE[table].strip("()").split(",")]
        now = int(time.time() * 1000)
        cols = ["ts_ingest_ms", "payload"] + keycols
        ph = ",".join("?" * len(cols))
        sql = (f"INSERT OR IGNORE INTO {table} ({','.join(cols)}) "
               f"VALUES ({ph})")
        n = 0
        with self.txn() as db:
            for r in rows:
                vals = [now, json.dumps(r, separators=(",", ":"),
                                        default=str)]
                vals += [r.get(k) for k in keycols]
                cur = db.execute(sql, vals)
                n += cur.rowcount or 0
        return n

    # ---------- reads ----------
    def count(self, table: str) -> int:
        return self.db.execute(f"SELECT COUNT(*) c FROM {table}").fetchone()["c"]

    def rows(self, table: str, since_seq: int = 0, limit: int = 100_000):
        q = (f"SELECT seq, ts_ingest_ms, payload FROM {table} "
             f"WHERE seq > ? ORDER BY seq LIMIT ?")
        for r in self.db.execute(q, (since_seq, limit)):
            d = json.loads(r["payload"])
            d["_seq"] = r["seq"]
            d["_ts_ingest_ms"] = r["ts_ingest_ms"]
            yield d

    def max_seq(self, table: str) -> int:
        v = self.db.execute(f"SELECT MAX(seq) m FROM {table}").fetchone()["m"]
        return int(v or 0)

    def sequence_gaps(self, table: str) -> list[tuple[int, int]]:
        """AUTOINCREMENT is dense unless rows were deleted. Any gap means
        tampering or corruption, and the day should be marked invalid."""
        seqs = [r["seq"] for r in
                self.db.execute(f"SELECT seq FROM {table} ORDER BY seq")]
        gaps = []
        for a, b in zip(seqs, seqs[1:]):
            if b != a + 1:
                gaps.append((a, b))
        return gaps

    # ---------- lifecycle markers ----------
    def mark_start(self, component: str) -> str:
        run_id = str(uuid.uuid4())
        self.db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)",
                        (f"run:{component}", json.dumps({
                            "run_id": run_id, "pid": os.getpid(),
                            "host": socket.gethostname(),
                            "started_ms": int(time.time() * 1000),
                            "clean_shutdown": False})))
        self.db.commit()
        return run_id

    def mark_clean_shutdown(self, component: str):
        row = self.db.execute("SELECT value FROM meta WHERE key=?",
                              (f"run:{component}",)).fetchone()
        if row:
            d = json.loads(row["value"])
            d["clean_shutdown"] = True
            d["stopped_ms"] = int(time.time() * 1000)
            self.db.execute("UPDATE meta SET value=? WHERE key=?",
                            (json.dumps(d), f"run:{component}"))
            self.db.commit()

    def last_run(self, component: str) -> dict | None:
        row = self.db.execute("SELECT value FROM meta WHERE key=?",
                              (f"run:{component}",)).fetchone()
        return json.loads(row["value"]) if row else None

    def close(self):
        try:
            self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass
        self.db.close()


class ProcessLock:
    """OS-level singleton. Prevents the duplicate-runner failure that has
    already happened once in this project (two runners, same state file)."""

    def __init__(self, name: str = "capture",
                 directory: Path | None = None):
        d = directory or Path(__file__).resolve().parent
        self.path = d / f".{name}.lock"
        self.fh = None

    def acquire(self) -> bool:
        """Lock byte 0 and never write into the locked file.

        msvcrt.locking() locks a byte range at the CURRENT file position, so
        writing or truncating the same file afterwards fails against our own
        lock. Owner metadata goes to a sidecar instead.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import msvcrt
            self.fh = open(self.path, "a+b")
            self.fh.seek(0)
            try:
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                self.fh.close()
                self.fh = None
                return False
        except ImportError:
            import fcntl
            self.fh = open(self.path, "a+b")
            try:
                fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                self.fh.close()
                self.fh = None
                return False
        try:
            self.path.with_suffix(".owner").write_text(
                f"{os.getpid()} {socket.gethostname()} {time.time():.0f}")
        except Exception:
            pass
        return True

    def release(self):
        if self.fh:
            try:
                import msvcrt
                self.fh.seek(0)
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                try:
                    import fcntl
                    fcntl.flock(self.fh, fcntl.LOCK_UN)
                except Exception:
                    pass
            try:
                self.fh.close()
            except Exception:
                pass
            self.fh = None

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError(
                f"another process holds {self.path}; refusing to start")
        return self

    def __exit__(self, *a):
        self.release()
