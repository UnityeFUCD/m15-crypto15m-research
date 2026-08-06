# Phase 1 — audit and implementation plan

Verified against the live API and against the code, not against comments.

---

## A. Critical correction: queue position IS available

`DATA_SPEC.md` claimed exchange-reported queue position does not exist. **That
was wrong.** Both endpoints verified working on 2026-08-06:

```
GET /portfolio/orders/{order_id}/queue_position
    -> 200 {"queue_position_fp": "0.00"}

GET /portfolio/orders/queue_positions?market_tickers=<csv>
GET /portfolio/orders/queue_positions?event_ticker=<ticker>
    -> 200 {"queue_positions": null}      (null only because nothing is resting)
```

The batch form **requires** `market_tickers` or `event_ticker` — calling it with
`order_ids` returns
`400 {"details": "Need to specify market_tickers or event_ticker"}`.

**How the error was made, so it is not repeated:** the runner's header asserts
*"Kalshi never emits L3, so queue position is unobservable to everyone."* That
is true of the *order book* (L2 only, `[price, aggregate_size]`) and false of
the *portfolio* endpoints. The absence was inferred from the book shape and
from a code comment instead of being probed. Phase 1 of the spec says *"Do not
trust comments as proof."* That rule was violated; the correction is above.

`queue_position_fp` is now the authoritative measurement. Tape-derived bounds
are demoted to a clearly-labelled fallback for when a read fails.

**Not yet validatable:** every order in the account is `executed`, so all reads
return `0.00`. A non-zero reading cannot be confirmed until an order actually
rests. The capture code must therefore treat `0.00` on a resting order as
suspicious and record `queue_read_ok` separately from the value.

---

## B. Audit findings — verified from code

### B1. State mutates before acknowledgement — CONFIRMED

`live/lsm_runner.py` lines 883–895:

```python
oid = place_maker(tk, side, px, q_use)
STATE["orders"] += 1                                   # unconditional
STATE["entered"].add(tk)                               # unconditional
STATE["per_window"][ct] = STATE["per_window"].get(ct, 0) + 1
STATE["deployed"] += q_use * px                        # unconditional
STATE["open_cost"].append([ct, q_use * px])
persist()                                              # persisted regardless
if oid:                                                # only this is guarded
    STATE["resting"][oid] = {...}
```

If `place_maker` returns `None` — rejection, timeout, or network failure — the
runner still counts the market as entered, still burns the per-window slot, and
still adds the cost to `deployed`. Exposure inflates and a market we never
entered is permanently blocked for the day. This is the root cause of the
`resting` double-count already recorded in the project history.

### B2. Pagination is entirely absent — CONFIRMED

`grep -n cursor live/lsm_runner.py` returns **nothing**. Every portfolio read
takes the first page only. With `limit=200` and more than 200 positions or
orders, state is silently truncated.

### B3. Money is float64 throughout — CONFIRMED

`STATE["deployed"] += q_use * px`, `fee = ceil(0.07*c*p*(1-p)*1e4)/1e4`,
`total_traded_dollars` parsed with `float()`. Every accumulator drifts.

### B4. `acct.py` discards error bodies — CONFIRMED

```python
return r.status_code, (r.json() if r.status_code == 200 else None)
```

Non-200 responses lose the body, so the `api_errors` table cannot be populated
from this client. It is also **GET-only**, which is why my first batch-queue
probe failed with a misleading 400.

`lsm_runner.py`'s own `api(method, path, params, body)` *does* support POST and
*does* preserve error bodies — the two clients disagree. The capture system
gets one client, not two.

### B5. Exposure derived from `total_traded_dollars` — CONFIRMED

`acct.py` and the runner both use `abs(total_traded_dollars)` as current
exposure. That field is **historical traded value**, not current cost basis: a
market traded in and out repeatedly accumulates it. Exposure must come from
`position_fp` × price.

### B6. Restart failure modes

- `ENT_F` is date-stamped (`lsm_entered_YYYYMMDD.json`); at UTC midnight the
  entered-set silently empties. Mitigated by `seed_entered_from_exchange()`,
  which only recovers markets with a **non-zero position** — a resting unfilled
  order is not recovered.
- `STATE["resting"]` is in-memory only. After a crash, resting orders are
  orphaned until the eligibility window passes.
- No process lock: two runners can run concurrently (this happened, PID 16564).

### B7. Schema mismatches vs the live API

| assumed | actual |
|---|---|
| queue position unavailable | `queue_position_fp`, both endpoints |
| settlement cost `*_fp` | only `yes_total_cost_dollars` / `no_total_cost_dollars` |
| `revenue` in dollars | **integer cents** |
| `expected_expiration_time` = close | close **+ 5 minutes** (39,453 markets verified) |
| `count` | always null; use `count_fp` |

---

## C. Build order

Priority is correctness of the foundation before breadth of tables.

| # | module | why first |
|---|---|---|
| 1 | `capture/money.py` | every other module stores money; fixed-point must exist first |
| 2 | `capture/kalshi.py` | one client: POST, pagination, error bodies, retries, rate limiting |
| 3 | `capture/store.py` | SQLite WAL ledger, idempotent upserts, sequence numbers |
| 4 | `capture/schemas.py` | corrected tables incl. authoritative queue |
| 5 | `capture/queue.py` | the corrected core — both endpoints + flow variables |
| 6 | `capture/treatments.py` | deterministic, persisted, restart-safe assignment |
| 7 | `capture/episodes.py` | decision-episode identity and cooldown |
| 8 | `tests/` + `tests/mock_exchange.py` | must pass with no credentials |
| 9 | `capture/collector.py` | capture-only runner |
| 10 | `capture/reconcile.py` | startup reconciliation, refuse-to-trade gate |
| 11 | `capture/report.py` | daily integrity report |
| 12 | `capture/export.py` | SQLite -> immutable parquet partitions |
| 13 | `analysis/` | survival estimation starter |

## D. Storage decision

**SQLite in WAL mode** as the authoritative ledger, exported to immutable
parquet partitions.

Rejected the current `capture/schemas.py` approach (read whole parquet →
concat → rewrite): it is O(n²) in a day's rows, and a crash mid-rewrite
destroys the entire partition. SQLite gives real transactions, `UNIQUE`
constraints for idempotent ingestion, and crash safety for free. Parquet
becomes an export artifact, never the write path.

## E. Scope note

Phases 8 (external venue data), 9 (private WebSocket) and 14 (analysis) depend
on 1–7 and 10. They are staged after the foundation is proven by tests. Nothing
is deployed at material size; the experimental runner is hard-capped at 2
contracts and defaults to capture-only.
