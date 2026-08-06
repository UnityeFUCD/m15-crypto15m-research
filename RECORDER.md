# Passive recorder — quick start

Collects the one thing the historical data cannot contain: **what would have
happened at prices we never posted at.**

Places **no orders**. Costs nothing. Cannot lose money. Does not touch the
live runner.

## Why it exists

All 608 orders in the live history used one policy — join the bid. You cannot
estimate `P(fill | action)` when the action never varied. This produces that
variation without trading, by opening *virtual* orders at several prices and
following the real tape to see whether the queue ahead of them got consumed.

## Run it

```bash
export KALSHI_CRED_DIR=/path/to/creds        # never committed
python -m capture.recorder --seconds 3600 --poll 5
python -m capture.recorder                   # runs until Ctrl-C
```

Writes to `capture/capture.db` (SQLite WAL). Restart-safe and idempotent —
re-running never duplicates a trade or a snapshot.

## What it records

| table | contents |
|---|---|
| `book_snapshots` | full L2 depth, both sides, every poll |
| `public_trades` | every trade: price, size, timestamp, aggressor side |
| `virtual_orders` | one per (market, price policy): the price, and the displayed size that was ahead at open |
| `virtual_queue` | per poll: cumulative volume traded at that exact price, current displayed size, and both fill bounds |

Policies tracked per market: `back_one_tick`, `join`, `improve_one_tick`,
`improve_two_ticks` (skipped when it would cross the spread).

## Reading the result

```
join  px 0.74 | ahead 326.4  traded@px 452.1  | OPT True  PESS True
```

326 contracts were ahead at 74c; 452 later traded there. A real order fills
even from the back of the queue — **observed, not modelled.**

- `filled_optimistic` — any trade at our price (front-of-queue assumption)
- `filled_pessimistic` — traded volume exceeded everything ahead of us

Truth is bracketed by the two. The pessimistic bound never advances the queue
on cancellations, because L2 cannot tell whether a size decrease happened
ahead of us or behind.

## Verified API details

```
GET /markets/{ticker}/orderbook
    -> {"orderbook_fp": {"yes_dollars": [[price,size],...],
                         "no_dollars":  [[price,size],...]}}
```

The key is **`orderbook_fp`** with **`*_dollars`** arrays — not `orderbook`
with `yes`/`no`. Parsing it wrong yields a silently empty book.

Prices ascend, so the best bid is the **last** entry. A YES ask is
`1 - (best NO bid)`.

## Gotcha

Git Bash rewrites `/tmp/...` arguments to the MSYS temp directory before a
Windows Python sees them, while `Path('/tmp')` inside Python resolves to
`C:\tmp`. Passing `--db /tmp/x.db` from bash therefore writes somewhere other
than where a Python reader looks. Use absolute Windows paths, or the default.
