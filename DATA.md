# Data dictionary — read this before using anything in `data/`

Four wrong verdicts were produced in this project, and **three of them came
from using a sampled file as if it were the population.** The files differ
enormously in trustworthiness. This page says which is which.

---

## TL;DR — what to use

| use this | for | why |
|---|---|---|
| `book_full.parquet` | any population claim about edge | unsampled, all 4 minutes, 24h, no band filter |
| `paths_full.parquet` | price evolution entry → settlement | unsampled, full ml 0–15 path |
| `spot_1m.parquet` | intra-window index movement | validated proxy, sign agreement 0.932 |
| `underlying.parquet` | outcomes, A0/A1 | ground truth, 41,334 markets |
| `trades_lsm.parquet` | queue reconstruction | public tape for our 301 traded markets |

**Never** base a population claim on `ladder_paths`, `premium_history2`,
`book_history`, or `book_minutes_15_45`. See the trap list.

---

## The population files (trustworthy)

### `book_full.parquet` — 39,428 rows, one per market
The unsampled book. Every market with a decodable close and a valid quote at
8–14 minutes to close. **No band filter applied at fetch time**, so band
sensitivity is testable.

`ticker, coin, close_ts, minute, side, bid, ask, vol, ml, won`

- `side` — the FAVOURITE side at entry (`yes`/`no`), frozen at first valid obs
- `bid`/`ask` — for the favourite side, already sign-corrected
- `won` — 1 if the favourite settled in the money
- `minute` — UTC close minute ∈ {0,15,30,45}, balanced: 9746/9996/9799/9887

In-band (0.65 ≤ bid < 0.80): **8,187 markets over 73 days.**

### `paths_full.parquet` — one row per market, full price path
Same population, but retains every 1-minute observation from ml 15 down to 0.

`ticker, coin, close_ts, minute, side, bid, ask, entry_ml, n_pts, won, path`

`path` is a JSON list, newest-last: `[{"ml":14.0,"yb":0.71,"ya":0.74,"v":120}, …]`
where `yb`/`ya` are **raw YES** bid/ask (not favourite-adjusted). The favourite
side is fixed once at entry and stored in `side` — do **not** re-derive it
later in the path, that leaks the outcome into the position definition.

### `spot_1m.parquet` — 632k rows, 1-minute Coinbase OHLC
`coin, ts, open, high, low, close, vol`. `ts` is a UNIX second.

**This is a PROXY, not the CF index.** Validated in
`research/validate_spot_proxy.py` against the 41,334 known A0/A1 pairs:

```
level correlation      0.99996
return correlation     0.9807
sign agreement a1>=a0  0.9320   (threshold set at 0.90 before testing)
```

Good enough for conditioning variables (distance to A0, realized vol). **Not**
good enough to redefine settlement — 6.8% of windows disagree on the winner.
Any result built on it must be labelled a proxy result.

### `underlying.parquet` — 41,334 markets, ground truth
`ticker, coin, wkey, series, close, result, a0, a1, …`

- `a0` = A0 (window open), `a1` = A1 (close). YES settles iff `a1 >= a0`,
  **ties pay YES**. Reconstructing this reproduces the printed result on
  **99.98%** of markets.
- `wkey` is `%y%b%d%H%M` in **ET**. True close UTC = `wkey + 4h` (summer).
- The `close` column is `expected_expiration_time`, which is the true close
  **plus five minutes**. Using it as the close silently shifts every window.

⚠️ **Snapshot truncation.** Markets that closed after the snapshot have no
`result` and are silently dropped by a naive merge. They are **not missing at
random** — the 17 affected LSM markets were worth **−$134** and moved a
measured result from +$143.38 to +$9.38. Recovered in
`lsm_missing_outcomes.parquet`; always merge it back.

### `trades_lsm.parquet` — public trade tape, 2,114,639 trades / 301 markets
`ticker, ts, count, yes_c, taker_book_side, taker_outcome_side`

- `ts` — epoch **milliseconds** (not seconds)
- `yes_c` — trade price on the **YES scale, in integer cents**
- `trade_id` was dropped after dedup; the file is zstd-compressed (93MB → 5.8MB)

Regenerate the raw form with `research/fetch_trades_lsm.py`.

### `queue_ahead.parquet` — 303 rows, one per LSM order
Derived by `research/queue_reconstruct.py`. Columns: `order_id, ticker, held,
yes_px, submitted, filled, did_fill, would_win, rest_seconds, queue_ahead,
n_trades_ahead`.

> **🔑 KEY RESULT — `queue_ahead` is 0.00 for all 303 orders.**
>
> This is not a bug. Exact-float and rounded-to-cent matching both give zero,
> and the reason is latency: median fill is **2.13s**, p25 is 0.78s, and 83 of
> 277 fills land in under a second. Nothing has time to trade ahead of us.
>
> **We are already at the front of the queue.** Therefore queue position
> *cannot* be the mechanism behind the +10.4pp fill bias, and gating on it
> cannot help. The adverse selection comes from the **taker's** information at
> the moment they lift us, not from our position in line.
>
> This kills arm B (queue-gate) of the Audit E experiment and leaves the
> signed-index cancel (arm C) as the only viable intervention.

### Execution records
- `orders_history.parquet` — 1,802 orders. **Only 303 are LSM**
  (`client_order_id` starts with `lsm`); the other 1,499 belong to different
  strategies and must be filtered out or every number is wrong.
  ⚠️ Side semantics: `action=sell` on `side=yes` means the position **held**
  is NO. The fills table reports the held side directly. They disagree on 163
  of 303 rows if you don't convert.
- `fills_history.parquet` — 2,280 fills, has `is_taker` and `fee_cost`.
- `lsm_missing_outcomes.parquet` — the 17 recovered outcomes above.

---

## ☠️ The trap list — files that produced wrong answers

### `ladder_paths.parquet` — **HOT 28.6% SAMPLE**
:00/:30 only, 5,624 markets. Its in-band subset shows **+3.99c** maker edge
against a true population **+2.37c** on the same minutes.

```
P(random draw of 1,091 markets >= +3.99c) = 0.0751
```

It is a **1.43σ fluctuation**, not a better market. It agrees with
`book_full` exactly where they overlap — it is simply a lucky subset. It
produced the retracted claim that the strategy was profitable. See
`research/reconcile_edge.py`.

### `premium_history2.parquet` — **minute :00 only, 8 of 24 hours**
2,780 rows. Testing a minute-:00 hypothesis on a minute-:00-only file gives
P=0.0002 for free. This produced the retracted HCR **PASS**.

### `book_minutes_15_45.parquet` — superseded
2,101 markets. An earlier version was fetched with `sort_values("close_utc")`
and covered only the earliest 18 days against :00/:30's 73 — that produced the
retracted HCR **FAIL**. Fixed to a random sample, then superseded entirely by
`book_full`.

### `book_history.parquet`, `grid`, `yesno`, `vol_entries`, `postshift_nofilter`
Older exploratory extracts. Sampling not characterised. Do not use for
population claims.

### `cf_index.parquet` — endpoints only, **not a path**
41,440 rows ≈ one per market: `window_open` / `window_close` at 900s spacing.
There is **no intra-window index path** here. Use `spot_1m.parquet`.

---

## Known results, so you don't re-derive them

**Established:**
- Settlement `a1 >= a0`, 60s trailing means, ties pay YES — 99.98%
- Favourite at 65–80c wins **~73%**; YES base rate across all markets 0.4950
- **Maker fill bias**: `P(fill|lose)=0.9884` vs `P(fill|win)=0.8848`,
  +10.4pp, CI [+5.5, +15.2], P=0.0001. Never-filled orders were **96.2%
  winners** vs a 71.6% base rate. Cost **$186.58 of $195.96** in 2 days.
- Population edge: maker **+2.29c**, fill-corrected **−0.43c**, taker −1.55c
- Per-market SD is **44× per-market EV** — daily P&L is ~all noise
- Taker fee `ceil(0.07·C·p·(1−p)·10⁴)/10⁴`; maker fee zero

**Refuted (don't reopen):** HCR (SOL was the whole effect), DRC, RACE timeout
tuning, price-band search (every band ≤ 0 fill-corrected), coin dropping,
volatility filters, hour filters, queue-depth filters, mean reversion
(endpoint artifact).

**Minute :00** — the strongest candidate, and it still **FAILS** the decision
rule: EV per **submitted** contract **+0.80c, 95% CI [−2.17, +3.73]**,
P(≤0)=0.2986. It beats :30 reliably (:30 is −0.99c/submitted, P(≤0)=0.965)
but "better than a loser" ≠ profitable. See `research/minute_zero.md`.

---

## Two arithmetic traps that cost real errors

**1. Edge per FILLED ≠ edge per SUBMITTED.**
```
corr()        = Σ(maker·pf)/Σ(pf) = E[maker | FILLED]
EV_submitted  = mean(maker·pf)    = E[maker | filled] × fill_rate
```
Multiplying an edge-per-filled by a count of *posted* markets assumes a 100%
fill rate — the exact assumption this project exists to reject.

**2. `(maker*pf).mean()` is already a fraction.** Multiplying by contracts
gives dollars. Dividing by 100 again makes every figure 100× too small.

---

## Reproducing

```bash
python research/audit_unsampled.py        # population audit, exactly 6 coins
python research/reconcile_edge.py         # why +3.99c was retracted
python research/prospective_execution.py  # 4 pre-registered fill predictions
python research/audit_a_minute_fills.py   # minute-specific execution
python research/audit_b_hostile.py        # hostile matching, frozen :00 rule
python research/validate_spot_proxy.py    # proxy validation
```

Signal construction requires **exactly six coins** (`nc == 6`) to match
`capture.hcr.common_return`, and the calm window needs **24 contiguous**
900s observations. Earlier scripts used `nc >= 4`, which is not the spec.
