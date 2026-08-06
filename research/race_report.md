# RACE / LSM ledger — corrected replay

## Status

| conclusion | verdict |
|---|---|
| Actual LSM P&L is **+$9.38**, not +$143.38 | **validated** |
| The missing −$134 was real settlement loss | **validated** |
| Those losses began filling fast | **validated** |
| Late-fill adverse selection exists | **supported, strongly** |
| RACE would have prevented the giveback | **refuted** — recovers $27 of $134 |
| 2 seconds is optimal | **not established** |
| Removing rank 3 is profitable | **unresolved** |
| Policies C–E deliver +$212…+$283 forward | **not established** |

## Authoritative ledger

```
all account orders                     1,802
LSM orders (client_order_id "lsm")       303
  with >=1 fill 277 | never filled 26
LSM fills                                432
unique traded markets                    275
  in snapshot 258 | recovered 17 | unresolved 0
duplicate same-ticker orders               2
FINAL contracts                        6,096
SUM OF ACTUAL EXCHANGE FILL P&L       $+9.38
unmatched fills 0   duplicate fill_ids 0
```

Elapsed time uses the **exchange** `created_time`, not a local pre-request clock.

## Rank (true submission sequence, all orders ranked)

| rank | orders | unique tickers | contracts | P&L |
|---|---|---|---|---|
| 1 | 116 | 116 | 2,508 | **+$71.68** |
| 2 | 111 | 111 | 2,395 | −$12.89 |
| 3 | 49 | 49 | 1,163 | −$28.11 |
| 4+ | 1 | 1 | 30 | −$21.30 |
| **total** | 277 | 275 | 6,096 | **+$9.38** |

cap-2 = +$58.79, gain **+$49.41** = rank3 + rank4+. Reconciles exactly.

**Superseded:** an earlier version ranked on *filled orders only*, shifting 15
of 277 orders and reporting rank3 = −$75.81 / gain = +$97.11. Both sets
reconcile internally; do not mix them.

## The mechanism — supported

Fill fraction by outcome, all 303 orders, unfilled counted as 0%:

| timeout | winner fill | loser fill | advantage |
|---|---|---|---|
| 2s | 42.42% | 38.29% | **+4.1pp** |
| 10s | 67.31% | 67.46% | −0.2pp |
| 30s | 77.14% | 86.86% | −9.7pp |
| **full** | **87.68%** | **98.26%** | **−10.6pp** |

The 26 never-filled orders **won 96.15%** (25/26) at an average posted price of
69.58c against 69.15c for filled orders — same price, opposite outcomes.

## Policy matrix — same chronological sequence

| policy | P&L | max DD | worst window | windows |
|---|---|---|---|---|
| A actual, all ranks, full wait | +$9.38 | $315.04 | −$73.50 | 124 |
| B cap2, full wait | +$58.79 | $237.03 | −$50.40 | 124 |
| C cap2, 2s/2s | +$212.07 | $133.64 | −$27.80 | 89 |
| D cap2, 5s/2s | +$227.50 | $150.04 | −$27.80 | 95 |
| E cap2, 8s/2s | +$262.81 | $146.54 | −$27.80 | 100 |

Drawdown reduction is partly **mechanical**: C–E hold positions in 89–100
windows against A's 124. Less participation, less exposure.

## Cancel-latency stress

| policy | +0s | +0.25s | +0.5s | +1s | +2s | +3s |
|---|---|---|---|---|---|---|
| C 2s/2s | +212.07 | +222.18 | +211.09 | +196.40 | +172.62 | +144.46 |
| D 5s/2s | +227.50 | +252.43 | +235.05 | +223.33 | +196.47 | +179.77 |
| E 8s/2s | +262.81 | +274.04 | +249.86 | +237.59 | +203.35 | +177.54 |

Survives realistic latency at roughly 70% of ideal. **But +0.25s of latency
*improves* all three policies** — a genuine monotone mechanism cannot do that.
Noise signature.

**The runner's main loop is 5 seconds. A 2-second cancel is not executable
without a dedicated scheduler that does not exist.**

## Significance — fails under every scheme

| bootstrap | P(≤0) |
|---|---|
| close-window, 7 seeds | 0.053 – 0.064 |
| moving block 30 min | 0.064 |
| moving block 60 min | 0.072 |
| moving block 90 min | 0.059 |

No multiple-comparison correction applied; ~15 timeout/rank policies were
examined, so the true figure is worse. **Policy E is the strongest in-sample
candidate, not a validated policy.**

Concentration: one close window (08-05 10:30) carries **56%** of the 2s effect.
The all-ranks uniform-timeout curve swings $94 between adjacent horizons
(2s +$141, 5s +$35, 8s +$113).

## The −$134 block

12 of 17 markets lost. First fills at 0.4, 0.4, 0.7, 0.7, 0.8, 0.9, 1.2, 1.4,
2.9, 3.7, 5.2, 15.9s — **losers filled fast**. Two of five winners filled at
89.4s and 35.8s. Six consecutive close windows (18:00 → 19:15): a sustained
directional regime, not late-fill toxicity. A 2s cancel recovers $27.40.

## Errors corrected across five passes

1. pooled all account strategies (10+ prefixes; LSM is 303 of 1,802 orders)
2. 17 markets missing outcomes (−$134)
3. pandas microsecond timestamps → 7-digit `close_ts`, collapsing 15-min spacing
4. win rate per fill instead of per market
5. policy A silently excluded rank 3
6. fill fractions excluded the 26 never-filled orders
7. orders-table `side` inverted vs fills-table for `action=sell` (163 of 303)
8. rank computed on filled orders only
9. single-seed p-value (0.034 → 0.053–0.064 across seeds)

Every one inflated a result. On two days of data the errors all pointed toward
a more exciting answer.

## Next

Do not tune further on these two days. Pre-register **two** candidates — 5s and
8s for rank 1, 2s for rank 2 — and log randomised timeout arms prospectively.
The passive recorder supports this at zero capital.
