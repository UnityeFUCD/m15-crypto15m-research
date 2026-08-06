# PTC — final verdict: **FAIL**

> **CORRECTION (2026-08-06).** Figures on this page quoting **-0.43c** as the
> fill-corrected population edge used the OLD fill model (0.843/0.962), which
> was estimated on 286 orders BEFORE the 17 late-closing outcomes were
> recovered. With the corrected model measured on all 303 (0.8848/0.9884) the
> population edge is **+0.02c per submitted contract, 95% CI [-1.57, +1.62],
> P(<=0) 0.49** on 8,187 in-band markets over 73 days.
>
> The corrected reading is **indistinguishable from zero**, not negative.
> Statements below that the strategy is "below zero" or "not profitable"
> overstate the evidence; the defensible claim is that no edge is measurable
> either way. The -0.43c figure was also per FILLED contract while +0.02c is
> per SUBMITTED - two different denominators that were being compared.


**Authoritative.** Supersedes `research/PTC_STATUS.md`,
`research/results/ptc_v2_report.md` and `research/ptc_adversarial_v2.py`, all
of which were computed on 220 of 303 orders and are now marked SUPERSEDED.

Binding reason, stated once:

> **The completed 303-order ledger cannot distinguish PTC from probe-only or
> from control, and the difference that does appear is carried by single
> orders.** Dropping the one best commitment turns PTC_60's IOC branch
> negative. One order — a single ETH probe filling at 61.69s — explains the
> entire "added latency helps" anomaly.

Deployment is not authorized. The architecture is built, tested and frozen so
the randomized trial can settle it.

---

## Part 2 — the data gate: PASSED

The blocking fault is fixed. All 82 missing markets (83 orders) were fetched;
`research/ptc_reconcile.py` exits zero on 13 hard assertions.

| check | result |
|---|---|
| LSM orders | 303 |
| resolved outcomes | 303 |
| **path coverage** | **303/303** |
| duplicate order IDs / fill IDs | 0 / 0 |
| unmatched fills | 0 |
| exchange fill P&L | **+$9.3814** |
| 17 recovered late-close markets | all carried |
| held-side agreement with fills table | **1.0000** (naive: 0.4765) |
| close derived from ticker | yes — `expected_expiration_time` is +5.0 min |
| timestamp units | seconds, all order-to-close in 0–900s |

**Missingness was outcome-correlated and is now empty.** Before: +$153.78
(matched, 220) vs −$144.40 (unmatched, 83) — path availability was effectively
a proxy for the result. After: one cohort, +$9.38, 303 orders.

Two causes, both structural: all 17 late-closing markets were excluded because
`paths_full` filtered on `result.isin(yes/no)` against a truncated snapshot,
and 65 more were dropped by an `8 ≤ ml ≤ 14` entry gate that is correct for
defining eligibility but wrong for recovery — the PTC audit reads only `path`
and takes the held side from the order.

## Part 3 — the complete historical double test

### A correction v2 still needed: decision clock vs observation clock

v2 scored `probe_filled = first_fill ≤ effective_wait`, where `effective_wait`
is the first complete one-minute candle after the nominal wait — **99.4s for a
60s design**. A probe filling at 75s was counted as filled although the frozen
rule cancels at 60s.

v3 separates them: **decision** `t_cancel = wait + latency` (a fill at or
before this is a real cancel-race fill), **observation** `t_obs` = first candle
at or after `t_cancel` (where the IOC is priced). This reclassified 4 orders
and changes the primary result, so it is applied and reported.

### Arms, all 303 orders, q15

| arm | submitted | probe fills | no-fills | commits | commit win | total P&L | $/assigned | max DD | worst window |
|---|---|---|---|---|---|---|---|---|---|
| CONTROL | 4,545 | 277 | 26 | 0 | — | +$2.96 | 0.0098 | $229.66 | −$36.75 |
| PROBE_ONLY_60 | 303 | 264 | 39 | 0 | — | +$0.58 | 0.0019 | $16.48 | −$2.45 |
| PTC_60 | 513 | 264 | 39 | 14 | 78.57% | +$2.87 | 0.0095 | $18.14 | −$12.55 |
| PTC_120 | 393 | 270 | 33 | 6 | 83.33% | **+$8.88** | 0.0293 | $13.18 | −$10.07 |

> Note what happened to CONTROL. On the 220-order cohort it was **+$140.41**;
> on the complete 303 it is **+$2.96**. The retracted "PTC minus control =
> −$119.70" was almost entirely the survivorship gap, not a property of PTC.

### The branch mechanism is real

| wait | branch | n | win rate | mean ask | eligible 60–80c | taker edge |
|---|---|---|---|---|---|---|
| 60 | filled | 264 | 69.32% | 68.31c | 156 | +1.52c |
| 60 | **no fill** | 39 | **87.18%** | 80.03c | 15 | +2.28c |
| 120 | filled | 270 | 69.26% | 69.96c | 139 | +0.21c |
| 120 | **no fill** | 33 | **90.91%** | 82.77c | 6 | +9.64c |

An unfilled probe predicts an ~18–22pp higher win rate. That is the confirmed
maker adverse selection used as information, and it survives the complete
ledger. **The mechanism is not what fails.**

### The comparisons all fail

Moving-block bootstrap, 5 fixed seeds, P(≤0) reported as a range:

| comparison | observed | 15 min | 30 min | 60 min | 90 min |
|---|---|---|---|---|---|
| PTC_60 − PROBE_ONLY | +$2.29 | .435–.444 | .432–.443 | .434–.446 | .464–.473 |
| PTC_120 − PROBE_ONLY | +$8.30 | .239–.246 | .234–.243 | .245–.251 | .244–.252 |
| PTC_60 − CONTROL | −$0.09 | .505–.513 | .516–.529 | .562–.580 | .593–.613 |
| PTC_120 − CONTROL | +$5.92 | .488–.497 | .498–.509 | .546–.563 | .575–.591 |

Every interval spans zero. Nothing approaches 5%.

*(An earlier version of this bootstrap re-centered resampled totals by
`observed/mean`; when the resampled mean landed near zero that ratio exploded
and printed a `[−85302, +83404]` interval. It now resamples the per-window
mean, which needs no such correction.)*

### Concentration — the disqualifying finding

| arm | commits | best commit | IOC branch without it |
|---|---|---|---|
| PTC_60 | 14 | +$5.16 | **−$1.41** (goes negative) |
| PTC_120 | 6 | +$5.16 = 59% | +$3.52 |

Leave-one-out on the complete ledger:

| comparison | drop day | drop coin | drop window-cluster |
|---|---|---|---|
| PTC_60 − PROBE_ONLY | all + | **min −8.37** | **min −4.96** |
| PTC_120 − PROBE_ONLY | all + | all + (min +2.24) | all + (min +3.15) |
| PTC_60 − CONTROL | min −91.65 | min −33.04 | — |
| PTC_120 − CONTROL | min −87.45 | min −23.90 | — |

PTC_120 vs probe-only is the only comparison surviving all three — on **six
commitments**.

## Part 4 — hostile sensitivity: incoherent, and diagnosed

**Cancel latency.** PTC_60 goes **+$2.87 → +$13.27** when 2s of latency is
added. A handicap that makes 4.6× the money is not robustness.

Cause, isolated to a single order: `KXETH15M-26AUG050415-15` fills at
**61.69s**. At ≥2s latency the longer cancel race catches it as a probe fill,
which deletes a commitment that lost **−$11.16**. The "improvement" is the
removal of one losing trade.

**Slippage.** Non-monotone in both arms (PTC_60: 2.87, 0.85, −6.17, **+7.18**,
−0.95). Cause: slippage is added to `exec_price` *before* the 60–80c
eligibility test, so more cost pushes candidates out of the band and silently
removes them. Holding the trade set fixed and only paying more gives the
correct monotone answer:

| slippage | as-implemented | **same trade set** |
|---|---|---|
| 0c | +$2.87 | +$2.87 |
| 1c | +$0.85 | +$0.85 |
| 2c | −$6.17 | **−$0.52** |
| 3c | +$7.18 | **−$2.22** |
| 5c | −$0.95 | **−$6.01** |

PTC_60 goes negative at **2c of slippage** on the same trades.

**IOC fill fraction** is monotone (25/50/75/100% → 1.04/1.65/2.26/2.87), the
only axis that behaves.

**Ask ceiling** stays frozen at 80c and is reported as sensitivity only. It is
wildly non-monotone (76c −0.42, 78c −9.03, 80c +2.87, 82c +10.94, 85c +12.75)
and 80c is not the best — which is precisely why it must not be re-chosen here.

## Parts 5–6 — architecture: built and tested

`capture/ptc.py`, 16 states, disabled by default (`PTC_ENABLED=0`,
`PTC_LIVE_ENABLED=0`), shadow/test mode needs no credentials.

**139 tests pass** (`tests/test_ptc.py` 51, `tests/test_ptc_replay_faults.py`
16, existing suite 72). Tests were written before the implementation.

Properties enforced: cancel timer starts from exchange `created_time`, never a
local clock; `commit_allowed()` fails closed without an affirmative
`CANCEL_CONFIRMED` with `filled_qty == 0`; a fill inside the cancel race
suppresses the commitment; `client_order_id` is a pure function of
(opportunity, phase) so a crash between POST and persist adopts rather than
resubmits; one commitment per close window under hostile repetition;
integer-micros money only; every skip, block and error stays in the
denominator; any side or timestamp integrity failure forces KILL.

Fault injection covers process death after submit and after cancel-request,
delayed acknowledgement, duplicate responses, paginated truncation, network
timeout after a possibly-accepted POST, stale book, exchange pause,
out-of-order fills and duplicate runner. **No fault produces duplicate or
excess exposure.** Deterministic replay of all 303 orders is byte-identical
across runs.

## Parts 7 & 9 — trial size and risk

Required sample, recomputed from the completed dataset (`ptc_prospective_spec.md`):

| comparison | windows/arm | days at 24/arm/day |
|---|---|---|
| PTC_120 − PROBE_ONLY | 2,272 | **95** |
| PTC_60 − PROBE_ONLY | 98,088 | 4,087 |
| PTC_60 − CONTROL | 2.1 billion | — |

PTC vs control is unmeasurable because the observed difference is ~zero
against CONTROL's $13.10/window dispersion. **PTC's one clearly measurable
effect is a 9× dispersion reduction** ($13.10 → $1.49 per window), not a
higher mean. That is a risk property, and it is worth having — but it is not
the edge the brief asks to establish.

Risk simulation, 100,000 chronological block-bootstrap paths per cell:

| qty | fails |
|---|---|
| **q5** | **none** |
| q10 | adverse_blocks |
| q15 | adverse_blocks |
| q20 | adverse_blocks, zero_edge |

q5 is the starting size — derived, not chosen.

*(The first version of this simulator allocated the full 100,000 × 2,160 path
matrix per cell, ~7 GB across 112 cells, and paged to disk. It now streams in
5,000-path chunks: same 100,000 paths, ~740 MB peak.)*

## Part 8 — the ten gates

| # | gate | result |
|---|---|---|
| 1 | all 303 paths recovered / missingness non-informative | **PASS** |
| 2 | PTC beats probe-only | **FAIL** — P(≤0) 0.234–0.473 |
| 3 | PTC beats standardized control | **FAIL** — PTC_60 −$0.09; P(≤0) ≥ 0.49 |
| 4 | positive day/block-clustered lower bound | **FAIL** — every CI spans zero |
| 5 | not carried by one coin/day/week/window | **FAIL** — PTC_60 negative without its best commit |
| 6 | IOC prices/depth/latency/partials preserve value | **FAIL** — negative at 2c slippage |
| 7 | added latency/cost do not improve incoherently | **FAIL** — 2s latency → 4.6× |
| 8 | no accounting/side/timestamp/pagination/restart fault | **PASS** — 139 tests |
| 9 | randomized prospective experiment passes | **NOT RUN** |
| 10 | risk simulation satisfies drawdown/ruin | **PASS at q5 only** |

**Verdict: FAIL.** Six gates fail on evidence, one is unrun.

This is a **mechanism PASS and a money FAIL**, and the two must not be
conflated. The no-fill branch genuinely predicts an 18–22pp higher win rate on
the complete ledger. What is unproven is that acting on it with an IOC makes
money after the ask, the fee and the loss of the maker rebate — on 6–14
commitments across 2 days, with one order able to flip the sign.

## What would change the verdict

Only the prospective trial: **PTC_120 vs probe-only, q5, ~95 days.** That is
the single comparison with both a coherent mechanism and a reachable sample
size. Everything needed to run it exists and is frozen.

## Preconditions, unchanged

The account is at **$136.27** against a **$211** static floor.
`commit_quantity()` returns 0 and the machine reports KILL. That is correct
behaviour and nothing here weakens it.
