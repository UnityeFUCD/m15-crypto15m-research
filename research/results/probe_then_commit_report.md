# Probe-Then-Commit (PTC) — clean-room audit

## Executive result

The strongest new mechanism is **not another directional filter**.  It is an
execution inversion:

> Post one contract as a diagnostic maker probe.  A quick fill is treated as a
> warning and receives no additional size.  A sufficiently long no-fill is
> treated as favorable information; cancel the probe, confirm cancellation, and
> take at most one bounded full-size position in the close window.

This uses the project's proven adverse selection as information instead of
letting it decide the entire position size.

The frozen primary candidate is:

```text
probe quantity          1
wait                    60 seconds
commit condition        probe still unfilled
commit ask              60¢ through 80¢
commit quantity         15
commit cap              one market per close window
selection               lowest observable ask, then fixed coin order
execution               cancel-confirm-reconcile, IOC, no retry, no chase
```

### Primary model-based result

| Metric | Estimate |
|---|---:|
| Live P(fill by 60s \| eventual win) | 84.33% |
| Live P(fill by 60s \| eventual loss) | 94.19% |
| Posterior P(win \| no fill by 60s) | **87.18%** |
| P(win \| fill by 60s) | 69.32% |
| Break-even all-in price after q15 fee | 86.3¢ |
| Mean commits over 73-day population | 219.5 |
| Mean modeled P&L | **$417.89** |
| Mean modeled P&L/day | **$5.72** |
| 95% interval from fill-rate uncertainty | [$0.73, $10.05]/day |
| P(mean day ≤ 0) | 0.012 |
| Median max drawdown | $39.20 |
| 95th-percentile max drawdown | $93.24 |
| P(hit $211 from $300 over historical sequence) | 0.850% |

**Classification:** this is a strong mechanistic candidate if the modeled
interval is positive, but it is not yet identified as live policy value.  The
fill branch is observed; the counterfactual IOC branch is not.  A randomized
prospective experiment is still required.

## Why the design is different

The old maker strategy allowed an informed taker to decide whether the account
received 10–30 contracts.  PTC makes the taker decide only whether the account
receives **one** diagnostic contract.  Full size is reserved for the branch in
which the market refuses to trade back to the maker bid.

The asymmetry is deliberate:

- toxic fill branch: at most one maker contract;
- favorable no-fill branch: at most one q15 IOC per close;
- no-fill but expensive ask: no trade;
- no duplicated position and no chase.

## Actual live discrimination by horizon

The following branch labels use actual first-fill timestamps and recovered
settlements, including never-filled orders as no-fill observations.

|   seconds |   filled_by_t |   not_filled_by_t |
|----------:|--------------:|------------------:|
|    1.0000 |        0.6988 |            0.7227 |
|    2.0000 |        0.7407 |            0.6964 |
|    5.0000 |        0.6949 |            0.7460 |
|   10.0000 |        0.7150 |            0.7191 |
|   15.0000 |        0.7004 |            0.7632 |
|   30.0000 |        0.6935 |            0.8182 |
|   45.0000 |        0.7016 |            0.8000 |
|   60.0000 |        0.6932 |            0.8718 |
|   90.0000 |        0.6914 |            0.9118 |
|  120.0000 |        0.6926 |            0.9091 |
|  180.0000 |        0.6912 |            0.9355 |
|  240.0000 |        0.6934 |            0.9310 |
|  300.0000 |        0.6920 |            0.9630 |
|  420.0000 |        0.6920 |            0.9630 |

The purpose of the table is not to select the best historical second.  It shows
when no-fill becomes informative enough to pay the spread and fee.  The primary
60-second wait was frozen because it is the first fully observable one-minute
price point and can be implemented by an event-driven scheduler.

## Frozen FR2 rerun on the full population

The earlier FR2 result came from the hot ``ladder_paths`` sample.  On
``paths_full`` the exact two-minute, 60–80¢ taker rule produces:

| Population | Trades | Edge/contract | P&L | Mean/day | Max DD |
|---|---:|---:|---:|---:|---:|
| No volume filter | 2214 | +0.39¢ | $174.64 | $2.39 | $306.80 |
| Volume ≥ 2,000 | 1052 | +0.43¢ | $90.48 | $1.24 | $290.80 |

Chronological result for the production-like volume-filtered rule:

| Split | Trades | Edge/contract | q20 P&L |
|---|---:|---:|---:|
| Train | 436 | +1.28¢ | $111.25 |
| Valid | 288 | -0.07¢ | $-4.30 |
| Test | 328 | -0.25¢ | $-16.47 |

This is the decisive full-population check on the previously proposed FR2 rule.

## PTC quantity sensitivity at the frozen 60s / 80¢ rule

|   commit_qty |   mean_pnl_per_day |   pnl_day_ci_lo |   pnl_day_ci_hi |   median_max_drawdown |   p_hit_211_over_history |
|-------------:|-------------------:|----------------:|----------------:|----------------------:|-------------------------:|
|      10.0000 |             3.8323 |          0.5725 |          6.4297 |               29.6774 |                   0.0033 |
|      15.0000 |             5.6550 |          0.2972 |         10.3782 |               39.5865 |                   0.0100 |
|      20.0000 |             7.6996 |          0.7264 |         13.2232 |               48.6084 |                   0.0150 |

## Wait sensitivity — descriptive only

|   wait_seconds |   delay_minutes |   live_nofill_win_rate |   mean_pnl_per_day |   pnl_day_ci_lo |   pnl_day_ci_hi |   median_max_drawdown |
|---------------:|----------------:|-----------------------:|-------------------:|----------------:|----------------:|----------------------:|
|        30.0000 |          1.0000 |                 0.8182 |             5.1686 |         -1.1981 |         10.4081 |               56.3396 |
|        60.0000 |          1.0000 |                 0.8718 |             5.6550 |          0.2972 |         10.3782 |               39.5865 |
|        90.0000 |          2.0000 |                 0.9118 |             5.0282 |          1.9107 |          7.7686 |               30.5264 |
|       120.0000 |          2.0000 |                 0.9091 |             5.0670 |          1.7327 |          8.1379 |               28.9975 |
|       180.0000 |          3.0000 |                 0.9355 |             4.4198 |          2.0965 |          6.5213 |               24.9730 |
|       300.0000 |          5.0000 |                 0.9630 |             2.9849 |          1.3511 |          4.3842 |               23.3870 |

A real policy should show a broad, economically coherent region rather than one
isolated timeout.  No timeout is promoted from this table.

## Audit C: can a one-minute signed-index cancel prevent the losses?

Among actual losing orders:

| Cutoff | Share whose first fill already occurred |
|---|---:|
| 1 second | 29.07% |
| 5 seconds | 62.79% |
| 30 seconds | 88.37% |
| 60 seconds | 94.19% |
| 120 seconds | 96.51% |

If most losing orders are already filled before one complete Coinbase bar is
available, a one-minute state cancel cannot be the main remedy.  Higher-frequency
spot would be needed to test a subsecond cancel.  PTC avoids that race by making
the pre-signal exposure one contract.

The descriptive reversal-hazard table is stored in
``research/results/reversal_hazard.csv``.  It uses Coinbase only as a proxy and
does not redefine settlement.

## Queue-position correction

``queue_ahead = 0`` is only a **lower bound**, not proof that initial queue was
zero.  Volume ahead can be canceled, and one aggressor trade can consume both
orders ahead and the first probe fill at the same timestamp.  The official
queue endpoint is the authoritative prospective measurement.  PTC does not
require the historical queue reconstruction to be exact; it requires only the
observed first-fill timestamp of a one-contract-equivalent probe.

## What must be tested live

Use a dedicated event-driven order lifecycle, not the five-second scan loop:

1. Place one post-only probe at the favorite bid.
2. Persist its treatment before submission.
3. At exactly 60 seconds, request cancellation if still resting.
4. Wait for authoritative cancellation confirmation.
5. Reconcile fills during the cancel race.
6. Refresh the book.
7. Among no-fill candidates in the close window, select the lowest ask no higher
   than 80¢.
8. Send one IOC for q15; no retry and no chase.
9. Maker probe plus IOC quantity may never exceed the configured target on one
   ticker.
10. Record every eligible opportunity, including no-fill, rejection, risk block,
    partial IOC, and API failure.

Randomize eligible close windows between:

- control: existing q15 maker;
- PTC: q1 probe plus frozen no-fill commit;
- diagnostic-only: q1 probe, never commit.

The diagnostic-only arm separates the value of reducing toxic exposure from the
value of the no-fill IOC branch.

## PASS requirements

PTC enters production size only if prospective data show:

- positive P&L per assigned opportunity;
- positive day/block-clustered lower bound;
- superiority to both control and diagnostic-only arms;
- no dependence on one coin or one week;
- actual IOC prices/depth inside the frozen ceiling;
- latency sensitivity that worsens monotonically;
- risk of hitting the configured floor within the approved limit.

Until then the result is **candidate, not deployable proof**.
