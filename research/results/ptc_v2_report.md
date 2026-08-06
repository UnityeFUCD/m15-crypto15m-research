> # ⛔ SUPERSEDED — DO NOT USE
>
> Computed on **220 of 303** LSM orders, where path availability was a proxy
> for the outcome (+$153.78 matched vs -$144.40 unmatched). That cohort is now
> complete and the numbers below did not survive it — in particular the
> control arm was **+$140.41** here and is **+$2.96** on the full ledger, so
> "PTC minus control = -$119.70" was the survivorship gap, not PTC.
>
> **Authoritative: [research/results/PTC_FINAL.md](PTC_FINAL.md) — verdict FAIL.**

# Probe-Then-Commit v2 — timestamp-corrected direct audit

## Critical correction

The first adversarial run was invalid.  Pandas stored the parsed datetimes at
microsecond resolution, while the script divided their integer representation
by 1e9 as though it were nanoseconds.  The effective 60-second wait became
about 1.8 billion seconds, so eventual no-fills were misclassified as
60-second no-fills.  This v2 uses ``Timestamp.timestamp()`` and hard assertions
on every time interval.

## Coverage and missingness

| Item | Value |
|---|---:|
| LSM orders | 303 |
| Full-path matched | 220 |
| Full-path unmatched | 83 |
| Matched actual P&L | $153.78 |
| Unmatched actual P&L | $-144.40 |
| Unmatched win rate | 60.24% |
| Unmatched order fill rate | 93.98% |

Direct PTC comparisons are restricted to the 220 matched orders.
The unmatched cohort is reported because path availability can be non-random.

## Frozen primary direct replay

```text
q1 diagnostic maker probe
nominal wait 60 seconds
first complete one-minute price snapshot after that wait
cancel-confirm-reconcile
one q15 IOC per close window, ask 60-80c
lowest ask first, fixed coin-order tie break
no retry, no chase
```

| Metric | Result |
|---|---:|
| Effective orders | 220 |
| Probe fills | 191 |
| No-fill branches | 29 |
| Full-size commitments | 8 |
| Commit win rate | 87.50% |
| Mean commit ask | 76.38c |
| Diagnostic-only P&L | $8.87 |
| PTC P&L | **$20.71** |
| Standardized q15 maker control | $140.41 |
| PTC minus diagnostic | $11.84 |
| PTC minus control | $-119.70 |
| PTC maximum drawdown | $13.46 |
| PTC worst close window | $-11.88 |

## Actual joint branch economics at the corrected 60-second decision

| Branch | n | Win rate | Mean ask | Ask in 60-80c | Direct q15 taker edge | Mean effective wait |
|---|---:|---:|---:|---:|---:|---:|
| Filled | 191 | 73.82% | 69.33c | 58.12% | +5.55c | 99.4s |
| No fill | 29 | **89.66%** | 82.15c | 31.03% | **+10.87c** | 99.3s |

This is the decisive correction to the original model.  Fill state and delayed
ask now come from the same order and the same market path.

## Moving-block concentration check

| comparison                |   block_minutes |   observed |     ci_lo |    ci_hi |   p_nonpositive |
|:--------------------------|----------------:|-----------:|----------:|---------:|----------------:|
| pnl_ptc_minus_pnl_diag    |              15 |    11.8374 |  -20.3935 |  37.6428 |          0.2013 |
| pnl_ptc_minus_pnl_diag    |              30 |    11.8374 |  -19.4622 |  37.6409 |          0.2044 |
| pnl_ptc_minus_pnl_diag    |              60 |    11.8374 |  -21.7159 |  39.3988 |          0.2133 |
| pnl_ptc_minus_pnl_diag    |              90 |    11.8374 |  -22.3590 |  40.9063 |          0.2258 |
| pnl_ptc_minus_pnl_control |              15 |  -119.6980 | -320.0093 |  93.4482 |          0.8752 |
| pnl_ptc_minus_pnl_control |              30 |  -119.6980 | -336.1540 | 117.1101 |          0.8457 |
| pnl_ptc_minus_pnl_control |              60 |  -119.6980 | -366.5437 | 149.9859 |          0.8173 |
| pnl_ptc_minus_pnl_control |              90 |  -119.6980 | -335.7603 | 124.2978 |          0.8387 |

Only two live days exist, so these intervals cannot validate a production
mean.  They test whether one isolated close block carries the result.

## Wait sensitivity

|     wait |        n |   no_fills |   commits |   commit_win_rate |   commit_mean_ask |   ptc_pnl |   ptc_max_dd |
|---------:|---------:|-----------:|----------:|------------------:|------------------:|----------:|-------------:|
|  60.0000 | 220.0000 |    29.0000 |    8.0000 |            0.8750 |            0.7637 |   20.7074 |      13.4602 |
| 120.0000 | 220.0000 |    27.0000 |    5.0000 |            1.0000 |            0.7460 |   26.6039 |       5.2200 |
| 180.0000 | 220.0000 |    24.0000 |    4.0000 |            0.7500 |            0.7675 |    7.6222 |      14.5569 |
| 300.0000 | 220.0000 |    22.0000 |    3.0000 |            1.0000 |            0.7700 |   18.8948 |       6.3900 |

## Cancel-latency sanity check

|   latency |   commits |   ptc_pnl |   ptc_max_dd |
|----------:|----------:|----------:|-------------:|
|    0.0000 |    8.0000 |   20.7074 |      13.4602 |
|    0.2500 |    8.0000 |   20.7074 |      13.4602 |
|    0.5000 |    8.0000 |   20.7074 |      13.4602 |
|    1.0000 |    8.0000 |   20.7074 |      13.4602 |
|    2.0000 |    8.0000 |   20.7074 |      13.4602 |
|    3.0000 |    8.0000 |   20.7074 |      13.4602 |

## Slippage stress

|   slippage_c |   commits |   ptc_pnl |   ptc_max_dd |
|-------------:|----------:|----------:|-------------:|
|       0.0000 |    8.0000 |   20.7074 |      13.4602 |
|       1.0000 |    8.0000 |   19.5526 |      13.6042 |
|       2.0000 |    6.0000 |   13.0226 |      13.7480 |
|       3.0000 |    5.0000 |   24.4696 |       7.3300 |
|       5.0000 |    4.0000 |   20.4856 |       7.3300 |

## IOC-fill stress

|   fill_fraction |   commits |   ptc_pnl |   ptc_max_dd |
|----------------:|----------:|----------:|-------------:|
|          0.2500 |    8.0000 |   11.2371 |       6.7349 |
|          0.5000 |    8.0000 |   14.3941 |       7.1241 |
|          0.7500 |    8.0000 |   17.5506 |      10.2922 |
|          1.0000 |    8.0000 |   20.7074 |      13.4602 |

## Verdict rule

PTC is promoted to a randomized prospective candidate only if:

1. the corrected no-fill branch remains positive after its own ask and fee;
2. PTC beats diagnostic-only, proving the IOC branch adds value;
3. PTC beats the standardized maker control;
4. latency/slippage worsen the result monotonically;
5. unmatched paths do not carry the opposite outcome;
6. no moving time block carries the effect.

Even a pass here is not permission to trade while the account is below its
kill floor.  It only justifies building the randomized control/diagnostic/PTC
experiment.
