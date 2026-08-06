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
| Full-path matched | 303 |
| Full-path unmatched | 0 |
| Matched actual P&L | $9.38 |
| Unmatched actual P&L | $0.00 |
| Unmatched win rate | nan% |
| Unmatched order fill rate | nan% |

Direct PTC comparisons are restricted to the 303 matched orders.
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
| Effective orders | 303 |
| Probe fills | 268 |
| No-fill branches | 35 |
| Full-size commitments | 10 |
| Commit win rate | 90.00% |
| Mean commit ask | 76.60c |
| Diagnostic-only P&L | $-0.38 |
| PTC P&L | **$17.84** |
| Standardized q15 maker control | $2.96 |
| PTC minus diagnostic | $18.22 |
| PTC minus control | $14.88 |
| PTC maximum drawdown | $12.91 |
| PTC worst close window | $-12.55 |

## Actual joint branch economics at the corrected 60-second decision

| Branch | n | Win rate | Mean ask | Ask in 60-80c | Direct q15 taker edge | Mean effective wait |
|---|---:|---:|---:|---:|---:|---:|
| Filled | 268 | 69.03% | 68.42c | 59.70% | +0.82c | 98.5s |
| No fill | 35 | **91.43%** | 80.58c | 31.43% | **+12.76c** | 98.6s |

This is the decisive correction to the original model.  Fill state and delayed
ask now come from the same order and the same market path.

## Moving-block concentration check

| comparison                |   block_minutes |   observed |     ci_lo |    ci_hi |   p_nonpositive |
|:--------------------------|----------------:|-----------:|----------:|---------:|----------------:|
| pnl_ptc_minus_pnl_diag    |              15 |    18.2216 |  -15.0435 |  45.6349 |          0.1247 |
| pnl_ptc_minus_pnl_diag    |              30 |    18.2216 |  -15.3363 |  45.8472 |          0.1205 |
| pnl_ptc_minus_pnl_diag    |              60 |    18.2216 |  -15.1922 |  47.0023 |          0.1308 |
| pnl_ptc_minus_pnl_diag    |              90 |    18.2216 |  -15.7643 |  46.3371 |          0.1253 |
| pnl_ptc_minus_pnl_control |              15 |    14.8772 | -238.5484 | 293.4253 |          0.4685 |
| pnl_ptc_minus_pnl_control |              30 |    14.8772 | -267.5337 | 318.4037 |          0.4607 |
| pnl_ptc_minus_pnl_control |              60 |    14.8772 | -287.0268 | 351.6175 |          0.4778 |
| pnl_ptc_minus_pnl_control |              90 |    14.8772 | -267.4874 | 324.7318 |          0.4760 |

Only two live days exist, so these intervals cannot validate a production
mean.  They test whether one isolated close block carries the result.

## Wait sensitivity

|     wait |        n |   no_fills |   commits |   commit_win_rate |   commit_mean_ask |   ptc_pnl |   ptc_max_dd |
|---------:|---------:|-----------:|----------:|------------------:|------------------:|----------:|-------------:|
|  60.0000 | 303.0000 |    35.0000 |   10.0000 |            0.9000 |            0.7660 |   17.8416 |      12.9102 |
| 120.0000 | 303.0000 |    32.0000 |    5.0000 |            1.0000 |            0.7460 |   17.6339 |      13.1800 |
| 180.0000 | 303.0000 |    29.0000 |    5.0000 |            0.8000 |            0.7540 |    2.9317 |      14.2769 |
| 300.0000 | 303.0000 |    27.0000 |    4.0000 |            1.0000 |            0.7275 |   15.6728 |      10.9900 |

## Cancel-latency sanity check

|   latency |   commits |   ptc_pnl |   ptc_max_dd |
|----------:|----------:|----------:|-------------:|
|    0.0000 |   10.0000 |   17.8416 |      12.9102 |
|    0.2500 |   10.0000 |   17.8416 |      12.9102 |
|    0.5000 |   10.0000 |   17.8416 |      12.9102 |
|    1.0000 |   10.0000 |   17.8416 |      12.9102 |
|    2.0000 |   10.0000 |   17.8416 |      12.9102 |
|    3.0000 |   10.0000 |   17.8416 |      12.9102 |

## Slippage stress

|   slippage_c |   commits |   ptc_pnl |   ptc_max_dd |
|-------------:|----------:|----------:|-------------:|
|       0.0000 |   10.0000 |   17.8416 |      12.9102 |
|       1.0000 |   10.0000 |   16.3986 |      13.0542 |
|       2.0000 |    8.0000 |   12.6404 |      13.1980 |
|       3.0000 |    7.0000 |   26.4238 |       8.2942 |
|       5.0000 |    5.0000 |   19.1614 |      11.2700 |

## IOC-fill stress

|   fill_fraction |   commits |   ptc_pnl |   ptc_max_dd |
|----------------:|----------:|----------:|-------------:|
|          0.2500 |   10.0000 |    3.2638 |      14.6833 |
|          0.5000 |   10.0000 |    8.1234 |      12.9807 |
|          0.7500 |   10.0000 |   12.9823 |      11.2783 |
|          1.0000 |   10.0000 |   17.8416 |      12.9102 |

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
