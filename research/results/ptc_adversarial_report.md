# Probe-Then-Commit — adversarial direct-live double test

## Why this rerun was necessary

The first PTC audit was promising, but it paired simulated fill/no-fill with
population delayed asks conditional only on the eventual outcome.  That can
invent cheap no-fill opportunities.  This audit instead joins each actual LSM
order to its own committed full path and keeps fill time, delayed ask, outcome,
and maker price together.

## Data integrity

| Item | Value |
|---|---:|
| LSM orders | 303 |
| Orders matched to a full path | 220 |
| Close windows | 125 |
| Calendar days | 2 |
| Reconciled actual P&L | $9.38 |

## Frozen primary direct replay

```text
probe quantity          1
nominal wait            60 seconds
actual decision point   first completed one-minute path point after 60s
cancel latency stress   0s in primary; 0.25-3s reported separately
commit condition        probe still unfilled at effective cancel time
commit ask              60-80c
commit quantity         15
commit cap              one per close window
IOC                     displayed ask, no retry, no chase
```

| Metric | Direct two-day counterfactual |
|---|---:|
| Probe fills | 199 |
| Full-size commitments | 11 |
| Commit win rate | 90.91% |
| Mean commit ask | 67.09c |
| Diagnostic q1 P&L | $9.43 |
| PTC P&L | **$46.20** |
| Standardized q15 maker control | $2.96 |
| PTC minus diagnostic | $36.77 |
| PTC minus standardized control | $-94.20 |
| PTC max drawdown | $10.33 |
| PTC worst close window | $-9.34 |

### The actual 60-second branches

| Branch | n | Win rate | Mean observed ask | Ask in 60-80c | Direct q15 taker edge |
|---|---:|---:|---:|---:|---:|
| Filled before decision | 199 | 73.87% | 61.86c | 59.30% | +3.77c |
| Still unfilled | 21 | **95.24%** | 62.38c | 57.14% | **+22.64c** |

This table is the central test.  If the no-fill branch remains profitable after
its own observed ask is included, the mechanism is not an artifact of pairing
no-fill with unrelated cheap quotes.

## Moving-block uncertainty

| comparison                   |   block_minutes |   observed |     ci_lo |    ci_hi |   p_nonpositive |   n_blocks |
|:-----------------------------|----------------:|-----------:|----------:|---------:|----------------:|-----------:|
| pnl_ptc_minus_pnl_diagnostic |              15 |    36.7722 |    2.0566 |  70.6891 |          0.0196 |        125 |
| pnl_ptc_minus_pnl_diagnostic |              30 |    36.7722 |    1.1313 |  72.6872 |          0.0229 |         63 |
| pnl_ptc_minus_pnl_diagnostic |              60 |    36.7722 |   -0.5073 |  73.9054 |          0.0267 |         32 |
| pnl_ptc_minus_pnl_diagnostic |              90 |    36.7722 |    3.6129 |  76.5368 |          0.0152 |         21 |
| pnl_ptc_minus_pnl_control    |              15 |    43.2378 | -224.8422 | 319.7353 |          0.3897 |        125 |
| pnl_ptc_minus_pnl_control    |              30 |    43.2378 | -261.3875 | 355.9620 |          0.3999 |         63 |
| pnl_ptc_minus_pnl_control    |              60 |    43.2378 | -267.9480 | 376.3657 |          0.4097 |         32 |
| pnl_ptc_minus_pnl_control    |              90 |    43.2378 | -259.5525 | 371.3415 |          0.4028 |         21 |

With only two live days, none of these intervals should be read as a durable
performance estimate.  They are a hostile concentration check, not prospective
proof.

## Sensitivity

### Wait

|   wait_seconds |   commits |   commit_win_rate |   commit_mean_ask |   ptc_pnl |   ptc_max_drawdown |
|---------------:|----------:|------------------:|------------------:|----------:|-------------------:|
|        60.0000 |   11.0000 |            0.9091 |            0.6709 |   46.2022 |            10.3348 |
|       120.0000 |   11.0000 |            0.9091 |            0.6709 |   46.2022 |            10.3348 |
|       180.0000 |   11.0000 |            0.9091 |            0.6709 |   46.2022 |            10.3348 |
|       300.0000 |   11.0000 |            0.9091 |            0.6709 |   46.2022 |            10.3348 |

### Cancel latency

|   cancel_latency_seconds |   commits |   ptc_pnl |   ptc_max_drawdown |
|-------------------------:|----------:|----------:|-------------------:|
|                   0.0000 |   11.0000 |   46.2022 |            10.3348 |
|                   0.2500 |   11.0000 |   46.2022 |            10.3348 |
|                   0.5000 |   11.0000 |   46.2022 |            10.3348 |
|                   1.0000 |   11.0000 |   46.2022 |            10.3348 |
|                   2.0000 |   11.0000 |   46.2022 |            10.3348 |
|                   3.0000 |   11.0000 |   46.2022 |            10.3348 |

### Slippage

|   slippage_cents |   commits |   ptc_pnl |   ptc_max_drawdown |
|-----------------:|----------:|----------:|-------------------:|
|           0.0000 |   11.0000 |   46.2022 |            10.3348 |
|           1.0000 |   12.0000 |   51.0778 |            10.4820 |
|           2.0000 |   14.0000 |   60.8180 |            10.6289 |
|           3.0000 |   17.0000 |   76.0129 |            10.7757 |
|           5.0000 |   17.0000 |   74.0971 |            11.0685 |

### IOC fill fraction

|   ioc_fill_fraction |   commits |   ptc_pnl |   ptc_max_drawdown |
|--------------------:|----------:|----------:|-------------------:|
|              0.2500 |   11.0000 |   16.7841 |             5.2200 |
|              0.5000 |   11.0000 |   26.5902 |             5.2200 |
|              0.7500 |   11.0000 |   36.3961 |             7.7495 |
|              1.0000 |   11.0000 |   46.2022 |            10.3348 |

A credible effect should deteriorate monotonically with latency and slippage.
Failure of that sanity check is a rejection signal, as it was for RACE.

## Interpretation limits

1. The q1 probe is inferred from first-fill time of original q10-30 orders.
   Visible size may affect taker behavior.
2. The first completed one-minute path point can be almost a minute later than
   the nominal wait; the effective wait is reported in the branch CSV.
3. IOC depth and partial fills were not historically observed.  Fill-fraction
   and slippage rows are stress tests, not estimates.
4. The sample has 303 orders and two days.  It cannot identify long-run P&L or
   ruin probability.
5. A real trial must randomize by close window between control, diagnostic-only,
   and PTC.  The account remains below its kill floor, so this audit does not
   authorize live orders.

## Decision rule

PTC deserves a prospective trial only when all of the following are true in
this direct audit:

- no-fill branch edge remains positive after its own observed ask and fee;
- PTC beats diagnostic-only, showing the commit branch adds value;
- PTC beats the standardized maker control;
- latency and slippage sensitivity are monotone;
- no single close block carries the result.

Passing those conditions makes PTC a strong candidate, not a deployment PASS.
