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
| Orders matched to a full path | 303 |
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
| Probe fills | 277 |
| Full-size commitments | 14 |
| Commit win rate | 92.86% |
| Mean commit ask | 66.50c |
| Diagnostic q1 P&L | $0.46 |
| PTC P&L | **$52.56** |
| Standardized q15 maker control | $2.96 |
| PTC minus diagnostic | $52.10 |
| PTC minus standardized control | $49.60 |
| PTC max drawdown | $11.65 |
| PTC worst close window | $-9.34 |

### The actual 60-second branches

| Branch | n | Win rate | Mean observed ask | Ask in 60-80c | Direct q15 taker edge |
|---|---:|---:|---:|---:|---:|
| Filled before decision | 277 | 69.31% | 60.77c | 54.87% | +1.01c |
| Still unfilled | 26 | **96.15%** | 61.92c | 57.69% | **+24.93c** |

This table is the central test.  If the no-fill branch remains profitable after
its own observed ask is included, the mechanism is not an artifact of pairing
no-fill with unrelated cheap quotes.

## Moving-block uncertainty

| comparison                   |   block_minutes |   observed |     ci_lo |    ci_hi |   p_nonpositive |   n_blocks |
|:-----------------------------|----------------:|-----------:|----------:|---------:|----------------:|-----------:|
| pnl_ptc_minus_pnl_diagnostic |              15 |    52.0997 |   14.6762 |  89.8235 |          0.0042 |        125 |
| pnl_ptc_minus_pnl_diagnostic |              30 |    52.0997 |   13.5335 |  90.3979 |          0.0046 |         63 |
| pnl_ptc_minus_pnl_diagnostic |              60 |    52.0997 |   10.4166 |  93.5123 |          0.0070 |         32 |
| pnl_ptc_minus_pnl_diagnostic |              90 |    52.0997 |   14.0797 |  93.9736 |          0.0027 |         21 |
| pnl_ptc_minus_pnl_control    |              15 |    49.5953 | -210.9423 | 316.4553 |          0.3678 |        125 |
| pnl_ptc_minus_pnl_control    |              30 |    49.5953 | -245.8637 | 354.6120 |          0.3791 |         63 |
| pnl_ptc_minus_pnl_control    |              60 |    49.5953 | -252.1295 | 372.4992 |          0.3899 |         32 |
| pnl_ptc_minus_pnl_control    |              90 |    49.5953 | -243.4110 | 368.0305 |          0.3827 |         21 |

With only two live days, none of these intervals should be read as a durable
performance estimate.  They are a hostile concentration check, not prospective
proof.

## Sensitivity

### Wait

|   wait_seconds |   commits |   commit_win_rate |   commit_mean_ask |   ptc_pnl |   ptc_max_drawdown |
|---------------:|----------:|------------------:|------------------:|----------:|-------------------:|
|        60.0000 |   14.0000 |            0.9286 |            0.6650 |   52.5597 |            11.6548 |
|       120.0000 |   14.0000 |            0.9286 |            0.6650 |   52.5597 |            11.6548 |
|       180.0000 |   14.0000 |            0.9286 |            0.6650 |   52.5597 |            11.6548 |
|       300.0000 |   14.0000 |            0.9286 |            0.6650 |   52.5597 |            11.6548 |

### Cancel latency

|   cancel_latency_seconds |   commits |   ptc_pnl |   ptc_max_drawdown |
|-------------------------:|----------:|----------:|-------------------:|
|                   0.0000 |   14.0000 |   52.5597 |            11.6548 |
|                   0.2500 |   14.0000 |   52.5597 |            11.6548 |
|                   0.5000 |   14.0000 |   52.5597 |            11.6548 |
|                   1.0000 |   14.0000 |   52.5597 |            11.6548 |
|                   2.0000 |   14.0000 |   52.5597 |            11.6548 |
|                   3.0000 |   14.0000 |   52.5597 |            11.6548 |

### Slippage

|   slippage_cents |   commits |   ptc_pnl |   ptc_max_drawdown |
|-----------------:|----------:|----------:|-------------------:|
|           0.0000 |   14.0000 |   52.5597 |            11.6548 |
|           1.0000 |   15.0000 |   56.9947 |            11.8020 |
|           2.0000 |   17.0000 |   66.2949 |            11.9489 |
|           3.0000 |   20.0000 |   81.0504 |            12.0957 |
|           5.0000 |   20.0000 |   78.5532 |            12.3885 |

### IOC fill fraction

|   ioc_fill_fraction |   commits |   ptc_pnl |   ptc_max_drawdown |
|--------------------:|----------:|----------:|-------------------:|
|              0.2500 |   14.0000 |   10.8795 |            11.8393 |
|              0.5000 |   14.0000 |   24.7730 |             9.6729 |
|              0.7500 |   14.0000 |   38.6663 |             9.0695 |
|              1.0000 |   14.0000 |   52.5597 |            11.6548 |

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
