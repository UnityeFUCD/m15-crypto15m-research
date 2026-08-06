# Settlement Probability Engine — clean-room audit

## Verdict: **FAIL / CANDIDATE ONLY**

All model and policy choices were made on TRAIN/VALID. TEST was evaluated once.

## Validation-selected policy

- Model: `full`
- Horizon: 12 minutes remaining
- Minimum predicted all-in edge: 3.0¢
- Validation trades: 283
- Validation P&L: $213.67

## Sealed test result

- Trades: 275
- Total P&L at q15: $-143.77
- Mean/day: $-7.19
- Daily SD: $31.65
- Edge/contract: -3.49¢
- Win rate: 0.4945
- Average entry: 51.33¢
- Maximum drawdown: $301.86
- Worst close window: $-13.31

## Probability increment on sealed test

- Δ log loss vs raw midpoint: +0.001708
- Δ Brier vs raw midpoint: +0.000776
- Δ log loss vs market-only calibrator: +0.000728
- Δ Brier vs market-only calibrator: +0.000350

Negative differences are improvements.

## Uncertainty

- Day-bootstrap mean/day 95% CI: [$-20.64, $6.04]
- P(mean day ≤ 0): 0.8456

| Block | CI on total P&L | P(nonpositive) |
|---:|---:|---:|
| 15 min | [$-383.21, $93.45] | 0.8807 |
| 30 min | [$-384.09, $92.72] | 0.8772 |
| 60 min | [$-389.74, $106.75] | 0.8696 |
| 90 min | [$-388.01, $105.47] | 0.8731 |

## Hard gates

- physical_information_beyond_market: **FAIL**
- positive_test_economics_and_n_ge_50: **FAIL**
- positive_day_ci: **FAIL**
- positive_all_block_cis: **FAIL**
- leave_one_coin_all_positive: **FAIL**
- leave_one_week_all_positive: **FAIL**
- leave_one_hour_all_positive: **FAIL**

## Interpretation

A profitable threshold backtest is not enough. Causal settlement state must
improve sealed-test probability quality beyond a market-only calibrator,
and the policy must survive clustered uncertainty and leave-one-group tests.

Nothing here authorizes live orders while the account KILL state is active.