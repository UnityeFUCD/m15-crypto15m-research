# Settlement Probability Engine — clean-room result

## Verdict: **FAIL / CANDIDATE ONLY**

The model and economic policy were selected on the chronological validation
slice. The final test slice was evaluated once.

## Data

- Feature rows: 255,621
- Unique markets: 38,070
- Calendar days: 73
- Fixed horizons: [13, 12, 10, 8, 6, 4, 2]

## Validation-selected policy

- Model: `struct`
- Horizon: 12 minutes remaining
- Minimum predicted all-in edge: 5.0¢
- Validation opportunities: 82
- Validation P&L: $68.24
- Validation mean/day: $3.79

## Sealed test policy

- Trades: 92
- Total P&L at q15: $-73.66
- Mean/day: $-3.68
- Daily SD: $15.09
- Edge/contract: -5.34¢
- Win rate: 0.4674
- Average entry: 50.40¢
- Maximum drawdown: $108.79
- Worst close window: $-10.57

## Probability increment on sealed test

- Δ log loss vs raw midpoint: +0.001364
- Δ Brier vs raw midpoint: +0.000730
- Δ log loss vs market-only calibrator: +0.000385

Negative deltas are improvements.

## Uncertainty

- Day-bootstrap mean/day 95% interval: [$-9.98, $2.86]
- P(mean day ≤ 0): 0.8718

| Block | CI on total P&L | P(nonpositive) |
|---:|---:|---:|
| 15 min | [$-206.10, $65.29] | 0.8482 |
| 30 min | [$-216.11, $66.54] | 0.8451 |
| 60 min | [$-216.10, $70.69] | 0.8375 |
| 90 min | [$-216.38, $69.20] | 0.8436 |

## Hard-gate components

- probability_increment: **FAIL**
- positive_test_economics_and_n_ge_50: **FAIL**
- positive_day_ci: **FAIL**
- positive_all_block_cis: **FAIL**
- leave_one_coin: **FAIL**
- leave_one_week: **FAIL**

## Interpretation

A probability model can be useful even when the final trading gate fails.
The key result is whether causal spot/settlement state improves sealed-test
calibration beyond the contemporaneous quote. A positive validation
backtest without that calibration increment is threshold search, not a
scientific breakthrough.

No result authorizes live orders while the repository's existing account
KILL state remains active.