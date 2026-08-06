# Paired Maker Lock — full-path audit

## Verdict: **FAIL / RESEARCH ONLY**

Two maker bids are posted on complementary YES and NO outcomes. If both
fill, the displayed spread is locked. Exactly one fill leaves directional
legging risk.

The policy was selected on validation and evaluated once on the sealed
chronological test period.

## Validation-selected policy

- Minimum spread: 5.0¢
- Cancel both if neither fills after: 1 minutes
- Maximum paired markets per close: 6
- Selection: widest
- Validation active markets: 157
- Validation P&L q15: $-332.70

## Sealed test

- Posted pairs: 82
- Active pairs: 57
- Both-fill rate: 35.37%
- Single-leg rate: 34.15%
- No-fill rate: 30.49%
- Edge per posted pair: -14.000¢
- Edge per active pair: -20.140¢
- Total P&L q15: $-172.20
- Mean/day q15: $-8.61
- Daily SD: $14.15
- Maximum drawdown: $172.20
- Worst close window: $-9.30

## Uncertainty

- Day-bootstrap mean/day CI: [$-15.05, $-3.04]
- P(mean day ≤ 0): 1.0000

| Block | Total-P&L CI | P(nonpositive) |
|---:|---:|---:|
| 15 min | [$-237.90, $-110.40] | 1.0000 |
| 30 min | [$-237.00, $-111.15] | 1.0000 |
| 60 min | [$-244.20, $-105.15] | 1.0000 |
| 90 min | [$-247.50, $-103.50] | 1.0000 |

## Interpretation

A positive both-fill branch is guaranteed arithmetically, but the strategy
passes only if locked-spread income exceeds one-leg losses after
chronological selection and concentration controls.

The fill reconstruction uses complete-minute quote crossings and positive
candle volume. It does not identify queue priority or q15 capacity, so even
a historical PASS requires q1 prospective execution before scaling.