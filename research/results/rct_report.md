# Runaway Confirmation Taker — full-population audit

## Verdict: **FAIL / CANDIDATE ONLY**

The strategy uses no maker-fill model and no Coinbase proxy. It waits on
the original favourite and pays the observed delayed ask only after the
favourite bid has strengthened. The grid was filtered on TRAIN, selected
on VALID, and TEST was evaluated once.

- Panel rows: 24,601
- Unique markets: 8,201
- Calendar days: 73

## Frozen primary rule

`d2_move2c_ask0.85_widen1c_vol2k`

## Train/validation-selected rule

`d2_move1c_ask0.90_widen0c_volall`

## Frozen primary

| Split | n | P&L | $/day | edge/ct | win | ask | max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 321 | $165.43 | $4.60 | +3.44¢ | 0.8287 | 78.25¢ | $72.52 |
| valid | 198 | $88.11 | $4.90 | +2.97¢ | 0.8283 | 78.70¢ | $52.45 |
| test | 227 | $86.20 | $4.31 | +2.53¢ | 0.8194 | 78.23¢ | $72.24 |

Test day-bootstrap mean/day: [$-4.44, $13.03], P(≤0)=0.1683

Matched edge lift after controlling week × coin × side × close-minute × delayed-ask bucket:

- train: 35 strata, win lift +0.89pp, edge lift +0.70¢
- valid: 24 strata, win lift -21.82pp, edge lift -22.05¢
- test: 24 strata, win lift +4.88pp, edge lift +4.81¢

Hard gates:
- train_valid_test_positive: **PASS**
- test_day_ci_positive: **FAIL**
- test_all_block_cis_positive: **FAIL**
- leave_one_group_positive: **PASS**
- matched_edge_lift_positive_all_splits: **FAIL**
- test_n_ge_50: **PASS**

## Selected configuration

| Split | n | P&L | $/day | edge/ct | win | ask | max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 736 | $326.26 | $9.06 | +2.96¢ | 0.8451 | 80.48¢ | $102.11 |
| valid | 442 | $286.11 | $15.90 | +4.32¢ | 0.8597 | 80.59¢ | $67.42 |
| test | 428 | $76.74 | $3.84 | +1.20¢ | 0.8224 | 79.95¢ | $139.90 |

Test day-bootstrap mean/day: [$-7.53, $14.99], P(≤0)=0.2579

Matched edge lift after controlling week × coin × side × close-minute × delayed-ask bucket:

- train: 53 strata, win lift +4.62pp, edge lift +4.40¢
- valid: 36 strata, win lift -6.91pp, edge lift -7.08¢
- test: 36 strata, win lift +7.59pp, edge lift +7.59¢

Hard gates:
- train_valid_test_positive: **PASS**
- test_day_ci_positive: **FAIL**
- test_all_block_cis_positive: **FAIL**
- leave_one_group_positive: **FAIL**
- matched_edge_lift_positive_all_splits: **FAIL**
- test_n_ge_50: **PASS**

## Interpretation

A rising quote is useful only if it predicts more than its new price
already says. The matched-state test is therefore load-bearing. A
profitable unconditioned rule with nonpositive matched edge lift is
price following, not a breakthrough.

No result here authorizes live orders while the account KILL state
remains active.