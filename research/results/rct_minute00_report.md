# Runaway Confirmation × minute :00 interaction audit

## Verdict: **FAIL / CANDIDATE ONLY**

Both components were fixed independently before this interaction test:

- minute :00 came from the cross-finding pricing audit;
- the RCT primary rule was frozen before the RCT configuration grid.

Frozen rule: `d2_move2c_ask0.85_widen1c_vol2k` restricted to close minute `:00`.

| Split | RCT n | RCT P&L | edge/ct | delayed-control P&L | incremental |
|---|---:|---:|---:|---:|---:|
| train | 79 | $149.12 | +12.58¢ | $111.36 | $37.76 |
| valid | 49 | $52.86 | +7.19¢ | $43.39 | $9.47 |
| test | 57 | $43.12 | +5.04¢ | $32.59 | $10.53 |

Test day-bootstrap RCT mean/day: [$-2.72, $6.44], P(≤0)=0.1810

Test incremental mean/day over delayed control: [$-2.64, $3.78], P(≤0)=0.3795

Matched edge lift after delayed-ask and state controls:
- train: 10 strata, win lift +24.24pp, edge lift +24.16¢
- valid: 4 strata, win lift +7.69pp, edge lift +7.84¢
- test: 4 strata, win lift +6.25pp, edge lift +6.37¢

## Hard gates

- rct_positive_train_valid_test: **PASS**
- incremental_over_delayed_control_positive_all_splits: **PASS**
- test_day_ci_positive: **FAIL**
- test_all_block_cis_positive: **FAIL**
- test_incremental_day_ci_positive: **FAIL**
- test_incremental_all_block_cis_positive: **FAIL**
- leave_one_coin_week_hour_positive: **PASS**
- matched_edge_lift_positive_all_splits: **PASS**
- test_n_ge_40: **PASS**

A positive absolute result is not enough. The interaction must beat
the same delayed-taker population without the quote-strengthening
condition; otherwise minute :00 or the new ask price explains the
result rather than runaway confirmation.

No result authorizes live orders while the account KILL state is active.