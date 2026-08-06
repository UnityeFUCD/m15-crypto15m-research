# Top-of-hour delayed-taker — independent proxy-free audit

## Verdict: **FAIL / CANDIDATE ONLY**

Frozen policy: initial favourite bid 65-80¢, wait 120 seconds, original
side ask 60-80¢, close minute :00, exact q15 taker fee.

## Population

- Initial in-band markets: 8,186
- Delayed-ask eligible: 4,051
- :00 candidates: 959
- :00 cap-one trades: 533

## Uncapped :00

- n: 959
- days: 73
- mean_ask: 0.713420229405631
- win_rate: 0.7664233576642335
- edge_per_contract: 0.038914028501911724
- total_pnl: 559.7783000000001
- mean_pnl_per_active_day: 7.668195890410961
- sd_active_day: 29.538234137552024
- max_drawdown: 149.07529999999986
- worst_window: -65.91329999999999

## Cap one per close window

- n: 533
- days: 73
- mean_ask: 0.6972045028142589
- win_rate: 0.7504690431519699
- edge_per_contract: 0.03873283302063791
- total_pnl: 309.66900000000004
- mean_pnl_per_active_day: 4.242041095890412
- sd_active_day: 14.061794165998121
- max_drawdown: 84.78529999999989
- worst_window: -12.168

## Held-out uncertainty

- all: edge +2.62¢, 95% CI [-2.59, +7.01]¢, P(≤0) 0.1510
- cap one: edge +3.30¢, 95% CI [-0.58, +7.16]¢, P(≤0) 0.0488

## Calibration

- n_train: 2007
- n_heldout: 2044
- base_log_loss: 0.5758038536225235
- minute_log_loss: 0.5761651563375456
- delta_log_loss: 0.00036130271502210487
- base_brier: 0.19429346563737931
- minute_brier: 0.1943672319352057
- delta_brier: 7.376629782637756e-05

## Hard gates

- heldout_all_lower_bound_positive: **FAIL**
- heldout_cap_one_lower_bound_positive: **FAIL**
- train_valid_test_all_positive: **PASS**
- best_of_four_p_below_005: **PASS**
- minute_improves_heldout_logloss_and_brier: **FAIL**
- leave_one_coin_positive: **PASS**
- leave_one_week_positive: **PASS**

The audit uses no fill proxy. A historical pass would still require a
frozen prospective test because :00 has been examined repeatedly.
Nothing authorizes live orders while the account KILL state is active.