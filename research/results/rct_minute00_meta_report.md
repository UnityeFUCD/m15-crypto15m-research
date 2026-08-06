# Minute-:00 delayed-taker / RCT convergent audit

## Verdict: **MINUTE-00 DELAYED EDGE PASS; RCT INCREMENT UNPROVEN**

The full 73-day population is used only after both components were fixed.
A within-day best-of-four permutation corrects for the availability of
four close-minute labels.

## Full-period q15 economics

| Policy | n | P&L | edge/contract | mean/day | max DD |
|---|---:|---:|---:|---:|---:|
| RCT minute :00 | 185 | $245.10 | +8.83¢ | $3.31 | $46.51 |
| Delayed control minute :00 | 293 | $187.34 | +4.26¢ | $2.53 | $59.68 |

## Full-period uncertainty

- RCT mean/day 95% CI: [$1.23, $5.28], P(≤0)=0.0011
- Delayed control mean/day 95% CI: [$-0.01, $5.00], P(≤0)=0.0254
- RCT minus control mean/day 95% CI: [$-0.86, $2.46], P(≤0)=0.1819

## Four-minute selection correction

- RCT absolute best-of-four p: 0.0360
- RCT incremental best-of-four p: 0.7755

## Rank capacity

| split   |   rank |   n |   days |   total_pnl |   mean_day |   sd_day |   edge_per_contract |   win_rate |   mean_ask |   max_drawdown |   worst_window |   positive_day_fraction |
|:--------|-------:|----:|-------:|------------:|-----------:|---------:|--------------------:|-----------:|-----------:|---------------:|---------------:|------------------------:|
| train   |      1 |  79 |     36 |    149.1236 |     4.1423 |   7.7175 |              0.1258 |     0.9114 |     0.7734 |        32.6247 |       -12.8839 |                  0.6944 |
| train   |      2 |   3 |     36 |     -6.3557 |    -0.1765 |   2.0526 |             -0.1412 |     0.6667 |     0.7967 |        11.5916 |       -11.5916 |                  0.0556 |
| train   |      3 |   0 |     36 |      0.0000 |     0.0000 |   0.0000 |            nan      |   nan      |   nan      |         0.0000 |         0.0000 |                  0.0000 |
| valid   |      1 |  49 |     18 |     52.8600 |     2.9367 |   9.2874 |              0.0719 |     0.8776 |     0.7943 |        25.0528 |       -12.7412 |                  0.6667 |
| valid   |      2 |  12 |     18 |     -8.8436 |    -0.4913 |   4.3143 |             -0.0491 |     0.7500 |     0.7875 |        31.2210 |       -11.7360 |                  0.3333 |
| valid   |      3 |   0 |     18 |      0.0000 |     0.0000 |   0.0000 |            nan      |   nan      |   nan      |         0.0000 |         0.0000 |                  0.0000 |
| test    |      1 |  57 |     20 |     43.1181 |     2.1559 |  10.8464 |              0.0504 |     0.8421 |     0.7798 |        46.5079 |       -12.7412 |                  0.7000 |
| test    |      2 |  12 |     20 |     -8.5571 |    -0.4279 |   3.8194 |             -0.0475 |     0.7500 |     0.7858 |        26.1278 |       -12.8839 |                  0.3000 |
| test    |      3 |   2 |     20 |    -10.0519 |    -0.5026 |   2.7861 |             -0.3351 |     0.5000 |     0.8250 |         0.0000 |       -12.1680 |                  0.0500 |

## Portfolio caps

| split   |   cap |   n |   days |   total_pnl |   mean_day |   sd_day |   edge_per_contract |   win_rate |   mean_ask |   max_drawdown |   worst_window |   positive_day_fraction |
|:--------|------:|----:|-------:|------------:|-----------:|---------:|--------------------:|-----------:|-----------:|---------------:|---------------:|------------------------:|
| train   |     1 |  79 |     36 |    149.1236 |     4.1423 |   7.7175 |              0.1258 |     0.9114 |     0.7734 |        32.6247 |       -12.8839 |                  0.6944 |
| train   |     2 |  82 |     36 |    142.7679 |     3.9658 |   7.5972 |              0.1161 |     0.9024 |     0.7743 |        29.5049 |       -12.8839 |                  0.6944 |
| train   |     3 |  82 |     36 |    142.7679 |     3.9658 |   7.5972 |              0.1161 |     0.9024 |     0.7743 |        29.5049 |       -12.8839 |                  0.6944 |
| valid   |     1 |  49 |     18 |     52.8600 |     2.9367 |   9.2874 |              0.0719 |     0.8776 |     0.7943 |        25.0528 |       -12.7412 |                  0.6667 |
| valid   |     2 |  61 |     18 |     44.0164 |     2.4454 |  10.8320 |              0.0481 |     0.8525 |     0.7930 |        48.2405 |       -24.0476 |                  0.6111 |
| valid   |     3 |  61 |     18 |     44.0164 |     2.4454 |  10.8320 |              0.0481 |     0.8525 |     0.7930 |        48.2405 |       -24.0476 |                  0.6111 |
| test    |     1 |  57 |     20 |     43.1181 |     2.1559 |  10.8464 |              0.0504 |     0.8421 |     0.7798 |        46.5079 |       -12.7412 |                  0.7000 |
| test    |     2 |  69 |     20 |     34.5610 |     1.7281 |  13.3673 |              0.0334 |     0.8261 |     0.7809 |        68.5632 |       -25.6251 |                  0.6500 |
| test    |     3 |  71 |     20 |     24.5091 |     1.2255 |  14.6566 |              0.0230 |     0.8169 |     0.7821 |        80.7312 |       -37.7931 |                  0.6500 |

## 30-day bankroll stress from $300

|   quantity |   mean_day |   p_hit_211_30d |   p_drawdown_ge_30pct_30d |   median_terminal_30d |   terminal_p05 |   terminal_p95 |   median_max_drawdown |   max_drawdown_p95 |
|-----------:|-----------:|----------------:|--------------------------:|----------------------:|---------------:|---------------:|----------------------:|-------------------:|
|    10.0000 |     2.2081 |          0.0001 |                    0.0002 |              367.3176 |       307.0566 |       421.5363 |               17.2289 |            41.2744 |
|    15.0000 |     3.3122 |          0.0022 |                    0.0068 |              401.0186 |       310.5850 |       482.8954 |               25.9518 |            61.8415 |
|    20.0000 |     4.4162 |          0.0102 |                    0.0358 |              434.7892 |       313.7892 |       542.9176 |               34.4575 |            82.9403 |

## Decision

- The minute-:00 delayed/RCT population clears the convergent gate: **True**.
- The 2¢ quote-strengthening condition adds independently proven value over the same delayed population: **False**.
- Exploratory rank 2 is positive in all three periods: **False**.

A full-period pass is not a prospective deployment pass. The rule must
still be logged forward without retuning, and actual IOC depth/latency
must be measured. No result bypasses the account KILL state.