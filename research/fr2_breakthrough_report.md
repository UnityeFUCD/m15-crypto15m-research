# FR2 — Two-Minute Favorite Repricing

## Classification

**Strong historical candidate, not yet a proven production edge.**

FR2 directly attacks the project's binding problem—maker fill bias—by removing
the full-sized passive order. It lets the market reprice for two complete
minutes, then buys the original favorite only when it remains inside a
moderate 60–80¢ ask band.

## Frozen rule

1. At the first complete observation between 14 and 8 minutes remaining,
   identify the favorite.
2. Require its bid to be at least 65¢ and below 80¢.
3. Preserve that original side.
4. Wait two complete minutes.
5. At the two-minute observation, require the same side's ask to be 60–80¢.
6. Take at most one market per close window in the existing production series
   order: BTC, ETH, SOL, XRP, DOGE, HYPE.
7. Submit an IOC at the displayed ask, never chase or retry, and hold fills to
   settlement.

## Clean-room historical result

Source: `ladder_paths`, a random 5,000-market sample from the even close slots
(:00 and :30) across 73 calendar days. The source contains real one-minute
bid/ask candlesticks and real settlement outcomes.

| Metric | FR2 |
|---|---:|
| Initial in-band markets reconstructed | 1,091 |
| Unique sampled close windows | 804 |
| Markets selected after delay and cap | 444 |
| Win rate | 77.93% |
| Average delayed ask | 71.19¢ |
| Average one-block fee | 1.41¢/contract |
| Net edge after ask and fee | **+5.32¢/contract** |
| q20 historical P&L | **+$472.77** |
| Mean per sampled calendar day | **+$6.48** |
| Daily standard deviation | $15.84 |
| Day-bootstrap 95% interval | **[+$2.93, +$10.17]/day** |
| Bootstrap P(mean ≤ 0) | **0.0002** |
| Historical maximum drawdown | $88.41 |
| Worst close window | −$16.22 |

Chronological partitions:

| Partition | Markets | Net edge | q20 P&L |
|---|---:|---:|---:|
| Train | 231 | +5.22¢ | +$241.36 |
| Validation | 105 | +1.75¢ | +$36.69 |
| Test | 108 | +9.01¢ | +$194.72 |

The train-versus-later difference is statistically indistinguishable
(clustered interaction P≈0.94), unlike DRC and HCR's earlier recent-only shape.

Robustness:

- 10 of 11 calendar weeks were positive.
- Both YES and NO sides were positive.
- Five of six coins were positive individually; every leave-one-coin-out
  portfolio remained positive.
- Minute :00 and minute :30 were both positive.
- The natural 60–80¢ configuration ranks 3rd of 441 delay/band combinations
  by daily t-statistic.
- A shared-day centered max-stat bootstrap across all 441 combinations gives
  an approximate search-adjusted P≈0.042.
- Nearby configurations form a plateau; this is not a single isolated cell.

## Same-sample comparison

All policies use one market per close window and q20.

| Policy | P&L | Mean/day | Max drawdown |
|---|---:|---:|---:|
| Initial maker, global fill correction | +$105.49 | +$1.45 | $301.71 |
| Immediate taker at initial ask | +$45.99 | +$0.63 | $342.83 |
| **FR2** | **+$472.77** | **+$6.48** | **$88.41** |

FR2's historical edge is about 3.9× the project's current +1.36¢
fill-corrected maker estimate while materially reducing the sampled drawdown.

## Mechanism

The initial favorite contains persistent information, but its first few minutes
contain noisy repricing. Immediate maker execution is selected against:
deteriorating favorites return and fill, while strengthening favorites escape.

FR2 waits through that first repricing and creates a confirmation band:

- Below 60¢: the original favorite has materially failed; skip it.
- Above 80¢: the outcome is too expensive; skip it.
- 60–80¢: the original side remains strong enough to win 77.9%, but the
  all-in price averages only about 72.6¢ after the fee.

This converts adverse passive selection into a bounded active entry.

## What can still kill it

1. `ladder_paths` covers only :00/:30 closes. :15/:45 needs an independent
   full-path reconstruction.
2. The path sample omits decision-time volume. The volume interaction must be
   reconstructed; low-volume quotes may lack q20 depth.
3. A candlestick close shows a displayed ask, not executable quantity.
4. Quote-to-order latency can cause no-fills that disproportionately miss
   winners.
5. The parameter search was broad. The max-stat correction is encouraging,
   not a substitute for prospective confirmation.
6. Only 12 historical live LSM markets overlap the path sample. On those,
   FR2 modeled +$0.38 while the actual maker orders lost $54.40, but the sample
   is too small for a claim.

## Required independent double test

Before material live deployment:

- Fetch a random, calendar-balanced full path sample from :15 and :45 closes.
- Run the exact frozen 2-minute/60–80¢ rule without changing parameters.
- Fetch the complete all-minute population if API limits permit.
- Record decision-time volume and displayed ask depth.
- Require positive day/block-clustered lower bounds in the independent sample.
- Run live IOC orders with no retry and record every no-fill and partial fill.
- Compare q10, q20 and q30 through randomized capacity arms.
- Keep maximum one FR2 position per close window during confirmation.

## Risk-aware starting profile

The historical sampled q20 path had a 29.5% maximum drawdown from a $300
reference balance. q15 scales that to roughly 22% while retaining the same
per-contract edge.

A day-block bootstrap on the sampled history estimated:

| Quantity | 30-day P(hit $211 from $300) | P(30% drawdown) |
|---:|---:|---:|
| 10 | 0.005% | 0.023% |
| 15 | 0.12% | 1.24% |
| 20 | 0.65% | 8.12% |

These numbers are conditional on the sampled opportunity frequency and must be
rerun on the complete all-minute stream. They support q15 as the risk-aware
confirmation tier, not an immediate large-scale claim.
