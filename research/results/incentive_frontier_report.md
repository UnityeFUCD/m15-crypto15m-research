# Kalshi liquidity-incentive frontier

## Verdict: **CANDIDATE — prospective reward capture justified**

Snapshot UTC: `2026-08-06T13:06:04.860692+00:00`

The audit ranks active liquidity programs using reward in dollars
(`period_reward / 10,000`), period length, target size, current public
books, and the worst possible loss when only one side of a two-sided
maker quote fills.

## Coverage

- Active liquidity programs: 3,793
- Programs with valid reward/target terms: 3,793
- Highest-density programs enriched with live market/book data: 300
- Open two-sided quote configurations: 1,582
- Predeclared hard-candidate markets: 5

## Hard hurdle

A row qualifies only when the conservative target-size share estimate
earns at least 25% of the worst one-leg loss per day at q20 or below.
This does not assume that the reward is riskless; it identifies markets
where a short prospective q1 quoting experiment is economically worth
running.

## Highest-ranked hard candidates

| Market | q/side | Reward/day lower | One-leg risk | Ratio | Spread |
|---|---:|---:|---:|---:|---:|
| KXGOLD15M-26AUG060915-15 | 1 | $6.38 | $0.70 | 9.11 | 1.0¢ |
| KXGOLD15M-26AUG060915-15 | 5 | $31.48 | $3.50 | 8.99 | 1.0¢ |
| KXGOLD15M-26AUG060915-15 | 10 | $61.94 | $7.00 | 8.85 | 1.0¢ |
| KXGOLD15M-26AUG060915-15 | 15 | $91.43 | $10.50 | 8.71 | 1.0¢ |
| KXGOLD15M-26AUG060915-15 | 20 | $120.00 | $14.00 | 8.57 | 1.0¢ |
| KXSILVER15M-26AUG060915-15 | 1 | $6.38 | $0.82 | 7.78 | 3.0¢ |
| KXWTI15M-26AUG060915-15 | 1 | $6.38 | $0.83 | 7.69 | 3.0¢ |
| KXSILVER15M-26AUG060915-15 | 5 | $31.48 | $4.10 | 7.68 | 3.0¢ |
| KXWTI15M-26AUG060915-15 | 5 | $31.48 | $4.15 | 7.58 | 3.0¢ |
| KXSILVER15M-26AUG060915-15 | 10 | $61.94 | $8.20 | 7.55 | 3.0¢ |
| KXWTI15M-26AUG060915-15 | 10 | $61.94 | $8.30 | 7.46 | 3.0¢ |
| KXSILVER15M-26AUG060915-15 | 15 | $91.43 | $12.30 | 7.43 | 3.0¢ |
| KXWTI15M-26AUG060915-15 | 15 | $91.43 | $12.45 | 7.34 | 3.0¢ |
| KXSILVER15M-26AUG060915-15 | 20 | $120.00 | $16.40 | 7.32 | 3.0¢ |
| KXWTI15M-26AUG060915-15 | 20 | $120.00 | $16.60 | 7.23 | 3.0¢ |

## Required prospective double test

1. Confirm account eligibility for the general liquidity program.
2. Randomize q1 two-sided quotes versus no-order observation windows.
3. Record official queue position, full books, fills, score terms, and
   eventual reward credits.
4. Never self-trade or manufacture volume.
5. Promote only when reward plus trading P&L per assigned window has a
   positive day/block-clustered lower bound.

The account's existing KILL state remains binding.