# User-specified `:00` strategy — tested exactly as written

**Verdict: it works, and it survives every correction — but not for the reason
it is built on.** The momentum condition, which is the headline idea, is inert.
What carries the result is `:00` plus the ask ceiling.

Implementation: `research/spec_momentum_00.py`. Close derived from the ticker,
never from `expected_expiration_time`. Favourite side frozen at entry and never
redefined. `won` verified against `underlying.parquet` at **1.000000**
agreement on 1,878 markets.

---

## The spec as written

```
161 trades over 66 days (2.44/day)   win rate 87.58%   mean ask 78.60c
total P&L  +$188.66 at q15   fees $28.09
per contract +7.81c   95% CI [+2.61, +12.65]   P(<=0) 0.0021
```

| check | result |
|---|---|
| train / valid / test | **+9.53c / +5.60c / +7.31c** — positive in all three |
| leave-one-coin-out | all six positive (+4.72 to +8.39c) |
| leave-one-week-out | all positive (+6.70 to +9.97c) |
| drop the best 5 trades | still **+7.15c** — not concentrated |
| Bonferroni over ~20 variants | 0.0021 < 0.0025 — survives, barely |
| **permutation over the entire search** | permuted best **−1.38c** (p95 +1.60c) vs observed +7.81c → **P = 0.0000** |

That last row is the one that matters. Shuffling outcomes within day and
re-running every configuration, the best config found by chance averages
*negative*. The observed result is nowhere near the noise distribution.

## Which conditions actually do work

Each row removes ONE condition and keeps the rest:

| policy | n | per contract | $/day |
|---|---|---|---|
| full spec | 161 | +7.81c | +2.86 |
| − momentum (bid rise ≥ 2c) | 182 | **+8.52c** | +3.37 |
| − volume (≥ 2000) | 325 | +7.75c | **+5.17** |
| − spread widen (≤ 1c) | 181 | +8.10c | +3.23 |
| − ask ceiling | 203 | **+5.54c** | +2.44 |

**Three of the four delayed conditions are inert:**

- **momentum** — removing it *improves* the edge (+8.52c vs +7.81c) on more
  trades. The 2c bid-rise requirement is not selecting anything.
- **volume ≥ 2000** — identical edge, half the frequency. It costs $2.31/day
  for nothing.
- **spread widening ≤ 1c** — no measurable effect.
- **ask ceiling** — this one is real. Removing it drops the edge to +5.54c.

## The 2-minute wait is a noise peak

| wait | n | win | per contract | 95% CI | P(≤0) |
|---|---|---|---|---|---|
| 1 min | 181 | 0.8122 | +2.10c | [−3.56, +7.52] | 0.2279 |
| **2 min** | 161 | 0.8758 | **+7.81c** | [+2.75, +12.47] | 0.0021 |
| 3 min | 138 | 0.8116 | +1.08c | [−6.76, +8.44] | 0.3719 |
| 4 min | 137 | 0.8248 | +2.20c | [−4.33, +8.24] | 0.2357 |

A sharp peak at exactly the specified value is a warning sign, so it was
tested directly: **wait-2 minus wait-1, day-paired on 62 shared days, is
+3.13c with 95% CI [−2.94, +9.72], P(≤0) = 0.1664.** Not significant. The
peak is real in-sample but cannot be distinguished from noise, so the
2-minute choice should not be treated as tuned or load-bearing.

## Highest-earning configuration

Dropping the two inert conditions nearly doubles the money:

```
drop volume + momentum
  362 trades   win 87.02%   +7.52c/contract   $5.59/day at q15
  95% CI [+4.13, +10.84]   P(<=0) 0.0000
  leave-one-coin-out min +6.14c (all positive)
  leave-one-week-out min +6.87c (all positive)
  drop best 5 trades: +7.19c
```

## ⚠ The one real warning: chronological decay

```
config "drop volume + momentum"
  train  n=160  win 0.9062  +11.26c
  valid  n=105  win 0.8667   +6.48c
  test   n= 97  win 0.8144   +2.47c
```

The edge falls monotonically across the three periods and the win rate drops
9 points from train to test. Two readings, and this data cannot separate them:

1. the effect is decaying as the market prices it, or
2. n≈100 per split is small and this is ordinary drift.

The full spec does not show the same pattern (+9.53 / +5.60 / +7.31), which
argues for noise. But a monotone decline to a test-period edge of +2.47c is
the single most important thing to watch, and it means the *magnitude* here
should not be planned around.

## What I would run

**`:00` + first quote at 8–14 min + favourite bid 65–80c + wait 2 minutes +
ask in [0.60, 0.85] + one per close window + IOC at the ask + q15.**

Drop the momentum, volume and spread-widening conditions — they cost
frequency and add nothing. Keep the ask ceiling; it is doing real work.

Expected, if the effect holds: **~$5/day at q15, ~5 trades/day.** Plan for the
test-period figure (+2.47c ≈ $1.80/day), not the full-sample one.

## Caveats not to gloss

1. **No mechanism.** `:00` is the top of the hour where other instruments
   settle and roll. That is a plausible story, not evidence.
2. **`:00` has been examined repeatedly** in this project, so these tests are
   not independent of earlier ones. The permutation covers this search, not
   the whole history of looking at `:00`.
3. **Mean ask is 78.60c.** At that price a loss costs 3.7× what a win pays;
   the 87% win rate is doing all the work and a small drop in it is expensive.
4. **The account is $136.27 against a $211 floor** and cannot trade this yet.
