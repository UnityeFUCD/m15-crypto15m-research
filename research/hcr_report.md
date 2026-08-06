# HCR — Hourly Common Reversal: independent test

## VERDICT: **PASS on the signal** (two corrections, three caveats)

Unlike DRC, HCR survives every falsification test applied, reproduces
near-exactly, and cleanly separates the live book.

## Rule as specified

At the first eligible 65–80c favourite between 14 and 8 minutes remaining:

```
r_common    = mean over 6 coins of A0(t)/A0(t-900) - 1   (completed at open)
d_favourite = +1 if YES favourite, -1 if NO
qualify     d_favourite * r_common <= -0.0015   (favourite OPPOSES the move)
            AND mean_24(|r_common|) <= 0.0030   (calm background)
            AND close minute == 00 UTC
```

No leakage: every input is an A0 value fixed at window open, and entry is 1–7
minutes after open.

## Reproduction — near exact

| | claimed | reproduced |
|---|---|---|
| candidates (premium_history2) | 521 | **520** |
| win rate | 0.7946 | **0.7923** |
| maker edge | — | **+9.18c** |
| non-HCR maker edge | — | +2.33c |
| day-clustered 95% CI | — | **[+3.93, +13.84]**, P(≤0) **0.0008** |

## The test DRC failed — matched controls

Matched on **same coin, same ISO week, same 2c price bucket, and SAME SIDE**,
so a "NO is cheaper" explanation is excluded by construction:

```
257 strata covering 364 HCR markets
win-rate lift    +10.69pp
maker-edge lift  +10.71c    95% CI [+5.18, +15.94]    P(<=0) 0.0002
```

The lift is in **win rate** — genuinely predictive. DRC's live advantage was a
0.88c price difference with an identical win rate. HCR's is not.

## Falsification

**Permutation** — shuffle `r_common` within day, preserving the calm filter:
observed +9.18c, permuted mean +4.24c, p95 +6.91c,
**P(permuted ≥ observed) = 0.0020**.

**Parameter plateau** — 120 configurations of (oppose, calm, lookback):

| oppose | calm≤20bp | calm≤25bp | calm≤30bp | calm≤40bp | no calm |
|---|---|---|---|---|---|
| 5bp | +6.90 | +7.82 | +7.46 | +7.09 | +7.36 |
| 10bp | +5.66 | +7.14 | +7.11 | +6.65 | +7.03 |
| **15bp** | +6.15 | +9.36 | **+9.18** | +8.30 | +8.65 |
| 20bp | +8.99 | +11.46 | +11.26 | +9.67 | +10.03 |
| 25bp | +18.00 | +19.86 | +16.83 | +14.59 | +14.85 |
| 30bp | +16.33 | +19.45 | +14.82 | +11.90 | +12.57 |

**99.2% of configurations beat the +3.59c baseline, and the specified one ranks
63rd of 120 — the median.** A rule tuned to this data would have picked 25bp
(+18.00c), not 15bp (+9.18c). The parameters are unremarkable within their own
family, which is strong evidence they were not fitted here.

Lookback is near-irrelevant: 12/24/48/96 → +8.83/+9.18/+9.80/+9.93.

## Live overlap — the decisive result

The flag applied to the 275 actual LSM markets:

| cohort | markets | contracts | win | P&L | c/ct |
|---|---|---|---|---|---|
| **HCR, any minute** | 60 | 1,253 | 0.7667 | **+$95.10** | **+7.59c** |
| HCR, minute-00 only | 12 | 240 | 0.8333 | +$47.60 | +19.83c |
| **non-HCR** | 215 | 4,843 | 0.6744 | **−$85.72** | −1.77c |

The entire LSM book netted **+$9.38**. HCR markets made **+$95.10**; everything
else lost **$85.72**. Real fills, real prices, real settlements.

## Real execution cost

`ladder_paths` carries genuine asks:

| cohort | n | win | maker | **taker (real ask + fee)** |
|---|---|---|---|---|
| HCR minute-00 | 120 | 0.7917 | +9.53c | **+5.44c** |
| HCR minute-30 | 103 | 0.7379 | +3.88c | +0.19c |
| HCR both | 223 | 0.7668 | +6.92c | **+3.02c** |
| non-HCR | 866 | 0.7298 | +3.29c | **−0.53c** |

Positive after real crossing costs. DRC was not.

## Fill-bias correction

Applying the measured model (P(fill|win)=0.843, P(fill|lose)=0.962):

| | raw maker | fill-corrected |
|---|---|---|
| HCR | +9.18c | **+6.94c** (76% of raw) |
| non-HCR | +2.33c | **−0.38c** |

**Fill-corrected lift +7.32c.** HCR survives the correction that reduces the
base strategy to roughly zero.

## CORRECTION 1 — drop the minute-00 filter

```
minute 00 taker +5.44c (n 120)   minute 30 taker +0.19c (n 103)
difference +5.25c   SE 5.68c   t = 0.92   NOT SIGNIFICANT
```

The filter is **not supported**. It discards half the opportunities for a
difference well inside one standard error. Removing it roughly doubles the
candidate rate.

## CORRECTION 2 — dataset sampling limits the minute test

```
premium_history2 minutes: {00: 2780}            <- minute-00 ONLY
ladder_paths     minutes: {00: 2806, 30: 2818}  <- :00 and :30 only
```

Both datasets are sampled. The minute-00 condition is **vacuous** on
premium_history2 (every row is already minute 00), so that dataset cannot test
it at all, and the claimed 493/521 counts require coverage this repo lacks.
Only :00 vs :30 is testable, and it is not significant.

## Caveats

1. **Chronological decline.** train +11.48c, valid +6.71c, test +6.92c — all
   positive and all beating baseline, but the *lift* narrows sharply in test
   (+6.92 vs non-HCR +5.74 = only +1.18c).
2. **Live overlap is not fully out-of-sample.** `premium_history2` spans
   through Aug 5, so the live Aug 4–5 markets share calendar time with the fit
   window, though not market selection.
3. **60 live markets** is still a small forward sample.

## Recommendation

**HCR is the strongest signal found in this project.** It is the only candidate
that passes matched controls on win rate, survives permutation, sits on a broad
parameter plateau, stays positive after real ask + fee + fill bias, and
separates the live book.

Deploy as a **logged flag with portfolio priority**, not an execution override:

- drop the minute-00 filter (unjustified, halves the sample)
- keep oppose ≤ −15bp; the calm filter adds little but costs little
- consider 20–25bp oppose if a stronger filter is wanted — the plateau supports it
- do **not** pair it with the unvalidated RACE timeout parameters

## Comparison with the other candidates

| candidate | matched controls | permutation | real-cost edge | live separation | verdict |
|---|---|---|---|---|---|
| **HCR** | **+10.69pp, P=0.0002** | **P=0.0020** | **+5.44c** | **+$95.10 vs −$85.72** | **PASS** |
| DRC | ~0pp win lift, price only | not run | +5.50c, CI spans 0 | +$20.58 vs −$11.20 | unresolved |
| RACE | n/a (execution) | n/a | n/a | mechanism only | mechanism yes, parameters no |
