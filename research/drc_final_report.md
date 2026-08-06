# DRC-15 Maker-Option Overlay — independent evaluation (v2, corrected)

## VERDICT: **CONDITIONAL PASS on the signal — FAIL on the overlay as specified**

The DRC-15 **directional signal reproduces and survives validation**. The
**execution overlay is unvalidated**, and the **economics at realistic size are
roughly a fifth of what a first pass suggested**.

Two corrections to my own earlier work are recorded below. Both were material
and both were found by re-checking rather than by the analysis succeeding.

---

## Correction 1 — v1 implemented the wrong statistic

v1 computed `sigma4 = r_prev.shift(1).rolling(4).std()`, i.e.
std(r[t-2]…r[t-5]) — **excluding the return being normalised**. The
specification is std(r[t-1]…r[t-4]), which **includes** it.

That is not cosmetic. The two select overlapping but materially different
populations, and the specified form performs consistently better:

| | `premium_history2` | `ladder_paths` |
|---|---|---|
| **DRC_INCLUDED** (specified) | **+10.19pp** | **+7.89pp** |
| DRC_BACKGROUND (v1's error) | +5.60pp | +5.39pp |

**The v1 FAIL verdict was based on a mis-implementation and is withdrawn.**

## Correction 2 — the opportunity rate was 3.6× too high

I computed P(NO favourite in 65–80c | window) = 0.488 from
`premium_history2`. That file is **already band-filtered** (px range
0.6500–0.7900, zero rows outside), so 0.488 is P(NO favourite | *in-band*
window) — the wrong conditional.

`ladder_paths` is unfiltered and gives the correct figure:

```
windows joined to signal        3,729
NO favourite                    1,965   52.7%
NO favourite AND 65-80c           358    9.6% of windows
of those, z_inc >= 1               94    2.52% of windows
=> 373.6 windows/day x 0.0252  =  9.4 DRC opportunities/day
```

**9.4/day, not 34/day.** Everything downstream changes accordingly.

---

## What reproduces — DRC_INCLUDED

### `premium_history2`, n = 192 (larger sample, no ask column)

| | claimed | reproduced |
|---|---|---|
| DRC win rate | 80.86% | **80.21%** |
| non-DRC win rate | 69.00% | **70.01%** |
| win lift | +11.86pp | **+10.19pp** |
| day-clustered 95% CI | — | **[+3.58, +16.06]**, P(≤0) = **0.0017** |
| close-window-clustered | — | **[+2.27, +17.25]**, P(≤0) = **0.0053** |

**Both intervals exclude zero.**

**Chronological (identical boundaries across datasets):**

| split | n | DRC | non-DRC |
|---|---|---|---|
| train 05-25 → 06-30 | 90 | **+9.89c** | +2.84c |
| valid 06-30 → 07-18 | 61 | **+7.48c** | −5.01c |
| test 07-18 → 08-07 | 41 | **+13.78c** | +0.59c |

Positive in all three and beating the baseline in all three.

**Per coin:** DOGE +7.49c, ETH +9.16c, SOL +12.13c, XRP +10.48c — all positive.
**Leave-one-week-out:** +8.58c to +13.23c over 11 weeks — never negative.

### `ladder_paths`, n = 94 (real asks and fees — the honest economics)

DRC taker edge **+5.50c** vs non-DRC −2.85c; win lift +7.89pp.
Day-clustered CI **[−3.56, +14.01]**, P(≤0) = 0.110 — **includes zero**.
XRP alone is **−7.07c** on n = 30; excluding XRP gives +11.39c.

The two datasets share only **12 tickers**, so these are near-independent
samples. Both positive; only the larger is significant.

---

## Economics at realistic scale

| edge basis | $/day at 1 contract | contracts/opp for $50/day |
|---|---|---|
| taker +3.56c (BACKGROUND, real ask) | $0.335 | **149** |
| taker +5.50c (INCLUDED, real ask) | $0.518 | **97** |
| maker +9.95c (INCLUDED, assumes fill) | $0.937 | **53** |

At **qty 20**: **+$10.36/day** (taker) or **+$18.74/day** (maker-assumed) —
not the +$68/day an uncorrected opportunity rate implied.

The 122–276 contract estimate supplied with the brief was **closer to correct
than my first answer**. Median displayed depth in-band is 80 contracts and the
25th percentile is 11, so 97–149 contracts per opportunity is **not**
supportable at top of book.

---

## Why the risk profile is still the interesting part

| | mean | SD | P(losing day) | edge/SD |
|---|---|---|---|---|
| **DRC**, qty 20, 9.4 opps | **+$18.74** | ~$32 | ~13% | **~0.58** |
| base (measured live) | +$36.59 | $157.51 | ~41% | 0.232 |

Per contract DRC is **+9.95c with SD 39.95c** against the base's **+0.41c with
SD 45.77c** — higher mean, lower dispersion, roughly **27× the per-contract
Sharpe**. DRC earns less per day than the base at the same size because it
trades a fifth as often, but it does so with a fraction of the variance. That
addresses the actual failure mode of the live account, which was variance, not
absent edge.

**On 2026-08-03 the base strategy went 0-for-3 (−70c) on windows where DRC went
3-for-3 (+31c).** Over the last three days: DRC +28.92c (n=13) vs base +0.44c
(n=18). Encouraging, and far too small a sample to lean on.

---

## Why this is not a full PASS

The brief's PASS criteria, honestly scored:

| criterion | status |
|---|---|
| directional edge survives chronological validation | **PASS** |
| beats base after realistic execution costs | **partial** — +5.50c on real asks, CI includes zero |
| positive P&L per eligible opportunity | **PASS** on the signal |
| portfolio drawdown within limits | **NOT TESTED** (Phases 5–7 not run) |
| no dependence on one coin / week / day | **PASS** on premium, **FAIL** on ladder (XRP −7.07c) |
| no accounting / side-mapping / leakage issue | **PASS** — A0 fixed at open, contiguity enforced |
| all tests pass | **PASS** — 24/24 |
| runner restart-safe | **NOT IMPLEMENTED** (Phase 10 not run) |

Phases 3–7 and 10 were not run. Tuning a 10-second timeout and a cross ceiling
on 94 real-ask observations would manufacture a false positive.

---

## Recommendation

**Deploy the DRC flag as a logged signal, not as an execution change.** Record
`z_inc`, `sigma4`, `r_prev` and DRC status on every opportunity so the sample
accumulates forward at ~9.4/day. At that rate a genuine 300-observation holdout
arrives in about a month.

**Do not build the maker-then-IOC overlay yet.** Its only real-ask evidence is
94 observations whose interval contains zero, and the size it would need for a
$50/day target is not supportable at measured book depth.

---

## Reproduction

```bash
python research/drc_reproduction_v2.py     # both definitions, full waterfall
python research/drc_reproduction.py        # v1, BACKGROUND only, superseded
```

Inputs: `data/underlying.parquet`, `data/ladder_paths.parquet`,
`data/premium_history2.parquet`. `grid.parquet` not used as primary evidence.
Outputs: `data/results/drc_v2_meta.json`, `drc_candidates.parquet`.

**Known data caveat:** `premium_history2` is band-filtered. Never use it as a
denominator for any rate conditioned on price. Use `ladder_paths`.
