# DRC-15 Maker-Option Overlay — independent evaluation

## VERDICT: **FAIL — do not integrate**

The signal is **not fabricated**. Its direction replicates on two independent
book sources. But it is roughly **half** the claimed magnitude, its confidence
interval comfortably contains zero, and — decisively — **it is negative in the
training period and positive only in the final 25% of the sample.**

The brief's own PASS criteria require that the edge *survive chronological
validation* and show *no dependence on one coin*. It fails both.

---

## What was reproduced, and what was not

### Directional core (41,334-window benchmark series, no book data needed)

| | claimed | reproduced |
|---|---|---|
| P(NO wins) lift from `z_up >= 1` | implies +11.86pp | **+1.68pp** |
| day-clustered 95% CI | — | **[−0.34, +3.68] pp**, P(≤0) = 0.0503 |

### The exact claimed population — NO favourites, 65–80c

| source | n | DRC win | non-DRC win | lift |
|---|---|---|---|---|
| `premium_history2` | 889 | **76.70%** | 71.11% | **+5.60pp** |
| `ladder_paths` (independent) | 358 | **76.92%** | 71.54% | **+5.39pp** |
| **claimed** | 371 | **80.86%** | 69.00% | **+11.86pp** |

Two independently constructed book sources agree to within 0.2pp. **The
direction and rough size of the effect are real.** The claimed magnitude is
not — it is about twice what the data supports.

Day-clustered 95% CI on the lift: **[−3.19, +14.08] pp**, P(≤0) = 0.0985.

### Economics — immediate taker, real asks, real fees

| cohort | n | win | avg ask | fee | **taker edge** |
|---|---|---|---|---|---|
| DRC | 91 | 0.7692 | 71.36c | 2.00c | **+3.56c** |
| non-DRC | 267 | 0.7154 | 71.64c | 1.99c | −2.09c |

| | claimed | reproduced |
|---|---|---|
| DRC taker edge | **+7.40c** | **+3.56c** |
| 95% CI | [+3.51, +11.27] | **[−5.04, +12.24]** |
| P(≤0) | — | **0.1947** |
| non-DRC | −4.90c | −2.09c |

The non-DRC baseline reproduces reasonably. **The DRC arm does not** — it is
roughly half the claimed edge and is statistically indistinguishable from zero.

---

## Why it fails — the two decisive tests

### 1. Chronological validation: negative in training

| split | dates | DRC n | DRC win | **DRC taker** | non-DRC taker |
|---|---|---|---|---|---|
| train (first 50%) | 05-25 → 06-28 | 42 | 0.7143 | **−2.33c** | +0.31c |
| valid (next 25%) | 06-29 → 07-16 | 24 | 0.7500 | +2.42c | −5.86c |
| test (last 25%) | 07-16 → 08-05 | 25 | 0.8800 | **+14.56c** | −3.40c |

**In the training period DRC is negative *and worse than the baseline it is
supposed to beat*.** The entire effect lives in the last quarter of the sample.
That is the signature of a regime artifact or of a threshold discovered on
recent data, not of a durable edge. The brief explicitly requires selection on
training data with evaluation on untouched validation and test; done in that
order, this signal would never have been selected.

### 2. Coin dependence — and it is not the coin the brief expected

| coin alone | n | win | taker edge |
|---|---|---|---|
| ETH | 25 | 0.8400 | **+11.04c** |
| SOL | 19 | 0.7895 | +6.37c |
| DOGE | 18 | 0.7778 | +4.28c |
| **XRP** | 29 | 0.6897 | **−5.17c** |

Leave-one-out swings the headline from **+0.73c** (drop ETH) to **+7.65c**
(drop XRP). With 18–29 observations per coin, the result is dominated by which
coin you happen to include. The brief anticipated a DOGE artifact; the actual
instability is ETH-positive / XRP-negative.

---

## What genuinely survives

Three findings argue the effect is real but small, not imaginary:

1. **Independent replication of direction and size.** `premium_history2` and
   `ladder_paths` are built by different code paths from different raw inputs
   and agree at +5.60pp and +5.39pp.
2. **Symmetry.** `z_up ≥ +1` predicts NO at 0.5193; `z_up ≤ −1` predicts YES at
   0.5311. A one-sided fluke would not produce a matching opposite tail. This
   is consistent with a genuine, weak, conditional reversal.
3. **No leakage found.** `A0(t)` is fixed at window open and entry is 1–7
   minutes after open, so `r_prev`, `sigma4` and `z_up` are all knowable at
   decision time. Contiguity was enforced — every `r_prev` comes from an
   adjacent 900-second pair, and windows with gaps or insufficient history were
   dropped rather than bridged.

---

## Sample-size discrepancy — stated plainly

The brief claims **371 candidates**; I find **91** in `ladder_paths` and 176 in
`premium_history2`. `ladder_paths` samples 5,624 markets, so after restricting
to four coins, NO favourites and the 65–80c band the base is simply smaller.

**This does not explain the failure.** A smaller sample widens intervals, but
it cannot flip a point estimate. The training-period estimate is **−2.33c** —
negative, not merely imprecise.

---

## Phases not run, and why

Phases 3–7 (live execution reconstruction, timeout optimisation, cross
ceilings, integration architectures, portfolio simulation) and Phase 10
(production code) were **not** run. The brief conditions them on survival:
*"If the strategy survives, integrate it into the existing runner."* It does
not survive Phase 2, and tuning a 10-second timeout on a signal that is
negative out-of-sample would manufacture a false positive rather than test one.

---

## Reproduction

```bash
python research/drc_reproduction.py
```

Inputs: `data/underlying.parquet` (41,334 markets, sha 4a3f8c21…),
`data/ladder_paths.parquet`, `data/premium_history2.parquet`.
`grid.parquet` was **not** used as primary evidence, per the brief.

Outputs: `data/results/drc_candidates.parquet`,
`data/results/drc_reproduction_meta.json`.

---

## What would change this verdict

Not more analysis of the same data — the training-period result is not a power
problem. It would take **new out-of-sample data**: if DRC's lift persists at
+5pp or better across the next several weeks, with ETH and XRP behaving
consistently, the signal becomes worth the execution work. Until then,
integrating it would be fitting a 10-second timeout and a cross ceiling to
25 observations from a single recent quarter.

**Recommendation: log the DRC flag on every opportunity so it accumulates
forward, and re-evaluate at n ≈ 300 with a genuine holdout. Do not trade it.**
