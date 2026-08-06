# How the findings interlock — and the one pattern that survived

Autonomous cross-analysis of everything measured in this project: how the
findings correlate, which are restatements of each other, and whether any
combination is stronger than its parts.

**Headline: one candidate survived, and it is proxy-free and directly
tradeable.** Everything else collapsed into a single underlying fact.

---

## 1. The map — almost everything is one finding

Pairwise correlation of every signal measured here, on 8,200 in-band markets:

```
          f_nofill  f_min00  f_calm  f_tight  f_cheap  f_yes
f_nofill     1.000   -0.003  -0.015   -0.006   -0.009  0.015
f_min00     -0.003    1.000  -0.003   -0.024    0.026  0.049
f_calm      -0.015   -0.003   1.000    0.181    0.039  0.007
```

They are mutually orthogonal — but only two carry any effect:

| signal | standalone win-rate lift | conditional on no-fill |
|---|---|---|
| **no-fill** | **+20.96pp** | (base) |
| **minute :00** | **+3.25pp** | **+3.37pp** — survives |
| calm tape | −0.76pp | +1.51pp |
| tight spread | −0.17pp | −0.47pp |
| cheap favourite (<70c) | −4.80pp | −4.09pp |
| YES side | +1.38pp | +1.94pp |

Calm, tight, cheap and side are noise. **Two real signals, uncorrelated
(r = −0.003), and each retains its effect after conditioning on the other.**
That orthogonality is what makes the combination worth anything.

## 2. The big negative: adverse selection does not vary

The hypothesis worth testing was that fill bias — the binding constraint on
everything — might be weaker somewhere. It is not.

```
filled-cohort maker edge, every dimension:   -2.24c to -7.88c
no-fill win-rate gap, every dimension:       +16.0pp to +24.4pp
```

Searched on train only, best cell frozen, evaluated held-out:

```
best train cell (coin == XRP, -2.24c)
  train -2.24c   valid -6.78c   test -6.87c
  held out CI [-12.31, -1.34]   P(<=0) 0.9918
```

It reverses completely out of sample. **There is no regime where the
counterparty knows less.** This explains why thirteen prior subset searches
failed: all of them searched inside a space where the binding constraint is
constant. Any "better markets" they found were noise around a fixed −5c drag.

## 3. The surviving pattern: the market does not price the close minute

The clean statement:

```
minute    n      mean ask    win rate    implied - actual
 :00     277      0.7615      0.8448        +8.32pp
 :15     329      0.7605      0.7477        -1.28pp
 :30     278      0.7551      0.7698        +1.47pp
 :45     290      0.7613      0.7966        +3.53pp
```

**The ask is identical across minutes (0.7551–0.7615). The win rate is not.**
That is a pricing failure, not a cheaper price — the market quotes `:00` the
same as every other minute while `:00` wins 5–10pp more often.

### The 2×2

| cohort | n | win | taker edge | 95% CI | P(≤0) |
|---|---|---|---|---|---|
| **:00 AND no-fill** | 277 | 0.8448 | **+6.32c** | [+2.33, +10.33] | **0.0013** |
| :00 AND filled | 686 | 0.7362 | +2.18c | [−2.47, +6.69] | 0.1752 |
| **:00 regardless** | 963 | 0.7674 | **+3.37c** | [+0.03, +6.80] | **0.0244** |
| other AND no-fill | 897 | 0.7703 | −0.87c | [−3.98, +2.15] | 0.7081 |
| other minutes | 3,095 | 0.7166 | −1.49c | [−3.78, +0.77] | 0.9024 |

The probe adds **+4.14c** at `:00` against **+0.86c** elsewhere — a genuine
interaction of +3.28c, not two separate effects being added.

### It survived every kill attempt but one

| check | result |
|---|---|
| day-clustered, full sample | +6.32c, P(≤0) **0.0013** |
| **held-out only** (never used to select) | +5.43c, CI [+0.06, +10.31], P **0.0240** |
| day-matched within-day vs other minutes | +9.71c, CI [+4.41, +14.92], P **0.0003** |
| **selection correction** (best-of-4 minutes, permuted) | P = **0.0158**, survives |
| leave-one-coin-out | all six positive (+4.55 to +8.06c) |
| leave-one-week-out | all positive (+5.18 to +8.06c) |
| drop the best 5 markets | still +5.89c |
| proxy accuracy by minute | 0.7250–0.7632, comparable — not an artifact |
| **hostile proxy stress** | **dies: −0.07c if the proxy is wrong on the best 25%** |

## 4. The version that needs no proxy

**(C) `:00` regardless of fill state: +3.37c, P(≤0) = 0.0244, 13.1 commits/day
held-out, +$4.30/day at q15.**

This is the important one. It requires no probe, no fill model and no proxy —
buy the favourite at the ask at minute `:00`. Every objection about the fill
proxy evaporates, and it is *directly* testable live with no new machinery.

It is weaker per contract than (A) but higher frequency, so it earns more per
day, and it rests on far less inference.

## 5. What I do not want glossed

1. **The lower bound is +0.03c.** (C) clears zero by a hair. It is significant,
   not comfortable.
2. **Minute :00 has been tested repeatedly in this project.** These tests are
   not independent of each other. The selection correction covers the four
   minutes, not the several times `:00` has been revisited.
3. **No mechanism is established.** `:00` coincides with the top of the hour
   where other instruments settle and roll, which plausibly changes who is
   quoting. That is a story. Nothing here tests it.
4. **(A) depends on the fill proxy**, whose no-fill bucket is only 23–33%
   truly unfilled. The between-minute contrast survives that because the proxy
   is equally inaccurate at every minute — but (A)'s absolute level does not.
5. **$4.30/day at q15** on an account that must first be funded above $211.

## 6. What this changes

The earlier verdict — no measurable edge anywhere — was **too strong**. It was
reached by testing policies (HCR, DRC, PTC, band shifts) rather than testing
*pricing*. The right question turned out to be the one the market answers
badly: it prices all four close minutes identically when they do not behave
identically.

Ranked by evidence:

1. **`:00` taker, no probe** — P=0.0244, proxy-free, ~$4.30/day at q15. The
   thing to run forward.
2. **`:00` + unfilled probe** — P=0.0013 and the strongest per contract, but
   conditional on a fill model that fails a hostile stress.
3. **Everything else** — dead, and now explained rather than merely refuted.

## 7. The correct next test

Not more history. `:00` has now been re-cut enough times that further slicing
of the same 73 days cannot add information.

**Run (C) forward, live, at q5, on `:00` only.** It needs no probe, no
proxy and no new architecture — it is one eligibility line. At 13 commits/day
a month produces ~390 out-of-sample observations, enough to see whether a
+3.37c edge is really there. That is the only remaining way to learn anything.

Scripts: `interplay_01_where_is_bias_weak.py`,
`interplay_02_underpriced.py`, `interplay_03_kill_the_candidate.py`.
