# HCR — final verdict: **FAILS** on the unsampled population

This file has carried three earlier verdicts — PASS, then FAIL, then NOT
ESTABLISHED — each computed on a sampled book. On the **complete unsampled
population** the answer is settled, and it is negative.

Evidence: `research/audit_unsampled.py`. Population: `data/book_full.parquet`,
39,428 markets with a valid quote at 8–14 min, **all four close minutes, all
24 hours, exactly six coins**. In-band with a valid signal: **7,504 markets,
1,556 HCR, 73 days** — 2.8x the sampled population every earlier verdict used.

---

## The finding that ends it: SOL is the entire effect

```
population        n       HCR     non-HCR      lift
all six coins   7504    +1.11c     -0.51c    +1.62c
drop SOL        6207    +0.16c     -0.49c    +0.65c   <- collapses
drop BTC        6114    +1.54c     -0.51c    +2.05c
drop XRP        6270    +1.55c     -0.84c    +2.39c
drop HYPE       6362    +1.34c     -0.47c    +1.81c
drop DOGE       6353    +0.93c     -0.46c    +1.39c
drop ETH        6214    +1.18c     -0.27c    +1.45c

day-clustered HCR lift EXCLUDING SOL:
  -0.40c   95% CI [-5.16, +4.07]   P(<=0) 0.5467
```

Removing any of the other five coins leaves the effect intact or strengthens
it. Removing **SOL** takes HCR's own edge from +1.11c to +0.16c and the lift
to nothing. Excluding SOL the lift is **negative**.

A signal built on a six-coin common factor that only works in one of the six
coins is not a common-factor signal. It is one coin's run.

## The effect shrank as the data grew

```
sampled    2,941 markets /   525 HCR : matched-control lift +2.43c  P=0.195
unsampled  7,504 markets / 1,556 HCR : matched-control lift +1.32c  P=0.181
```

**2.8x the data, and the effect fell 46%.** Regression toward zero under more
data is the signature of a spurious finding. A real effect holds its point
estimate and tightens its interval.

## Every other test, on the full population

| test | result | verdict |
|---|---|---|
| matched controls (coin × week × 2c bucket × side) | +1.32c, 95% CI [−1.50, +4.18], **P=0.181** | fails |
| permutation (shuffle r_common within day) | **P=0.093** | fails |
| day-clustered HCR against zero | +1.11c, CI [−2.34, +4.47], **P=0.264** | fails |
| chronological train | **−0.50c** (n=760, the largest slice) | fails |
| chronological valid / test | +3.12c / +3.60c | passes |
| per coin | BTC −0.32c, XRP −2.12c negative; **SOL +6.44c dominates** | fails |
| leave-SOL-out | **−0.40c, P=0.547** | fails |

## Two things that do NOT explain it away

**Coin selection is not the cause.** Dropping the two coins the live strategy
trades least (HYPE 4% of orders, DOGE 11.6%) barely moves the result:
+1.62c → +1.57c. The result does not depend on including them.

**The fill model is not the cause.** The lift is stable across every fill
assumption, including none at all:

```
0.843 / 0.962  (original estimate)   lift +1.62c
0.8848 / 0.9884 (corrected on 303)   lift +1.61c
1.000 / 1.000  (no bias at all)      lift +1.54c
```

## The minute filter is noise, confirmed

With a balanced population the per-minute lifts are:

```
:00  -0.05c    :15  -5.59c    :30  +4.50c    :45  +6.52c
```

An 12-cent spread across four minutes with no mechanism that distinguishes
them, and a pattern that does not match the sampled data's pattern. This is
noise being sliced. `capture/hcr.py` ignores `close_minute`; that stays.

## The larger finding: no price band survives fill correction

Band sensitivity is testable for the first time, because `book_full.parquet`
applies **no band filter at fetch time**:

| band | n | win | maker | fill-corrected | taker |
|---|---|---|---|---|---|
| 0.55–0.65 | 15,346 | 0.6146 | +2.47c | **−0.69c** | −1.74c |
| 0.60–0.70 | 10,625 | 0.6680 | +2.99c | **+0.01c** | −1.04c |
| **0.65–0.80** (current) | 7,504 | 0.7224 | +2.54c | **−0.17c** | −1.30c |
| 0.70–0.80 | 3,324 | 0.7512 | +1.77c | **−0.77c** | −1.97c |
| 0.80–0.90 | 577 | 0.8423 | +1.46c | **−0.37c** | −1.48c |

**Every band is at or below zero after fill correction.** The current band is
not badly chosen — no band is good. 0.60–0.70 is nominally best at +0.01c,
which is zero.

This is a bigger result than HCR. The strategy has no profitable price region
once realistic fills are applied, and no signal tested in this project changes
that.

## Recommendation

**Do not deploy HCR.** Not as a signal, and not as the filter the previous
revision suggested — the filter's apparent value was SOL.

`capture/hcr.py` is correct as specified and covered by 70 passing tests. Keep
it as reference; do not size on it.

The one question worth working remains **maker fill bias**, quantified
prospectively in `research/prospective_execution.py`: it cost **$186.58 of
$195.96** — 95% of theoretical profit — across two live days.

## All four verdicts, and why they differed

| verdict | dataset | why it was wrong |
|---|---|---|
| PASS | `premium_history2` | minute-:00 only, 8 of 24 hours |
| FAIL | + earliest 18 days at :15/:45 | `sort_values` instead of random sample |
| NOT ESTABLISHED | 2,941 sampled, all minutes | correct but underpowered, and `nc>=4` not `nc==6` |
| **FAILS** | **7,504 unsampled, exactly 6 coins** | **definitive** |

Three of four were artifacts of sampling. The lesson is in `SYNTHESIS.md`
Part 7: a sampled dataset lies about the thing it was sampled on, and the only
cure is to stop sampling.
