# HCR — final verdict: **NOT ESTABLISHED**

**This file previously carried a PASS verdict. That verdict was wrong and is
retracted.** It was computed on a dataset that could not support it. The
history of the error is kept below on purpose, because the way it failed is
more useful than the conclusion it replaced.

Superseded by `hcr_final_audit.py` (taker), `hcr_maker_audit.py` (maker),
`hcr_minute_daymatched.py` (minute confound). Previous revision of this file:
`git log -- research/hcr_report.md`.

---

## The number that decides it

Calendar-balanced population: **2,941 in-band markets, 525 HCR, 73 days, all
four close minutes, all 24 hours.** The production design is post-only maker,
so maker is the metric that counts. Fill correction uses the measured
P(fill|winner)=0.843 / P(fill|loser)=0.962.

| cohort | n | win | maker | fill-corrected |
|---|---|---|---|---|
| HCR | 525 | 0.7314 | +3.53c | **+0.88c** |
| non-HCR | 2,416 | 0.7148 | +1.64c | **−1.10c** |

Fill-corrected lift **+1.99c**. At q15, on the observed 7.2 HCR firings/day,
that is **+$0.83/day.**

Even granting the signal in full, it earns about a dollar a day.

## Why it is not established

| test | result | verdict |
|---|---|---|
| matched controls (coin × week × 2c bucket × side) | +2.43c, 95% CI [−3.32, +8.20], **P=0.195** | fails |
| day-clustered HCR cohort alone | +3.53c, CI [−1.85, +8.78], **P=0.098** | fails |
| permutation (shuffle r_common within day) | **P=0.085** | fails |
| chronological **train** | **−0.15c lift** on the largest slice (304 markets) | fails |
| chronological valid / test | +1.98c / +7.50c | passes |
| per coin | BTC −1.51c, XRP −2.44c; other four positive | mixed |

Nothing clears 5%. Three independent tests land in 0.09–0.20 — the region
where an effect is neither present nor excluded.

**Correction — the chronological argument is weaker than first stated here.**
An earlier revision called the train null "disqualifying". Tested directly
(`hcr_why_not_trade.py`), the train-vs-test gap is +7.65c with 95% CI
[−2.95, +17.77], **P=0.076** — itself inside noise. The data cannot tell the
two periods apart, so "absent in train" is **uninformative, not disproving**.
It remains suggestive, because it is the same shape DRC-15 showed, but it is
not evidence the signal is fake.

The real objection is in the next section.

## The minute-:00 filter is wrong and should stay dropped

The original brief conditioned on minute :00. Day-matched, within-day — the
only comparison not confounded by calendar:

```
all minutes      days 69  taker lift +3.67c  P(<=0) 0.0755  win +3.23pp
minute :00 only  days 49  taker lift -0.57c  P(<=0) 0.5305  win -0.89pp
excluding :00    days 68  taker lift +4.36c  P(<=0) 0.0776  win +3.81pp
```

HCR is **stronger without** :00 than with it. `capture/hcr.py` accepts
`close_minute` and deliberately ignores it; `test_hcr_has_no_minute_filter`
pins that behaviour.

## How the PASS happened — the actual failure

Three compounding data faults, each of which alone inverted the answer.

**1. `premium_history2` is minute-:00 only, and covers 8 of 24 hours.**
Every figure in the retracted version above (+9.18c, +10.69pp, matched-control
P=0.0002) came from it. Conditioning the test on the same minute the
hypothesis names guarantees the hypothesis looks right.

**2. My fix sampled the wrong markets.** Fetching :15/:45 I wrote
`todo.sort_values("close_utc")`, which took the earliest 18 days against
:00/:30's 73. That produced the FAIL — equally invalid, because it compared
minutes across different calendars. Fixed to
`todo.sample(LIMIT, random_state=...)`, `fetch_missing_minutes.py:57`.

**3. Only after both fixes** does the population support a verdict, and the
verdict is neither PASS nor FAIL.

Sequence: PASS (biased minute) → FAIL (biased calendar) → NOT ESTABLISHED
(balanced). Two of my three verdicts were artifacts of my own sampling, not
findings about the market.

## The real objection: the edge is unmeasurable, not absent

"HCR is more positive than non-HCR" is true. It is also not sufficient.

```
HCR's own edge   +0.88c   95% CI [-4.93, +6.46]   P(<=0) 0.36
at q15           +$0.95/day mean, SD $16.38/day
days to distinguish +0.88c from 0.00c at 80% power:  2,321  (6.4 years)
```

The uncertainty on +0.88c is **seven times the estimate**, and there is a 36%
chance the true value is zero or negative. Daily noise is 17x daily signal.
No amount of live trading resolves this on a useful horizon.

## What IS worth keeping — and it is bigger than the edge

| policy | per contract | $/day at q15 |
|---|---|---|
| trade everything | −0.75c | **−$3.97** |
| HCR only | +0.88c | **+$0.83** |
| non-HCR only | −1.10c | −$4.81 |

Day-matched within-day, HCR-only vs trade-everything: **+3.35c, P=0.078.**

> The value of HCR is **~$4.80/day in losses avoided**, not $0.95/day in edge
> earned. An earlier revision of this file said "worth about a dollar a day" —
> that was the wrong quantity. The non-HCR cohort loses money, and HCR's job
> is to decline it.

This is consistent with everything else established in this repo: population
edge ~+3.5c, realized +1.54c, the gap entirely maker fill bias. HCR's role is
not *adding* edge but *declining* the markets where fill bias eats it.

That is a reason to trade **less**, not a reason to trade. Fill-corrected, the
base strategy is underwater at −0.75c; HCR moves it to roughly **break-even**.
Break-even does not compound.

## Recommendation

Do not size on HCR. `capture/hcr.py` is correct as specified and fully covered
(56/56 tests passing) — keep it, run it in **shadow**, and let it accumulate
an out-of-sample record. The training-period null is the thing to watch: if
the signal is real, forward data will look like valid/test; if it was a search
artifact, forward data will look like train.

About 300 more HCR firings — roughly six weeks — separates those two.

## Comparison with the other candidates

| candidate | matched controls | permutation | train period | verdict |
|---|---|---|---|---|
| HCR | +2.43c, P=0.195 | P=0.085 | **−0.15c** | not established |
| DRC | ~0pp win lift, price only | not run | effect late-only | not established |
| RACE | n/a (execution) | n/a | n/a | mechanism yes, parameters no |

Three candidates, three non-results. The base strategy remains at its
structural limit, and the limit is maker fill bias.
