# What this project established

Everything below is re-derivable from `data/` via `research/project_synthesis.py`.
Figures are quoted per contract in cents, because dollars depend on size and
size has changed.

---

## Part 1 — How the strategy actually trades

**The market.** Kalshi CRYPTO15M. Every 15 minutes, for each of 6 coins, a
market opens asking: will this coin's index be higher at close than at open?

- YES settles if `A1 >= A0`
- `A0` and `A1` are 60-second trailing means of the CF Benchmarks Real-Time
  Index, at open and at close
- **Ties pay YES**

Reconstructing `A1 >= A0` reproduces the printed settlement on **99.98% of
41,334 markets**. The mechanism is fully understood — there is no ambiguity
left about what these contracts do.

Two field mappings that are not documented and cost real money to learn:
`floor_strike` is `A0`, `expiration_value` is `A1`, and
`expected_expiration_time` is the true close **plus five minutes** (verified
on 39,453 markets). Using it as the close silently shifts every window.

**The trade.** Between 14 and 8 minutes before close, find markets where the
favourite trades between 65c and 80c. Rest a **post-only bid** at the
favourite's price. Maker fee is zero. Hold to settlement.

**Why it should work.** The YES base rate across all markets is 0.4950 — the
market is a coin flip and correctly priced as one. But conditional on a
favourite having emerged at 65–80c, that favourite wins **~73%** of the time,
which is more than 65–80c implies. That gap is the edge.

**Why it does not work.** On the **unsampled** population, 39,428 markets:

| | per contract |
|---|---|
| maker edge, if every order filled | **+2.29c** |
| maker edge, fill-corrected | **−0.43c** |
| taker edge, real ask + real fee | **−1.55c** |

> **CORRECTION.** An earlier revision gave +3.99c / +1.36c / +0.15c and called
> the strategy marginally profitable. Those came from `ladder_paths`, a 28.6%
> subset, and were a **1.43σ fluctuation** — a random draw of that size
> reaches +3.99c **7.5%** of the time (`research/reconcile_edge.py`). The two
> files agree *exactly* where they overlap; the subset simply ran hot.
> **The strategy is not profitable after realistic fills.**

The gap between the first two lines is fill bias (Part 3). But the second line
is now **negative**, which is a different claim than the earlier revision made.

The one exception is **minute :00** — see Part 6c and `research/minute_zero.md`.

---

## Part 2 — Why the results feel so violent

This is the answer to "why did we win yesterday like crazy but lose so bad
today, there's no consistency."

At q15 and a 72c entry:

```
a win  pays  +$4.20
a loss costs -$10.80
expected value per market  +$0.15
standard deviation         +$6.66
```

**Noise is 44x signal, per market.** You risk $10.80 to make $4.20, and you're
right 73% of the time, which nets fifteen cents.

A 73% win rate loses four in a row 0.5% of the time — that's once every 200
sequences, and at q15 it costs **$43.20**. Nothing has gone wrong when that
happens. It is the strategy operating normally.

There is no version of this where daily results look consistent. Day-to-day
P&L is almost entirely noise, and the edge only becomes visible over months.
Any explanation of a single bad day in terms of *what changed* is a story
fitted to noise — including the ones I offered earlier in this project.

---

## Part 3 — The one real mechanism: maker fill bias

Measured on live resting orders:

```
P(fill | the order would LOSE)   0.962
P(fill | the order would WIN)    0.843
```

**Your resting bid is 14% more likely to fill when it is about to lose.**

This is not bad luck and it does not average out. It is adverse selection: the
counterparty who lifts your bid is disproportionately someone who just saw the
index move against your side. You get filled on the markets you'd rather skip
and miss the ones you want.

Arithmetically it converts **+2.29c into −0.43c** — it removes the entire edge
and then some, pushing the strategy below zero. Every backtest in this repo
that assumes 100% fill is overstated by 27.6% pre-shift and 58.9% post-shift.

**This is the binding constraint on the entire strategy.** Not signal quality,
not timing, not coin selection. Everything in Part 5 failed because it tried
to add edge to a system whose problem is that it can't collect the edge it
already has.

---

## Part 4 — What is SOLID

Build on these.

| finding | status |
|---|---|
| settlement is `A1 >= A0`, 60s trailing means, ties pay YES | 99.98% on 41,334 markets |
| `expected_expiration_time` = close + 5 min | 39,453 markets |
| `floor_strike` = A0, `expiration_value` = A1 | exact |
| favourite at 65–80c wins ~73% | 41,334 markets |
| maker fill bias 0.962 / 0.843 | live orders, the core mechanism |
| taker fee `ceil(0.07·C·p·(1−p)·10⁴)/10⁴`, maker fee zero | exact |
| queue position IS available via the API | `/portfolio/orders/{id}/queue_position` |
| per-market SD is 44x per-market EV | arithmetic |
| the account runs multiple strategies — 303 of 1,802 orders are LSM | must filter or every number is wrong |

---

## Part 5 — What is DEAD

Tested and refuted. **Do not revisit these without new information.**

| idea | why it's dead |
|---|---|
| **improve fill rate** | premise backwards — 92.86% on unfilled orders is survivorship, not opportunity |
| **cross the spread (taker)** | −$5.58/day over 73 days, P(worse) = **1.0000** |
| **quote one tick better** | fees eat it; `P(fill \| LOSE)` = 1.0000 at every rung |
| **trade only high-volatility windows** | volatility is correctly priced; no residual edge |
| **mean reversion after a move** | endpoint artifact — non-overlapping pairs flip *positive* in 5 of 6 coins |
| **filter on queue depth** | no relationship |
| **drop the worst coin** | in-sample selection; doesn't hold out |
| **filter by hour of day** | no stable pattern once multiple-comparison corrected |
| **shift the price band** | edge is flat across the band; shifting trades size for nothing |
| **minute-:00 as an HCR filter** | HCR's *lift* is not concentrated at :00 — see the note below |

The pattern: every one of these tried to find a *better subset* of markets.
None addressed fill bias. That's why they all failed.

> **Do not confuse the last row with Part 6c.** Two different quantities share
> the word "minute". HCR's *lift* is not concentrated at :00 — that filter is
> dead, and `capture/hcr.py` correctly ignores `close_minute`. But the
> *absolute edge* at :00 is the strongest result in the project. The first is
> about HCR; the second is about the base strategy. Both are true.

---

## Part 5b — No price band survives fill correction

Measured on the **unsampled** population (39,428 markets, all minutes, no band
filter applied at fetch time — so this is testable for the first time):

| band | n | win | maker | **fill-corrected** | taker |
|---|---|---|---|---|---|
| 0.55–0.65 | 15,346 | 0.6146 | +2.47c | **−0.69c** | −1.74c |
| 0.60–0.70 | 10,625 | 0.6680 | +2.99c | **+0.01c** | −1.04c |
| **0.65–0.80** (current) | 7,504 | 0.7224 | +2.54c | **−0.17c** | −1.30c |
| 0.70–0.80 | 3,324 | 0.7512 | +1.77c | **−0.77c** | −1.97c |
| 0.80–0.90 | 577 | 0.8423 | +1.46c | **−0.37c** | −1.48c |

**Every band is at or below zero after fill correction.** The current band is
not badly chosen — no band is good. The best, 0.60–0.70, is +0.01c, which is
zero.

This is the largest single result in the project. It says the strategy has no
profitable price region once realistic fills are applied, and that no amount
of band tuning changes it.

## Part 6 — What is DEAD, second wave (the signal candidates)

All three candidates are now refuted on the unsampled population.

**HCR** — fade the favourite when it opposes a common 6-coin move in a calm
tape. Full write-up in `research/hcr_report.md`. **SOL is the entire effect:**

```
all six coins   HCR +1.11c   lift +1.62c
drop SOL        HCR +0.16c   lift +0.65c
day-clustered excluding SOL:  -0.40c  95% CI [-5.16, +4.07]  P(<=0) 0.5467
```

Removing any of the other five leaves it intact or strengthens it. Removing
SOL kills it. A six-coin common-factor signal that works in one of six coins
is not a common-factor signal.

It also **shrank as the data grew** — matched-control lift +2.43c on 2,941
markets became +1.32c on 7,504. Regression toward zero under more data is what
a spurious effect does. Nothing clears 5%: matched controls P=0.181,
permutation P=0.093, day-clustered P=0.264, train −0.50c.

Not coin selection (dropping HYPE+DOGE moves it +1.62c → +1.57c) and not the
fill model (lift is +1.62 / +1.61 / +1.54c across all three assumptions,
including no bias at all).

**DRC-15** — same shape: effect concentrated late, ~0pp win-rate lift, the
advantage is a 0.88c price difference rather than better prediction.

**RACE** — the *mechanism* (late fills are adversely selected) is real and is
just fill bias restated. The specific timeout parameters are unvalidated.

**Rank-3 removal** — 2 days of live data says drop it, 73 days of population
data says keep it. Unresolved, and 2 days can't settle it.

## Part 6c — Minute :00, the one positive result

Full write-up: `research/minute_zero.md`. Two claims, only one established.

**ESTABLISHED — minute is a real discriminator.**

```
day-matched :00 vs the other three, within day
  +4.27c   95% CI [+0.75, +7.80]   P(<=0) 0.0090

selection-corrected permutation (shuffle the minute LABEL within day and
record the BEST of four, because :00 was CHOSEN as best-of-four):
  observed +4.27c   permuted best-of-four mean +1.58c   p95 +2.92c
  P = 0.0020        Bonferroni 0.0090 x 4 = 0.036
```

**NOT ESTABLISHED — that :00 is profitable in absolute terms.**

```
:00 fill-corrected  +2.37c   95% CI [-0.97, +5.69]   P(<=0) 0.0833
:00 taker           +1.04c   95% CI [-2.14, +4.18]   P(<=0) 0.2585
```

Reliably better than a losing alternative is not the same as making money.

**Why this is not another HCR:**

| test | HCR | minute :00 |
|---|---|---|
| chronological **train** | **−0.50c** | **+3.33c** |
| valid / test | +3.12c / +3.60c | +3.11c / +4.70c |
| leave-one-coin-out | collapses without SOL (+1.62c → +0.65c) | +1.85c to +2.75c, stable |
| per coin | BTC, XRP negative | **all six positive** |

Full per-minute picture:

```
minute    n     win    maker  fill-corrected   taker    day-clustered P(<=0)
  :00   1874  0.7444   +4.95c    +2.37c       +1.04c         0.0833
  :15   2268  0.7244   +2.63c    -0.05c       -1.16c         0.5047
  :30   1942  0.6957   -0.12c    -2.97c       -3.96c         0.9652
  :45   2103  0.7161   +1.78c    -0.95c       -2.06c         0.7548
```

**The immediately actionable part needs no further validation: stop trading
`:30`.** It is −2.97c fill-corrected with P(≤0)=0.965 — the most confidently
negative cohort found anywhere in this project, and dropping it does not
depend on the :00 result being real.

**No mechanism is identified.** `:00` coincides with the top of the hour where
other instruments settle and roll, which plausibly changes who is quoting —
but that is a story, not evidence. A finding without a mechanism that survives
selection correction is worth forward-testing, not sizing up on.

At q15 and 25.7 in-band `:00` markets/day, +2.37c is **+$9.11/day** — if real.

## Part 6b — What the live record PROVED (prospective, held separate)

`research/prospective_execution.py`. Four predictions registered before
computing anything, each derived from the fill-bias model, each able to fail.
A power check ran first: 303 orders resolve 10.7pp at 80% power against a
predicted 11.9pp effect, so a null would have meant something.

| prediction | result |
|---|---|
| P1 fill bias exists | `P(fill\|lose)` **0.9884** vs `P(fill\|win)` **0.8848** = **+10.4pp**, CI [+5.5, +15.2], **P=0.0001** |
| P2 realized ≈ fill-corrected, not raw | realized **+0.15c**; 1.21c from corrected, 3.84c from raw |
| P3 the ones that got away were winners | never-filled orders **96.15% winners** vs **71.62%** base rate |
| P4 full-fill overstates | actual **+$9.38** vs **+$195.96** if all filled — **fill bias cost $186.58** |

All four hold. **This is the only hypothesis in the project that survived a
pre-registered test.** Fill bias destroyed 95% of theoretical profit in two
days.

---

## Part 7 — What the experiments taught, as method

The research findings are mostly negative. The method findings are not.

**1. Sampled data lies about the thing it was sampled on.**
`premium_history2` is minute-`:00` only and covers 8 of 24 hours. Testing a
minute-`:00` hypothesis on it produced P=0.0002. The effect was the sampling.
My fix then sorted by close time and pulled the earliest 18 days against 73,
which produced the opposite error. **Three verdicts on HCR — PASS, FAIL, NOT
ESTABLISHED — and two were artifacts of my own sampling.**

**2. Check what else is in the account.** 1,499 of 1,802 orders belong to
other strategies. Pooling them made every early number wrong.

**3. Missing outcomes are not missing at random.** 17 markets that closed
after the snapshot were worth **−$134** — they moved the baseline from
+$143.38 to +$9.38. Markets absent from a dataset are absent for reasons
correlated with their result.

**4. Compute the rank on all orders, not the filled ones.** Ranking within
filled orders only shifted 15 of 277 and changed the rank-3 penalty from
−$75.81 to −$28.11. Survivorship hides inside intermediate steps.

**5. One seed is not a p-value.** 0.034 on one seed was 0.053–0.064 across
seven. Any p-value near 0.05 from a single seed is unreported noise.

**6. Chronological splits catch what significance tests miss** — but a train
vs test gap must itself be tested. HCR's was P=0.076, i.e. inside noise, which
makes "absent in train" *uninformative* rather than *disproving*. I initially
called it disqualifying. That was wrong.

**7. When the mechanism is understood, stop adding signals.** Fill bias was
identified early and every subsequent experiment ignored it in favour of
looking for a better filter. Ten dead ends in Part 5 is the cost of that.

---

## Part 8 — Where this actually stands

The strategy is **past its structural limit — it is below zero.** On the
unsampled population every price band is at or under zero after fill
correction, and the in-band figure is **−0.17c/contract**. Three independent
signal candidates returned three refutations. The dead-end list is ten deep
and every entry attacked the wrong constraint.

The account is at **$136.27** against a kill floor of **$398.25** (75% of the
$531 strategy high-water mark). `capture/hcr.py` sizes it to 0 and returns
KILL, which is correct — it should not trade here.

**Two questions are worth working on.**

**(a) Minute :00** — the only cohort that is positive, the only result in the
project to clear 5% after selection correction, and the only one that holds in
the training period. See `research/minute_zero.md`. The immediately actionable
part needs no further validation: **stop trading :30**, which is −2.97c
fill-corrected with P(≤0)=0.965.

**(b) Fill bias**, because it is what stands between +2.29c and −0.43c.
Recovering it does not require a new signal — it requires collecting edge
already measured. Two approaches have not been tried:

1. **Queue position is available via the API** and has never been used. If
   adverse fills correlate with queue position at fill time, that is directly
   observable and directly actionable.
2. **Cancel-and-replace on index movement** — the fills that hurt arrive after
   the index moves against you. That's observable in real time from the same
   RTI feed the settlement uses.

Both attack the mechanism. Everything in Part 5 did not.
