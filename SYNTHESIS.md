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

**Why it barely works:**

| | per contract |
|---|---|
| maker edge, if every order filled | **+3.99c** |
| maker edge, fill-corrected | **+1.36c** |
| taker edge, real ask + real fee | **+0.15c** |

The first line is the backtest. The second is reality. **The gap between them
is the entire problem**, and Part 3 explains it.

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

Arithmetically it converts **+3.99c into +1.36c** — it removes about two
thirds of the edge. Every backtest in this repo that assumes 100% fill is
overstated by 27.6% pre-shift and 58.9% post-shift.

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
| **minute-:00 filter (HCR)** | affirmatively wrong — HCR is **stronger without it** (+4.36c vs −0.57c) |

The pattern: every one of these tried to find a *better subset* of markets.
None addressed fill bias. That's why they all failed.

---

## Part 6 — What is UNRESOLVED

Genuinely uncertain. Not proven, not refuted.

**HCR** — fade the favourite when it opposes a common 6-coin move in a calm
tape. Full write-up in `research/hcr_report.md`.

```
HCR      +0.88c fill-corrected   95% CI [-4.93, +6.46]   P(<=0) 0.36
non-HCR  -1.10c fill-corrected
matched controls P=0.195   permutation P=0.085   day-clustered P=0.098
```

Nothing clears 5%. Its real value is as a **filter, not a signal**: trading
everything is −$3.97/day, HCR-only is +$0.83/day, so it's worth **~$4.80/day
in losses avoided**, not in edge earned. But **6.4 years** of trading are
needed to distinguish +0.88c from zero at 80% power. Shadow only.

**DRC-15** — same shape: effect concentrated late, ~0pp win-rate lift, the
advantage is a 0.88c price difference rather than better prediction.

**RACE** — the *mechanism* (late fills are adversely selected) is real and is
just fill bias restated. The specific timeout parameters are unvalidated.

**Rank-3 removal** — 2 days of live data says drop it, 73 days of population
data says keep it. Unresolved, and 2 days can't settle it.

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

The strategy is **at its structural limit**. Population edge +3.99c, realized
+1.36c, and the gap is fill bias. Three independent signal candidates returned
three non-results. The dead-end list is ten deep and every entry attacked the
wrong constraint.

The account is at **$136.27** against a kill floor of **$398.25** (75% of the
$531 strategy high-water mark). `capture/hcr.py` sizes it to 0 and returns
KILL, which is correct — it should not trade here.

**The only question worth working on is fill bias**, because it is the only
thing standing between +1.36c and +3.99c. Tripling realized edge does not
require a new signal — it requires collecting the edge already measured. Two
approaches have not been tried:

1. **Queue position is available via the API** and has never been used. If
   adverse fills correlate with queue position at fill time, that is directly
   observable and directly actionable.
2. **Cancel-and-replace on index movement** — the fills that hurt arrive after
   the index moves against you. That's observable in real time from the same
   RTI feed the settlement uses.

Both attack the mechanism. Everything in Part 5 did not.
