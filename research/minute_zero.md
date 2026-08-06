# Minute :00 — the first result in this project to clear 5%

Found while reconciling an error, not by searching for it. Every number here
is on the **unsampled** population: `data/book_full.parquet`, 39,428 markets,
all four close minutes, all 24 hours, 73 days.

---

## The claim, stated precisely

There are two separate claims and only one of them is established.

**ESTABLISHED — minute is a real discriminator.**

```
day-matched :00 vs the other three minutes, within each day
  +4.27c   95% CI [+0.75, +7.80]   P(<=0) 0.0090

selection-corrected permutation (shuffle the minute LABEL within day,
3,000 times, record the BEST of four each time - because :00 was CHOSEN
as the best of four, not specified in advance):
  observed best-of-four      +4.27c
  permuted best-of-four mean +1.58c   p95 +2.92c
  P(permuted best >= observed) = 0.0020

Bonferroni (4 minutes tested): 0.0090 x 4 = 0.036
```

**NOT ESTABLISHED — that :00 is profitable in absolute terms.**

```
minute :00 fill-corrected  +2.37c   95% CI [-0.97, +5.69]   P(<=0) 0.0833
minute :00 taker           +1.04c   95% CI [-2.14, +4.18]   P(<=0) 0.2585
```

Being reliably better than a losing alternative is not the same as making
money. The absolute level's interval contains zero.

## Why this is not HCR

HCR failed three specific tests. Minute :00 passes all three.

| test | HCR | minute :00 |
|---|---|---|
| chronological **train** period | **−0.50c** (nothing) | **+3.33c** |
| valid / test | +3.12c / +3.60c | +3.11c / +4.70c |
| leave-one-coin-out | collapses without SOL (+1.62c → **+0.65c**) | **+1.85c to +2.75c**, stable |
| per coin | BTC and XRP negative | **all six positive** |
| effect vs sample size | shrank 46% as data grew | measured once, on the full population |

The chronological row is the important one. HCR's effect lived entirely in
valid and test; :00 is present in the training period at roughly the same size
as everywhere else. And where HCR was one coin (SOL) wearing a six-coin
costume, :00 holds when any single coin is removed.

## Full per-minute picture

```
minute    n     win    maker  fill-corrected   taker     day-clustered P(<=0)
  :00   1874  0.7444   +4.95c    +2.37c       +1.04c          0.0833
  :15   2268  0.7244   +2.63c    -0.05c       -1.16c          0.5047
  :30   1942  0.6957   -0.12c    -2.97c       -3.96c          0.9652
  :45   2103  0.7161   +1.78c    -0.95c       -2.06c          0.7548
```

`:30` is actively bad — −2.97c fill-corrected, P(≤0)=0.965. Roughly half the
strategy's losses come from trading a minute that loses money reliably.

## Economics, if the absolute level is real

```
in-band :00 markets      25.7 per day
fill-corrected edge      +2.37c
at q15                   +$9.11/day
```

The 2-per-window and 4-open caps do not bind: :00 offers 24 windows a day and
25.7 in-band markets, so the caps allow more than the supply.

**This is conditional on +2.37c being real, which is exactly what P=0.0833
says is unresolved.** The defensible reading is that trading only :00 removes
a reliably negative cohort; whether the remainder is positive needs forward
data.

## No mechanism identified

This is the weakest part of the finding and it should not be glossed.

`:00` closes coincide with the top of the hour, where hourly candles, funding
intervals, and other instruments' settlement cluster. That plausibly changes
who is quoting and how well the favourite is priced. **This is a story, not
evidence.** Nothing here tests it.

A finding without a mechanism that survives selection correction is worth
forward-testing, not worth sizing up on.

## What to do

1. **Stop trading `:30`.** It is −2.97c fill-corrected with P(≤0)=0.965 — the
   most confidently negative cohort in the entire project. This does not
   depend on the :00 result being real.
2. **Shadow `:00`.** Log it live and accumulate an out-of-sample record. At
   25.7 markets/day the population accrues quickly.
3. **Do not size up.** The absolute-level test is P=0.083, and the account is
   below its kill floor regardless.

## How it was found

While reconciling why an earlier figure (+3.99c "population maker edge") did
not match the unsampled data. That figure came from `ladder_paths`, a 28.6%
subset, and was a 1.43σ fluctuation: `P(random draw of 1,091 ≥ +3.99c) =
0.0751`. Correcting it exposed the per-minute structure that the sampled data
had been averaging over.

The error was real and is corrected in `SYNTHESIS.md`. The finding is a
by-product of fixing it.
