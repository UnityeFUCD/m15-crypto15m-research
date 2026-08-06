# Triple-check of the `:00` decay — the "competed away" story is not supported

I reported that the `:00` edge was being arbitraged away. Checking it properly
reverses the interpretation. The decay in the *strategy* is real but marginal;
the decay in the *`:00` mispricing itself* is not established; and the specific
signature of "the market learned" is **absent**.

## 1. The decisive test: what actually fell?

`edge = win_rate − ask − fee`. If participants learned, the **ask rises** toward
the true probability. If the regime changed, the **win rate falls with the ask
flat**. These imply opposite futures.

```
period          n    mean ask   win rate      edge
May25-Jun14   532     0.7202     0.8026     +6.26c
Jun15-Jun30   426     0.7150     0.6690     -6.60c
Jul01-Jul15   422     0.7137     0.7299     -0.39c
Jul16-Aug05   494     0.7067     0.7591     +3.24c

change:  ask  -1.35c      win rate  -4.35pp
```

**The ask FELL by 1.35c. It did not rise.** The market never repriced `:00`
upward — which is precisely what "competed away" requires. That story fails its
own test.

## 2. The full `:00` population did not decay

| series | slope | 95% CI | P(≥0) |
|---|---|---|---|
| strategy's selected trades (n=450) | −0.1395 c/day | [−0.3017, +0.0069] | **0.0307** |
| full `:00` in-band population (n=1,874) | −0.0349 c/day | [−0.1846, +0.1068] | **0.2923** |

The population is also not monotone — it goes **+6.26 → −6.60 → −0.39 → +3.24**.
That is noise, not a trend. First half +0.54c, second half **+1.57c**: the
underlying mispricing is, if anything, slightly *better* now.

So the decay lives in the **strategy's 450 selected trades**, not in the `:00`
effect it exploits.

## 3. What did change: volatility collapsed

```
period          |A1/A0 - 1|    strategy edge
May25-Jun14        27.18 bp        +8.82c
Jun15-Jun30        24.43 bp        +8.87c
Jul01-Jul15        18.73 bp        +6.35c
Jul16-Aug05        15.10 bp        -0.68c
```

Market movement fell **44%** over the same window. Per-coin 1-minute vol roughly
halved (BTC 6.83 → 4.66bp, HYPE 16.66 → 9.30bp). Daily correlation between
market movement and strategy edge: **+0.21**.

The mechanism is coherent: the strategy buys a favourite 3 minutes in and needs
the move to **hold**. In a quiet tape the early move is smaller and the
favourite less genuinely established — while the ask stays near 71–76c
regardless. You pay the same price for a less certain outcome.

## 4. What is ruled out

- **Composition drift** — entry bid 0.6945 → 0.6953, spread 2.34c → 1.46c,
  markets/day 25.9 → 25.4, coin mix stable within ±3pp.
- **A market-wide deterioration** — other minutes moved +2.11c, +1.07c, −0.92c
  over the same period. Only the strategy subset fell consistently.

## Revised verdict

**Not established:** that the `:00` mispricing is disappearing. The ask never
rose, and the population trend is P=0.29.

**Supported:** the strategy's realised edge tracks market volatility, and
volatility fell hard in July.

That flips the implication for the live run. A competed-away edge does not
return. A **volatility-regime effect does**, whenever crypto gets moving again.

The live run is now worth more than I said an hour ago — but its result will be
confounded by whatever regime it lands in, so record realised market movement
alongside P&L or the outcome will be uninterpretable.

Evidence: `research/decay_triple_check.py`, `research/spec_is_decay_real.py`.
