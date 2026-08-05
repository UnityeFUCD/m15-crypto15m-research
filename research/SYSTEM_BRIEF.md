# CRYPTO15M Longshot-Seller Maker — Technical Brief

Handoff document for an independent researcher. Written 2026-08-05.
Every number here is either exchange-authoritative or from a stated backtest;
where a figure is uncertain the interval is given, not just the point estimate.

---

## 1. The market

Kalshi lists 15-minute binary markets on five crypto assets:
`KXBTC15M`, `KXETH15M`, `KXSOL15M`, `KXXRP15M`, `KXDOGE15M`.

**Settlement rule.** YES pays iff `A1 >= A0`, where `A0` and `A1` are each a
60-second trailing mean of the CF Benchmarks Real Time Index taken at the
window's opening and closing boundary. **Equality pays YES.** This is a
symmetric "will it be higher in 15 minutes" contract with a tie-break toward
YES.

**Windows.** 96 per day per asset, 24/7 → 480 asset-windows/day.

**Fees.**
- Maker fee: **zero**.
- Taker fee: `ceil(0.07 · C · P · (1−P) · 1e4) / 1e4`, rounded UP to the
  centicent, charged **once per fill block, not per contract**.

The zero maker fee is load-bearing. The strategy never crosses the spread;
there is no taker code path in the runner at all.

**Tick grid** is tapered: 0.1c below 10c and above 90c, 1c in between.

**API.** V2, `POST /portfolio/events/orders`. Side is `bid`/`ask` expressed in
YES terms. `count`/`price` are strings. Note `count` is always null — the real
field is `count_fp`; similarly `position_fp`, `volume_fp`,
`remaining_count_fp`, `total_traded_dollars`, `market_exposure_dollars`.
Kalshi **trims trailing zeros** from fractional-second timestamps (`.5Z`,
`.167Z`), which broke a naive parser and silently discarded 621,647 trades
(10% of the sample) in the original discovery run.

---

## 2. The edge

**What we do.** Post a maker bid on the **favourite** — the side trading at
65–85c — between 8 and 14 minutes before the window closes, and hold to
settlement. No exits.

**Why it pays.** `maker_px > 50c` means the *taker* on the other side bought
the *cheap* side. The edge is payment for absorbing impatient longshot buying:
someone wants the 25c lottery ticket now, and we sell it.

**Mechanism confirmed, not assumed.** The hypothesis makes a sharp prediction
that raw volume does not: the edge should come specifically from taker flow
*buying the longshot*, and not from taker flow buying the favourite — the same
volume with the opposite meaning. Decomposing volume into `v_long` and `v_fav`
and racing them, `v_long` carries the effect and `v_fav` does not. This is the
single strongest structural result in the project.

**Backtest performance** (10 days, 898 windows, 25,359-row per-minute grid,
window-equal-weighted):

| volume floor | entries/day | win rate | edge/contract | $/day @ qty 30 |
|---|---|---|---|---|
| 0 | 143.7 | 0.758 | +4.84c | $208 |
| 500 | 128.1 | 0.798 | +8.34c | $321 |
| **2000** | **107.7** | **0.838** | **+10.60c** | **$343** |
| 4000 | 63.8 | 0.875 | +12.85c | $246 |

With per-window cap 3 at floor 2000: 121.6 entries/day, win 0.846,
**+11.29c/contract, $412/day**.

---

## 3. Live results — and their uncertainty

Exchange-authoritative, attributed by ticker (not by date — an older strategy
shared the same series on 2026-08-04 and date-slicing misattributes it):

```
104 settlements · 2,024 contracts · +$113 · +6.27c/contract
95% CI [-4.24c, +15.03c]     P(edge <= 0) = 0.133
F_Q (quantity-weighted passive fill fraction) = 0.9164
zero taker fills across all orders
```

**Read this carefully.** The point estimate moved +9.27c → +5.59c → +6.27c
within one hour as windows settled. Per-settlement standard deviation is
**41.23c**. The correct resampling unit is the *settlement*, not the contract —
all 30 contracts in a position share one outcome, and resampling contracts
gives a spuriously tight [+3.66, +7.41].

**The edge is not yet statistically confirmed.** It is directionally consistent
with the backtest, and the backtest's +8.34c sits inside the CI — but so does
zero.

Sample size required:

| target | settlements | days at 122/day |
|---|---|---|
| distinguish +5.6c from zero | 166 | 1.4 |
| CI half-width ±4c | 408 | 3.4 |
| CI half-width ±2c | 1,632 | 13.4 |
| CI half-width ±1c | 6,531 | 53.7 |

---

## 4. Capacity — the answer to "can this make $100k/year"

**Capital is not the constraint.** The strategy holds ~$110 gross at a time and
turns it over every 15 minutes. Funding qty 20 needs ~$44 of standing exposure.

**Fill capacity is the constraint**, because the edge *is* the liquidity
premium — our size cannot exceed the flow arriving to be absorbed.

Median flow in an entered window: 2,911 contracts cumulative at entry, of which
1,484 is longshot-buying. In the 8–14 minute slice while we rest, a median
5,126 contracts trade. A 30-lot is **~1.2%** of the longshot flow.

| qty | % of tradeable flow | $/day | $/year | assessment |
|---|---|---|---|---|
| 30 | 1.2% | $412 | $150k | flow absorbs it |
| 60 | 2.3% | $824 | $301k | flow absorbs it |
| 100 | 3.9% | $1,373 | $501k | flow absorbs it |
| 200 | 7.8% | $2,746 | $1.0M | fill rate will degrade |
| 500 | 19.5% | $6,864 | $2.5M | we *are* the flow; edge decays |

**$100k/year = $274/day**, which needs **qty 20** at the backtest edge or
**qty 40** at the live edge — 0.8%–1.6% of available flow.

**So size is not what stands between here and $100k.** What stands in the way
is edge durability: whether +6c/contract survives another 10,000 settlements,
more competition, and a different volatility regime. The extrapolation above
holds edge constant, which it will not be. Treat the table as a capacity
ceiling, not a forecast.

Caveats a researcher should press on:
- the flow figure counts *all* volume in the slice, not volume at our price
  level; true absorbable flow is smaller
- F_Q already degrades with queue depth: 0.937 with <100 ahead, **0.748 with
  >3k ahead**. A larger order rests deeper.
- adverse selection should worsen with size — the flow we win is increasingly
  the flow we should not want
- 10 backtest days is a narrow regime sample

---

## 5. Risk structure

**Positions within a window are correlated.** This is the most important risk
fact and it is easy to miss:

| simultaneous positions | P(all lose) observed | if independent | ratio |
|---|---|---|---|
| 2 | 0.0649 | 0.0237 | **2.7×** |
| 3 | 0.0226 | 0.0036 | **6.2×** |

Concentration risk grows superlinearly. Worst observed consecutive-window pair:
**−$88.20 = 26% of a $339 account**. This is why gross exposure is capped at
$110 and was *not* raised even though the per-window cap of 3 would otherwise
justify it.

By contrast, windows within a day are near-independent (ICC = −0.002; within an
hour, +0.04), so daily variance is close to the i.i.d. figure: daily sd moves
only $135 → $164 across bootstrap block choices.

**Live risk controls:**
- hard pre-submit drawdown envelope — Kalshi reserves cash on *rest* as well as
  on fill, so `worst_case_equity = cash − proposed_cost` is checked against
  `(1 − 0.30) × peak_equity` before every submission
- $150 runner stop, $150 watchdog halt threshold
- watchdog computes equity from **cost basis**, never market mark; an earlier
  version halted falsely at $76.80 on unrealised marks when realised drawdown
  was $63.00
- peak equity requires two confirming samples (settlement briefly credits cash
  while the mark persists, which once produced a phantom $251.57 peak)

---

## 6. The knobs — and whether they are dynamic

**All of them are static except one.** This is the honest answer and probably
the most promising research direction.

| knob | value | dynamic? |
|---|---|---|
| `LSM_QTY` | 30 (hard cap `MAX_QTY` 35) | static |
| price band | 65–85c | static |
| time window | 8–14 min remaining | static |
| `LSM_MIN_VOLUME` | 2000 contracts | static |
| `LSM_PER_WINDOW` | 3 | static |
| `LSM_MAX_GROSS` | $110 | static |
| `LSM_MAX_LOSS` | $150 | static |
| `LSM_MAX_DD_FRAC` | 0.30 | **dynamic** — floor tracks peak equity |
| series | 5 coins | static |

Measured sensitivity (out-of-sample, walk-forward):
- **volume floor** — dominant. Walk-forward selected 2000 on 7/7 held-out days.
- **per-window cap** — the 3rd entry carries **+16.65c**, CI [+12.23, +21.40],
  p < 0.0001, better on 9/10 days. Removing the three largest contributions
  *raises* it to +18.84c, so it is not outlier-driven. Mechanism: a 3rd entry
  only exists when three coins clear the volume floor in the same window, which
  selects high-agreement windows.
- **price band** — nearly flat. 65–85c is not special; do not tune it.
- **time window** — wider is worse.

Nothing adapts to time of day, realised volatility, per-coin behaviour, or
recent fill quality. That is unexplored territory, not a considered decision.

---

## 7. Already ruled out — do not redo these

- **Static arbitrage does not exist here.** 858,173 pairs → 1 violation worth
  0.52c. An earlier "$572,047 arbitrage" was a bug: `strike_type: 'between'`
  is a *range*, not a threshold.
- **Favourite-longshot bias does not generalise** outside CRYPTO15M.
- **The price path is a random walk** — below the arcsine law at every horizon.
- **z-score is a price proxy** with no independent content. (First test was
  underpowered and wrongly killed it; the corrected, properly powered test
  killed it for real.)
- **Take-profit is strictly worse at every exit level.** Losers never reach the
  target, so the worst case is identical and the upside is truncated.
- **Historical queue replay is impossible.** No order-book data overlaps the
  discovery period, and Kalshi never emits L3 — `client_order_id` is "only
  visible for your own orders."
- **D3/calm effect** was a measurement artifact.

**Methodological warning.** Contract-weighting over-samples busy windows where
the favourite ran away, and produced **three separate false findings** in this
project (a fake +3.18c spread effect, fake thin-coin unprofitability, a fake
z-score result) — all Simpson's paradox. **Window-equal-weight everything.**

---

## 8. Open questions worth a researcher's time

1. **Should any knob be dynamic?** Volume floor by time-of-day; band by
   realised vol; size by recent F_Q. Nothing here has been tried.
2. **Is "number of coins qualifying in a window" a signal in its own right?**
   The rank-3 result hints strongly that it is.
3. **Where does the edge actually decay with size?** Requires live
   experimentation at qty 60–100; the capacity table is inference, not
   measurement.
4. **Does the edge survive a different volatility regime?** 10 days is one
   regime.
5. **Optimal queue placement.** F_Q 0.916 overall but 0.748 when >3k ahead —
   is there a better price/queue tradeoff than always joining the favourite's
   bid?
6. **Adverse selection by counterparty type**, if it can be inferred at all
   from public trade data.
