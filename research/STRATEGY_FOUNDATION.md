# CRYPTO15M Longshot-Seller Maker — Complete Structural Foundation

**Purpose of this document.** A complete statement of the market's structure, the
strategy's mechanism, every knob and its justification, every measured number,
and every dead end — written so a researcher can convert the qualitative
understanding into a quantitative research programme.

**Status conventions used throughout.** Every claim is tagged:

- **[ESTABLISHED]** — verified by multiple independent tests, survived
  out-of-sample and permutation testing
- **[MEASURED]** — single measurement, point estimate given with its interval
- **[REFUTED]** — tested and rejected
- **[UNKNOWN]** — open, with the measurement that would close it

Written 2026-08-05. Live equity at time of writing: **$509.11**.

---

# PART 1 — THE MARKET

## 1.1 The instrument

Kalshi lists 15-minute binary contracts on crypto prices. Series tickers follow
`KX<ASSET>15M`. Ticker format: `KXDOGE15M-26AUG050245-45` = asset DOGE, close
2026-Aug-05 02:45 **Eastern**, strike suffix.

**Settlement rule.** YES pays iff `A1 >= A0`, where:

- `A0` = simple average of the CF Benchmarks Real Time Index over the **60
  seconds before the window opens**
- `A1` = simple average of the same index over the **60 seconds before the
  window closes**

**Ties pay YES.** `strike_type` is `greater_or_equal`.

This is a symmetric "is it higher in 15 minutes" contract with a tie-break
toward YES. The 60-second averaging matters: it damps the final-second jitter
that would otherwise make the last moments a coin flip on microstructure noise.

**Cadence.** 96 windows/day/asset, 24/7, closing at :00 :15 :30 :45.

## 1.2 The fee structure — load-bearing

- **Maker fee: ZERO.**
- Taker fee: `ceil(0.07 · C · P · (1−P) · 1e4) / 1e4`, charged **once per fill
  block, not per contract**, rounded UP to the centicent.

**[ESTABLISHED]** The zero maker fee is why this strategy exists. The gross edge
is ~+8-11c/contract; a taker fee at 70c would be ~1.5c/contract and would not
kill it, but the strategy's whole design — patient limit orders, never crossing
— is only optimal because passivity is free. There is no taker code path in the
runner at all.

## 1.3 Tick grid

Tapered: **0.1c below 10c and above 90c, 1c between.** Our operating band
(65-85c) is entirely in 1c territory, so prices snap to whole cents.

## 1.4 API structure and its traps

Endpoint base: `https://api.elections.kalshi.com/trade-api/v2`. RSA-PSS signed
requests (`KALSHI-ACCESS-KEY`, `-SIGNATURE`, `-TIMESTAMP`).

**Field-name traps that have caused real bugs:**

| trap | correct field |
|---|---|
| `count` is **always null** | `count_fp` |
| `position` | `position_fp` |
| `volume` | `volume_fp` |
| `remaining_count` | `remaining_count_fp` |
| settlement `revenue` is an **integer in CENTS** (3000 = $30.00) | — |
| settlement cost is split | `yes_total_cost_dollars` + `no_total_cost_dollars` |
| no `revenue_dollars` field exists | — |

**Timestamp trap.** Kalshi **trims trailing zeros** from fractional seconds
(`.5Z`, `.167Z`). A parser assuming 6 digits silently discarded **621,647
trades (10% of the sample)** in the original discovery run. Any replay must
handle variable-length fractional seconds.

**Order submission.** `POST /portfolio/events/orders`. Side is `bid`/`ask`
expressed in YES terms. `count` and `price` are **strings**.

**Post-only rejection.** Orders that would cross return HTTP 400 with
`details: "post only cross"`. This is the guard working — the market moved
between price read and submit. Observed ~2 per 150 orders. These orders never
rest and must be **excluded from fill-rate statistics**.

**Trades feed.** `GET /markets/trades?ticker=X&limit=1000`, returned
**newest-first**, cursor-paginated. Each trade carries `taker_side`
(`yes`/`no`), `yes_price_dollars`, `count_fp`.

**Cash reservation.** Kalshi reserves cash the moment an order **rests**, not
when it fills, and again on fill. This is central to the risk model: if every
open position and every resting order went to zero, what remains is exactly the
cash balance. Therefore `worst_case_equity = cash`.

**[ESTABLISHED] No L3 data ever.** `client_order_id` is documented as "only
visible for your own orders." Queue position for other participants is
unobservable. **Historical queue replay is impossible** — no order-book depth
data overlaps the discovery period. Anything about queue behaviour must be
measured live.

---

# PART 2 — THE MECHANISM (why the numbers work)

## 2.1 The strategy in one sentence

Post a **maker bid on the favourite** — the side trading at 65-85c — between 8
and 14 minutes before close, only in markets that have already traded ≥2,000
contracts, and hold to settlement. Never cross. No exits.

## 2.2 Where the edge comes from — the longshot premium

**[ESTABLISHED]** When the maker price of a side exceeds 50c, the *taker* on the
other side is buying the *cheap* side. Someone wants the 25c lottery ticket and
wants it **now**. We sell it to them. The edge is compensation for supplying
immediacy to impatient longshot buyers.

**This was proved, not assumed.** The hypothesis makes a prediction that raw
volume does not: the edge should come specifically from taker flow *buying the
longshot*, and **not** from taker flow buying the favourite — numerically the
same volume, opposite meaning. Decomposing volume into `v_long` and `v_fav` and
racing them:

- `v_long` carries the entire effect
- `v_fav` carries none

This is the single strongest structural result in the project and it is what
distinguishes a real mechanism from a curve fit.

**[ESTABLISHED] The buyers are NOT informed.** Holding price fixed, the excess
(actual win rate minus price-implied) *rises* with longshot participation:

| longshot share quartile | 0.26 | 0.44 | 0.57 | 0.74 |
|---|---|---|---|---|
| excess | +8.91c | +9.32c | +11.03c | **+15.89c** |

If they knew something, excess would fall as they piled in. It climbs
monotonically. They are the source of the edge, not a threat to it.

## 2.3 What we pay for it — adverse selection at the fill

**[MEASURED]** We post at the favourite's bid and wait. Two outcomes:

- favourite **strengthens** → price walks up away from our bid → **no fill**.
  We miss a winner.
- favourite **weakens** → sellers come down and hit our bid → **fill**. We buy
  at 73c something the market now says is 68c.

Fills are selected toward losers; non-fills toward winners.

| | win rate | n |
|---|---|---|
| orders we FILLED | 0.7767 | 103 |
| orders that NEVER filled | **0.9167** | 12 |

**+14.00pp gap**, 95% CI [−5.58, +28.16], p = 0.1358. Direction and consistency
match the mechanism (65-70c: +15.28pp, 70-75c: +19.05pp) but the magnitude is
**not pinned down** at n=12 unfilled. This resolves as unfilled orders
accumulate.

## 2.4 The master equation

```
realised edge  =  longshot premium collected  −  adverse selection paid
```

Everything else in this document is a consequence of that equation.

**Consequence A — F_Q is an EDGE metric, not a throughput metric.**
Every unfilled order is a winner selected away. A drop in fill rate is edge
being taken *before it appears in P&L*. This makes F_Q the only leading
indicator we have, since realised edge needs hundreds of settlements to resolve.

**Consequence B — there is a structural price ceiling.**
The premium shrinks as price rises (less longshot left to sell) while the
selection cost does not. Where they cross, a price bucket turns negative. This
**predicted** the 80-85c bucket going negative (−11.20c) before it happened, and
says it is structural, not bad luck.

**Consequence C — entering early is doubly correct.**
More resting time → higher fill rate → fewer winners lost to selection.
**[MEASURED]** Fill rate by entry time:

| minutes remaining at entry | orders | F_Q | never filled |
|---|---|---|---|
| 13.5-14 | 81 | **0.9091** | 7.4% |
| 12-13.5 | 29 | 0.8345 | 13.8% |
| 10-12 | 4 | 0.5500 | 25.0% |

Later entry fills far worse (−0.4494, CI [−0.8464, −0.0461], p = 0.0230). The
dominant effect is *time available to fill*, not price walk-away. The runner
already enters at the first qualifying moment (median 13.75 min), which is the
ceiling given the band opens at 14.

## 2.5 The correlation structure — why drawdown happens

**[ESTABLISHED]** Coins inside one window do **not** fail independently:

| simultaneous positions | P(all lose) observed | if independent | ratio |
|---|---|---|---|
| 2 | 0.0649 | 0.0237 | **2.7×** |
| 3 | 0.0226 | 0.0036 | **6.2×** |

Concentration risk grows superlinearly. The worst 5% of windows carry −$1,705
against $1,630 of total profit — they lose more than everything else makes — and
they average **2.30 positions with 2.30 losses**. Not "mostly losses": *every*
position in a bad window loses.

**Drawdown is not accumulated small losses. It is a handful of windows where
the whole basket goes down together.**

**[ESTABLISHED] But windows within a day are near-independent**: ICC = −0.002
(within day), +0.04 (within hour). So daily variance is close to the i.i.d.
figure — daily sd moves only $135 → $164 across bootstrap block choices. The
clustering is *within window*, not across the day.

## 2.6 The cross-sectional signal — why it works

**[ESTABLISHED]** The other coins' favourite prices at our entry instant predict
our outcome, at an identical own-price:

| others' mean favourite px | our px | our win | our excess |
|---|---|---|---|
| 59.8c | 73.6c | 0.8199 | +8.43c |
| 68.0c | 73.7c | 0.8382 | +10.12c |
| 74.0c | 73.6c | 0.8806 | +14.43c |
| 81.6c | 73.8c | 0.8905 | **+15.25c** |

p = 0.0000, survives walk-forward at +5.31c. Our own price is flat across all
four rows, so this is not a price effect in disguise.

**Interpretation:** it measures **market decisiveness**. When the whole complex
is strongly priced, favourites hold and adverse selection is mild. When
everything sits near a coin flip, ours is fragile too. This is why a
cross-coin signal beat every within-coin one — the risk is market-wide, and
within-coin measurements are structurally blind to it.

---

# PART 3 — EVERY KNOB

Current live configuration (`lsm_config.json`):

```
LSM_QTY=30              LSM_MAX_GROSS=110.0     LSM_MAX_LOSS=150.0
LSM_MAX_DD_FRAC=0.3     LSM_MIN_VOLUME=2000.0   LSM_PER_WINDOW=3
LSM_DEAD_FILTER=1       LSM_OTHERS_MAX=0.71     LSM_FLONG_MAX=0.5046
LSM_SERIES=KXBTC15M,KXETH15M,KXSOL15M,KXXRP15M,KXDOGE15M,KXHYPE15M
```

## 3.1 Entry-selection knobs

### `BAND = (0.65, 0.85)` — favourite's bid must sit here

**Why it exists.** Below 65c the "favourite" is barely favoured and the premium
is thin relative to variance. Above 85c the premium is too small to cover
adverse selection (Consequence B).

**[MEASURED]** Edge is remarkably flat inside the band — this knob is **not**
where the money is:

| bucket | win | edge/ct | capital-normalised score |
|---|---|---|---|
| 55-65c | 0.6915 | +8.81c | 0.1323 |
| 65-70c | 0.7942 | +11.32c | 0.1505 |
| 70-75c | 0.8233 | +9.55c | 0.1188 |
| 75-80c | 0.9185 | **+13.92c** | **0.1619** |
| 80-85c | 0.8961 | +6.60c | 0.0720 |
| 85-95c | 0.9524 | +5.47c | 0.0552 |

**[REFUTED] Trimming to 65-80c**: costs **−$38.73/day**, better on 2/10 days,
edge diff p=0.13, permutation p=0.13, sign flips on XRP. The capital-score
argument (80-85c scores 0.0973 vs 0.1499, p=0.0200) is a *ratio* that improves
because 80-85c reserves more capital — it only converts to dollars if freed
capital gets reused, and it does not.

**[REFUTED] Extending to 55-65c**: the slice is genuinely profitable
(+8.93c, CI [+5.23, +12.40], **P(edge≤0)=0.0000**, works in all three coins),
but adding it gains only **+$15/day on $412** because the marginal entries carry
lower edge. Arithmetic, not a capital effect.

**Is 75-85c distinguishable from 65-75c?** No. Excess +10.25c vs +10.42c,
difference +0.17c, CI [−3.54, +3.65], **p = 0.88**. Treat as one population.

### `MIN_LEFT = (8, 14)` minutes remaining

**Why it exists.** Earlier than 14 min the price hasn't settled into a stable
favourite. Later than 8 min there isn't enough resting time to fill
(Consequence C).

**[REFUTED] Entering later**: 8-14 gives $411.99/day vs 8-11 $404.46, 8-10
$386.13, 8-9 $331.83. And the ranking **does not flip at tighter capital**
($110 → $70 → $45 → $30, 8-14 wins at every level). Waiting buys
capital-minutes but loses more edge than it frees.

### `MIN_VOLUME = 2000` contracts traded before entry

**Why it exists.** This is the **strongest single conditioning variable found**.
In a market nobody has traded there is no impatient demand to be paid for — the
quote is a market maker's guess, not a consensus.

**[ESTABLISHED]** Monotone in edge:

| floor | entries/day | win | edge/ct | $/day @30 |
|---|---|---|---|---|
| 0 | 143.7 | 0.758 | +4.84c | $208 |
| 500 | 128.1 | 0.798 | +8.34c | $321 |
| 1000 | 122.5 | 0.812 | +9.23c | $339 |
| **2000** | **107.7** | **0.838** | **+10.60c** | **$343** |
| 4000 | 63.8 | 0.875 | +12.85c | $246 |

Walk-forward selected 2000 on **7/7** held-out days. Survives every test that
killed the z-score: works within all three coins, correlates *negatively* with
price (−0.23, so not a price proxy), holds within coin AND price
(+10.77, CI [+7.35, +14.74]).

### `MAX_PER_WINDOW = 3`

**[ESTABLISHED]** The 3rd entry is the *best* one, not a dilution: rank-3
entries carry **+16.65c** (CI [+12.23, +21.40], p<0.0001), beat cap 2 on
**9/10 days**, and removing the three largest contributions *raises* it to
+18.84c — so not outlier-driven.

**Mechanism:** a 3rd entry only exists when three coins clear the volume floor
in the same window, which selects the highest-agreement windows. This is the
same market-decisiveness effect as `others_px`.

**[REFUTED] cap 2** — makes drawdown *worse* (95%DD $449 → $473, ruin 4.4% →
6.2%). It removes 14 entries/day of within-day diversification and the mean
falls faster than the variance. The obvious "fewer correlated bets" move
backfires.

### `LSM_DEAD_FILTER` — the conjunction filter

Skip when **both**: `others_px ≤ 0.71` AND `frac_long ≤ 0.5046`.

- `others_px` = mean favourite price of the *other* coins in the same window
- `frac_long` = share of traded volume where the taker bought the cheap side

**[ESTABLISHED]** Those entries earn **+2.03c** vs **+13.41c** for everything
else. Difference −13.41c, CI [−15.29, −11.31], worse on **7/7** out-of-sample
days, **permutation p = 0.0005** (1 of 2,000 shuffles this extreme, outcomes
shuffled within coin × price bucket, so not "worst of four cells"). Present in
every coin: DOGE −12.05, SOL −7.50, XRP −20.87.

**The conjunction is required** — `others_px` alone gives −6.68c, `frac_long`
alone −5.81c. Global thresholds work nearly as well as per-cell ones (−11.85c
vs −13.41c), which is why they are deployed as fixed numbers.

**Effect:** Sharpe 1.20 → 1.42, 95% drawdown $332 → $253, **ruin 1.52% →
0.52%**. It is a *risk* filter — the P&L change was inside noise.

**Why this one filter works when others don't:** the discarded entries earn
approximately **zero**, so discarding them costs nothing. Every other filter
discards entries earning +6 to +9c, which we cannot afford (see §5.1).

### `LSM_SERIES` — which assets

Currently 6: BTC, ETH, SOL, XRP, DOGE, HYPE.

**[MEASURED] Per-coin, API replay, n≈170 markets each:**

| coin | replay edge | live edge | live n | backtest rows |
|---|---|---|---|---|
| ETH | +5.79c | +6.54c | 40 | THIN (256) |
| SOL | +3.59c | +1.51c | 23 | 3,994 |
| HYPE | +3.75c | — | 1 | **none** |
| XRP | +2.36c | +8.61c | 20 | 3,723 |
| DOGE | +2.16c | +18.67c | 18 | 3,515 |
| BNB | **+1.01c** | — | — | **none** → **REMOVED** |
| BTC | ~0 qualifying entries | +14.36c | 39 | **1 row** |

**BNB was removed** because +1.01c is at the level of the dead-entry cell
(+2.03c) that we already refuse to trade. Trading a coin at the edge we
explicitly filter out is inconsistent.

**[UNKNOWN] Per-coin ranking is not currently possible.** To resolve a 5c gap
you need ±2.5c, which needs **1,044 settlements per coin ≈ 35 days**. We have
18-40. Current CI half-widths: ETH ±12.8c, BTC ±12.9c, SOL ±16.9c, XRP ±18.1c,
DOGE ±19.0c. **Every per-coin number quoted anywhere is inside its own noise.**

**BTC anomaly worth explaining:** BTC produced **zero** qualifying entries in 60
settled markets and 1 row in the entire historical 8-14 minute grid. Its
favourite almost never sits at 65-85c that late — BTC is priced too efficiently
and sits near 50c or resolves early. The BTC fills we *do* get are rare
exceptions, and they have been our best coin live (+14.36c) — an unresolved
tension worth investigating.

## 3.2 Risk knobs

### `MAX_GROSS = 110` — total simultaneous exposure

**Why it exists.** Caps how much can be lost in one correlated window
(§2.5, 6.2× joint-loss).

**[MEASURED]** The worst observed consecutive-window pair is **−$88.20 = 26% of
a $339 account**. Doubling gross would scale that toward ~52% in one bad pair.

**[UNKNOWN] Whether it currently binds at all.** The backtest shows zero
blocking at *every* level ($110 to $220) because it settles positions at window
boundaries. Live, positions overlap windows. Current instrumentation shows
**zero** gross-cap blocks with equity at $509 — so it is probably *not* binding
now, and $110 was sized for a $341 account.

### `MAX_DD_FRAC = 0.30` — pre-submit drawdown envelope

**Why it exists — and why it must be pre-submit.** The watchdog is a 60-second
*poll*, an alarm after the fact. Between polls the runner could submit orders
pushing exposure past any limit, and a cancel/fill race can turn a resting order
into a position regardless. So the limit is enforced **before** the order is
sent:

```
worst_case_equity = cash − cost_of_proposed_order
require: worst_case_equity ≥ (1 − MAX_DD_FRAC) × peak_equity
```

This is exact because Kalshi reserves cash on rest *and* on fill (§1.4).

**This is the only risk control that does not depend on knowing the regime**,
which matters because the drawdown estimate ranges from $19.50 to $469.80
depending on assumptions the data cannot yet resolve.

### `MAX_LOSS = 150` (runner) and `WATCH_MAX_DD = 150` (watchdog)

Hard stops. The watchdog additionally:

- computes equity from **cost basis, never market mark** — an earlier version
  halted falsely at $76.80 on unrealised marks when realised drawdown was
  $63.00
- requires **two confirming samples** for a new peak — settlement briefly
  credits cash while the mark persists, which once produced a phantom $251.57
  peak
- restarts a dead runner **using the config the runner published**, and refuses
  to restart rather than invent one (it previously fell back to hardcoded
  defaults and silently reverted tuned settings mid-session)

### `QTY = 30` (hard cap `MAX_QTY = 35`)

**[MEASURED] Capacity is not the constraint.** Median flow while we rest is
5,126 contracts; a 30-lot is **~1.2%** of the longshot flow. Size could rise
substantially before moving the market.

**[REFUTED] Sizing up on the best signal cell**: raises $/day but pushes 95%DD
$253 → $289 and ruin 0.52% → 0.77%. Mean and variance scale together.

---

# PART 4 — EVERYTHING RULED OUT

Do not re-derive these.

| hypothesis | verdict | evidence |
|---|---|---|
| Static arbitrage exists | **[REFUTED]** | 858,173 pairs → 1 violation worth 0.52c. An earlier "$572,047 arbitrage" was a bug: `strike_type: 'between'` is a *range*, not a threshold |
| Favourite-longshot bias generalises beyond CRYPTO15M | **[REFUTED]** | tested, does not |
| Price path is predictable | **[REFUTED]** | below the arcsine law at every horizon — a random walk |
| z-score (standardised index margin) adds information | **[REFUTED, then softened]** | conditioning on *exact* price, coin and time: +1.38pp overall (p=0.274), +1.95pp below 80c (p=0.108), +2.29pp below 75c (p=0.105). All CIs include zero, but point estimates are positive and **monotone as price falls** — the shape the theory predicts. Status: **unresolved, not dead** |
| Take-profit exits improve consistency | **[REFUTED]** | strictly worse at every exit level — losers never reach the target, so the worst case is identical and the upside is truncated |
| Informed traders are picking us off | **[REFUTED]** | excess *rises* with longshot participation (§2.2); no pre-entry flow feature predicts outcome once price is fixed |
| Deep-queue fills are toxic | **[REFUTED]** | ≥1000 ahead: +11.57c vs <1000: +7.11c (p=0.62). Sign is *opposite* to the fear — deep queue means many makers want the favourite, the same decisiveness `others_px` rewards |
| YES/NO asymmetry (tie-break or retail bullishness) | **[REFUTED]** | headline −2.56c (p=0.18) but sign flips across buckets (65-70c +2.34, 70-75c −11.12, 80-85c +2.47) and lives in one coin (XRP −7.62). Mechanism check fails: NO-favourite windows carry only 0.5139 longshot share vs 0.4863 |
| Ranking candidates by predicted edge beats first-come | **[REFUTED at current capital]** | zero benefit at gross $110; +$22.84/day only below gross $45 |
| Hour-of-day gating | **[REAL but UNPROFITABLE]** | effect survives permutation (**p = 0.0007**, +2.94c/contract, 9/10 days) but gating loses **−$111.84/day** and wins in dollars on 1/10 days, because it halves entries |
| Day-to-day adaptive sizing | **[REFUTED]** | daily premium lag-1 autocorrelation +0.1613, **permutation p = 0.5752**. Yesterday does not predict today; any adaptive rule chases noise and cuts size after unlucky days |
| Historical queue replay | **[IMPOSSIBLE]** | no order-book data overlaps the discovery period; Kalshi never emits L3 |

---

# PART 5 — THE DEEP STRUCTURAL FINDINGS

## 5.1 The binding constraint is OPPORTUNITY, not capital

**[ESTABLISHED]** With equity at ~$509 and a $110 gross cap, the runner deploys
$20-66 and logs **zero** gross-cap and **zero** drawdown-envelope blocks. It
cannot find enough qualifying markets.

**This single fact explains why four separate real signals all failed to make
money.** Volume, `frac_long`, `others_px`, hour-of-day are each genuinely
predictive — and gating on any of them *loses* money, because discarding an
opportunity buys nothing back when opportunities are the scarce resource.

**The rule this implies:** a filter is only worth deploying if the entries it
discards earn approximately **zero**. The dead-entry conjunction is the only one
that clears that bar (+2.03c). Everything else discards entries earning +6 to
+9c.

**Corollary:** the lever for more profit is **more markets**, not better
selection. That is why adding HYPE was correct and why the research programme
should prioritise finding more qualifying series.

**Caution against a superseded claim.** An earlier reading of "177 refusals per
36 fills" was **wrong** — those counts were inflated by the same market being
re-evaluated every 5 seconds. Any event-count metric in this system must be
de-duplicated by ticker before interpretation. The same error inflated the
dead-entry skip count (84 events = 7 distinct markets).

## 5.2 The premium level-shifted down ~78% after Aug 1

**[ESTABLISHED]** Measured at trade prices with **no fill conditioning**, so
adverse selection cannot contaminate it:

| period | premium |
|---|---|
| Jul 23 – Aug 1 (backtest) | **+10.40c** |
| Aug 2–3 | **+0.81c** |
| Aug 4–5 | **+4.39c** |

**Method verified:** the replay logic reproduces the grid/backtest method to
within **0.89c** on the same period (+10.40c vs +11.29c). So this is a genuine
market change, not a measurement artifact.

**No decay *trend* within the recent window** — slope +4.75c/day, CI [−3.25,
+12.80], P(slope ≥ 0) = 0.94. It reads as a level shift, possibly transient.

**The alarming correlate:** daily qualifying-entry counts roughly **doubled**
(119-166/day → 272-330/day) exactly as the premium collapsed. `corr(daily n,
daily premium) = −0.8252` excluding the partial day. More markets qualifying
while the premium falls is the signature of **more participants arriving**.
This is one event described two ways, not 14 independent observations — but it
is the first evidence pointing at competition.

**Consequence: the backtest is stale.** Every projection built on +11.29c —
$412/day, the drawdown percentiles, the ruin probabilities — is too optimistic.
On current-regime numbers the expectation is closer to **$80-150/day**.

## 5.3 Live performance vs the backtest

**[MEASURED]** Live, attributed by ticker (date-slicing misattributes, because
an older strategy shared the same series on Aug 4):

```
141 settlements · 3,186 contracts · +$290.62 · +9.12c/contract
F_Q 0.8517 · zero taker fills across every order
```

**But the interval is what matters.** At 124 settlements the 95% CI was
**[−2.36c, +14.81c]**, P(edge ≤ 0) = 6.98%. The point estimate has swung
+9.27 → +5.59 → +6.27 → +7.41 → +9.12 within hours. Per-settlement sd is
**41.23c**.

**The correct resampling unit is the SETTLEMENT, not the contract** — all 30
contracts in a position share one outcome. Resampling contracts gives a
spuriously tight [+4.92, +8.23].

**Note the tension:** live realises +9.12c while the recent replay premium is
+2.32c. The live filters (volume floor, dead-entry) select better-than-average
entries by construction, but that plausibly cannot account for 7c. **+2.32c
sits inside the live CI**, so they are not formally in conflict — live simply
has error bars wide enough to hide the difference, with 141 observations against
replay's ~1,200.

## 5.4 Sample sizes required

| target | settlements | days at current rate |
|---|---|---|
| distinguish +5.6c from zero | 166 | 1.4 |
| CI half-width ±4c | 408 | 3.4 |
| CI half-width ±2c | 1,632 | 13.4 |
| CI half-width ±1c | 6,531 | 53.7 |
| **rank one coin against another (±2.5c)** | **1,044 per coin** | **35 per coin** |

---

# PART 6 — METHODOLOGICAL WARNINGS

These caused real false findings in this project. A researcher repeating this
work will hit them.

**1. Contract weighting causes Simpson's paradox.** It over-samples busy windows
where the favourite ran away, and produced **three separate false findings**: a
fake +3.18c spread effect, fake thin-coin unprofitability, and a fake z-score
result. **Window-equal-weight everything.**

**2. Event counts are inflated by re-evaluation.** The runner scans every 5
seconds, so one persistently-blocked market generates dozens of log events.
De-duplicate by ticker before interpreting any count.

**3. Day-clustered inference is mandatory.** Entries within a day share market
conditions. Naive CIs are far too tight.

**4. Units errors in ratio metrics.** A "drawdown removed per $/day given up"
ratio divides a *level* by a *rate*, yielding units of days — it cannot be
compared against 1, and it inflated an intervention's apparent efficiency by
roughly the horizon length. Put both sides on the same horizon.

**5. Single-seed stress results are not estimates.** A stress simulation that
randomly selects which windows to flip gave ratios from 0.03 to 0.77 across 24
seeds. Report the distribution.

**6. Small out-of-sample day counts make drawdown meaningless.** Bootstraps
reporting maxDD $0.00 were reporting the *absence of a bad day in 5-10
observations*, not the absence of risk.

**7. Overlapping training windows break independence.** A walk-forward where
each day reuses all prior days does not give independent trials — 9/10 successes
can arise from one lucky parameter propagating. Permutation-test the whole
walk-forward procedure.

**8. Freeze/hash verification can silently pass.** A parity check once hashed the
**empty string** (`e3b0c442…`) because a regex anchored on a comment that did
not exist, making every check trivially pass.

---

# PART 7 — OPEN QUESTIONS, WITH THE MEASUREMENT THAT CLOSES EACH

| # | question | why it matters | measurement needed |
|---|---|---|---|
| 1 | Is the premium drop competition or transient? | **Existential** | track daily qualifying-entry count vs premium for 2-3 weeks. Rising n with depressed premium = competition |
| 2 | What is the true current edge? | Sets every projection | 408 more settlements → ±4c; 1,632 → ±2c |
| 3 | Should the dead-entry thresholds be refit? | They were fitted on the pre-shift regime | refit `others_px`/`frac_long` cut points on Aug 2+ data only, compare to 0.71 / 0.5046 |
| 4 | Is the volume floor of 2000 still right post-shift? | Same staleness risk | re-run the floor ladder on post-shift data |
| 5 | How large is adverse selection really? | Determines the value of any fill-rate improvement | accumulate unfilled orders; n=12 → need ~60 for ±5pp |
| 6 | Why is BTC our best live coin but produces almost no qualifying entries? | Suggests a different, rarer, richer setup | characterise the BTC windows that *do* qualify |
| 7 | Are there more qualifying series? | The only proven profit lever (§5.1) | monitor ADA/BCH/TON/NEAR/ZEC volumes; they currently fail the 2,000 floor |
| 8 | Does z survive with more data? | Point estimates are positive and monotone | it needs ~4× current data to resolve ±1pp |
| 9 | Does F_Q predict realised edge? | Would make F_Q a true leading indicator | correlate per-period F_Q against subsequent realised edge |

---

# PART 8 — WHAT A QUANTITATIVE BRIEF SHOULD PIN DOWN

Framing for the receiving researcher. The strategy is fully described by:

```
E[profit per window] = P(qualify) × P(fill | posted) × [ premium(p, v, x) − selection(p, F_Q) ] × qty
```

where `p` = entry price, `v` = prior volume, `x` = cross-sectional state
(`others_px`, `frac_long`).

**Each term needs a number and a functional form:**

1. **`P(qualify)`** — currently ~1.8 candidates per window across 6 coins. This
   is the *binding* term (§5.1) and the least studied.
2. **`P(fill | posted)`** — measured 0.8517, decomposes by queue depth
   (0.894 <100, 0.748 >3k) and by entry time (0.909 at 13.5-14 min).
3. **`premium(p, v, x)`** — the best-characterised term. Monotone in `v`,
   roughly flat in `p` across 65-85c, monotone in `others_px` and `frac_long`.
   **Level is regime-dependent and currently unstable (§5.2).**
4. **`selection(p, F_Q)`** — the least-measured term. Estimated at +14pp of win
   rate at n=12. Theory says it grows relative to the premium as `p` rises,
   which is what makes the price ceiling structural.

**The highest-value quantitative contribution** would be a joint model of terms
3 and 4 that explains the 80-85c bucket turning negative *from the mechanism*
rather than from curve-fitting — because if `premium(p)` and `selection(p)` can
be separately estimated, the optimal band falls out analytically instead of
being tuned, and it will adapt automatically when the regime shifts.
