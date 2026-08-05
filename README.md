# CRYPTO15M longshot-premium maker — full research handoff

Live market-making strategy on Kalshi's 15-minute crypto up/down markets, plus
every experiment run against it. Written for another agent picking this up cold.

**No credentials are in this repo.** `live/acct.py` reads an API key id and a
private key from a path outside the tree. You cannot run anything live from
this alone, by design. All analysis scripts work on the bundled parquets.

---

## 1. What the strategy is

Kalshi lists a market every 15 minutes per coin: *"will the price be higher at
the close than at the open?"* Settlement is `A1 >= A0`, where both are
**60-second trailing means of the CF Benchmarks Real Time Index**. Ties pay YES.

The bot:

1. Watches 6 series — `KXBTC15M, KXETH15M, KXSOL15M, KXXRP15M, KXDOGE15M, KXHYPE15M`
2. With **8–14 minutes left**, finds the **favourite** (the side priced ≥ 50c)
3. If the favourite's bid is in **65–80c** and window volume ≥ 2000, posts a
   **maker bid at the favourite's own bid** for `QTY` contracts
4. Max 3 positions per window, `MAX_GROSS` $110 of capital deployed
5. Cancels anything unfilled at 7 minutes left, then holds to settlement

**Verified live: 608 of 608 orders were the favourite side at its own bid.**
Maker fees are zero on Kalshi; taker fees are
`ceil(0.07·C·p·(1−p)·1e4)/1e4` per fill block.

### Why there is an edge at all

The **longshot premium**. Retail overpays for the cheap side of a binary, so
the favourite is systematically underpriced. Verified by decomposition: only
the longshot leg carries the premium, the favourite leg carries none. It is a
behavioural premium, not a volatility or information edge — which matters,
because it means the usual quant levers do not apply (see §4).

---

## 2. Current live state (2026-08-05 23:16 UTC)

| | |
|---|---|
| equity | **$216.27** |
| floor (hard stop) | $211.00 |
| room | **$5.27** |
| peak | $457.87 (real max $531.31) |
| drawdown | $241.60 (52.8%) |
| deployed config | `QTY=20 MAX_GROSS=110 MIN_VOLUME=2000 PER_WINDOW=3 LIVE_HI=0.80 AGREE_RULE=1 FLOOR=211` |

Settled book: **269 fills, +$73.18, +1.22c/contract.** Both trading days were
individually profitable (Aug 4 +$51.52, Aug 5 +$21.66). The drawdown is
variance, not decline — see §5.

---

## 3. THE MOST IMPORTANT THING TO KNOW

**`research/final_minute_favorite_maker_validation/grid.parquet` is not
trustworthy.** Almost every early conclusion in this project was derived from
it, and reconstruction from the exchange's own candlesticks contradicts it
twice, both times in the same direction:

| quantity | grid.parquet (10 days) | reconstruction (73 days) |
|---|---|---|
| population edge | +11.76c | **~+3.5c** |
| 00–07 UTC hour effect | +8.12c | **+0.09c** |

It overstates effects roughly 3×. This explains why *every* live price band
reads "below research prediction," and why three deployed changes had to be
reversed. **Prefer `data/underlying.parquet` and `data/ladder_paths.parquet`,
which are reconstructed directly from market metadata and candlesticks.**

---

## 4. Everything tested, and the result

Every one of these was tested against the bar: *what it discards must be
verifiably NEGATIVE (not merely smaller), baseline is keep-all, measured in
dollars, day- or window-clustered.*

### Rejected — market is efficient

| idea | result |
|---|---|
| **Fill-rate improvement** | The 92.86% win rate on unfilled orders is **survivorship**, not lost edge. Markets that never return won 100%; reachable ones won 46.7% at 68.1c = −21.47c |
| **Hold past the 7-min cancel** | −21.47c/contract even assuming front-of-queue |
| **Taker (cross the spread)** | −$5.58/day over 73 days, P(worse)=1.0000. True spread is 2.08c mean, not the 1.21c a 2-day live sample suggested |
| **Quote improvement (bid+1..+4)** | +$1.17/day, CI [−0.35,+1.10], P=0.21. `fill\|LOSE = 1.0000` at **every** rung — see §5 |
| **Underlying volatility** | Correctly priced. 65-70c band: calm +13.68c vs wild +13.03c. All correlations null |
| **Mean reversion** | Apparent effect was overlapping-endpoint bias. Non-overlapping pairs flip **positive** in 5/6 coins |
| **Queue depth** | Deep queue is the *best* bucket; skipping it loses $12/day |
| **Hour-of-day tilt** | +8.12c on 10 days → **+0.09c on 73 days**. Noise |
| **Coin selection (drop DOGE/SOL)** | No coin is a drop candidate; the negative readings were pre-cap-selection artifacts |
| **Cap 1 or 2 per window** | Cap 3 best. Marginal entries **+4.18c, CI [+1.29,+7.13]** |
| **Reverse to the longshot on wide spreads** | Favourite wins in **every** spread bucket inside the band. Split-half −8.30c, dollars −$10.18/day |
| **Add 6 more coins** | Only BNB/NEAR/ZEC are liquid enough, and coins correlate **0.768** — effective independent bets go 1.24 → 1.33 |
| **Adaptive/percentile dead-entry cuts** | The filter's discards were worth −0.35c ≈ zero. Disabled |

### Confirmed correct as deployed

- **65–70c is the best slice in 7 of 11 weeks**; 60-65c averages negative, 80-85c badly negative → the 65–80c band and `LIVE_HI=0.80` are right
- **14-minute entries are best (+3.67c)** and the runner already enters as early as the window allows (median 13.3 min)
- `MAX_PER_WINDOW=3`, `MIN_VOLUME=2000`, agreement rule — all survive re-testing
- The capital cap **never binds** (mean gross $35.54 of $110, max $103.30)

---

## 5. The two findings that actually explain the P&L

### (a) Maker fill bias — the only real leak, and it is structural

Measured on live orders, then confirmed exactly on 73 days of book data:

```
fill | LOSER   = 0.962   →  1.0000 at every posting price
fill | WINNER  = 0.843
```

A favourite that goes on to **win** strengthens and runs away from a resting
bid, so we never fill it. One that goes on to **lose** weakens, comes back
through our price, and fills us. **We fill 100% of losers no matter where we
post.** Price only changes how many winners we catch, and the market charges
almost exactly what those extra winners are worth.

This converts a population premium of ~+3.65c into a realised +1.54c, and it
**fully accounts for the gap** — predicted realised win rate 0.714 vs observed
0.7063. It also means every 100%-fill backtest in this project overstates edge
by **27.6% (pre-shift) to 58.9% (post-shift)**.

### (b) The noise is enormous relative to the edge

```
mean P&L per window   +$0.60
SD   P&L per window   $20.17
daily mean           +$36.59
daily SD             $157.51     ← 56% of account equity
days to establish the edge is positive:  74
```

Sharpe is genuinely good (0.232 daily ≈ 4.4 annualised). But a one-standard-
deviation day is ±$157 on a $250 account. **Consistency is mathematically
unavailable at this size.** The Aug 5 session went +$272 by 09:00 UTC and
finished +$21 — a 1.7σ round trip, entirely ordinary.

**The premium is NOT decaying.** 73-day trend +0.0617c/day (positive), last 14
days +8.74c — the strongest stretch in the sample.

---

## 6. Layout

```
live/
  lsm_runner.py          the deployed bot
  risk.py                single source of truth for the stop; $211 hard floor
  lsm_watch.py           watchdog; restarts only from published config
  acct.py                standalone account view (does NOT import the runner)
  lsm_report.py          fill rate by queue depth, F_Q
  lsm_price_report.py    per-price-bucket economics
  RISK_RUNBOOK.md        halt/restart procedure

  # the reconstruction toolchain — start here, this is the trustworthy data
  fetch_underlying.py    rebuilds the true BRTI series from market metadata
  premium_history.py     replays the strategy over 73 days of candlesticks
  quote_ladder.py/2.py   exact fill-vs-price ladder from the observed book
  maker_vs_taker_73d.py  the taker comparison
  both_sides.py          favourite vs longshot, cap tests, band drift
  spread_reversal.py     verification that killed the reversal idea
  skip_audit.py          what every deployed filter actually discarded

research/                77 numbered experiments, 15_* .. 73_*
data/                    9 parquets (see §3 for which to trust)
```

### Reproducing the trustworthy numbers

```bash
python live/fetch_underlying.py    # 41,334 settled markets, 73 days
python live/premium_history.py     # the population premium, day by day
python live/quote_ladder2.py       # the fill ladder with correct fees
```

### Two traps that cost hours

- `expected_expiration_time` is the window close **plus 5 minutes**. Deriving
  the close from the `wkey` (ET ticker clock + 4h in summer) is correct — this
  is verified on 39,453 markets. Getting it wrong reads prices 5 minutes late,
  deep into resolution.
- Settlement `revenue` is an **integer in cents**; cost splits across
  `yes_total_cost_dollars` / `no_total_cost_dollars`. There is no `*_fp`
  variant for cost — using one silently yields 0 and makes every winner look
  like +100c/contract.

---

## 7. Open questions worth another set of eyes

1. **Is there any way around the fill bias?** Pricing is closed (§5a). Taker is
   closed. That leaves order size and queue mechanics, which have never been
   separable from the regime.
2. **Position sizing.** At qty 20, 30-day floor risk is 4.14%; at qty 15 it is
   1.31%. The edge is real but small and the account is now $5 above its floor.
3. **Does the YES/NO asymmetry inform the true favourite?** Untested as of this
   writing — whether the *unfavoured* side's book shape predicts outcomes
   better than the favourite's price alone.
