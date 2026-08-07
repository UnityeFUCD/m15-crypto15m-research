# Breakthrough Search — Authoritative Status

## Executive verdict

The search did **not** validate another predictive edge in the crypto-15m
outcome space. It did identify a genuinely different structural candidate:

> **Queue-Shielded Target-Frontier Reward Farming (QSTRF)** — rest low-price
> liquidity inside an incentive program's Target Size frontier, behind existing
> queue where possible, and require the conservative reward subsidy to dominate
> the maximum directional fill loss.

This is a **mechanism breakthrough candidate**, not a production PASS. Its edge
comes from an external liquidity subsidy rather than predicting settlement.
Actual eligibility, queue stability, reward-credit semantics and fill timing
remain prospective facts.

No result authorizes live orders or weakens an account KILL state.

---

## What was independently killed

### Cross-market quote-lag / relative value (CMQL)

A preregistered audit used 114,138 synchronous observations across 19,023
window-minute states. Peer-coin quote features slightly helped validation but
worsened sealed-test settlement log loss versus own-path features. The selected
peer policy lost $68.26 at q15 in sealed TEST, with -8.75c/contract edge and
negative/cross-zero block intervals. Predicted edge was not economically
monotone. The dominance-pair lane generated no qualifying TEST trades.

**Verdict: FAIL.** One-minute cross-coin Kalshi quote lag is not the missing
executable edge in this population.

### Gold one-cent historical rule

A frozen q64, 1c KXGOLD15M rule found zero historical eligible entries in the
four available reward-program days. The current 1c book state was not a
recurrent historical start-state.

**Verdict: no historical confirmation.**

---

## The new structural mechanism

Kalshi's public liquidity programs specify:

- a reward pool `R` for an incentive period;
- a Target Size `T`;
- a distance multiplier `m` for quote quality;
- proportional reward allocation from qualifying resting score.

Under deliberately hostile competition — both outcome sides continuously
filled to Target Size — a q-contract order resting for fraction `f` of the
period has conservative modeled reward:

```text
reward_lower(q) = R * (q * m * f) / (2*T + q*m*f)
```

Maximum directional loss on a filled bid at price `p` is:

```text
loss_max(q) = q * p
```

The key sufficient arithmetic condition is:

```text
reward_lower(q) >= loss_max(q)
```

with all of:

1. the order lies inside the currently displayed Target Size frontier;
2. q fits behind displayed liquidity at that level;
3. the modeled reward clears the stated payout minimum;
4. enough incentive-period time remains after submission.

The queue provides the safety layer. A new order joins behind displayed size at
its price. It can still add qualifying liquidity inside Target Size while the
contracts ahead must be consumed before it can fill. This potentially
decouples reward score from fill risk.

---

## Current-book evidence

A public snapshot screened 3,745 active liquidity programs. Among the 1,200
highest-density programs with usable books it found:

- 759 one-sided Target Size frontier rows;
- 34 q1 rows whose full-period modeled reward exceeded one-contract maximum
  loss;
- 1 two-sided arithmetic-cover row;
- 11 markets where the quantity modeled to reach a $1 payout at a frozen
  half/quarter-period rest fraction fit inside the displayed frontier and had
  reward greater than maximum one-fill loss.

Examples from that snapshot included:

| Market | Quote | Queue/displayed ahead | Frozen quantity/rest | Modeled reward | Max one-fill loss |
|---|---:|---:|---:|---:|---:|
| Presidential approval bucket | 1c | 258.03 | q10 at 50% period | $1.033 | $0.10 |
| Daily gas threshold | 1c | 1.01 | q41 at 50% period | $1.015 | $0.41 |
| Boston rain | 1c | 471.26 | q41 at 50% period | $1.015 | $0.41 |
| China AI outcome | 2c | 25.00 | q30 at 50% period | $1.005 | $0.60 |
| Presidential-actions bucket | 2c, 2c from best | 219.54 | q49 at 50% period | $1.011 | $0.98 |

These are not guaranteed profits. The snapshot calculation initially used full
or fractional program duration, while a current opportunity may appear after
part of that duration has elapsed. A separate remaining-time correction is
required before any row is actionable.

The KXGOLD15M q64 example illustrates the correction. Under a full 50% period,
modeled reward was about $1.013 against a $0.64 maximum loss. But the observed
1c state appeared with only a few minutes remaining, so it could not earn the
assumed half-period score. It is **not** a current production candidate merely
because the full-period arithmetic passes.

---

## Why this is different from the failed crypto research

The failed crypto candidates required one of these to be true:

- settlement probability was misestimated;
- a signal was not already priced at the ask;
- maker fills could be made less endogenous;
- cross-coin information propagated slowly enough to trade.

QSTRF requires none of them. It asks whether a posted order receives an
external subsidy large enough to cover its bounded fill loss. The primary
unknowns are program/accounting and execution facts, not the direction of the
underlying event.

This is therefore a genuinely orthogonal mechanism family.

---

## Binding unknowns

Before a PASS, all of the following must be resolved:

1. **Remaining time:** use only score available from submission until program
   end.
2. **Minimum payout:** establish whether the $1 threshold applies per incentive
   period or to aggregated reward credits.
3. **Actual score credit:** verify a real resting order receives the modeled
   qualifying score.
4. **Queue semantics:** record queue ahead and cancel before the protective
   buffer is consumed.
5. **Fill timing:** an early fill can occur before enough reward has accrued.
6. **Book/frontier stability:** Target Size eligibility can change as other
   orders enter or leave.
7. **Program persistence:** pools and terms are temporary and can change.
8. **Capacity:** scale across independent program/market frontiers, not by
   blindly increasing one order.

---

## Smallest decisive prospective experiment

The first experiment should not attempt to maximize profit.

1. Run a shadow scanner that computes the live Target Size frontier,
   conservative current-to-end reward, queue ahead, quantity required for the
   payout minimum and maximum fill loss.
2. Confirm the payout-minimum interpretation with official support or account
   activity.
3. Select one low-price market where:
   - minimum-payout quantity fits the frontier;
   - current-to-end modeled reward exceeds maximum loss by a wide margin;
   - substantial queue is ahead;
   - worst possible directional loss is no more than a preapproved tiny
     research budget.
4. Log every second of queue/frontier status and the eventual reward credit.
5. Do not scale until credited reward and queue behavior reconcile exactly to
   the model.

A q1 experiment can validate score accounting if rewards aggregate across
periods. If the minimum is per period, use only the mathematically derived
minimum quantity and keep worst possible loss below the research cap.

---

## Present classification

| Finding | Status |
|---|---|
| Cross-market quote-lag alpha | REFUTED |
| Cross-coin dominance pair | REFUTED / no qualifying TEST trades |
| First-observation provenance | audit pending |
| Generic reward-funded commodity policies | historical audit pending |
| Fixed q64 one-cent Gold rule | FAIL historically; zero eligible rows |
| Target-frontier subsidy arithmetic | ESTABLISHED from current public terms/book snapshot |
| Queue-shielded reward farming | STRONG MECHANISM CANDIDATE |
| Executable positive-EV strategy | NOT YET ESTABLISHED |

The breakthrough is the shift from trying to outpredict the market to searching
for situations where the exchange subsidy itself can dominate a deliberately
capped trading loss. It is the strongest remaining path because it can be
validated with cents-to-dollars of bounded downside rather than months of noisy
settlement outcomes.
