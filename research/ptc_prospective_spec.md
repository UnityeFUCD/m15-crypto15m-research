# PTC randomized prospective experiment — frozen specification

Frozen before any data is collected. Machine-readable form:
`config/ptc_frozen.json`. Nothing in this document authorizes live orders; the
account is below its kill floor and the historical verdict is FAIL
(`research/results/PTC_FINAL.md`).

## Why a trial is required at all

Historical replay cannot settle this. Changing from a q15 maker order to a q1
probe changes what the counterparty sees and therefore changes who fills us.
Fills are endogenous: the thing being measured is altered by the intervention.
The replay in `research/ptc_v3.py` is a counterfactual on 303 orders where
that feedback is assumed away.

## Assignment

- deterministic HMAC over the **close-window identifier**, mod 4
- persisted **before** any order is sent
- identical after restart — assignment is a pure function of the window, never
  of state
- no interim parameter tuning
- no early stopping on a favourable interim result

Randomizing per close window rather than per day means all arms share the same
market regime. Per-day assignment would confound arm with regime, which is the
error that produced three wrong verdicts earlier in this project.

## Arms

| arm | probe | cancel | commit |
|---|---|---|---|
| CONTROL | — | — | full-size maker at the validation quantity |
| PROBE_ONLY_60 | q1 | 60s | never |
| PTC_60 | q1 | 60s | one bounded IOC |
| PTC_120 | q1 | 120s | one bounded IOC |

60s and 120s are **both** retained. The two-day history must not choose
between them: `PTC_60 − PROBE_ONLY` has P(≤0) 0.434–0.443 and `PTC_120 −
PROBE_ONLY` has P(≤0) 0.234–0.252 across five seeds. Neither is resolved, and
selecting the better one on this sample is exactly how the retracted HCR PASS
was produced.

## Quantity

Start at **q5**. This is not a round number — it is the only size that clears
every stress scenario in `research/ptc_risk_sim.py`:

| qty | fails |
|---|---|
| q5 | — |
| q10 | adverse_blocks |
| q15 | adverse_blocks |
| q20 | adverse_blocks, zero_edge |

CONTROL uses the same q5, so the comparison is size-matched. q10/q15 capacity
is tested only after q5 execution mechanics pass.

## Endpoints

**Primary:** realized P&L per **assigned opportunity**, day/window-clustered.

Per-assigned is the only honest denominator. Per-filled-contract divides by a
quantity the strategy influences — a policy that fills only its best orders
looks excellent per fill and can still lose money.

**Secondary:** P&L per submitted contract; PTC − probe-only; PTC − control;
no-fill-branch win probability; actual IOC price and depth; IOC fill fraction;
drawdown; worst close-window loss.

## Required sample size

Recomputed from the completed 303-order dataset at q5, paired on the close
window:

| comparison | observed $/window | SD | windows/arm | days at 24/arm/day |
|---|---|---|---|---|
| PTC_120 − PROBE_ONLY | +0.0664 | 1.130 | **2,272** | **95** |
| PTC_60 − PROBE_ONLY | +0.0183 | 2.049 | 98,088 | 4,087 |
| PTC_60 − CONTROL | −0.0007 | 12.13 | 2.1 billion | — |

Read this honestly. **PTC_120 vs probe-only is reachable in about three
months. PTC_60 is not reachable at all**, and PTC vs control is unreachable
because the observed difference is essentially zero against enormous
dispersion — CONTROL's SD is $13.10 per window against PTC_120's $1.49.

That last row is itself informative: PTC's main measurable effect is a **9×
reduction in per-window dispersion**, not an increase in mean.

## Operational minimums (regardless of the calculation)

- ≥ 300 assigned close windows per active arm
- ≥ 15 separate calendar days
- ≥ 1 adverse directional episode
- no single coin or week carrying the result

Where the powered target exceeds these, the powered target governs.

## Stopping and analysis

Fixed horizon. No peeking. Analysis is pre-specified: day-clustered and
moving-block (30/60/90 min) bootstrap across five fixed seeds, reporting the
range rather than the best seed; leave-one-day-out, leave-one-coin-out and
leave-one-window-cluster-out for concentration.

A result is reported as positive only if the **absolute lower bound** on P&L
per assigned opportunity exceeds zero. Beating a losing alternative is not
evidence of profit.

## Preconditions before a single live order

1. account funded above the **$211** static floor (currently $136.27)
2. `PTC_ENABLED=1` and `PTC_LIVE_ENABLED=1` set deliberately
3. startup reconciliation clean; no unmanaged orders or positions
4. singleton process lock held
5. side-semantics and timestamp integrity checks passing
6. `research/ptc_reconcile.py` exiting zero
