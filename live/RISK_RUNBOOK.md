# Risk System — Runbook

Unified 2026-08-05. Every threshold is a **fraction of a peak equity that
survives restarts**. Nothing is denominated in fixed dollars any more.

## Why it was rebuilt

Three controls existed and all three were weak for the same two reasons.

**Fixed dollars are not a constant risk policy.** $75 was 22% of the bankroll
when set at $341, 14% at $527, and would be 3.8% at $2,000. Simulated, a $75
stop fires with probability **1.0000 within 20 days at every bankroll** with the
edge fully alive — median ~72 windows, under a day. Window sd is $16.84, so over
72 windows the random walk has sd ≈ $143; $75 sits well inside ordinary noise.
It was not a risk control, it was a near-daily interruption.

**Session baselines forget.** Every control measured from equity captured at
process start. The runner restarted **17 times** and the watchdog re-baselined
**13 times** on 2026-08-05, so none ever accumulated a real measurement window,
and a decline after a restart read as smaller than it was. Deploying the
persistent mark moved the reported drawdown from **$0.00 to its true $60.10** in
one step.

## The layers

| layer | threshold | basis | action |
|---|---|---|---|
| **Envelope** | 30% of peak | worst case: `cash − proposed_cost` | Refuses **one order**, keeps running. Self-restoring |
| **Runner stop** | 20% of peak | actual equity | Cancels all, exits process |
| **Watchdog** | 22% of peak | actual equity | Writes `LSM_KILL` |
| Absolute ceiling | `WATCH_MAX_DD` $150 | dollars | Backstop for a single catastrophic move |

The watchdog is deliberately **looser** than the runner stop so that in normal
operation the runner stops itself and the watchdog only fires when the runner is
wedged or dead.

Envelope and stop are **not comparable as numbers** — the envelope asks "if
every open position went to zero, where would I be", the stop asks "where am I
now". Both are wanted.

At peak **$526.61**: envelope floor $368.63 · runner stops below **$421.29** ·
watchdog halts below **$410.76**.

## Fail-safe direction

If `lsm_hwm.json` is missing or corrupt, `peak()` returns 0 and **no stop can
fire** — the unsafe direction. Both processes therefore check `risk.hwm_ok()`
every cycle and re-seed from current equity, logging `hwm_reseeded` /
`hwm_unreadable`. The system self-heals within one cycle, losing only history.

The mark **only ever rises**, so concurrent writers cannot corrupt it — the
worst case is a lost update the next cycle re-applies. Writes are atomic
(temp file + `os.replace`).

## When to resume after a halt

**Rule: resume immediately, unless the sequential test says the edge is dead.**

Run `python resume_check.py`. It reports halt state, risk state, and the
edge-evidence verdict.

The justification is a measured fact: **window P&L is serially independent** —
autocorrelation null at every lag (permutation p = 0.87, 0.28, 0.095, 0.68,
0.77). A drawdown carries no information about the next window, so halts split
into two kinds needing opposite responses:

| policy (20% halt, 20 days) | edge ALIVE | edge DEAD |
|---|---|---|
| resume immediately | **$12,213** | **−$3,146** |
| **resume unless evidence says dead** | **$11,924** | **+$441** |
| wait 1 day | $11,447 | +$410 |
| wait 3 days | $10,613 | +$447 |
| never resume | $9,598 | +$448 |

Evidence-gating captures **97.6%** of the immediate-resume upside while avoiding
the −$3,146 branch entirely. Time-based waiting is worse than both — it loses
when alive *and* leaks when dead. Neither elapsed time nor equity recovery
carries information about which case you are in; only accumulated evidence does.

## Files

| file | role |
|---|---|
| `risk.py` | single source of truth: fractions, persistent peak, all checks |
| `lsm_hwm.json` | the persistent peak. Only ever rises |
| `resume_check.py` | the restart decision, reports only |
| `edge_monitor.py` | sequential edge-health test (LLR), reports only |
| `lsm_config.json` | published by the runner so a watchdog restart reproduces the exact config |

## Operations

```bash
python risk.py                    # current thresholds and headroom
python risk.py check 450          # what would happen at equity 450
python resume_check.py            # should we restart?
python edge_monitor.py            # is the edge still there?
rm LSM_KILL                       # resume (after resume_check says so)
```

**Never** hand-edit `lsm_hwm.json` downward. Lowering the mark loosens every
control at once. To reset it deliberately after a withdrawal, delete the file
and let the next cycle re-seed from live equity.

The one documented exception is removing a **phantom peak** — see below.

## The settlement spike, and why it is the most dangerous bug here

Equity = cash + **cost basis** of open positions. At settlement the exchange
credits the payout to cash while the settled position is still returned by
`/portfolio/positions` for a moment, so one sample counts both and equity reads
high by exactly the cost basis of the settling positions.

Observed live on 2026-08-05:

```
09:15:11  cash 417.31 + deployed 60.00 = 477.31   true
09:15:1x  cash 507.31 + deployed 60.00 = 567.31   PHANTOM  (+$60 = the cost basis)
09:15:16  cash 507.31 + deployed  0.00 = 507.31   true
```

Because the mark **only ever rises and persists**, a single blip poisons it
permanently: every later drawdown is measured from a peak that never existed and
every stop fires early, forever. It is invisible until a stop fires for no
reason.

Two phantoms were found and removed on 2026-08-05 — a $567.31 written by the
runner, and a $526.61 that appeared for exactly one sample between $436.61 and
$462.41 (a $90 spike = 3 positions × 30 × $1.00). The true peak was **$524.71**,
sustained across seven consecutive samples.

**Defence:** `risk.observe_confirmed()` raises the mark only on two consecutive
readings, so a one-sample spike cannot raise it while a genuine rise still does,
one cycle later. Every caller must use `observe_confirmed()`, never `observe()`.

**To identify a phantom:** look at the neighbouring samples. A genuine high is
sustained; a phantom appears once and drops back by exactly the cost basis of
whatever just settled.

## What is deliberately NOT automated

The sequential edge test does **not** halt trading. Tested against the static
drawdown rule across three regimes, no calibration dominated it: static costs
$156 when the edge is alive versus the SPRT's $240–297, with a lower false-stop
rate (1.99% vs 2.91–3.66%), and saves essentially the same when the edge is dead
($2,451 vs $2,378–2,488). Adding a second automated kill path that does not
dominate would add risk, not remove it.

Its one genuine advantage is **speed** — it identifies a truly dead edge in ~30
windows against the drawdown rule's ~43, roughly three hours earlier. That is
worth having as information with a human deciding, which costs nothing and
cannot false-stop the strategy.
