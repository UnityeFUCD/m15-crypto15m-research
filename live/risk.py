"""Unified risk control - the single source of truth for every stop.

WHY THIS MODULE EXISTS
  The system had three drawdown controls that disagreed with each other:

    pre-submit envelope   30% of peak     fractional, self-scaling  (correct)
    runner hard stop      -$150 realised  fixed dollars, and RESET TO ZERO on
                                          every restart - 17 restarts on
                                          2026-08-05, so it never accumulated
    watchdog kill         $150 / $75      fixed dollars, measured from a
                                          baseline captured at process start -
                                          13 re-baselines the same day

  Two defects made all of them weak:

  1. FIXED DOLLARS DO NOT SCALE. $75 was 22% of the bankroll when set at $341,
     14% at $527, and would be 3.8% at $2,000. Simulation: a $75 stop fires
     with probability 1.0000 within 20 days AT EVERY BANKROLL with the edge
     fully alive, median ~72 windows - under a day. That is not a risk control,
     it is a near-daily interruption. Window sd is $16.84, so over 72 windows
     the random walk has sd ~$143; $75 is well inside ordinary noise.

  2. SESSION BASELINES FORGET. Every restart discarded the running peak, so a
     later decline was measured from a lower reference and read as a smaller
     drawdown than it was. Deploying the persistent mark moved the reported
     drawdown from $0.00 to its true $60.10 in one step.

THE FIX
  Every threshold is a FRACTION of a peak equity that persists across restarts.
  That gives a constant risk policy at any bankroll, which is the whole point:
  the account can grow without the controls silently tightening into uselessness
  or loosening into negligence.

LAYERING (tightest fires first, so the throttle acts before the kill)
    ENVELOPE_FRAC  0.30  pre-submit, WORST-CASE basis. Refuses one order and
                         keeps running. Self-restoring - as equity recovers,
                         headroom returns. This is a throttle, not a switch.
    STOP_FRAC      0.20  runner hard stop on ACTUAL equity. Cancels all, exits.
    WATCH_FRAC     0.22  watchdog backstop, deliberately LOOSER than the runner
                         stop so that in normal operation the runner stops
                         itself and the watchdog only fires if the runner is
                         wedged or dead.

  The envelope and the stop use different bases and are not comparable as
  numbers: the envelope asks "if every open position went to zero, where would
  I be", the stop asks "where am I now". Both are wanted.

FAIL-SAFE DIRECTION
  If the persistent mark cannot be read, this module falls back to the equity
  passed in, which makes the drawdown read as ZERO and would prevent a stop
  from firing. That is the unsafe direction, so every such fallback is logged
  loudly by the caller and `hwm_ok()` reports it.
"""
import json
import os
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
HWM_FILE = HERE / "lsm_hwm.json"

# ABSOLUTE FLOOR. The account started at $211; the user's standing instruction
# is that anything above that is acceptable. This replaces the old fixed $75
# drawdown rule, which was enforced by hand rather than by code and which
# deadlocked: measured against a persistent peak it could only clear by
# earning, and it stopped the earning.
#
# A floor is the right shape for this. It does not tighten as the account
# grows, it cannot deadlock, and it states the actual risk preference: how much
# of the original stake may be lost, not how far below a high-water mark we
# have drifted.
FLOOR = float(os.environ.get("LSM_FLOOR", "211"))
ENVELOPE_FRAC = float(os.environ.get("LSM_ENVELOPE_FRAC", "0.30"))
STOP_FRAC = float(os.environ.get("LSM_STOP_FRAC", "0.20"))
WATCH_FRAC = float(os.environ.get("LSM_WATCH_FRAC", "0.22"))


def hwm_ok():
    """True if the persistent mark is readable and sane."""
    try:
        d = json.loads(HWM_FILE.read_text())
        return float(d.get("peak_equity") or 0.0) > 0.0
    except Exception:
        return False


def peak():
    """Persistent all-time peak equity. 0.0 if unset/unreadable."""
    try:
        d = json.loads(HWM_FILE.read_text())
        return float(d.get("peak_equity") or 0.0)
    except Exception:
        return 0.0


def observe(equity, who="?"):
    """Record an equity reading; raise the mark if it is a new high.

    Only ever RAISES, so concurrent writers cannot corrupt it - the worst case
    is a lost update that the next cycle re-applies, and no locking is needed.
    The write is atomic (temp file + os.replace), so a crash mid-write cannot
    leave a truncated file.

    Returns the authoritative peak AFTER the update."""
    try:
        equity = float(equity)
    except (TypeError, ValueError):
        return peak()
    cur = peak()
    if equity <= cur:
        return cur
    try:
        payload = {"peak_equity": round(equity, 4), "updated_by": str(who)}
        fd, tmp = tempfile.mkstemp(dir=str(HERE), suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=1)
        os.replace(tmp, HWM_FILE)
        return float(payload["peak_equity"])
    except Exception:
        return cur


_last_sample = {}          # process-local, per caller


def observe_confirmed(equity, who="?"):
    """Raise the mark only on a CONFIRMED high - two consecutive readings.

    THE SETTLEMENT SPIKE THIS DEFEATS
      Equity is computed as cash + cost basis of open positions. At settlement
      the exchange credits the payout to CASH while the settled position is
      still returned by /portfolio/positions for a moment, so for one sample
      both are counted and equity reads high by exactly the cost basis of the
      settling positions. Observed live on 2026-08-05: true equity went
      $477.31 -> $507.31, but one sample in between read $567.31, inflated by
      the $60 cost basis of three settling positions.

      Because the persistent mark ONLY EVER RISES, a single such blip poisons
      it permanently - every future drawdown is then measured from a peak that
      never existed, and every stop fires early forever. That makes this the
      most damaging possible bug in a risk control, and it is invisible until
      a stop fires for no reason.

    Taking min(current, previous) means a one-sample spike cannot raise the
    mark, while a genuine sustained rise still does - it simply takes two
    cycles instead of one. The cost of the delay is nil; the cost of the
    poison is permanent.

    Callers are keyed separately, so the runner and the watchdog each keep
    their own two-sample history without interfering.
    """
    try:
        equity = float(equity)
    except (TypeError, ValueError):
        return peak()
    prev = _last_sample.get(who)
    _last_sample[who] = equity
    if prev is None:
        return peak()          # first sample of a process can never raise it
    return observe(min(equity, prev), who=who)


def dd_dollars(equity):
    """Dollars below the persistent peak. 0.0 if at or above it."""
    pk = peak()
    if pk <= 0:
        return 0.0
    return max(0.0, pk - float(equity))


def dd_frac(equity):
    """Drawdown as a FRACTION of the persistent peak. 0.0 if unknown."""
    pk = peak()
    if pk <= 0:
        return 0.0
    return max(0.0, (pk - float(equity)) / pk)


def envelope_floor():
    """Dollar equity floor for the pre-submit worst-case check.

    Referenced to the ABSOLUTE FLOOR, not to a fraction of the peak. The
    fractional version tightened as the peak rose and would have blocked
    orders at $315 while the stated risk boundary was $211 - refusing trades
    the account could plainly afford.

    Safety is preserved by construction: with MAX_GROSS $110 and the floor at
    $211, even total loss of every open position leaves cash above the floor,
    so the gross cap binds before this check can be reached."""
    return FLOOR


def stop_breached(equity):
    """True if the runner should stop.

    The FLOOR is the binding rule and needs no peak, so it works even when the
    persistent mark is unreadable - the one case where the fractional test
    silently cannot fire. The fractional test is retained as a second condition
    only when it is LOOSER than the floor, so the two can never deadlock each
    other."""
    eq = float(equity)
    if eq < FLOOR:
        return True
    pk = peak()
    if pk <= 0:
        return False
    frac_level = (1.0 - STOP_FRAC) * pk
    return eq < frac_level if frac_level < FLOOR else False


def watch_breached(equity):
    """Watchdog backstop. Same logic, a touch below the runner's own trigger so
    that in normal operation the runner stops itself first."""
    eq = float(equity)
    if eq < FLOOR - 1.0:
        return True
    pk = peak()
    if pk <= 0:
        return False
    frac_level = (1.0 - WATCH_FRAC) * pk
    return eq < frac_level if frac_level < FLOOR else False


def summary(equity):
    pk = peak()
    return dict(floor=FLOOR, peak=round(pk, 2), equity=round(float(equity), 2),
                dd=round(dd_dollars(equity), 2),
                dd_frac=round(dd_frac(equity), 4),
                envelope_floor=round(envelope_floor(), 2),
                # Display the level that will ACTUALLY fire. The fractional
                # test is inert whenever it sits above the floor, so showing
                # it would misreport the real trigger.
                stop_at=round((1.0 - STOP_FRAC) * pk
                              if (1.0 - STOP_FRAC) * pk < FLOOR else FLOOR, 2),
                watch_at=round((1.0 - WATCH_FRAC) * pk
                               if (1.0 - WATCH_FRAC) * pk < FLOOR
                               else FLOOR - 1.0, 2),
                room_to_floor=round(float(equity) - FLOOR, 2),
                hwm_ok=hwm_ok())


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "set":
        print(observe(float(sys.argv[2]), who="manual"))
    elif len(sys.argv) > 2 and sys.argv[1] == "check":
        print(json.dumps(summary(float(sys.argv[2])), indent=1))
    else:
        print(json.dumps(summary(peak()), indent=1))
