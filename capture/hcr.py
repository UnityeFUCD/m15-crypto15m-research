"""HCR production stack: signal, sizing, portfolio caps, circuit breakers.

Pure functions, no I/O, no credentials. Every rule here is covered by
tests/test_hcr_stack.py, which was written before this module.

DESIGN NOTES WHERE A NAIVE IMPLEMENTATION WOULD BE WRONG

  common_return   requires ALL six coins. A mean over five is a different
                  statistic and would silently change the threshold's meaning.

  calm_mean       requires 24 CONTIGUOUS windows. Twenty-four rows spanning a
                  gap is not a 6-hour background estimate.

  dynamic_qty     uses E = min(current, day-start) so an intraday run-up cannot
                  raise size, and floors (never rounds) so a constraint is
                  never breached by rounding.

  breaker_state   giveback is (day_high - equity) against a percentage of
                  DAY-START equity, per spec - not a percentage of the high.
                  The kill floor is max(static, 75% of strategy HWM), so it
                  RISES as the account grows and can exceed $211.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

N_COINS = 6
CALM_WINDOWS = 24
WINDOW_SECONDS = 900

OPPOSE_THRESHOLD = -0.0015      # d_favourite * r_common must be <= this
CALM_MAX = 0.0030               # mean_24(|r_common|) must be <= this

WINDOW_RISK_FRAC = 0.08         # <=8% of equity across one close window
TOTAL_RISK_FRAC = 0.18          # <=18% across all overlapping windows
MAX_PER_WINDOW = 2
MAX_TOTAL_OPEN = 4
BUFFER_FRAC = 0.05              # of launch equity
STATIC_FLOOR = 211.0
HWM_FLOOR_FRAC = 0.75

THROTTLE_DAY_LOSS = 0.08
THROTTLE_GIVEBACK = 0.10
HALT_DAY_LOSS = 0.15
HALT_GIVEBACK = 0.12
MAX_API_FAILURES = 3


# ------------------------------------------------------------------ signal
def common_return(returns: Sequence[float | None]) -> float:
    """Simple mean of the six completed A0-to-A0 returns.

    Raises if any coin is missing or non-finite: the spec requires all six
    inputs, and a five-coin mean would silently redefine the threshold.
    """
    if returns is None or len(returns) != N_COINS:
        raise ValueError(f"need exactly {N_COINS} coin returns, "
                         f"got {0 if returns is None else len(returns)}")
    out = []
    for r in returns:
        if r is None:
            raise ValueError("missing coin return; skip the window")
        r = float(r)
        if not math.isfinite(r):
            raise ValueError("non-finite coin return; skip the window")
        out.append(r)
    return sum(out) / N_COINS


def calm_mean(history: Sequence[tuple[int, float]]) -> float:
    """mean_24(|r_common|) over the 24 most recent CONTIGUOUS windows.

    `history` is [(close_ts_seconds, r_common), ...] oldest-first. Adjacent
    entries must be exactly 900 seconds apart.
    """
    if history is None or len(history) < CALM_WINDOWS:
        raise ValueError(f"need {CALM_WINDOWS} windows, "
                         f"got {0 if history is None else len(history)}")
    recent = list(history)[-CALM_WINDOWS:]
    for (t0, _), (t1, _) in zip(recent, recent[1:]):
        if int(t1) - int(t0) != WINDOW_SECONDS:
            raise ValueError("calm history is not contiguous at 900s")
    vals = []
    for _, r in recent:
        r = float(r)
        if not math.isfinite(r):
            raise ValueError("non-finite r_common in calm history")
        vals.append(abs(r))
    return sum(vals) / len(vals)


def hcr_qualifies(favourite_side: str, r_common: float, calm: float,
                  close_minute: int | None = None) -> bool:
    """True when the favourite OPPOSES a completed common move, in a calm tape.

    `close_minute` is accepted and deliberately IGNORED: the independent test
    could not establish that minute :00 is superior, and filtering on it
    discards roughly half the opportunities.
    """
    if favourite_side not in ("yes", "no"):
        raise ValueError("favourite_side must be 'yes' or 'no'")
    if not (math.isfinite(r_common) and math.isfinite(calm)):
        raise ValueError("non-finite signal inputs")
    d = 1.0 if favourite_side == "yes" else -1.0
    return (d * r_common <= OPPOSE_THRESHOLD) and (calm <= CALM_MAX)


# ------------------------------------------------------------------ sizing
def dynamic_qty(equity: float, price: float, floor: float,
                launch_equity: float, q_max: int = 20,
                day_start_equity: float | None = None) -> int:
    """Contracts per position, bound by the tightest of four constraints.

        q = floor(min(q_max,
                      WINDOW_RISK_FRAC*E / (MAX_PER_WINDOW*p),
                      TOTAL_RISK_FRAC*E  / (MAX_TOTAL_OPEN*p),
                      (E - floor - buffer) / (MAX_TOTAL_OPEN*p)))

    E is min(current, day-start) so an intraday gain cannot raise size.
    Returns 0 when there is no room; the caller must not trade on 0.
    """
    if price <= 0 or price >= 1:
        raise ValueError("price must be in (0,1)")
    if q_max <= 0:
        return 0
    E = equity if day_start_equity is None else min(equity, day_start_equity)
    if E <= 0:
        return 0
    buffer = BUFFER_FRAC * launch_equity
    room = E - floor - buffer
    if room <= 0:
        return 0
    caps = (float(q_max),
            WINDOW_RISK_FRAC * E / (MAX_PER_WINDOW * price),
            TOTAL_RISK_FRAC * E / (MAX_TOTAL_OPEN * price),
            room / (MAX_TOTAL_OPEN * price))
    return max(0, int(math.floor(min(caps))))


# --------------------------------------------------------------- portfolio
def portfolio_admits(in_window: int, total_open: int,
                     throttled: bool = False) -> bool:
    """Cap 2 per close window and 4 open overall; 1 per window when throttled."""
    per_window = 1 if throttled else MAX_PER_WINDOW
    return in_window < per_window and total_open < MAX_TOTAL_OPEN


# ---------------------------------------------------------------- breakers
@dataclass
class State:
    equity: float
    day_start: float
    day_high: float
    strategy_hwm: float
    launch_equity: float
    consecutive_api_failures: int = 0
    side_mapping_ok: bool = True
    unmanaged_orders: int = 0
    unmanaged_positions: int = 0
    duplicate_runner: bool = False
    ledger_reconciles: bool = True


def kill_floor(strategy_hwm: float) -> float:
    """max(static floor, 75% of strategy-era high-water equity).

    This RISES as the account grows: at a $1,000 HWM the floor is $750, not
    $211. Inheriting an obsolete peak from an earlier strategy version would
    set this wrongly, so the HWM must be reset at each funded launch.
    """
    return max(STATIC_FLOOR, HWM_FLOOR_FRAC * float(strategy_hwm))


def breaker_state(s: State) -> str:
    """One of NORMAL, THROTTLED, HALTED, KILL. KILL dominates."""
    if (not s.side_mapping_ok or s.duplicate_runner
            or s.unmanaged_orders > 0 or s.unmanaged_positions > 0
            or not s.ledger_reconciles
            or s.equity < kill_floor(s.strategy_hwm)):
        return "KILL"

    day_loss = s.day_start - s.equity
    giveback = s.day_high - s.equity

    if (s.consecutive_api_failures >= MAX_API_FAILURES
            or day_loss >= HALT_DAY_LOSS * s.day_start
            or giveback >= HALT_GIVEBACK * s.day_start):
        return "HALTED"

    if (day_loss >= THROTTLE_DAY_LOSS * s.day_start
            or giveback >= THROTTLE_GIVEBACK * s.day_start):
        return "THROTTLED"

    return "NORMAL"
