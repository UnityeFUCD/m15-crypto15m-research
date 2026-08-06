"""Probe-Then-Commit order-lifecycle state machine.

DISABLED BY DEFAULT. PtcConfig.enabled and PtcConfig.live_enabled are both
False unless set explicitly, and every order-building entry point returns None
while disabled or killed. Shadow and test modes need no credentials and never
construct an API client.

WHAT PTC IS
  1  post exactly one maker contract as a diagnostic probe
  2  a fill before cancellation confirmation is a WARNING - hold that one
     contract and never add size
  3  if the probe is confirmed unfilled, cancel and refresh the book
  4  commit at most one bounded IOC per close window
  5  never retry, never chase

It REPLACES full-size maker execution for the same eligible opportunity; it is
not an extra layer on top of it.

DESIGN RULES THAT ARE NOT OBVIOUS

  Two clocks, never merged. The cancel timer starts at the exchange's
  authoritative created_time, never at a local pre-request timestamp. A local
  clock that is ahead makes the probe rest too briefly; one that is behind
  leaves it exposed past the horizon.

  Fail closed. commit_allowed() requires an affirmative CANCEL_CONFIRMED with
  filled_qty == 0. A cancel request that was merely SENT is not permission -
  a fill can land inside the race, and that fill means we are already long one
  contract and must not add more.

  No optimistic state. Only an authoritative exchange acknowledgement advances
  the machine. An API error moves to ERROR and flags the record for
  reconciliation; it never assumes the order did or did not rest.

  Idempotent identity. client_order_id is a pure function of (opportunity,
  phase), so a process that dies between POST and local persistence adopts its
  own order on restart instead of submitting a second one.

  Everything counts. Skips, risk blocks, errors, no-fills and partial fills all
  remain in the denominator. A denominator that silently drops its failures is
  how this project previously turned -$134 of real losses into a +$143 report.
"""
from __future__ import annotations

import hashlib
import hmac
import math
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Any

MICROS = 1_000_000
FEE_RATE = 0.07


class TimestampUnitError(ValueError):
    """Raised when an epoch cannot be interpreted as s/ms/us/ns."""


class S(IntEnum):
    """Lifecycle states. Ordered so restart tests can compare progress."""
    ELIGIBLE = 0
    TREATMENT_ASSIGNED = 1
    PROBE_SUBMITTING = 2
    PROBE_ACKNOWLEDGED = 3
    PROBE_RESTING = 4
    PROBE_FILLED = 5
    CANCEL_REQUESTED = 6
    CANCEL_CONFIRMED = 7
    CANCEL_RACE_RECONCILED = 8
    COMMIT_ELIGIBLE = 9
    IOC_SUBMITTING = 10
    IOC_COMPLETE = 11
    SETTLED = 12
    BLOCKED = 90
    ERROR = 91
    KILLED = 99


# --------------------------------------------------------------- timestamps
def detect_epoch_unit(value: float) -> str:
    a = abs(float(value))
    if a < 1e11:
        return "s"
    if a < 1e14:
        return "ms"
    if a < 1e17:
        return "us"
    if a < 1e20:
        return "ns"
    raise TimestampUnitError(f"epoch out of range for s/ms/us/ns: {value!r}")


def to_epoch_seconds(value: float) -> float:
    """Normalise any epoch to SECONDS, refusing to guess when out of range.

    PTC v1 divided microsecond integers by 1e9 as though they were nanoseconds
    and turned a 60-second wait into 1.8 billion seconds. Silent rescaling is
    the failure mode, so an impossible magnitude raises instead.
    """
    unit = detect_epoch_unit(value)
    return float(value) / {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}[unit]


# ------------------------------------------------------------ side & quotes
def held_side(action: str, side: str) -> str:
    """The side actually HELD, from the orders table's (action, side).

    action=sell on side=yes is a NO position. Deriving this wrongly agrees
    with the fills table only 47.65% of the time.
    """
    if action not in ("buy", "sell"):
        raise ValueError(f"action must be buy/sell, got {action!r}")
    if side not in ("yes", "no"):
        raise ValueError(f"side must be yes/no, got {side!r}")
    if action == "buy":
        return side
    return "no" if side == "yes" else "yes"


def held_quote_micros(side: str, yes_bid: int, yes_ask: int) -> tuple[int, int]:
    if side == "yes":
        return int(yes_bid), int(yes_ask)
    if side == "no":
        return MICROS - int(yes_ask), MICROS - int(yes_bid)
    raise ValueError(f"side must be yes/no, got {side!r}")


def ioc_fee_micros(price_micros: int, qty: int) -> int:
    """Kalshi taker fee in micros: ceil to 4dp of 0.07 * C * p * (1-p)."""
    if qty <= 0:
        return 0
    _require_int_micros(price_micros, "price_micros")
    p = price_micros / MICROS
    dollars = math.ceil(FEE_RATE * qty * p * (1 - p) * 10_000 - 1e-12) / 10_000
    return int(round(dollars * MICROS))


def _require_int_micros(v: Any, name: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError(f"{name} must be integer micros, got {type(v).__name__}"
                        f" ({v!r}) - binary float money is not permitted")
    return v


# ------------------------------------------------------------------ config
@dataclass(frozen=True)
class RiskLimits:
    """Production risk envelope. Part 9 values, not tuned on any dataset."""
    window_risk_frac: float = 0.06      # worst-case loss in one close window
    total_risk_frac: float = 0.12       # worst-case across all open exposure
    max_commit_qty: int = 20
    max_commits_per_window: int = 1
    static_floor_micros: int = 211 * MICROS
    hwm_floor_frac: float = 0.80        # kill at 20% off the strategy HWM
    throttle_day_loss: float = 0.06
    halt_day_loss: float = 0.10
    halt_giveback: float = 0.08
    max_scale_step: float = 1.20        # never raise target qty >20% at once


FROZEN = RiskLimits()


@dataclass(frozen=True)
class PtcConfig:
    enabled: bool = False
    live_enabled: bool = False
    wait_seconds: float = 60.0
    commit_qty: int = 5
    equity_micros: int = 0
    ask_floor_micros: int = 600_000
    ask_ceil_micros: int = 800_000
    kill: bool = False
    limits: RiskLimits = field(default_factory=RiskLimits)
    hmac_secret: bytes = b"ptc-frozen-v1"


def commit_quantity(equity_micros: int, price_micros: int,
                    limits: RiskLimits = FROZEN,
                    floor_micros: int | None = None) -> int:
    """Contracts for the IOC, bound by the tightest constraint.

    Never hard-codes a size: derived from equity, price and the risk budget.
    Floors rather than rounds, so no budget is breached by rounding.
    """
    _require_int_micros(equity_micros, "equity_micros")
    _require_int_micros(price_micros, "price_micros")
    if price_micros <= 0 or price_micros >= MICROS:
        raise ValueError("price must be in (0, 1) dollars")
    floor = limits.static_floor_micros if floor_micros is None else floor_micros
    room = equity_micros - floor
    if room <= 0:
        return 0
    caps = (
        float(limits.max_commit_qty),
        limits.window_risk_frac * equity_micros / price_micros,
        limits.total_risk_frac * equity_micros / price_micros,
        room / price_micros,
    )
    return max(0, int(math.floor(min(caps))))


# ------------------------------------------------------------------ record
@dataclass
class Record:
    opportunity_id: str
    ticker: str
    close_ts: int
    coin: str
    side: str
    bid_micros: int
    ask_micros: int
    state: S = S.ELIGIBLE
    probe_order_id: str | None = None
    probe_ack_ts: float | None = None
    probe_submitted: int = 0
    probe_filled: int = 0
    fill_ids: tuple = ()
    cancel_requested_ts: float | None = None
    cancel_confirmed_ts: float | None = None
    cancel_confirmed_filled: int | None = None
    committed: bool = False
    ioc_filled: int = 0
    ioc_price_micros: int | None = None
    cash_out_micros: int = 0
    needs_reconciliation: bool = False
    block_reason: str | None = None


class PtcMachine:
    """In-memory lifecycle manager. Persistence is the caller's concern; use
    snapshot()/restore() with the append-only store."""

    def __init__(self, cfg: PtcConfig):
        self.cfg = cfg
        self._r: dict[str, Record] = {}
        self._committed_windows: set[int] = set()
        self.killed: bool = bool(cfg.kill)
        self.kill_reason: str | None = "config" if cfg.kill else None
        self._startup_dirty: bool = False

    # ---------------------------------------------------------- registration
    def register(self, opportunity_id: str, ticker: str, close_ts: int,
                 coin: str, side: str, bid_micros: int,
                 ask_micros: int) -> str:
        _require_int_micros(bid_micros, "bid_micros")
        _require_int_micros(ask_micros, "ask_micros")
        if opportunity_id not in self._r:
            self._r[opportunity_id] = Record(
                opportunity_id=opportunity_id, ticker=ticker,
                close_ts=int(close_ts), coin=coin, side=side,
                bid_micros=bid_micros, ask_micros=ask_micros,
                state=S.TREATMENT_ASSIGNED)
        return opportunity_id

    def record(self, oid: str) -> Record:
        return self._r[oid]

    def state(self, oid: str) -> S:
        if self.killed:
            return S.KILLED
        return self._r[oid].state

    def denominator(self) -> int:
        """Every registered opportunity, including skips, blocks and errors."""
        return len(self._r)

    # --------------------------------------------------------------- identity
    def client_order_id(self, oid: str, phase: str) -> str:
        if phase not in ("probe", "commit"):
            raise ValueError("phase must be probe/commit")
        digest = hmac.new(self.cfg.hmac_secret,
                          f"{oid}|{phase}".encode(), hashlib.sha256
                          ).hexdigest()[:24]
        return f"ptc-{phase}-{digest}"

    # ------------------------------------------------------------- integrity
    def report_integrity_failure(self, reason: str) -> None:
        self.killed = True
        self.kill_reason = reason

    def _blocked(self) -> bool:
        return self.killed or not self.cfg.enabled

    # ----------------------------------------------------------------- probe
    def build_probe(self, oid: str) -> dict | None:
        if self._blocked():
            return None
        r = self._r[oid]
        if r.state not in (S.TREATMENT_ASSIGNED, S.ELIGIBLE):
            return None
        return {"client_order_id": self.client_order_id(oid, "probe"),
                "ticker": r.ticker, "action": "buy", "side": r.side,
                "count": 1, "type": "limit", "post_only": True,
                "price_micros": r.bid_micros}

    def mark_submitting(self, oid: str) -> None:
        if self._blocked():
            return
        self._r[oid].state = S.PROBE_SUBMITTING

    def on_probe_ack(self, oid: str, order_id: str,
                     exchange_created_ts: float) -> None:
        r = self._r[oid]
        if r.probe_order_id == order_id and r.probe_submitted:
            return                                   # duplicate delivery
        r.probe_order_id = order_id
        r.probe_ack_ts = to_epoch_seconds(exchange_created_ts)
        r.probe_submitted = 1
        r.state = S.PROBE_RESTING
        r.needs_reconciliation = False

    def adopt_from_exchange(self, oid: str, order_id: str,
                            exchange_created_ts: float,
                            filled_qty: int) -> None:
        """Startup reconciliation: claim an order we submitted but never
        persisted, instead of submitting a second one."""
        self.on_probe_ack(oid, order_id, exchange_created_ts)
        if filled_qty > 0:
            self._r[oid].probe_filled = int(filled_qty)
            self._r[oid].state = S.PROBE_FILLED

    def should_cancel(self, oid: str, now: float) -> bool:
        r = self._r[oid]
        if r.state is not S.PROBE_RESTING or r.probe_ack_ts is None:
            return False
        return to_epoch_seconds(now) >= r.probe_ack_ts + self.cfg.wait_seconds

    def on_fill(self, oid: str, qty: int, price_micros: int, ts: float,
                fill_id: str | None = None) -> None:
        r = self._r[oid]
        if fill_id is not None:
            if fill_id in r.fill_ids:
                return                               # out-of-order duplicate
            r.fill_ids = r.fill_ids + (fill_id,)
        r.probe_filled += int(qty)
        r.cash_out_micros += int(qty) * _require_int_micros(price_micros,
                                                           "price_micros")
        if r.state in (S.CANCEL_REQUESTED, S.CANCEL_CONFIRMED):
            r.state = S.CANCEL_RACE_RECONCILED
        else:
            r.state = S.PROBE_FILLED

    # ---------------------------------------------------------------- cancel
    def on_cancel_requested(self, oid: str, ts: float) -> None:
        r = self._r[oid]
        r.cancel_requested_ts = to_epoch_seconds(ts)
        if r.state is S.PROBE_RESTING:
            r.state = S.CANCEL_REQUESTED

    def on_cancel_confirmed(self, oid: str, ts: float, filled_qty: int) -> None:
        r = self._r[oid]
        r.cancel_confirmed_ts = to_epoch_seconds(ts)
        r.cancel_confirmed_filled = int(filled_qty)
        if filled_qty or r.probe_filled or r.state is S.CANCEL_RACE_RECONCILED:
            r.state = S.CANCEL_RACE_RECONCILED
        else:
            r.state = S.COMMIT_ELIGIBLE

    def on_cancel_rejected(self, oid: str, ts: float, reason: str) -> None:
        r = self._r[oid]
        r.state = S.ERROR
        r.block_reason = reason
        r.needs_reconciliation = True

    def on_api_error(self, oid: str, reason: str) -> None:
        r = self._r[oid]
        r.state = S.ERROR
        r.block_reason = reason
        r.needs_reconciliation = True

    def mark_blocked(self, oid: str, reason: str) -> None:
        r = self._r[oid]
        r.state = S.BLOCKED
        r.block_reason = reason

    # ---------------------------------------------------------------- commit
    def refresh_book(self, oid: str, ask_micros: int) -> None:
        _require_int_micros(ask_micros, "ask_micros")
        self._r[oid].ask_micros = ask_micros

    def commit_allowed(self, oid: str) -> bool:
        if self._blocked():
            return False
        r = self._r[oid]
        if r.state is not S.COMMIT_ELIGIBLE or r.committed:
            return False
        if r.probe_filled or (r.cancel_confirmed_filled or 0) > 0:
            return False
        if r.cancel_confirmed_ts is None:
            return False                              # fail closed
        if r.close_ts in self._committed_windows:
            return False
        return (self.cfg.ask_floor_micros <= r.ask_micros
                <= self.cfg.ask_ceil_micros)

    def try_commit(self, oid: str) -> dict | None:
        if not self.commit_allowed(oid):
            return None
        r = self._r[oid]
        qty = self.cfg.commit_qty
        if qty <= 0:
            return None
        r.committed = True
        r.state = S.IOC_SUBMITTING
        self._committed_windows.add(r.close_ts)
        return {"client_order_id": self.client_order_id(oid, "commit"),
                "ticker": r.ticker, "action": "buy", "side": r.side,
                "count": qty, "type": "market", "time_in_force": "ioc",
                "price_micros": r.ask_micros}

    def on_ioc_result(self, oid: str, filled_qty: int,
                      price_micros: int) -> None:
        r = self._r[oid]
        _require_int_micros(price_micros, "price_micros")
        r.ioc_filled = int(filled_qty)
        r.ioc_price_micros = price_micros
        if filled_qty > 0:
            r.cash_out_micros += (int(filled_qty) * price_micros
                                  + ioc_fee_micros(price_micros,
                                                   int(filled_qty)))
        r.state = S.IOC_COMPLETE

    # -------------------------------------------------------------- restart
    def needs_startup_reconciliation(self) -> bool:
        return any(r.state in (S.PROBE_SUBMITTING, S.IOC_SUBMITTING,
                               S.ERROR, S.CANCEL_REQUESTED)
                   for r in self._r.values())

    def snapshot(self) -> dict:
        return {"records": {k: vars(v).copy() for k, v in self._r.items()},
                "committed_windows": sorted(self._committed_windows),
                "killed": self.killed, "kill_reason": self.kill_reason}

    def restore(self, snap: dict) -> None:
        self._r = {}
        for k, v in snap["records"].items():
            d = dict(v)
            d["state"] = S(d["state"])
            d["fill_ids"] = tuple(d.get("fill_ids") or ())
            self._r[k] = Record(**d)
        self._committed_windows = set(snap.get("committed_windows") or [])
        self.killed = bool(snap.get("killed"))
        self.kill_reason = snap.get("kill_reason")
