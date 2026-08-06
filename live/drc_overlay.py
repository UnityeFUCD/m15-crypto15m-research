"""DRC-15 signal and safe overlay primitives.

This module is intentionally independent from credentials and order submission.
It provides the causal signal calculation, strict contiguous-history checks,
configuration parsing, and deterministic maker-option planning.

IMPORTANT
---------
The repository currently has two DRC definitions under review:

DRC_INCLUDED
    sigma includes the immediately previous return being tested.

DRC_BACKGROUND
    sigma uses the four returns preceding the immediately previous return.

The original candidate used DRC_INCLUDED.  The independent FAIL reproduction
implemented DRC_BACKGROUND.  Both are supported explicitly so they cannot be
silently confused again.

Nothing in this module places an order.  Live execution remains disabled unless
the runner explicitly integrates and enables it after reproduction.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, getcontext
from enum import Enum
from statistics import stdev
from typing import Iterable, Literal, Sequence

getcontext().prec = 40

Side = Literal["yes", "no"]


class DRCDefinition(str, Enum):
    INCLUDED = "included"
    BACKGROUND = "background"


@dataclass(frozen=True)
class FloorPoint:
    """One market boundary and its A0/floor strike.

    `close_ts_s` is the contract close timestamp. Adjacent contracts in the
    same 15-minute series must differ by exactly 900 seconds.
    """

    close_ts_s: int
    floor: Decimal


@dataclass(frozen=True)
class DRCSignal:
    qualifies: bool
    definition: DRCDefinition
    previous_return: Decimal
    sigma: Decimal
    z_score: Decimal
    favorite_side: Side
    reason: str


@dataclass(frozen=True)
class MakerOptionPlan:
    target_qty: int
    maker_timeout_seconds: float
    initial_ask: Decimal
    relative_ceiling: Decimal
    absolute_ceiling: Decimal
    cross_ceiling: Decimal

    def remaining(self, filled_qty: int) -> int:
        if filled_qty < 0:
            raise ValueError("filled_qty cannot be negative")
        return max(0, self.target_qty - filled_qty)

    def may_cross(self, current_ask: Decimal, signal_still_valid: bool) -> bool:
        return (
            signal_still_valid
            and Decimal("0") < current_ask < Decimal("1")
            and current_ask <= self.cross_ceiling
        )


def decimal_value(value: object) -> Decimal:
    """Convert API numeric strings/ints without accepting binary floats."""
    if isinstance(value, float):
        raise TypeError("binary float rejected; pass an API string or Decimal")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"invalid decimal value: {value!r}") from exc
    if not result.is_finite():
        raise ValueError("value must be finite")
    return result


def _validate_contiguous(points: Sequence[FloorPoint]) -> None:
    for point in points:
        if point.floor <= 0:
            raise ValueError("floor strikes must be positive")
    for left, right in zip(points, points[1:]):
        if right.close_ts_s - left.close_ts_s != 900:
            raise ValueError("floor history is not contiguous at 900 seconds")


def completed_returns(points: Sequence[FloorPoint]) -> list[Decimal]:
    """Return consecutive floor-to-floor returns, oldest to newest."""
    _validate_contiguous(points)
    return [
        right.floor / left.floor - Decimal("1")
        for left, right in zip(points, points[1:])
    ]


def compute_drc_signal(
    floors: Sequence[FloorPoint],
    *,
    favorite_side: Side,
    definition: DRCDefinition = DRCDefinition.INCLUDED,
    z_threshold: Decimal = Decimal("1"),
    epsilon: Decimal = Decimal("1e-12"),
) -> DRCSignal:
    """Compute DRC using only completed, contiguous A0/floor observations.

    INCLUDED needs five floor points, producing four completed returns. The
    newest return is both the numerator and one member of sigma4.

    BACKGROUND needs six floor points, producing five returns. The newest
    return is the numerator; sigma4 uses the four preceding returns.
    """
    if favorite_side not in ("yes", "no"):
        raise ValueError("favorite_side must be yes or no")
    if z_threshold <= 0 or epsilon <= 0:
        raise ValueError("z_threshold and epsilon must be positive")

    needed = 5 if definition is DRCDefinition.INCLUDED else 6
    if len(floors) < needed:
        return DRCSignal(
            False, definition, Decimal("0"), Decimal("0"), Decimal("0"),
            favorite_side, f"need_{needed}_contiguous_floor_points",
        )

    points = list(floors[-needed:])
    try:
        returns = completed_returns(points)
    except ValueError as exc:
        return DRCSignal(
            False, definition, Decimal("0"), Decimal("0"), Decimal("0"),
            favorite_side, str(exc),
        )

    previous_return = returns[-1]
    sigma_returns = returns[-4:] if definition is DRCDefinition.INCLUDED else returns[-5:-1]
    sigma = decimal_value(stdev(sigma_returns))
    z_score = previous_return / (sigma + epsilon)

    # DRC-15 is the NO-favorite reversal after a sufficiently positive prior
    # move. Symmetric YES analysis is research-only and is not silently traded.
    if favorite_side != "no":
        reason = "favorite_not_no"
        qualifies = False
    elif z_score < z_threshold:
        reason = "z_below_threshold"
        qualifies = False
    else:
        reason = "qualified"
        qualifies = True

    return DRCSignal(
        qualifies=qualifies,
        definition=definition,
        previous_return=previous_return,
        sigma=sigma,
        z_score=z_score,
        favorite_side=favorite_side,
        reason=reason,
    )


def make_plan(
    *,
    target_qty: int,
    initial_ask: object,
    timeout_seconds: float = 10.0,
    relative_ceiling_cents: object = "3",
    absolute_ceiling: object = "0.85",
) -> MakerOptionPlan:
    """Create a maker-first, conditional-IOC plan without placing orders."""
    if target_qty <= 0:
        raise ValueError("target_qty must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    ask = decimal_value(initial_ask)
    rel = ask + decimal_value(relative_ceiling_cents) / Decimal("100")
    absolute = decimal_value(absolute_ceiling)
    if not Decimal("0") < ask < Decimal("1"):
        raise ValueError("initial ask must be between zero and one")
    if not Decimal("0") < absolute < Decimal("1"):
        raise ValueError("absolute ceiling must be between zero and one")

    return MakerOptionPlan(
        target_qty=target_qty,
        maker_timeout_seconds=timeout_seconds,
        initial_ask=ask,
        relative_ceiling=rel,
        absolute_ceiling=absolute,
        cross_ceiling=min(rel, absolute),
    )


def floor_points_from_markets(markets: Iterable[dict]) -> list[FloorPoint]:
    """Normalize market dictionaries containing close timestamp and floor.

    Accepted timestamp keys: `close_ts`, `close_time_ts`, `close_ts_s`.
    Accepted floor keys: `floor_strike`, `floor_strike_dollars`.
    Malformed rows are skipped; duplicates are resolved deterministically.
    """
    by_close: dict[int, FloorPoint] = {}
    for market in markets:
        raw_ts = (
            market.get("close_ts_s")
            or market.get("close_time_ts")
            or market.get("close_ts")
        )
        raw_floor = market.get("floor_strike_dollars", market.get("floor_strike"))
        if raw_ts is None or raw_floor is None:
            continue
        try:
            ts = int(raw_ts)
            floor = decimal_value(raw_floor)
        except (TypeError, ValueError):
            continue
        if ts > 0 and floor > 0:
            by_close[ts] = FloorPoint(ts, floor)
    return [by_close[key] for key in sorted(by_close)]
