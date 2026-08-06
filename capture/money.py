"""Exact numeric types. No binary float touches authoritative money.

WHY
  The existing runner accumulates exposure as float64:
      STATE["deployed"] += q_use * px
  Every add carries representation error, and the error compounds across a
  day's orders. Comparisons against a hard floor ($211) then depend on drift.

UNITS
  price_micros      1e-6 dollars   a 1c tick is 10_000 micros
  quantity_hundredths  1e-2 contracts  Kalshi quantities are *_fp with 2dp
  cost/fee/payout/balance/exposure  all *_micros, int64

  int64 micros holds +/- 9.2e12 dollars. No overflow concern.

RULE
  Parse API strings with Decimal, never float(). "0.6900" -> 690000 exactly.
  float() of an API string is the bug this module exists to prevent.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP, localcontext

MICROS = 1_000_000
HUNDREDTHS = 100

# Kalshi tick sizes: 0.1c below 10c and above 90c, 1c in between.
TICK_FINE_MICROS = 1_000      # 0.001
TICK_COARSE_MICROS = 10_000   # 0.01


def to_micros(v) -> int:
    """API string / Decimal / int -> integer micros. Never accepts float."""
    if v is None:
        return 0
    if isinstance(v, float):
        raise TypeError(
            "refusing float for money: pass the raw API string or a Decimal. "
            "float('0.6900') is not exactly 0.69 and the error compounds.")
    if isinstance(v, int):
        return v * MICROS
    d = v if isinstance(v, Decimal) else Decimal(str(v).strip())
    with localcontext() as ctx:
        ctx.prec = 34
        return int((d * MICROS).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def from_micros(m: int) -> Decimal:
    return (Decimal(int(m)) / MICROS).quantize(Decimal("0.000001"))


def dollars_str(m: int) -> str:
    return f"{from_micros(m):.6f}"


def qty_to_hundredths(v) -> int:
    """Kalshi *_fp quantities ('20.00') -> integer hundredths of a contract."""
    if v is None:
        return 0
    if isinstance(v, float):
        raise TypeError("refusing float for quantity; pass the API string")
    d = v if isinstance(v, Decimal) else Decimal(str(v).strip())
    with localcontext() as ctx:
        ctx.prec = 34
        return int((d * HUNDREDTHS).quantize(Decimal(1),
                                             rounding=ROUND_HALF_UP))


def qty_from_hundredths(h: int) -> Decimal:
    return (Decimal(int(h)) / HUNDREDTHS).quantize(Decimal("0.01"))


def tick_micros(price_micros: int) -> int:
    """Tick at a given price. Boundaries are inclusive of the fine band."""
    if price_micros < 100_000 or price_micros > 900_000:
        return TICK_FINE_MICROS
    return TICK_COARSE_MICROS


def snap_to_tick(price_micros: int) -> int:
    """Round to the nearest valid tick. Uses the tick of the ROUNDED result at
    band boundaries, so snapping 0.0995 does not land on an invalid price."""
    t = tick_micros(price_micros)
    snapped = ((price_micros + t // 2) // t) * t
    t2 = tick_micros(snapped)
    if t2 != t:
        snapped = ((price_micros + t2 // 2) // t2) * t2
    return snapped


def complement_micros(price_micros: int) -> int:
    """YES price -> NO price. Exact: they must sum to exactly $1."""
    return MICROS - price_micros


def taker_fee_micros(price_micros: int, quantity_hundredths: int) -> int:
    """Kalshi taker fee: ceil(0.07 * C * P * (1-P) * 100) cents, per block.

    Computed in Decimal and rounded UP to the cent, matching the exchange.
    Maker fees are zero.
    """
    with localcontext() as ctx:
        ctx.prec = 34
        p = Decimal(price_micros) / MICROS
        c = Decimal(quantity_hundredths) / HUNDREDTHS
        raw_cents = Decimal("0.07") * c * p * (Decimal(1) - p) * 100
        cents = int(raw_cents.to_integral_value(rounding="ROUND_CEILING"))
        return cents * 10_000       # 1 cent = 10_000 micros


def notional_micros(price_micros: int, quantity_hundredths: int) -> int:
    """Cost of a position. Exact integer arithmetic, no rounding drift."""
    return (price_micros * quantity_hundredths) // HUNDREDTHS
