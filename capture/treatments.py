"""Structured execution experiment. Deterministic, persisted, restart-safe.

WHY THE A-H LIST WAS REPLACED
  The original arms mixed three independent dimensions into one label, so
  "improve one tick" and "quantity 2" could never be observed together and the
  effects were not separable. Price, cancellation and quantity are now
  orthogonal factors.

DETERMINISM
  Assignment is a pure function of (opportunity_id, experiment_version). No RNG
  state is carried, so a crash mid-episode cannot reassign: recomputing after
  restart yields the same arm. The draw is HMAC-SHA256 over the key, which is
  uniform and reproducible across machines and Python versions - unlike
  hash(), which is salted per process.

INVERSE-PROBABILITY ANALYSIS
  The realised probability of the chosen arm is stored on the assignment row,
  so an analyst can reweight. Without it, an unbalanced or mid-experiment
  weight change silently biases every downstream estimate.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, asdict, field

EXPERIMENT_VERSION = "exp-001"

PRICE_POLICIES = ("shadow", "join", "back_one_tick", "improve_one_tick")
CANCEL_POLICIES = ("early", "standard", "late")
QUANTITIES = (1, 2)


@dataclass(frozen=True)
class Arm:
    price_policy: str
    cancel_policy: str
    quantity: int

    @property
    def label(self) -> str:
        return f"{self.price_policy}/{self.cancel_policy}/q{self.quantity}"


@dataclass
class ExperimentConfig:
    """Versioned. Changing ANY field must change `version`, because the
    assignment hash is keyed on it - otherwise two different experiments share
    an assignment stream and the arms are silently confounded."""
    version: str = EXPERIMENT_VERSION
    started_ms: int = 0
    max_quantity: int = 2                 # hard cap; never raise without the user
    min_spread_ticks_for_improve: int = 2  # improving needs room inside the spread
    cancel_seconds_left: dict = field(default_factory=lambda: {
        "early": 540, "standard": 420, "late": 240})
    # active arms and their weights; weights need not sum to 1 (normalised)
    weights: dict = field(default_factory=lambda: {
        "shadow/standard/q1": 0.20,       # the counterfactual arm
        "join/standard/q1": 0.30,         # control = current behaviour
        "join/standard/q2": 0.10,
        "join/early/q1": 0.10,
        "join/late/q1": 0.10,
        "back_one_tick/standard/q1": 0.10,
        "improve_one_tick/standard/q1": 0.10,
    })
    disabled: tuple = ()

    def config_hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True,
                          separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def active_arms(self) -> list[tuple[Arm, float]]:
        out = []
        for label, w in sorted(self.weights.items()):
            if label in self.disabled or w <= 0:
                continue
            pp, cp, q = label.split("/")
            qty = int(q.lstrip("q"))
            if pp not in PRICE_POLICIES or cp not in CANCEL_POLICIES:
                raise ValueError(f"bad arm label {label!r}")
            if qty > self.max_quantity:
                raise ValueError(
                    f"arm {label} exceeds max_quantity={self.max_quantity}")
            out.append((Arm(pp, cp, qty), float(w)))
        if not out:
            raise ValueError("no active arms")
        return out


def assign(opportunity_id: str, cfg: ExperimentConfig,
           spread_ticks: int | None = None) -> dict:
    """Deterministic assignment. Same inputs always give the same arm.

    Arms that are infeasible for this market (improving with no room inside
    the spread) are removed BEFORE the draw, and the stored probability is the
    one from the feasible set - so the weighting an analyst reverses is the
    weighting that actually applied.
    """
    arms = cfg.active_arms()
    if spread_ticks is not None and spread_ticks < cfg.min_spread_ticks_for_improve:
        arms = [(a, w) for a, w in arms if a.price_policy != "improve_one_tick"]
        if not arms:
            arms = cfg.active_arms()
    total = sum(w for _, w in arms)
    key = f"{opportunity_id}|{cfg.version}|{cfg.config_hash()}".encode()
    digest = hmac.new(b"treatment-assignment-v1", key,
                      hashlib.sha256).digest()
    draw = int.from_bytes(digest[:8], "big") / float(1 << 64)
    acc = 0.0
    chosen, prob = arms[-1][0], arms[-1][1] / total
    for a, w in arms:
        acc += w / total
        if draw < acc:
            chosen, prob = a, w / total
            break
    return {
        "opportunity_id": opportunity_id,
        "treatment": chosen.label,
        "price_policy": chosen.price_policy,
        "cancel_policy": chosen.cancel_policy,
        "quantity": chosen.quantity,
        "assignment_probability": prob,
        "experiment_version": cfg.version,
        "config_hash": cfg.config_hash(),
        "draw": draw,
        "feasible_arms": len(arms),
    }


def resolve_price_micros(price_policy: str, fav_bid_micros: int,
                         fav_ask_micros: int, tick_micros_fn) -> int | None:
    """Map a price policy to an actual limit price. None means do not submit."""
    if price_policy == "shadow":
        return None
    if price_policy == "join":
        return fav_bid_micros
    t = tick_micros_fn(fav_bid_micros)
    if price_policy == "back_one_tick":
        return fav_bid_micros - t
    if price_policy == "improve_one_tick":
        px = fav_bid_micros + t
        return px if px < fav_ask_micros else None   # never cross
    raise ValueError(f"unknown price policy {price_policy!r}")
