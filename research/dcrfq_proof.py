from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

Leg = str
LegSet = frozenset[Leg]


@dataclass(frozen=True)
class Instrument:
    ticker: str
    legs: LegSet
    side: str
    ask: float
    depth: int
    fee_per_contract: float = 0.0

    @property
    def all_in_unit_cost(self) -> float:
        return self.ask + self.fee_per_contract


@dataclass(frozen=True)
class Allocation:
    ticker: str
    quantity: int
    unit_cost: float
    legs: tuple[str, ...]


def powerset(values: Sequence[Leg]) -> list[LegSet]:
    return [
        frozenset(combo)
        for r in range(1, len(values) + 1)
        for combo in itertools.combinations(values, r)
    ]


def combo_payout(legs: LegSet, state: dict[Leg, float]) -> float:
    out = 1.0
    for leg in legs:
        x = state[leg]
        if not (0.0 <= x <= 1.0):
            raise ValueError(f"settlement outside [0,1]: {leg}={x}")
        out *= x
    return out


def arm_a_terminal_value(
    target: LegSet,
    hedges: Sequence[tuple[float, LegSet]],
    quantity: float,
    state: dict[Leg, float],
) -> float:
    """Subset-YES hedges plus target NO."""
    return sum(weight * combo_payout(legs, state) for weight, legs in hedges) + quantity * (
        1.0 - combo_payout(target, state)
    )


def arm_b_terminal_value(
    target: LegSet,
    cover: Sequence[LegSet],
    quantity: float,
    state: dict[Leg, float],
) -> float:
    """Target YES plus NO on every selected covering subset."""
    return quantity * combo_payout(target, state) + quantity * sum(
        1.0 - combo_payout(legs, state) for legs in cover
    )


def is_subset_hedge(target: LegSet, hedge: LegSet) -> bool:
    return bool(hedge) and hedge.issubset(target)


def is_integer_cover(target: LegSet, cover: Iterable[LegSet]) -> bool:
    union: set[Leg] = set()
    for legs in cover:
        if not legs or not legs.issubset(target):
            return False
        union.update(legs)
    return union == set(target)


def verify_arm_a_statewise(
    target: LegSet,
    hedges: Sequence[tuple[float, LegSet]],
    quantity: float,
    states: Iterable[dict[Leg, float]],
    tol: float = 1e-12,
) -> float:
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    total_weight = sum(weight for weight, _ in hedges)
    if total_weight + tol < quantity:
        raise ValueError("subset hedge quantity is smaller than target quantity")
    if any(weight < 0 or not is_subset_hedge(target, legs) for weight, legs in hedges):
        raise ValueError("invalid Arm-A hedge")
    minimum = float("inf")
    for state in states:
        value = arm_a_terminal_value(target, hedges, quantity, state)
        minimum = min(minimum, value)
        if value + tol < quantity:
            raise AssertionError({"arm": "A", "value": value, "quantity": quantity, "state": state})
    return minimum


def verify_arm_b_statewise(
    target: LegSet,
    cover: Sequence[LegSet],
    quantity: float,
    states: Iterable[dict[Leg, float]],
    tol: float = 1e-12,
) -> float:
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    if not is_integer_cover(target, cover):
        raise ValueError("Arm-B subsets do not cover the target")
    minimum = float("inf")
    for state in states:
        value = arm_b_terminal_value(target, cover, quantity, state)
        minimum = min(minimum, value)
        if value + tol < quantity:
            raise AssertionError({"arm": "B", "value": value, "quantity": quantity, "state": state})
    return minimum


def scalar_grid_states(legs: Sequence[Leg], grid: Sequence[float]) -> Iterable[dict[Leg, float]]:
    for values in itertools.product(grid, repeat=len(legs)):
        yield dict(zip(legs, values, strict=True))


def random_state(legs: Sequence[Leg], rng: random.Random) -> dict[Leg, float]:
    return {leg: rng.random() for leg in legs}


def cheapest_subset_yes_allocation(
    target: LegSet,
    quantity: int,
    instruments: Sequence[Instrument],
) -> tuple[list[Allocation], float] | None:
    """Depth-conserving Arm-A allocator.

    Every admitted YES instrument has payout at least target YES.  One unit of
    any admitted instrument hedges one target-NO contract.  Depth can be pooled
    across instruments without reusing a unit.
    """
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    eligible = [
        inst
        for inst in instruments
        if inst.side == "yes"
        and inst.depth > 0
        and 0.0 < inst.ask < 1.0
        and is_subset_hedge(target, inst.legs)
    ]
    eligible.sort(key=lambda x: (x.all_in_unit_cost, x.ticker))
    remaining = quantity
    allocations: list[Allocation] = []
    total = 0.0
    used: dict[str, int] = {}
    for inst in eligible:
        take = min(remaining, inst.depth)
        if take <= 0:
            continue
        used[inst.ticker] = used.get(inst.ticker, 0) + take
        if used[inst.ticker] > inst.depth:
            raise AssertionError("depth reused")
        allocations.append(
            Allocation(inst.ticker, take, inst.all_in_unit_cost, tuple(sorted(inst.legs)))
        )
        total += take * inst.all_in_unit_cost
        remaining -= take
        if remaining == 0:
            return allocations, total
    return None


def minimum_cost_integer_no_cover(
    target: LegSet,
    instruments: Sequence[Instrument],
) -> tuple[list[Instrument], float] | None:
    """Exact minimum-cost integer set cover by dynamic programming.

    The returned NO instruments are purchased one-for-one with each unit of the
    target-YES position.  Candidate depth is checked later against accepted q.
    """
    legs = sorted(target)
    index = {leg: i for i, leg in enumerate(legs)}
    full = (1 << len(legs)) - 1
    candidates: list[tuple[int, Instrument]] = []
    for inst in instruments:
        if (
            inst.side != "no"
            or inst.depth <= 0
            or not (0.0 < inst.ask < 1.0)
            or not is_subset_hedge(target, inst.legs)
        ):
            continue
        mask = 0
        for leg in inst.legs:
            mask |= 1 << index[leg]
        candidates.append((mask, inst))

    inf = float("inf")
    dp = [inf] * (full + 1)
    parent: list[tuple[int, int] | None] = [None] * (full + 1)
    dp[0] = 0.0
    for candidate_index, (cover_mask, inst) in enumerate(candidates):
        previous = dp.copy()
        previous_parent = parent.copy()
        for mask in range(full + 1):
            if not math.isfinite(previous[mask]):
                continue
            new_mask = mask | cover_mask
            new_cost = previous[mask] + inst.all_in_unit_cost
            if new_cost + 1e-15 < dp[new_mask]:
                dp[new_mask] = new_cost
                parent[new_mask] = (mask, candidate_index)
        # Preserve states not improved in this 0/1 iteration.
        for mask in range(full + 1):
            if dp[mask] == previous[mask] and parent[mask] is None:
                parent[mask] = previous_parent[mask]

    if not math.isfinite(dp[full]):
        return None
    selected: list[Instrument] = []
    mask = full
    seen: set[int] = set()
    while mask:
        step = parent[mask]
        if step is None:
            raise AssertionError("cover parent chain broken")
        prior_mask, candidate_index = step
        if candidate_index in seen:
            raise AssertionError("candidate reused in 0/1 cover")
        seen.add(candidate_index)
        selected.append(candidates[candidate_index][1])
        mask = prior_mask
    selected.reverse()
    if not is_integer_cover(target, [x.legs for x in selected]):
        raise AssertionError("optimizer returned a non-cover")
    return selected, dp[full]


def arm_b_cover_for_quantity(
    target: LegSet,
    quantity: int,
    instruments: Sequence[Instrument],
) -> tuple[list[Allocation], float] | None:
    eligible = [inst for inst in instruments if inst.depth >= quantity]
    solved = minimum_cost_integer_no_cover(target, eligible)
    if solved is None:
        return None
    selected, unit_cost = solved
    allocations = [
        Allocation(inst.ticker, quantity, inst.all_in_unit_cost, tuple(sorted(inst.legs)))
        for inst in selected
    ]
    return allocations, quantity * unit_cost


def invalid_cover_counterexample(target: LegSet, cover: Sequence[LegSet]) -> dict[Leg, float]:
    union = set().union(*cover) if cover else set()
    missing = sorted(set(target) - union)
    if not missing:
        raise ValueError("cover is valid; no missing-leg counterexample")
    state = {leg: 1.0 for leg in target}
    state[missing[0]] = 0.0
    return state


def all_unique_covers_up_to_three(target: LegSet) -> Iterable[tuple[LegSet, ...]]:
    subsets = powerset(sorted(target))
    for size in (1, 2, 3):
        for combo in itertools.combinations(subsets, size):
            if is_integer_cover(target, combo):
                yield combo


def run_exhaustive_checks() -> dict[str, int | float]:
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    arm_a_cases = 0
    arm_a_state_checks = 0
    min_a = float("inf")
    for n in range(1, 7):
        legs = tuple(f"L{i}" for i in range(n))
        target = frozenset(legs)
        states = list(scalar_grid_states(legs, grid))
        for subset in powerset(legs):
            minimum = verify_arm_a_statewise(target, [(1.0, subset)], 1.0, states)
            min_a = min(min_a, minimum)
            arm_a_cases += 1
            arm_a_state_checks += len(states)

    arm_b_cases = 0
    arm_b_state_checks = 0
    min_b = float("inf")
    for n in range(1, 6):
        legs = tuple(f"L{i}" for i in range(n))
        target = frozenset(legs)
        states = list(scalar_grid_states(legs, grid))
        for cover in all_unique_covers_up_to_three(target):
            minimum = verify_arm_b_statewise(target, cover, 1.0, states)
            min_b = min(min_b, minimum)
            arm_b_cases += 1
            arm_b_state_checks += len(states)

    return {
        "arm_a_cases": arm_a_cases,
        "arm_a_state_checks": arm_a_state_checks,
        "arm_a_minimum_terminal_value": min_a,
        "arm_b_integer_cover_cases_up_to_three_sets": arm_b_cases,
        "arm_b_state_checks": arm_b_state_checks,
        "arm_b_minimum_terminal_value": min_b,
    }


def run_random_checks(seed: int = 20260813) -> dict[str, int | float]:
    rng = random.Random(seed)
    checks_a = 0
    checks_b = 0
    min_a = float("inf")
    min_b = float("inf")
    for n in range(2, 16):
        legs = tuple(f"L{i}" for i in range(n))
        target = frozenset(legs)
        subsets = powerset(legs) if n <= 10 else None
        for _ in range(1000):
            state = random_state(legs, rng)
            if subsets is not None:
                hedge = rng.choice(subsets)
            else:
                hedge = frozenset(rng.sample(legs, rng.randint(1, n)))
            value_a = arm_a_terminal_value(target, [(1.0, hedge)], 1.0, state)
            if value_a < 1.0 - 1e-12:
                raise AssertionError(("A", n, hedge, state, value_a))
            min_a = min(min_a, value_a)
            checks_a += 1

            # Build a random valid overlapping cover.
            shuffled = list(legs)
            rng.shuffle(shuffled)
            cover: list[LegSet] = []
            cursor = 0
            while cursor < n:
                width = rng.randint(1, max(1, n // 2))
                block = set(shuffled[cursor : min(n, cursor + width)])
                if cover and rng.random() < 0.7:
                    block.update(rng.sample(legs, rng.randint(0, min(2, n))))
                cover.append(frozenset(block))
                cursor += width
            if not is_integer_cover(target, cover):
                raise AssertionError("random cover builder failed")
            value_b = arm_b_terminal_value(target, cover, 1.0, state)
            if value_b < 1.0 - 1e-12:
                raise AssertionError(("B", n, cover, state, value_b))
            min_b = min(min_b, value_b)
            checks_b += 1
    return {
        "seed": seed,
        "arm_a_random_checks": checks_a,
        "arm_a_minimum_terminal_value": min_a,
        "arm_b_random_checks": checks_b,
        "arm_b_minimum_terminal_value": min_b,
    }


def run_optimizer_checks() -> dict[str, object]:
    target = frozenset({"A", "B", "C", "D"})
    instruments = [
        Instrument("A-YES", frozenset({"A"}), "yes", 0.72, 2, 0.01),
        Instrument("AB-YES", frozenset({"A", "B"}), "yes", 0.61, 4, 0.01),
        Instrument("ABC-YES", frozenset({"A", "B", "C"}), "yes", 0.55, 3, 0.01),
        Instrument("D-YES", frozenset({"D"}), "yes", 0.67, 20, 0.01),
        Instrument("OUTSIDE", frozenset({"X"}), "yes", 0.01, 100, 0.0),
        Instrument("AB-NO", frozenset({"A", "B"}), "no", 0.20, 20, 0.01),
        Instrument("CD-NO", frozenset({"C", "D"}), "no", 0.22, 20, 0.01),
        Instrument("ABC-NO", frozenset({"A", "B", "C"}), "no", 0.25, 20, 0.01),
        Instrument("D-NO", frozenset({"D"}), "no", 0.10, 20, 0.01),
        Instrument("A-NO", frozenset({"A"}), "no", 0.12, 20, 0.01),
        Instrument("B-NO", frozenset({"B"}), "no", 0.13, 20, 0.01),
        Instrument("C-NO", frozenset({"C"}), "no", 0.14, 20, 0.01),
    ]

    arm_a = cheapest_subset_yes_allocation(target, 8, instruments)
    if arm_a is None:
        raise AssertionError("Arm-A allocator unexpectedly failed")
    allocations_a, cost_a = arm_a
    if sum(row.quantity for row in allocations_a) != 8:
        raise AssertionError("Arm-A quantity not conserved")
    if any(row.ticker == "OUTSIDE" for row in allocations_a):
        raise AssertionError("non-subset hedge admitted")

    arm_b = arm_b_cover_for_quantity(target, 7, instruments)
    if arm_b is None:
        raise AssertionError("Arm-B optimizer unexpectedly failed")
    allocations_b, cost_b = arm_b
    chosen_b = {row.ticker for row in allocations_b}
    if chosen_b != {"AB-NO", "CD-NO"}:
        raise AssertionError(("unexpected minimum cover", chosen_b))

    invalid = [frozenset({"A", "B"}), frozenset({"C"})]
    state = invalid_cover_counterexample(target, invalid)
    invalid_value = arm_b_terminal_value(target, invalid, 1.0, state)
    if invalid_value >= 1.0 - 1e-12:
        raise AssertionError("invalid cover was not falsified")

    return {
        "arm_a_allocations": [row.__dict__ for row in allocations_a],
        "arm_a_total_cost": cost_a,
        "arm_b_allocations": [row.__dict__ for row in allocations_b],
        "arm_b_total_cost": cost_b,
        "invalid_cover_counterexample": state,
        "invalid_cover_terminal_value": invalid_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    exhaustive = run_exhaustive_checks()
    random_checks = run_random_checks()
    optimizers = run_optimizer_checks()
    report = {
        "status": "PASS",
        "identity": {
            "arm_a": "C_S + (1-C_T) >= 1 for every nonempty S subset T",
            "arm_b": "C_T + sum_j(1-C_Sj) >= 1 for every integer cover union S_j=T",
        },
        "exhaustive": exhaustive,
        "random": random_checks,
        "optimizer": optimizers,
        "limitations": [
            "This proves payoff dominance and optimizer invariants, not market prices or fills.",
            "Arm B intentionally uses integer covers only.",
            "Rule-signature validation and exact Kalshi fee overrides remain required at runtime.",
        ],
    }
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    print(text)


if __name__ == "__main__":
    main()
