# GTR — Gold Tail-Reward Audit

Frozen after the current MTF screen identified a 1c KXGOLD15M target-frontier
quote and before its historical fill/outcome path is evaluated.

## Mechanism

Each KXGOLD15M incentive period currently offers a $20 liquidity pool with
Target Size 300. A q64 order at a 1c qualifying bid, resting for about half of
the 15-minute period under maximum two-sided target competition, earns just
over the stated $1 minimum payout while its maximum directional loss is 64c.

This creates a potential subsidy-dominates-risk policy. It does not require a
settlement prediction.

## Frozen policy

- Series: KXGOLD15M only.
- At the first complete quote after program start, identify the cheaper outcome
  bid.
- Participate only when that bid is exactly 1c.
- Quantity: 64 contracts, derived before historical testing as the minimum
  quantity clearing $1 near a 50% rest fraction under maximum competition.
- Join the 1c bid; maker only.
- Cancel at the first complete eight-minute deadline if unfilled.
- If filled before cancellation, hold to settlement.
- Reward share uses actual modeled rest time, maximum target-size competition
  on both sides, and the exact program pool.
- If modeled reward is below $1, credited reward is zero. Otherwise floor to
  the nearest cent.
- Evaluate strict close-through and touch high/low fill models.

## Gates

A strong prospective candidate requires, under both fill models:

1. at least 50 historical eligible periods;
2. combined trading plus payout-adjusted reward P&L positive;
3. positive 15/30/60/90-minute moving-block lower 95% bounds;
4. positive after removing any one calendar day;
5. maximum drawdown below $10 on q64;
6. result remains positive if only 75% of the modeled score is credited.

Historical paths do not reveal whether our order would actually remain inside
the Target Size frontier. Passing therefore justifies a prospective shadow/q1
score-credit measurement, not production deployment.

No live action is authorized.
