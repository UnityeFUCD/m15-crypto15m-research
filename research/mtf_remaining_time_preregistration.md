# MTF-R — Remaining-Time Correction

Frozen before a new public incentive/book snapshot is fetched.

## Why this correction is required

The first MTF screen compared reward under a full or half incentive period with
current order books. For short programs, especially 15-minute commodities, a
frontier may appear only late in the period. Reward that could have been earned
before the snapshot is unavailable. The valid upper bound is the score that can
be earned from the current snapshot until program end.

## Frozen necessary-condition calculation

For every active liquidity program and each side whose current q1 order is
guaranteed inside the displayed Target Size frontier:

- compute exact remaining fraction of the incentive period;
- assume the order rests without filling through program end;
- maximum competitor score remains target size on both sides throughout;
- calculate current-to-end reward for q1;
- solve the minimum integer quantity needed to reach the stated $1 minimum
  payout;
- require that quantity to fit the current guaranteed frontier;
- compare reward at that quantity with the maximum directional loss
  `quantity * quote price`.

Two interpretations are reported:

1. **Per-program minimum:** raw reward must reach $1 in this incentive period.
2. **Aggregate minimum:** q1 reward is economically meaningful if rewards are
   aggregated across periods before the $1 payout threshold.

## Candidate definitions

- `per_program_cover`: minimum quantity for $1 fits the target frontier and its
  current-to-end reward covers maximum directional loss.
- `q1_cover`: q1 current-to-end reward covers one-contract maximum loss,
  irrespective of payout aggregation.

These are necessary arithmetic conditions only. They do not establish queue
stability, fill probability, actual credited score, reward aggregation, or
future program availability. No order is authorized.
