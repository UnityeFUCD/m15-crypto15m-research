# MTF-S — Target-Frontier Scaling and Minimum-Payout Audit

Frozen after the q1 MTF screen identified arithmetic-cover rows and before any
quantity result is calculated.

## Purpose

The q1 screen ignored the official program's stated $1 minimum payout. This
audit asks whether a quantity can remain inside the currently displayed Target
Size frontier, clear a $1 payout under conservative score competition, and
still have the reward cover the worst directional loss.

## Frozen calculation

For every one-sided MTF row:

- guaranteed capacity at the same price is `target_size - cumulative_ahead`;
- rest-fraction scenarios: 25%, 50%, 75%, 100%;
- own score-time is `quantity * distance_multiplier * rest_fraction`;
- competitor score-time is maximum `2 * target_size`;
- reward is `pool * own / (2*target + own)`;
- minimum payout target is $1;
- worst directional loss is `quantity * quote_price`.

For each rest fraction, solve the smallest integer quantity that reaches $1,
then report whether it fits the target frontier and whether reward covers worst
loss. No quantity or market is selected by historical P&L.

## Candidate definition

A **scalable arithmetic-cover candidate** requires:

1. quantity needed for $1 payout fits the currently guaranteed target frontier;
2. modeled reward at that quantity is at least the worst one-fill loss;
3. rest fraction is no more than 50%;
4. quote price is no more than 5c.

Actual eligibility, queue changes, per-program versus aggregate minimum payout,
resting time, fills, and credited rewards remain prospective facts. No result
authorizes orders.
