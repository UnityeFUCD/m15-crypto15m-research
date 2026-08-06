# Incentive-adjusted CRYPTO15M audit

## Verdict: **FAIL — no active CRYPTO15M incentive period**

Snapshot UTC: `2026-08-06T13:02:06.192662+00:00`

## Current program inventory

- Active programs exchange-wide: 3793
- Upcoming programs exchange-wide: 219
- Active/upcoming CRYPTO15M programs: 0
- Active CRYPTO15M liquidity programs: 0
- Active CRYPTO15M volume programs: 0
- Active series: none

## Economic hurdle

The current unsampled population estimate is approximately **−0.43¢ per
submitted contract** after maker fill selection. The general volume
program caps rewards at **0.50¢ per eligible contract**, so the absolute
best case would be approximately **+0.07¢**. That upper bound is not an
expected payout; proportional pool competition can make the actual reward
much smaller.

No active CRYPTO15M market was returned by the public incentive API.
Rewards therefore cannot repair the present strategy economics at this
snapshot. Upcoming schedules remain in the attached CSV/JSON evidence.

## Limits

Liquidity payout depends on the account's share of random one-second
snapshots. Current aggregate order books can only bound that share; they
cannot identify future participant scores. A live candidate requires a
prospective recorder that captures program terms, full books, own resting
orders, actual reward credits, and fills without self-trading or artificial
volume.

No result authorizes changing the repository's existing KILL state.