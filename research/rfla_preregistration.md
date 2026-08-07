# RFLA — Reward-Funded Liquidity Audit

Frozen before the historical reward audit is run.

## Why this is a genuinely different mechanism

The crypto favorite research repeatedly found real informational structure that
was consumed by the spread, fees, or endogenous maker fills. RFLA does not ask
for another predictive subset. It asks whether an **external liquidity subsidy**
can pay more than the adverse directional loss created by supplying liquidity.

The public incentive frontier identified recurring 15-minute GOLD, SILVER and
WTI programs with a $20 pool and target size 300. This audit uses public
program terms, real resolved markets, and historical one-minute quote paths.
No credentials or orders are used.

## Certified economic convention

- Program reward is `period_reward / 10_000` dollars.
- Maximum qualifying competitor score is conservatively treated as target size
  on both outcome sides for the entire period.
- Own reward is proportional to actual modeled resting time.
- Maker fees are zero.
- Trading P&L and reward P&L are reported separately.

Because actual account reward credits are not historically observable, the
modeled reward is stressed to 100%, 50%, and 25% of the conservative score-share
estimate.

## Frozen policies

### P1 — complementary join

- q1 YES bid and q1 NO bid at the first complete quote after program start;
- maximum one market per shared close;
- select the market with greatest reward density;
- if neither side fills, cancel both after 5, 8, 11, or 14 minutes (all four
  horizons are reported; no best horizon is selected);
- once one side fills, leave the complement through program end;
- fill models: strict minute close-through and touch high/low.

This policy may overstate reward eligibility when displayed best depth already
fills the target. It is therefore mechanism evidence, not the primary PASS
lane.

### P2 — price-priority cheap-side quote

- identify the cheaper outcome side at the first complete quote;
- improve its bid by exactly one tick while remaining strictly inside the
  spread;
- q1 only;
- maximum one market per shared close;
- select by modeled reward-to-worst-loss ratio;
- quote-price risk tiers: <=20c and <=30c;
- cancel after 2, 3, 5, 8, 11, or 14 minutes (all reported, no selection);
- fill models: strict and touch.

Because this order becomes the best bid at submission, its liquidity should be
inside the target-size qualifying frontier. P2 is the primary economic lane.

## Required outputs

For every policy, fill model, horizon, and reward-credit stress:

- markets quoted and active fills;
- fill rate and fill win rate;
- trading P&L;
- reward P&L;
- combined P&L;
- mean per day;
- 15/30/60/90-minute block-bootstrap intervals;
- maximum drawdown and worst close window;
- leave-one-series results.

## Candidate gate

P2 becomes a **strong historical candidate** only if one of the two frozen risk
tiers (20c or 30c), without choosing a cancellation horizon, is positive for
**every** cancellation horizon under both strict and touch fill models at only
25% of modeled reward credit, and remains positive after removing any one
series.

This is deliberately severe. Four calendar days cannot establish a durable
annual edge. Passing would show that the reward subsidy dominates modeled
trading losses by a large enough margin to justify a q1 prospective reward-
credit test. Failing closes this mechanism under the present program terms.

No result authorizes live trading or weakens any existing KILL state.
