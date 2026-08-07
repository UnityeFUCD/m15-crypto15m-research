# MTF — Marginal Target-Frontier Liquidity Audit

Frozen before the current incentive/book snapshot is fetched.

## Mechanism

Kalshi liquidity rewards score only orders that help fill a program's Target
Size. Quoting at the best bid maximizes score but also maximizes adverse fill
risk. The target-size rule creates a different opportunity: place q1 at the
**deepest price level that is still guaranteed to lie inside the qualifying
Target Size frontier**. Such an order earns a distance-discounted reward while
sitting behind more price protection than a best bid.

This is an external-subsidy mechanism, not a prediction filter.

## Frozen construction

For every active public liquidity program with a two-sided order book:

1. Read the full YES and NO bid books and the program Target Size.
2. For each outcome side, walk prices from best to worse using raw displayed
   size.
3. Find the deepest price where a new q1 order is guaranteed to remain within
   cumulative Target Size. Never assume queue priority at a level that already
   crosses the target.
4. Require the quote to remain strictly below the opposing ask (post-only).
5. Calculate distance multiplier from the program discount factor.
6. Conservative reward share denominator is maximum target-size score on both
   sides plus own score.
7. Report one-sided q1 and two-sided q1 constructions.

## Outputs

- program/market/side;
- target-frontier quote price and distance from best;
- qualifying score multiplier;
- lower-bound reward per program period and per day;
- worst directional loss if filled and wrong;
- reward/loss ratio under full-period and half-period resting;
- displayed depth and frontier stability inputs.

## Candidate levels

- **Arithmetic cover:** full-period lower reward exceeds worst one-fill loss.
- **Strong prospective candidate:** half-period lower reward covers at least
  25% of worst one-fill loss and the quote is at least two cents behind best.
- Otherwise research only.

Actual queue order, snapshots, fills, eligibility and credited rewards remain
prospective. No result authorizes orders or weakens a KILL state.
