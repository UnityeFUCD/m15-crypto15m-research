# CMQL — Cross-Market Quote-Lag / Relative-Value Audit

Frozen before results are generated.

## Motivation

The broad favorite-maker strategy is approximately break-even after endogenous
fills; no-fill information is real but priced at the taker ask; the original
minute-:00 effect does not generalize unconditionally to thin new series; and
pre-entry public trade flow is usually absent. The remaining plausible
information wedge is asynchronous **quote formation across simultaneously
settling coins**.

This audit asks whether one coin's Kalshi quote underreacts to quote movements
already visible in the other coins, and whether any such residual can be
monetized after the executable taker ask and exact fee.

No Coinbase/spot data are used. This avoids the known one-minute candle
alignment trap.

## Population

Primary population: `data/paths_full.parquet`, the unsampled six-coin,
all-minute, 73-day quote-path population described in `DATA.md`.

A quote observation at minute-left `m` may use only:

- that market's quote at `m` and older observations (`m+1`, `m+2`), and
- other coins' quotes at the same `m` and older observations.

The next quote (`m-1`) and settlement are labels only.

## Chronology

Unique UTC dates are split chronologically:

- TRAIN: first 50%
- VALIDATION: next 25%
- TEST: final 25%

Models are fit on TRAIN only. A small, frozen policy grid is selected on
VALIDATION. TEST is evaluated once.

## Hypothesis H1 — cross-coin quote lead-lag

After peers' YES log-odds move, a target coin whose quote has moved less in the
same direction should catch up during the next complete minute.

Primary information test:

- baseline model: current target quote state + coin/minute controls
- own-path model: baseline + target's own prior quote changes
- peer model: own-path + contemporaneous peer quote changes and target-vs-peer
  response gap

The peer model must improve next-minute quote-change error and settlement log
loss on both VALIDATION and TEST relative to the own-path model.

## Hypothesis H2 — executable residual alpha

For every target observation, each model produces a settlement probability.
The all-in predicted edge is computed against the displayed ask on YES and NO
using the exact q15 taker-fee rounding convention.

Frozen policy grid:

- decision minute-left: 12, 11, or 10
- minimum predicted all-in edge: 2c, 4c, 6c, or 8c
- maximum one selected market per close window
- choose the candidate with the largest predicted all-in edge
- IOC at displayed ask; no retry/chase; hold to settlement

The best peer-model configuration is selected on VALIDATION only, requiring at
least 40 validation trades. The sealed TEST must satisfy all of:

1. positive total and per-contract edge;
2. positive day-bootstrap lower 95% bound;
3. positive 30/60/90-minute moving-block lower bounds;
4. peer policy beats the own-path policy and market-only policy;
5. positive after removing any one coin, week, or close-minute label;
6. predicted-edge deciles are economically monotone.

## Hypothesis H3 — low-variance cross-coin dominance pair

At the selected decision minute, construct ordered pairs:

- buy YES on coin i;
- buy NO on coin j;
- one pair maximum per close window.

The pair pays 1 when both outcomes agree, 2 when i=YES/j=NO, and 0 only when
i=NO/j=YES. Model edge is predicted pair payoff minus both asks and fees.

Frozen threshold grid: 2c, 4c, 6c, 8c predicted pair edge. Select on
VALIDATION; evaluate TEST once. A PASS requires positive day/block lower bounds
and lower maximum drawdown than the corresponding single-leg peer policy.

## Hard failure conditions

The audit fails if:

- peer features do not improve both validation and test information metrics;
- economic residuals are not monotone;
- the peer policy does not beat simpler baselines on sealed TEST;
- a result is carried by one coin/week/minute;
- or realistic asks/fees eliminate the value.

No threshold, coin, minute, or feature may be added after seeing results.
