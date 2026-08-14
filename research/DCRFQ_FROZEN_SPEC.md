# DCRFQ Frozen Specification

## Name

**DCRFQ: Deterministic Cover RFQ Engine**

DCRFQ is a generalized payoff-dominance engine for Kalshi multivariate-event
(MVE) markets.  It does not forecast which leg will win.  It constructs a
portfolio whose terminal payout is at least the payout owed on the accepted
combo side in every admissible settlement state.

HBCD (selected component YES + target combo NO) is the simplest one-hedge
instance of DCRFQ.

## Canonical payout representation

For a target combo with selected leg set \(T\), write each selected leg's
settlement value as \(x_i\in[0,1]\).  The target YES payout is

\[
C_T=\prod_{i\in T}x_i.
\]

Every hedge instrument is represented by the exact set of selected legs in its
MVE metadata.  Titles, rounded labels, and inferred event relationships are not
admitted.

An atomic selected-leg YES contract is represented by a singleton set.  A
subset combo \(S\subseteq T\) pays

\[
C_S=\prod_{i\in S}x_i.
\]

## Arm A: requester buys target YES; maker acquires target NO

The maker owes the economics of target NO, \(1-C_T\).  A hedge portfolio must
pay at least \(C_T\).

For every nonempty subset \(S\subseteq T\),

\[
C_S\ge C_T.
\]

Therefore one unit of any subset YES plus one unit of target NO pays at least
one dollar:

\[
C_S+(1-C_T)\ge1.
\]

For quantity \(q\), depth may be split across multiple admissible subset-YES
instruments.  If nonnegative hedge quantities \(h_j\) satisfy

\[
\sum_j h_j=q,
\]

then

\[
\sum_j h_jC_{S_j}+q(1-C_T)\ge q.
\]

The cheapest causal hedge is obtained by sorting all sequence-valid subset-YES
asks by all-in cost and consuming depth until total hedge quantity equals the
accepted target quantity.

Special cases:

- singleton subset: original HBCD component hedge;
- proper subset combo: nested-combo dominance;
- identical selected-leg set in another market: duplicate-payoff lock.

## Arm B: requester buys target NO; maker acquires target YES

The maker must hedge \(1-C_T\).  Let \(S_1,\ldots,S_m\subseteq T\) be a set
cover of \(T\):

\[
\bigcup_{j=1}^{m}S_j=T.
\]

Buying one unit of NO on every selected subset pays

\[
\sum_{j=1}^{m}(1-C_{S_j}).
\]

Because

\[
\prod_{j=1}^{m}C_{S_j}\le C_T
\]

when the subsets cover \(T\), and because

\[
1-\prod_{j=1}^{m}z_j\le\sum_{j=1}^{m}(1-z_j)
\quad\text{for }z_j\in[0,1],
\]

we have the statewise guarantee

\[
C_T+\sum_{j=1}^{m}(1-C_{S_j})\ge1.
\]

The first implementation uses an **integer set cover** only.  Fractional covers
are excluded until separately proved and execution-tested.  The optimizer
minimizes total all-in NO ask cost subject to every exact target leg being
covered at least once.

Special cases:

- all singleton subsets: combo YES plus NO on every component;
- one proper subset combo plus the remaining atomic legs;
- several disjoint or overlapping subset-combo NO hedges.

## Admissible hedge instruments

A hedge instrument is admissible only when all of the following hold:

1. Its exact selected-leg tuples are available from official market/RFQ
   metadata.
2. Its leg set is a nonempty subset of the target's exact leg set.
3. The target and hedge use the normal MVE product settlement rule.
4. Every relevant market is open and not paused.
5. Its order book snapshot and deltas are sequence-valid.
6. The economic acquisition ask and displayed quantity are positive.
7. The book age, price grid, fee override, and risk state pass frozen gates.
8. The same displayed quantity is not allocated to two simultaneous accepted
   targets.

No semantic implication is inferred from names.  Threshold, deadline, or
mutual-exclusion implications require a separate rule-signature verifier.

## Quote economics

### Arm A

For accepted quantity \(q\), let the all-in causal subset-YES hedge cost be
\(H_A(q)\), including fees, rounding, and depth walking.  The maker's target-NO
bid \(n\) is admissible only when

\[
q(1-n)-H_A(q)-F_{target}(q,n)\ge M(q),
\]

where \(M(q)\) is the frozen minimum guaranteed profit.

Equivalently, the requester-facing target-YES price is \(1-n\).

### Arm B

Let \(H_B(q)\) be the all-in cost of buying \(q\) units of every member of the
minimum-cost integer NO-cover.  The maker's target-YES bid \(y\) is admissible
only when

\[
q(1-y)-H_B(q)-F_{target}(q,y)\ge M(q).
\]

All fee calculations use the actual market fee override when available and
cash rounding at executed quantity.  Missing fee metadata fails closed.

## Frozen execution state machine

1. Receive authenticated `rfq_created` and exact MVE legs.
2. Build the exact subset lattice from currently known atomic and combo
   markets.
3. Read sequence-valid books and solve both arms using only locally received
   states.
4. Quote only the arm with the larger guaranteed net value; the other bid is
   zero.
5. A quote is sent only when the complete hedge was genuinely executable at
   quote time.
6. On `quote_accepted`, freeze accepted side and quantity, refresh every hedge
   book, and recompute the proof and all-in margin.
7. Submit the complete hedge using FOK orders.  Arm A may split quantity across
   several markets; Arm B may require a batch of cover orders.
8. Confirm the combo quote only after every required hedge fill is authoritative
   and enough confirmation time remains.
9. If any hedge fails, do not confirm.
10. If any hedge fills but the target quote does not execute, record every
    orphan position and emergency-unwind loss.
11. Reconcile private order IDs, quote ID, RFQ ID, fill IDs, fees, settlement,
    and final cash PnL.

The confirmation window is risk control, not an option to issue insincere
quotes.  Repeated acceptance without confirmation is a strategy failure.

## Mechanical acceptance tests

The proof library must pass all of these before any market test:

1. Exhaustive scalar-grid verification for Arm A.
2. Exhaustive scalar-grid verification for every integer set cover in Arm B on
   target sizes up to seven legs.
3. Random continuous verification on larger targets.
4. Duplicate and overlapping subset-cover tests.
5. Invalid non-cover counterexamples must be detected.
6. Side-semantic tests for requester YES/NO versus maker NO/YES bids.
7. Price-grid, fee-rounding, and depth-allocation conservation tests.
8. No unit of book depth may be reused across simultaneous locks.

## Economic evidence ladder

- **Mechanical PASS:** statewise payout inequalities and optimizer invariants
  pass.
- **Tier C contestability:** a causal deterministic quote was strictly better
  than a later public trade, with conservative public hedge prices.  Not a fill.
- **Tier B shadow:** authenticated RFQs and sequence-valid live hedge books show
  the quote was executable at decision and acceptance time.  No order sent.
- **Tier A q=1:** actual quote, acceptance, full hedge FOK, confirmation,
  execution, fees, settlement, and cash PnL are joined.
- **Capacity certified:** every quantity rung is scored with accepted RFQs,
  failed hedges, orphan losses, depth walking, and capital occupancy included.

## Frozen promotion gates

1. Mechanical suite passes with zero exceptions.
2. At least 500 authenticated eligible RFQs are shadow-scored, including every
   skip and terminal state.
3. q=1 live: at least 100 accepted opportunities and 30 completed locks.
4. The lower 95% confidence bound on net dollars per accepted opportunity is
   positive after orphan hedges and emergency exits.
5. Scale sequentially through q=1, 2, 5, 10, 20, 50.  Every rung must pass before
   the next begins.
6. Annual profit claims use actual accepted RFQ frequency and contemporaneous
   hedge depth, never public print volume or total RFQ size.
7. Coin, category, collection, requester, day, and largest-lock concentration
   are reported.

## $80K throughput requirement

An $80,000 calendar-year target requires approximately $219.18 net per day.
Required completed locked contracts per day are:

- 1 cent net: 21,918
- 2 cents net: 10,959
- 5 cents net: 4,384
- 10 cents net: 2,192
- 20 cents net: 1,096

DCRFQ is not an $80K engine until Tier-A data demonstrates this combination of
accepted flow, hedge depth, confirmation reliability, capital turnover, and
net margin.
