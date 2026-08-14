# HBCD Frozen Specification

## Name

**HBCD: Hedge-Before-Confirm Dominance Lock**

This is a deterministic cross-market lock between one selected component leg of a Kalshi multivariate-event (combo) market and the complementary side of the combo.

## Mechanical identity

Let a combo YES settlement be

\[
C=\prod_{i=1}^{k}L_i,\qquad L_i\in[0,1],
\]

where \(L_i\) is the settlement value of selected leg \(i\). For any selected leg \(j\),

\[
C\le L_j.
\]

Therefore one long selected leg \(L_j\) plus one long combo NO has terminal value

\[
L_j+(1-C)=1+L_j-C\ge1.
\]

The inequality is pathwise and does not require a probabilistic forecast. It also holds when one or more component markets settle to scalar values in \([0,1]\).

If the selected-leg acquisition cost is \(a\), the combo-NO cost is \(n\), and all fees/rounding/slippage are \(f\), the minimum terminal profit is

\[
\pi_{\min}=1-a-n-f.
\]

Writing the requester-facing combo YES price as \(y=1-n\),

\[
\pi_{\min}=y-a-f.
\]

A quote is admissible only when \(\pi_{\min}\) clears the frozen minimum margin.

## Why the RFQ state machine matters

Combo RFQs use a two-stage execution lock:

1. The requester accepts a maker quote.
2. The maker has the high-volatility-market confirmation window to confirm.
3. Only after confirmation does the execution timer begin.

The strategy does **not** pre-hedge every quote. It waits for `quote_accepted`, buys the exact selected-leg quantity with a FOK order, and confirms the combo quote only after a full hedge fill. If the hedge does not fill, the quote is not confirmed.

This turns acceptance into a conditional commitment opportunity. It does **not** remove all risk: a filled hedge followed by a failed confirmation is residual execution risk and must be measured explicitly.

## Frozen public-history screen

The public-history audit is a **contestability screen**, not an execution backtest.

### Causal inputs

At the combo market's public `created_time` proxy:

- selected component legs are read from `mve_selected_legs`;
- the component acquisition price is the conservative immediate-ask proxy from the last completed one-minute candlestick;
- no future combo trade is used to set the quote;
- a requester-facing YES quote is set to the lowest valid tick that clears all frozen fee stress and margin.

### Primary cell

- Market universe: crypto-only combo markets whose selected legs are from BTC, ETH, SOL, XRP, DOGE, HYPE, or BNB 15-minute series.
- Direction: acquire combo NO and one selected component leg.
- Quote latency from public market creation proxy: **1.0 second**.
- Quantity: **q=1**.
- Minimum guaranteed profit after fee stress: **1.0 cent per contract**.
- Fee stress: both acquisitions charged the taker quadratic rate plus the maximum sub-cent rounding charge per order. This intentionally overcharges the combo RFQ leg.
- Price evidence: a later public taker-YES trade must execute at least one full valid tick worse for the requester than our frozen quote.
- Component price freshness: no more than 90 seconds between the completed candle and the decision proxy.
- Capacity allocation: one selected combo per component market, choosing the greatest guaranteed margin. This prevents the same component contract from being counted as a hedge for multiple combo contracts.
- Missing candle, price, trade, quantity, or timing data is an explicit non-certificate, never a zero-cost fill.

### Sensitivity only

No sensitivity row may replace the primary cell after results are observed.

- Latency: 0.25, 0.50, 1.0, 2.0, 3.0, 5.0 seconds.
- Minimum profit: 0.5, 1.0, 2.0, 3.0 cents.
- Quantity caps: 1, 5, 10, 20, 50.
- Quote comparison: strict one-tick improvement and weak tie-or-better.

## Evidence labels

- **Mechanical PASS:** the settlement inequality is verified algebraically and by exhaustive grid checks.
- **Contestable:** the causal quote would have been strictly better than a later public trade, subject to the public market-creation and candle proxies.
- **Tier C shadow:** public trade/candlestick evidence only; no private RFQ or hedge lifecycle.
- **Tier A execution:** authenticated RFQ receipt, quote, acceptance, component L2, FOK hedge response/fill, quote confirmation, quote execution, fees, settlement, and cash PnL are joined on one clock.

No public-history result may be called a fill, realized PnL, or executable capacity.

## Frozen q=1 prospective state machine

1. Receive `rfq_created` on the authenticated communications WebSocket.
2. Verify combo metadata, selected legs, price grid, RFQ size, available balance, and risk limits.
3. Read sequence-valid component orderbooks. For each selected leg:
   - YES acquisition ask = \(1-\) best NO bid.
   - NO acquisition ask = \(1-\) best YES bid.
4. Select the cheapest fully hedgeable leg and calculate the exact maximum combo-NO cost.
5. Send a quote declining the combo-YES acquisition side and bidding for combo NO only.
6. On `quote_accepted`, re-read the component book.
7. Send q=1 component FOK at the frozen maximum hedge price.
8. Confirm the quote only after a full component fill is acknowledged and while a conservative confirmation deadline remains.
9. If the hedge fails, do not confirm.
10. If the hedge fills but confirmation fails, enter the frozen emergency-unwind path and record the complete loss.
11. Reconcile `quote_executed`, private fills, fee fields, settlements, and final cash PnL.

## Promotion gates

The strategy may advance only in this order:

1. Mechanical identity and side-semantics tests pass.
2. Public contestability screen has at least 30 independent days and positive day-clustered lower bound.
3. Authenticated shadow collector records at least 500 eligible RFQs with no missing terminal states.
4. q=1 live test records at least 100 accepted opportunities and at least 30 executed locks.
5. The lower 95% confidence bound of net dollars per accepted opportunity is positive after charging all orphan-hedge losses.
6. Scale q through 1, 2, 5, 10, 20 only after each prior rung passes.
7. Capacity is measured from exact contemporaneous component depth and RFQ accepted size, not public trade volume.

## Profit-target arithmetic

An $80,000 annual target requires approximately $219.18 per calendar day. Required daily locked contracts are:

- 1 cent net: 21,918 contracts/day
- 2 cents net: 10,959 contracts/day
- 5 cents net: 4,384 contracts/day
- 10 cents net: 2,192 contracts/day

These are throughput requirements, not forecasts. The strategy is an $80K engine only if Tier-A data demonstrates sufficient accepted RFQ flow, hedge depth, confirmation reliability, and capital turnover.
