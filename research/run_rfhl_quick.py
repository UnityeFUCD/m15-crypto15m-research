"""Seven-day rapid discovery pass for Reward-Funded Hedged Liquidity.

This is intentionally a discovery pass. The 21-day audit remains the formal
confirmation run.

The public liquidity-program rules impose a $1.00 minimum payout and round
payments down to the nearest cent. This runner applies that rule per incentive
program before any policy is evaluated. Earlier fractional-reward runs are not
valid for a production claim.
"""
from pathlib import Path
import math
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research import reward_hedged_liquidity as audit
from research import commodity15m_reward_data_fast as data

# Fast discovery window only. Formal confirmation keeps the longer history.
data.LOOKBACK_DAYS = 7

# One-sided score must be large enough to have a realistic chance of clearing
# the official $1 minimum under maximum target-size competition. Quantities
# below this range can be useful for telemetry but cannot support the modeled
# reward economics.
audit.QTY_GRID = [16, 20, 32, 40, 50, 75, 100]


def minimum_payout_size_states(frame, qty):
    """Conservative reward accounting with block fee recomputation.

    - Competing score assumes both sides are filled to target size for the
      entire period.
    - Program reward is paid only when modeled entitlement is at least $1.
    - Eligible rewards are rounded down to the nearest cent.
    - Trading P&L uses the original immediate-complement hedge and exact
      block-rounded taker fee for the requested quantity.
    """
    x = frame.copy()
    own_score_time = qty * x["rest_seconds"]
    competitor_score_time = 2.0 * x["target_size"] * x["period_seconds"]
    share = own_score_time / (competitor_score_time + own_score_time)
    raw_reward = x["period_reward_dollars"] * share
    x["reward_pnl_raw"] = raw_reward
    x["reward_pnl"] = np.where(
        raw_reward >= 1.0,
        np.floor(raw_reward * 100.0 + 1e-12) / 100.0,
        0.0,
    )

    fees = np.zeros(len(x), dtype=float)
    filled = x["maker_filled"] & x["hedge_price"].notna()
    if filled.any():
        fees[filled.to_numpy()] = [
            audit.taker_fee_total(qty, float(price))
            for price in x.loc[filled, "hedge_price"]
        ]
    x["hedge_fee"] = fees
    x["trading_pnl"] = np.where(
        x["maker_filled"],
        qty * (1.0 - x["maker_price"] - x["hedge_price"])
        - x["hedge_fee"],
        0.0,
    )
    x["combined_pnl"] = x["reward_pnl"] + x["trading_pnl"]
    x["qty"] = qty
    x["reward_cost"] = x["reward_pnl"] / np.maximum(
        -x["trading_pnl"], 0.0001
    )
    x["reward_density"] = (
        x["period_reward_dollars"] / x["target_size"]
    )
    x["reward_share_lower"] = share
    return x


audit.load_programs = data.load_programs
audit.fetch_markets = data.fetch_markets
audit.fetch_candles = data.fetch_candles
audit.size_states = minimum_payout_size_states

if __name__ == "__main__":
    audit.main()
