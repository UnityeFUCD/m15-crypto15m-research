"""Run Reward-Funded Hedged Liquidity with the fast filtered loader."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research import reward_hedged_liquidity as audit
from research import commodity15m_reward_data_fast as data

audit.load_programs = data.load_programs
audit.fetch_markets = data.fetch_markets
audit.fetch_candles = data.fetch_candles

if __name__ == "__main__":
    audit.main()
