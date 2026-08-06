"""Run the one-sided reward audit with the fast filtered data loader."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research import reward_adjusted_commodity15m_onesided as audit
from research import commodity15m_reward_data_fast as data

audit.load_programs = data.load_programs
audit.fetch_markets = data.fetch_markets
audit.fetch_candles = data.fetch_candles

if __name__ == "__main__":
    audit.main()
