"""Export the full historical account record and the CF index series.

Produces, into data/ and data_csv/:
  orders_history          every order ever placed, fully paginated
  fills_history           every fill, with is_taker and fee
  cf_index                the CF Benchmark BRTI as a continuous series per coin
  fill_model.json         the measured fill model, with its provenance

Nothing here writes a credential anywhere. Reads them from KALSHI_CRED_DIR.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture.kalshi import KalshiClient          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, CSV = ROOT / "data", ROOT / "data_csv"
DATA.mkdir(exist_ok=True)
CSV.mkdir(exist_ok=True)


def dump(df: pd.DataFrame, name: str):
    df.to_parquet(DATA / f"{name}.parquet", index=False)
    df.to_csv(CSV / f"{name}.csv", index=False)
    print(f"  {name:22} {len(df):>7,} rows  "
          f"{(DATA / f'{name}.parquet').stat().st_size/1048576:5.2f} MB")


def main():
    api = KalshiClient()

    print("orders (full pagination)...")
    orders = list(api.paginate("/portfolio/orders", "orders"))
    if orders:
        dump(pd.DataFrame(orders), "orders_history")

    print("fills (full pagination)...")
    fills = list(api.paginate("/portfolio/fills", "fills"))
    if fills:
        dump(pd.DataFrame(fills), "fills_history")

    # ---- CF index as a continuous series -------------------------------
    # a1 of a window and a0 of the next are the same instant, so stringing
    # the pairs together gives the index at every 15-minute boundary.
    print("CF Benchmark index series...")
    u = pd.read_parquet(DATA / "underlying.parquet")
    u["close_utc"] = (pd.to_datetime(u.wkey, format="%y%b%d%H%M", utc=True)
                      + pd.Timedelta(hours=4))
    rows = []
    for coin, g in u.sort_values("close_utc").groupby("coin"):
        g = g.reset_index(drop=True)
        for r in g.itertuples():
            rows.append({"coin": coin, "ts_utc": r.close_utc - pd.Timedelta(minutes=15),
                         "index_value": r.a0, "point": "window_open"})
            rows.append({"coin": coin, "ts_utc": r.close_utc,
                         "index_value": r.a1, "point": "window_close"})
    cf = (pd.DataFrame(rows).drop_duplicates(subset=["coin", "ts_utc"])
          .sort_values(["coin", "ts_utc"]).reset_index(drop=True))
    cf["ret"] = cf.groupby("coin").index_value.pct_change()
    dump(cf, "cf_index")

    # ---- fill model ----------------------------------------------------
    print("fill model...")
    model = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": "measured on live orders, cross-checked on 73 days of book data",
        "n_live_orders": 120,
        "model": {
            "p_fill_given_loser": 0.962,
            "p_fill_given_winner": 0.843,
            "derivation": (
                "Reachability split on 120 resolved live orders: 53 markets "
                "whose price returned to our bid filled 51/53 = 0.962 and won "
                "0.5472; 67 whose price ran away filled 53/67 = 0.791 and won "
                "1.0000. P(reachable|won) = 53*0.5472/(53*0.5472+67) = 0.302, "
                "P(reachable|lost) = 1.0. Hence "
                "P(fill|win) = 0.302*0.962 + 0.698*0.791 = 0.843."),
            "validation": (
                "Predicts a realised win rate of 0.736*0.894/0.922 = 0.714 "
                "against 0.7063 observed on 269 settled fills."),
        },
        "consequences": {
            "backtest_overstatement_preshift": 0.276,
            "backtest_overstatement_postshift": 0.589,
            "note": ("Any backtest assuming 100% fill overstates edge by these "
                     "fractions. Population +3.65c becomes realised +1.54c."),
        },
        "ladder": {
            "note": ("fill|LOSE is 1.0000 at EVERY posting price tested. The "
                     "loss side is fully adversely selected and pricing cannot "
                     "change it; price only alters how many winners are caught."),
            "fill_given_winner_by_offset_ticks": {
                "0": 0.8562, "1": 0.9031, "2": 0.9531, "3": 0.9843},
            "source": "quote_ladder2.py over 73 days, wide-spread subset",
        },
        "caveats": [
            "Estimated from 120 orders; P(fill|win) 95% CI is [0.7703, 0.9100].",
            "All 608 historical orders used ONE policy (join the bid), so this "
            "model is only identified at that policy. Other prices are "
            "simulated from candle touches, not observed.",
            "queue_position_fp was never recorded; queue here is displayed size.",
        ],
    }
    (ROOT / "fill_model.json").write_text(json.dumps(model, indent=2))
    print(f"  fill_model.json        {len(json.dumps(model)):>7,} bytes")


if __name__ == "__main__":
    main()
