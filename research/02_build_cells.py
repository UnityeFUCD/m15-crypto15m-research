"""STAGE 3+4 - settlement/clock audit, then reproduce with honest denominators.

Builds ONE aggregate table at (window, 5-second bin, 5-cent maker-price bin)
granularity, carrying enough to compute every denominator Stage 4 requires and
every clustering Stage 5 requires, without ever holding 11M per-trade rows.

CLOCK, stated explicitly because the protocol warns against conflating them
  contract close T        parsed from the ticker; ticker time is ET = UTC-4
  A1 averaging window     [T-59, T]  -> settlement information STARTS
                          accumulating 60s before close, not at close
  A0 averaging window     [T-900-59, T-900]
  seconds_to_close        T - trade timestamp (exchange `created_time`, µs)
  seconds_into_A1         60 - seconds_to_close, when positive
Only exchange timestamps exist in this tape; no local receipt time, so
submission latency is NOT measurable here and is not modelled.

FAVOURITE is defined ONLY from the contemporaneous traded price
(maker_px > $0.50). Settlement is never used to define it.

GUARD BANDS on |A1-A0| are carried per window so Stage 3 can exclude
near-tie settlements without recomputing.
"""
import calendar
import glob
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_s = importlib.util.spec_from_file_location(
    "ta", Path(__file__).resolve().parent / "01_trade_accounting.py")
ta = importlib.util.module_from_spec(_s)
_s.loader.exec_module(ta)

ROOT = Path("C:/Users/fycin/m15-claude-independent")
OUT = ROOT / "research/final_minute_favorite_maker_validation"
H, W = 60, 900
LOOK = int(os.environ.get("LOOK_S", "180"))
TBIN = 5
MON = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def parse_ts(ct):
    """Exchange timestamp -> float epoch seconds.

    Kalshi TRIMS TRAILING ZEROS from the fractional part, so the field appears
    as '...:27.5Z', '...:54.167Z', '...:59.940763Z' and even bare '...:59Z'.
    An earlier parser assumed exactly six fractional digits and silently
    discarded 621,647 trades - 10% of the tape, every trade whose microseconds
    ended in a zero. That defect was present in the original discovery run.
    """
    if not ct or len(ct) < 19:
        return None
    try:
        base = calendar.timegm(time.strptime(ct[:19], "%Y-%m-%dT%H:%M:%S"))
    except Exception:
        return None
    rest = ct[19:].rstrip("Zz")
    if not rest:
        return float(base)
    if not rest.startswith("."):
        return None
    frac = rest[1:]
    if not frac.isdigit():
        return None
    return base + int(frac) / (10.0 ** len(frac))


def close_utc(tk):
    p = tk.split("-")[1]
    return calendar.timegm((2000 + int(p[:2]), MON[p[2:5]], int(p[5:7]),
                            int(p[7:9]), int(p[9:11]), 0, 0, 0, 0)) + 4 * 3600


def second_grid(coin):
    fs = sorted(glob.glob(str(ROOT / f"data/sare/cf/{coin}_*.npz")),
                key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
    if not fs:
        return None, None
    h0 = int(os.path.basename(fs[0]).split("_")[1].split(".")[0])
    h1 = int(os.path.basename(fs[-1]).split("_")[1].split(".")[0]) + 3600
    g = np.full(h1 - h0, np.nan)
    for f in fs:
        z = np.load(f)
        t, v = z["t"], z["v"]
        s = (t // 1000).astype(np.int64) - h0
        ok = (s >= 0) & (s < len(g))
        if ok.any():
            g[s[ok]] = v[ok]
        z.close()
        del z, t, v, s, ok
    gc.collect()
    return g, h0


# official settlements from the validated frame, for the Stage 3 cross-check
sys.path.insert(0, str(ROOT))
from claude_frame_ext import load_wide
fr = load_wide(blocks=("A013", "E", "H"))
fr = fr[fr.source != "A013"]
fr = fr[np.isin(fr["y"].to_numpy(), (0, 1))]
OFFICIAL = dict(zip(fr.ticker.values, fr["y"].to_numpy().astype(int)))
print(f"official settlements available for {len(OFFICIAL):,} markets", flush=True)

rows = []
audit = defaultdict(int)
mismatch = []
for coin in ["BTC", "ETH", "SOL", "XRP", "DOGE"]:
    grid, h0 = second_grid(coin)
    if grid is None:
        audit[f"{coin}_no_index"] += 1
        continue
    files = sorted(glob.glob(str(ROOT / f"data/sare/trades/KX{coin}15M-*.json")))
    print(f"{coin}: {len(files)} windows", flush=True)
    for fi, fp in enumerate(files):
        tk = os.path.basename(fp)[:-5]
        audit["windows_seen"] += 1
        T = close_utc(tk)
        ci, oi = T - h0, T - W - h0
        if oi - H + 1 < 0 or ci >= len(grid):
            audit["excl_index_out_of_range"] += 1
            continue
        a1, a0 = grid[ci - H + 1:ci + 1], grid[oi - H + 1:oi + 1]
        if not np.isfinite(a1).all():
            audit["excl_A1_incomplete"] += 1
            continue
        if not np.isfinite(a0).all():
            audit["excl_A0_incomplete"] += 1
            continue
        A1, A0 = float(a1.mean()), float(a0.mean())
        y_cf = 1 if A1 >= A0 else 0
        y_off = OFFICIAL.get(tk)
        if y_off is None:
            audit["no_official_settlement"] += 1
        elif y_off != y_cf:
            audit["settlement_disagreement"] += 1
            mismatch.append({"ticker": tk, "official": int(y_off), "cf": y_cf,
                             "margin": A1 - A0})
        y = y_off if y_off is not None else y_cf
        margin_bp = (A1 - A0) / A0 * 1e4 if A0 else float("nan")
        day = time.strftime("%Y-%m-%d", time.gmtime(T))
        agg = defaultdict(lambda: [0, 0, 0, 0])   # trades, cnt, maker_net, maker_wins_cnt
        try:
            tr = json.loads(Path(fp).read_text())
        except Exception:
            audit["excl_unparseable"] += 1
            continue
        for t in tr:
            ts = parse_ts(t.get("created_time", ""))
            if ts is None:
                audit["trade_bad_timestamp"] += 1
                continue
            s = T - ts
            if not (0 < s <= LOOK):
                continue
            rec, why = ta.reconstruct(t)
            if rec is None:
                audit["trade_rejected_" + why.split("(")[0].strip()[:24]] += 1
                continue
            audit["trades_used"] += 1
            _, _, _, mn = ta.pnl(rec, y)
            maker_won = int((rec["maker_side"] == "yes") == bool(y))
            tb = int(s // TBIN)
            pb = rec["maker_px"] // 50_000          # 5-cent bin, 0..19
            k = (tb, pb)
            a = agg[k]
            a[0] += 1
            a[1] += rec["cnt"]
            a[2] += mn
            a[3] += rec["cnt"] if maker_won else 0
        for (tb, pb), a in agg.items():
            rows.append((coin, tk, day, T, y, margin_bp, tb, pb,
                         a[0], a[1], a[2], a[3]))
        if (fi + 1) % 300 == 0:
            print(f"   {fi+1}/{len(files)}  {len(rows):,} cells", flush=True)
    del grid
    gc.collect()

D = pd.DataFrame(rows, columns=["coin", "ticker", "day", "close_utc", "y",
                                "margin_bp", "tbin", "pbin", "trades",
                                "cnt_h", "maker_net_micros", "maker_win_cnt_h"])
D.to_parquet(OUT / "discovery_cells.parquet", index=False)
(OUT / "settlement_clock_audit.json").write_text(json.dumps(
    {"audit": dict(audit), "settlement_mismatches": mismatch[:50],
     "n_mismatches": len(mismatch), "lookback_s": LOOK, "tbin_s": TBIN},
    indent=1, default=str))

print("\n=== STAGE 3 AUDIT ===")
for k in sorted(audit):
    print(f"  {k:38s} {audit[k]:,}")
print(f"  settlement mismatches (official vs CF): {len(mismatch)}")
print(f"\n  windows retained: {D.ticker.nunique():,} of {audit['windows_seen']:,}")
print(f"  cells: {len(D):,}   trades used: {D.trades.sum():,}   "
      f"contracts: {D.cnt_h.sum()/100:,.0f}")
print(f"  coins: {dict(D.groupby('coin').ticker.nunique())}")
print(f"  days: {D.day.nunique()}   {D.day.min()} .. {D.day.max()}")
print(f"\n  wrote {OUT/'discovery_cells.parquet'}")
