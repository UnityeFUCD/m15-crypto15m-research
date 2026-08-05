"""What did EVERY filter we added actually throw away?

THE QUESTION
  Five mechanisms currently discard entries, and only two have ever been
  measured against outcomes:

      volume_filter_skip       7304   MIN_VOLUME=2000
      dead_entry_skip          1258   the dead-entry cuts (now disabled)
      drawdown_envelope_block   506   our own risk envelope
      gross_cap                 174   MAX_GROSS
      shadow_band               132   entries above LIVE_HI=0.80
      direction_disagree_skip    26   the agreement rule

  Each was added for a reason that sounded good. None was checked for the
  thing that actually matters: whether the entries it removes were WINNERS.

  This matters more than it looks. Today established that the maker already
  loses money by systematically failing to fill winners - 84.3% of winners fill
  against 96.2% of losers. If our own filters are ALSO biased against winners,
  that is a second helping of the same problem, and it is entirely
  self-inflicted.

THE ONE I EXPECT TO BE WORST
  drawdown_envelope_block. It fires when equity is DOWN, which is exactly when
  the runner has just taken losses. If losses cluster in bad windows and the
  envelope then blocks the following windows, it is systematically standing
  aside during the recovery. That is a plausible mechanism for a filter to be
  anti-correlated with returns, and nothing about its design prevents it.

THE BAR, unchanged
  A filter earns its keep only if what it discards is verifiably NEGATIVE. Not
  smaller than what we keep - negative. Anything discarding positive-edge
  entries is costing money and should come off.
"""
import glob
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from acct import api

LIVE = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261057)
MAXFETCH = 260          # per filter, to keep the API load modest


def ts(x):
    try:
        return datetime.fromisoformat((x or "").replace("Z", "+00:00"))
    except Exception:
        return None


EV = {}
for fp in sorted(glob.glob("lsm_*.jsonl")) + sorted(glob.glob("lsm_*.log")):
    try:
        for ln in open(fp, errors="ignore"):
            ln = ln.strip()
            if not ln.startswith("{"):
                continue
            try:
                d = json.loads(ln)
            except Exception:
                continue
            e, t = d.get("ev"), d.get("ticker")
            if not t:
                continue
            if e in ("drawdown_envelope_block", "dead_entry_skip",
                     "volume_filter_skip", "shadow_band",
                     "direction_disagree_skip"):
                EV.setdefault(e, {}).setdefault(t, d)
    except Exception:
        pass
for e in sorted(EV):
    print(f"  {e:>26}: {len(EV[e]):>5} distinct markets")

# outcomes, batched
RESULT = {}


def load_results(tickers):
    todo = [t for t in tickers if t not in RESULT]
    for i in range(0, len(todo), 20):
        c, j = api("/markets", {"tickers": ",".join(todo[i:i + 20])})
        for m in ((j or {}).get("markets") or []):
            if m.get("result") in ("yes", "no"):
                RESULT[m["ticker"]] = m["result"]
        time.sleep(0.03)


def candle_price(ticker, coin, close, minute):
    """Favourite side and bid at the logged minute, from the book."""
    cts = int(close.timestamp())
    try:
        c, j = api(f"/series/KX{coin}15M/markets/{ticker}/candlesticks",
                   {"period_interval": 1, "start_ts": cts - 16 * 60,
                    "end_ts": cts})
    except Exception:
        return None
    best = None
    for k in ((j or {}).get("candlesticks") or []):
        ml = (cts - int(k.get("end_period_ts") or 0)) / 60.0
        if abs(ml - minute) > 0.6:
            continue
        try:
            yb = float(k["yes_bid"]["close_dollars"])
            ya = float(k["yes_ask"]["close_dollars"])
        except (KeyError, TypeError, ValueError):
            continue
        if yb <= 0 or ya >= 1 or yb >= ya:
            continue
        best = ("yes", yb) if yb >= 0.5 else ("no", 1.0 - ya)
    return best


def close_from_ticker(t):
    return (pd.to_datetime(t.split("-")[1], format="%y%b%d%H%M", utc=True)
            + pd.Timedelta(hours=4))


def audit(name, price_fn, needs_book=False):
    d = EV.get(name, {})
    if not d:
        print(f"\n  {name}: no events")
        return None
    keys = sorted(d)
    if len(keys) > MAXFETCH:
        keys = list(RNG.choice(keys, MAXFETCH, replace=False))
    load_results(keys)
    rows = []
    for t in keys:
        res = RESULT.get(t)
        if res is None:
            continue
        rec = d[t]
        coin = t.split("-")[0].replace("KX", "").replace("15M", "")
        if needs_book:
            got = candle_price(t, coin, close_from_ticker(t),
                               float(rec.get("min_left") or 11))
            if not got:
                continue
            side, px = got
        else:
            got = price_fn(rec)
            if not got:
                continue
            side, px = got
        if not (0.60 <= px < 0.90):
            continue
        won = 1 if side == res else 0
        rows.append(dict(ticker=t, coin=coin, side=side, px=px, won=won,
                         edge=won - px, window=t.split("-")[1]))
    if len(rows) < 15:
        print(f"\n  {name}: only {len(rows)} resolved - too few")
        return None
    R = pd.DataFrame(rows)
    ws = sorted(R.window.unique())
    gw = {w: g for w, g in R.groupby("window")}
    bs = np.sort(np.array([
        pd.concat([gw[ws[i]] for i in RNG.integers(0, len(ws), len(ws))]).edge.mean() * 100
        for _ in range(6000)]))
    e = R.edge.mean() * 100
    print(f"\n  {name}")
    print(f"    n {len(R)}   avg px {R.px.mean()*100:.1f}c   "
          f"win {R.won.mean():.4f}   edge {e:+.2f}c/contract")
    print(f"    window-clustered 95% CI [{bs[150]:+.2f}, {bs[5849]:+.2f}]   "
          f"P(discards POSITIVE) {(bs > 0).mean():.4f}")
    verdict = ("EARNS ITS KEEP" if bs[5849] < 0 else
               "COSTING MONEY" if bs[150] > 0 else "UNRESOLVED")
    print(f"    -> {verdict}")
    return dict(name=name, n=len(R), edge=e, lo=bs[150], hi=bs[5849],
                verdict=verdict)


print("\n" + "=" * 78)
print("WHAT EACH FILTER DISCARDED")
print("=" * 78)
out = []
for nm, fn, book in (
    ("drawdown_envelope_block",
     lambda r: (("yes" if float(r.get("price") or 0) >= 0.5 else "no"),
                float(r["price"])) if r.get("price") else None, False),
    ("shadow_band",
     lambda r: (r.get("side"), float(r["fav_bid"])) if r.get("fav_bid") else None,
     False),
    ("direction_disagree_skip",
     lambda r: (r.get("side"), float(r["fav_bid"])) if r.get("fav_bid") else None,
     False),
    ("dead_entry_skip", None, True),
    ("volume_filter_skip", None, True),
):
    try:
        res = audit(nm, fn, book)
        if res:
            out.append(res)
    except Exception as ex:
        print(f"\n  {nm}: failed - {ex}")

print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"{'filter':>26} {'n':>6} {'discarded edge':>16} {'verdict':>16}")
for r in out:
    print(f"{r['name']:>26} {r['n']:>6} {r['edge']:>+15.2f}c "
          f"{r['verdict']:>16}")
print("\n  Anything reading COSTING MONEY is discarding positive-edge entries")
print("  and should come off. Anything UNRESOLVED needs more data before it is")
print("  touched - the default is to leave a validated filter alone.")
