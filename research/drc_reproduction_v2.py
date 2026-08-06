"""DRC-15 corrected reproduction. Both sigma definitions, full waterfall.

THE CORRECTION
  v1 computed  sigma4 = r_prev.shift(1).rolling(4).std()
  which is std(r[t-2], r[t-3], r[t-4], r[t-5]) - it EXCLUDES the return being
  normalised. The specification is

      sigma4 = std(r[t-1], r[t-2], r[t-3], r[t-4])
      z_up   = r[t-1] / (sigma4 + eps)

  i.e. r_prev is INSIDE its own denominator. Both are computed here:

      DRC_INCLUDED    sigma includes r_prev   <- the specified rule
      DRC_BACKGROUND  sigma excludes r_prev   <- v1, kept as a neighbouring
                                                 falsification test

  This is not a cosmetic difference. Including a value in its own sample
  standard deviation makes the ratio self-limiting: with n=4 the statistic
  cannot run away the way the background version can, so the tail selected by
  z >= 1.0 is a different population, not merely a shifted one.

CONTIGUITY
  Four returns require FIVE consecutive A0 observations, each exactly 900s
  apart. Enforced for both definitions (BACKGROUND needs six).

NO LEAKAGE EITHER WAY
  A0(t) is fixed at window open; entry is 1-7 minutes after open. Every input
  to both statistics is knowable at decision time.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "results"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260806)

COINS = ["DOGE", "ETH", "SOL", "XRP"]
EPS = 1e-12
Z = 1.0
LOOKBACK = 4
BAND = (0.65, 0.80)
# identical boundaries for EVERY dataset, not per-dataset row splits
SPLITS = {"train": ("2026-05-25", "2026-06-30"),
          "valid": ("2026-06-30", "2026-07-18"),
          "test":  ("2026-07-18", "2026-08-07")}


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(ROOT), text=True).strip()
    except Exception:
        return "unknown"


def sha(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def one_contract_fee(price: float) -> float:
    """Kalshi taker fee for ONE contract: ceil(0.07*p*(1-p)*100) cents."""
    return math.ceil(0.07 * 1 * price * (1 - price) * 100) / 100.0


# ---------------------------------------------------------------- signal
def build_signals():
    u = pd.read_parquet(DATA / "underlying.parquet")
    wf = {"raw_markets": len(u)}
    u["close_utc"] = (pd.to_datetime(u.wkey, format="%y%b%d%H%M", utc=True)
                      + pd.Timedelta(hours=4))
    u = u[u.result.isin(["yes", "no"])]
    u = u.dropna(subset=["a0", "a1"])
    u = u[(u.a0 > 0) & (u.a1 > 0)]
    u = u.sort_values(["coin", "close_utc"]).drop_duplicates(
        ["coin", "close_utc"])
    wf["valid_settled"] = len(u)
    u4 = u[u.coin.isin(COINS)]
    wf["four_coins"] = len(u4)

    parts = []
    for c, g in u.groupby("coin"):
        g = g.sort_values("close_utc").reset_index(drop=True)
        contig = (g.close_utc.diff().dt.total_seconds() == 900)
        g["r_prev"] = np.where(contig, g.a0 / g.a0.shift(1) - 1.0, np.nan)
        r = g.r_prev

        # DRC_INCLUDED: sigma over r[t-1..t-4] -> the CURRENT r_prev plus 3
        # prior. Needs 4 contiguous returns = 5 consecutive A0 points.
        g["sigma_inc"] = r.rolling(LOOKBACK).std(ddof=1)
        ok_inc = contig.rolling(LOOKBACK).min()
        g.loc[ok_inc != 1, "sigma_inc"] = np.nan
        g["z_inc"] = g.r_prev / (g.sigma_inc + EPS)

        # DRC_BACKGROUND: sigma over r[t-2..t-5], excluding r_prev.
        g["sigma_bg"] = r.shift(1).rolling(LOOKBACK).std(ddof=1)
        ok_bg = contig.shift(1).rolling(LOOKBACK).min()
        g.loc[(ok_bg != 1) | (~contig), "sigma_bg"] = np.nan
        g["z_bg"] = g.r_prev / (g.sigma_bg + EPS)
        parts.append(g)
    s = pd.concat(parts, ignore_index=True)
    s = s[s.coin.isin(COINS)]
    wf["has_z_included"] = int(s.z_inc.notna().sum())
    wf["has_z_background"] = int(s.z_bg.notna().sum())
    return s, wf


# ---------------------------------------------------------------- books
def book_from_ladder():
    """ONE candidate per ticker, at the FIRST eligible minute (14 -> 8)."""
    L = pd.read_parquet(DATA / "ladder_paths.parquet")
    wf = {"ladder_rows": len(L), "unique_tickers": L.ticker.nunique()}
    rows, no_entry, dup_check = [], 0, set()
    for r in L.drop_duplicates(subset=["ticker"]).itertuples():
        try:
            pth = json.loads(r.path)
        except Exception:
            continue
        # FIRST eligible = LARGEST minutes-left within [8,14]
        elig = sorted([k for k in pth if 8 <= k["ml"] <= 14],
                      key=lambda x: -x["ml"])
        if not elig:
            no_entry += 1
            continue
        e = elig[0]
        assert r.ticker not in dup_check, "duplicate ticker emitted"
        dup_check.add(r.ticker)
        yb, ya = e["bc"], e["ac"]
        if yb <= 0 or ya >= 1 or yb >= ya:
            continue
        fy = yb >= 0.5
        rows.append(dict(
            ticker=r.ticker, coin=r.coin, entry_ml=e["ml"],
            side="yes" if fy else "no",
            bid=yb if fy else 1.0 - ya, ask=ya if fy else 1.0 - yb,
            won=1 if (("yes" if fy else "no") == r.result) else 0,
            close_utc=pd.to_datetime(r.ticker.split("-")[1],
                                     format="%y%b%d%H%M", utc=True)
                      + pd.Timedelta(hours=4)))
    wf["no_eligible_minute"] = no_entry
    wf["one_row_per_ticker"] = True
    b = pd.DataFrame(rows)
    wf["valid_book"] = len(b)
    return b, wf


def book_from_premium():
    P = pd.read_parquet(DATA / "premium_history2.parquet")
    wf = {"premium_rows": len(P), "unique_tickers": P.ticker.nunique()}
    P = P.drop_duplicates(subset=["ticker"]).copy()
    P["close_utc"] = pd.to_datetime(P.close, utc=True)
    P = P.rename(columns={"px": "bid"})
    P["ask"] = np.nan          # premium_history2 has no ask
    wf["valid_book"] = len(P)
    mlcol = "mins_left" if "mins_left" in P.columns else (
        "ml" if "ml" in P.columns else None)
    if mlcol:
        P = P.rename(columns={mlcol: "entry_ml"})
    else:
        P["entry_ml"] = np.nan
    return P[["ticker", "coin", "side", "bid", "ask", "won",
              "close_utc", "entry_ml"]], wf


def waterfall(b, s, zcol, label, has_ask):
    w = {}
    w["1_book_rows"] = len(b)
    m = b.merge(s[["coin", "close_utc", zcol, "r_prev"]],
                on=["coin", "close_utc"], how="inner")
    w["2_joined_to_signal"] = len(m)
    m = m[m.coin.isin(COINS)]
    w["3_four_coins"] = len(m)
    m = m[m[zcol].notna() & np.isfinite(m[zcol])]
    w["4_valid_contiguous_history"] = len(m)
    m = m[m.side == "no"]
    w["5_no_favourite"] = len(m)
    m = m[(m.bid >= BAND[0]) & (m.bid < BAND[1])]
    w["6_bid_in_65_80"] = len(m)
    pre_z = m.copy()
    m = m[m[zcol] >= Z]
    w["7_z_ge_threshold"] = len(m)
    if has_ask:
        m = m[m.ask.notna() & (m.ask > m.bid) & (m.ask < 1.0)]
    w["8_valid_ask"] = len(m)
    w["9_final_candidates"] = len(m)
    print(f"\n  EXCLUSION WATERFALL - {label}")
    for k, v in w.items():
        print(f"    {k:<32} {v:>7,}")
    return m, pre_z, w


def stats(g, has_ask, tag):
    if not len(g):
        return None
    d = {"n": len(g), "win": g.won.mean(), "avg_bid": g.bid.mean()}
    if has_ask:
        d["avg_ask"] = g.ask.mean()
        d["fee"] = g.ask.map(one_contract_fee).mean()
        d["taker_edge"] = (g.won - g.ask - g.ask.map(one_contract_fee)).mean()
    d["maker_edge"] = (g.won - g.bid).mean()
    return d


def clustered_ci(g, keycol, valfn, n=6000):
    if len(g) < 10:
        return (np.nan, np.nan, np.nan)
    gd = {k: v for k, v in g.groupby(keycol)}
    ks = sorted(gd)
    out = []
    for _ in range(n):
        smp = pd.concat([gd[ks[i]] for i in RNG.integers(0, len(ks), len(ks))])
        v = valfn(smp)
        if v is not None and np.isfinite(v):
            out.append(v)
    a = np.sort(np.array(out))
    if not len(a):
        return (np.nan, np.nan, np.nan)
    return a[int(.025 * len(a))], a[int(.975 * len(a))], (a <= 0).mean()


def main():
    print("=" * 78)
    print("DRC-15 CORRECTED REPRODUCTION - both sigma definitions")
    print("=" * 78)
    s, wf_sig = build_signals()
    print("\n  SIGNAL CONSTRUCTION")
    for k, v in wf_sig.items():
        print(f"    {k:<32} {v:>8,}")

    print("\n  distribution of z under each definition (4 coins):")
    for col, lab in (("z_inc", "DRC_INCLUDED  "), ("z_bg", "DRC_BACKGROUND")):
        v = s[col].replace([np.inf, -np.inf], np.nan).dropna()
        print(f"    {lab} n {len(v):>7,}  median {v.median():+.3f}  "
              f"p95 {v.quantile(.95):+.3f}  max {v.max():+.3f}  "
              f"share>=1 {(v >= Z).mean():.4f}")
    print("\n  NOTE: including r_prev in its own sigma bounds the statistic.")
    print("  With n=4 the maximum attainable |z| is limited, so z>=1 selects a")
    print("  materially different (and larger) tail than the background form.")

    books = {}
    lb, wl = book_from_ladder()
    books["ladder_paths"] = (lb, True, wl)
    pb, wp = book_from_premium()
    books["premium_history2"] = (pb, False, wp)
    print("\n  BOOK SOURCES")
    for nm, (b, ask, w) in books.items():
        print(f"    {nm}: {w}")

    results = {}
    for zcol, zlab in (("z_inc", "DRC_INCLUDED"), ("z_bg", "DRC_BACKGROUND")):
        print("\n" + "=" * 78)
        print(f"{zlab}   (sigma {'INCLUDES' if zcol=='z_inc' else 'EXCLUDES'} r_prev)")
        print("=" * 78)
        for nm, (b, has_ask, _) in books.items():
            cand, pre_z, w = waterfall(b, s, zcol, f"{zlab} / {nm}", has_ask)
            base = pre_z[pre_z[zcol] < Z]
            if has_ask:
                base = base[base.ask.notna() & (base.ask > base.bid)]
            sc, sb = stats(cand, has_ask, "drc"), stats(base, has_ask, "base")
            if not sc or not sb:
                continue
            print(f"\n    RESULTS  {zlab} / {nm}")
            print(f"      DRC      n {sc['n']:>5}  win {sc['win']:.4f}  "
                  f"bid {sc['avg_bid']*100:.2f}c" +
                  (f"  ask {sc['avg_ask']*100:.2f}c  fee {sc['fee']*100:.2f}c"
                   f"  TAKER {sc['taker_edge']*100:+.2f}c" if has_ask else
                   f"  MAKER {sc['maker_edge']*100:+.2f}c"))
            print(f"      non-DRC  n {sb['n']:>5}  win {sb['win']:.4f}  "
                  f"bid {sb['avg_bid']*100:.2f}c" +
                  (f"  ask {sb['avg_ask']*100:.2f}c  fee {sb['fee']*100:.2f}c"
                   f"  TAKER {sb['taker_edge']*100:+.2f}c" if has_ask else
                   f"  MAKER {sb['maker_edge']*100:+.2f}c"))
            lift = (sc["win"] - sb["win"]) * 100
            print(f"      win lift {lift:+.2f} pp")

            cand = cand.copy()
            cand["day"] = cand.close_utc.dt.date
            cand["cw"] = cand.close_utc
            metric = ((lambda x: (x.won - x.ask - x.ask.map(one_contract_fee)).mean() * 100)
                      if has_ask else (lambda x: (x.won - x.bid).mean() * 100))
            for keyc, kl in (("day", "day-clustered"),
                             ("cw", "close-window-clustered")):
                lo, hi, p0 = clustered_ci(cand, keyc, metric)
                print(f"      {kl:<24} 95% CI [{lo:+.2f}, {hi:+.2f}]  "
                      f"P(<=0) {p0:.4f}")

            print(f"      chronological (identical boundaries all datasets):")
            for sp, (a, bnd) in SPLITS.items():
                sl = cand[(cand.close_utc >= a) & (cand.close_utc < bnd)]
                slb = base[(base.close_utc >= a) & (base.close_utc < bnd)]
                if len(sl) < 5:
                    print(f"        {sp:<6} n {len(sl):>4}  (too few)")
                    continue
                v = metric(sl)
                vb = metric(slb) if len(slb) >= 5 else float("nan")
                print(f"        {sp:<6} n {len(sl):>4}  win {sl.won.mean():.4f}  "
                      f"DRC {v:+7.2f}c   non-DRC {vb:+7.2f}c")

            print(f"      per coin / leave-one-coin-out:")
            for c in COINS:
                only = cand[cand.coin == c]
                lo1 = cand[cand.coin != c]
                if len(only) >= 5:
                    print(f"        {c:<5} alone n {len(only):>4} "
                          f"{metric(only):+7.2f}c   "
                          f"excl n {len(lo1):>4} {metric(lo1):+7.2f}c")
            cand["week"] = cand.close_utc.dt.isocalendar().week
            wk = [(int(w_), metric(cand[cand.week != w_]))
                  for w_ in sorted(cand.week.unique())
                  if len(cand[cand.week != w_]) > 20]
            if wk:
                vals = [v for _, v in wk]
                print(f"      leave-one-week-out: min {min(vals):+.2f}c  "
                      f"max {max(vals):+.2f}c  ({len(wk)} weeks)")
            results[f"{zlab}|{nm}"] = {
                "waterfall": w, "drc": sc, "base": sb,
                "tickers": sorted(cand.ticker.tolist())}

    # ---- ticker set comparison ----
    print("\n" + "=" * 78)
    print("CANDIDATE TICKER SET COMPARISON")
    print("=" * 78)
    keys = list(results)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            A = set(results[keys[i]]["tickers"])
            B = set(results[keys[j]]["tickers"])
            if not A and not B:
                continue
            print(f"  {keys[i]}  vs  {keys[j]}")
            print(f"    intersection {len(A & B):>5}   only-A {len(A - B):>5}"
                  f"   only-B {len(B - A):>5}")

    meta = {"git_commit": git_commit(),
            "inputs": {p.name: sha(p) for p in
                       [DATA / "underlying.parquet", DATA / "ladder_paths.parquet",
                        DATA / "premium_history2.parquet"]},
            "params": {"coins": COINS, "z": Z, "lookback": LOOKBACK,
                       "band": BAND, "splits": SPLITS, "eps": EPS},
            "signal_waterfall": wf_sig,
            "results": {k: {"waterfall": v["waterfall"], "drc": v["drc"],
                            "base": v["base"]} for k, v in results.items()}}
    (OUT / "drc_v2_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"\nwrote {OUT/'drc_v2_meta.json'}")


if __name__ == "__main__":
    main()
