"""DRC-15 Phase 1-2: independent reproduction of the directional signal.

CANDIDATE
  coins   DOGE ETH SOL XRP
  r_prev  = A0(t) / A0(t-1) - 1          previous completed window return
  sigma4  = sample std of the last 4 completed returns
  z_up    = r_prev / (sigma4 + eps)
  qualify z_up >= 1.0 AND favourite is NO AND 0.65 <= fav_bid < 0.80

CLAIMED
  ~371 candidates, 80.86% win, +7.40c net taker edge, day-clustered
  95% CI [+3.51c, +11.27c]; comparable non-DRC NO favourites 69.00% / -4.90c

PRIOR THAT MUST BE OVERCOME
  15-minute benchmark returns are a random walk on this data: lag-1
  autocorrelation -0.0086 against a 2-sigma band of +/-0.0103 on 41,334
  windows. A conditional reversal signal is not excluded by that, but it starts
  from a strong prior of "no".

LEAKAGE AUDIT
  A0(t) is the 60-second benchmark mean at window OPEN, so it is fixed before
  we ever evaluate. Entry is 8-14 minutes before close, i.e. 1-7 minutes AFTER
  open. Therefore r_prev, sigma4 and z_up are all knowable at decision time.
  That part of the construction is sound.

  What is NOT automatically sound: contiguity. r_prev is meaningless across a
  gap. Every window pair is checked to be exactly 900s apart and rows with any
  gap are dropped rather than silently bridged.

THIS SCRIPT DELIBERATELY TESTS THE DIRECTIONAL CORE FIRST
  Win rate given z_up needs only the benchmark series - 41,334 windows, far
  more power than the ~2,800 rows with book data. If the directional edge is
  absent here, the overlay cannot work and the remaining phases are moot.
"""
from __future__ import annotations

import hashlib
import json
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
Z_THRESHOLD = 1.0
VOL_LOOKBACK = 4


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(ROOT), text=True).strip()
    except Exception:
        return "unknown"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def build_series() -> pd.DataFrame:
    """Benchmark series per coin with STRICT contiguity enforcement."""
    u = pd.read_parquet(DATA / "underlying.parquet")
    n_raw = len(u)
    # window close from the wkey (ET ticker clock + 4h in summer), NOT
    # expected_expiration_time which is close PLUS FIVE MINUTES.
    u["close_utc"] = (pd.to_datetime(u.wkey, format="%y%b%d%H%M", utc=True)
                      + pd.Timedelta(hours=4))
    u["open_utc"] = u.close_utc - pd.Timedelta(minutes=15)
    u = u[u.result.isin(["yes", "no"])]
    u = u.dropna(subset=["a0", "a1"])
    u = u[(u.a0 > 0) & (u.a1 > 0)]
    u = u.sort_values(["coin", "close_utc"]).reset_index(drop=True)
    u = u.drop_duplicates(subset=["coin", "close_utc"], keep="first")

    parts = []
    for coin, g in u.groupby("coin"):
        g = g.sort_values("close_utc").reset_index(drop=True)
        gap = g.close_utc.diff().dt.total_seconds()
        g["contiguous_1"] = gap == 900
        # r_prev needs A0(t) and A0(t-1); both require the pair to be adjacent
        g["a0_prev"] = g.a0.shift(1)
        g["r_prev"] = g.a0 / g.a0_prev - 1.0
        g.loc[~g.contiguous_1, "r_prev"] = np.nan
        # sigma4 over the 4 most recent COMPLETED returns, strictly prior
        r = g.r_prev
        g["sigma4"] = r.shift(1).rolling(VOL_LOOKBACK).std(ddof=1)
        # every one of those 4 must itself be from a contiguous pair
        ok = g.contiguous_1.shift(1).rolling(VOL_LOOKBACK).min().astype("float")
        g.loc[ok != 1, "sigma4"] = np.nan
        g["z_up"] = g.r_prev / (g.sigma4 + EPS)
        parts.append(g)
    s = pd.concat(parts, ignore_index=True)
    s["usable"] = s.z_up.notna() & np.isfinite(s.z_up)
    print(f"  raw rows {n_raw:,} -> resolved {len(u):,} -> "
          f"with z_up {int(s.usable.sum()):,}")
    dropped = len(s) - int(s.usable.sum())
    print(f"  dropped {dropped:,} for gaps / insufficient history")
    return s


def main():
    print("=" * 78)
    print("PHASE 1 - REBUILD THE SIGNAL FROM THE BENCHMARK SERIES")
    print("=" * 78)
    s = build_series()
    s = s[s.usable].copy()
    s["yes_won"] = (s.result == "yes").astype(int)
    s["no_won"] = 1 - s.yes_won

    print("\n  leakage audit:")
    print("    A0(t) is fixed at window open; entry is 1-7 min after open")
    print("    -> r_prev, sigma4, z_up all knowable at decision time: OK")
    print("    contiguity enforced: every r_prev from an adjacent 900s pair")

    print("\n" + "=" * 78)
    print("PHASE 2a - DOES z_up PREDICT DIRECTION AT ALL?  (all coins, 41k base)")
    print("=" * 78)
    print(f"{'z_up bucket':>16} {'n':>7} {'P(NO wins)':>12} {'P(YES wins)':>12}")
    s["zb"] = pd.cut(s.z_up, [-np.inf, -2, -1, -0.5, 0, 0.5, 1, 2, np.inf])
    for k, g in s.groupby("zb", observed=True):
        print(f"{str(k):>16} {len(g):>7,} {g.no_won.mean():>12.4f} "
              f"{g.yes_won.mean():>12.4f}")
    base = s.no_won.mean()
    print(f"\n  unconditional P(NO wins) = {base:.4f}")

    print("\n" + "=" * 78)
    print("PHASE 2b - THE EXACT DRC DIRECTIONAL CLAIM")
    print("=" * 78)
    d = s[s.coin.isin(COINS)].copy()
    hit = d[d.z_up >= Z_THRESHOLD]
    rest = d[d.z_up < Z_THRESHOLD]
    print(f"  coins {COINS}, z_up >= {Z_THRESHOLD}")
    print(f"    DRC windows      n {len(hit):>6,}   P(NO wins) {hit.no_won.mean():.4f}")
    print(f"    non-DRC windows  n {len(rest):>6,}   P(NO wins) {rest.no_won.mean():.4f}")
    diff = (hit.no_won.mean() - rest.no_won.mean()) * 100
    print(f"    difference       {diff:+.2f} percentage points")

    # day-clustered bootstrap on the difference
    d["day"] = d.close_utc.dt.date
    gd = {k: v for k, v in d.groupby("day")}
    days = sorted(gd)
    bs = []
    for _ in range(6000):
        smp = pd.concat([gd[days[i]] for i in RNG.integers(0, len(days), len(days))])
        a = smp[smp.z_up >= Z_THRESHOLD]
        b = smp[smp.z_up < Z_THRESHOLD]
        if len(a) > 10 and len(b) > 10:
            bs.append((a.no_won.mean() - b.no_won.mean()) * 100)
    bs = np.sort(np.array(bs))
    lo, hi = bs[int(.025 * len(bs))], bs[int(.975 * len(bs))]
    print(f"    day-clustered 95% CI [{lo:+.2f}, {hi:+.2f}] pp   "
          f"P(<=0) {(bs <= 0).mean():.4f}")

    print("\n  per coin:")
    print(f"{'coin':>7} {'n DRC':>7} {'P(NO|DRC)':>11} {'n rest':>8} "
          f"{'P(NO|rest)':>11} {'diff pp':>9}")
    for c, g in d.groupby("coin"):
        a, b = g[g.z_up >= Z_THRESHOLD], g[g.z_up < Z_THRESHOLD]
        if len(a) < 20:
            continue
        print(f"{c:>7} {len(a):>7,} {a.no_won.mean():>11.4f} {len(b):>8,} "
              f"{b.no_won.mean():>11.4f} "
              f"{(a.no_won.mean()-b.no_won.mean())*100:>+8.2f}")

    print("\n" + "=" * 78)
    print("PHASE 2c - IS THE EFFECT SYMMETRIC?  (falsification check)")
    print("=" * 78)
    print("  If z_up>=1 genuinely predicts NO, then by symmetry z_up<=-1 should")
    print("  predict YES by a similar margin. If only one tail 'works', that is")
    print("  the signature of a fluke rather than a mechanism.\n")
    up = d[d.z_up >= Z_THRESHOLD]
    dn = d[d.z_up <= -Z_THRESHOLD]
    mid = d[d.z_up.abs() < Z_THRESHOLD]
    print(f"  z_up >= +1 : n {len(up):>6,}  P(NO wins)  {up.no_won.mean():.4f}")
    print(f"  |z_up| < 1 : n {len(mid):>6,}  P(NO wins)  {mid.no_won.mean():.4f}")
    print(f"  z_up <= -1 : n {len(dn):>6,}  P(YES wins) {dn.yes_won.mean():.4f}")

    print("\n" + "=" * 78)
    print("PHASE 2d - CHRONOLOGICAL SPLIT")
    print("=" * 78)
    d = d.sort_values("close_utc")
    n = len(d)
    for name, sl in (("train  (first 50%)", d.iloc[:n // 2]),
                     ("valid  (next 25%)", d.iloc[n // 2:3 * n // 4]),
                     ("test   (last 25%)", d.iloc[3 * n // 4:])):
        a, b = sl[sl.z_up >= Z_THRESHOLD], sl[sl.z_up < Z_THRESHOLD]
        if not len(a) or not len(b):
            continue
        print(f"  {name}  {sl.close_utc.min():%Y-%m-%d} .. "
              f"{sl.close_utc.max():%Y-%m-%d}  "
              f"DRC n {len(a):>5,} P(NO) {a.no_won.mean():.4f}   "
              f"rest P(NO) {b.no_won.mean():.4f}   "
              f"diff {(a.no_won.mean()-b.no_won.mean())*100:+.2f}pp")

    meta = {
        "git_commit": git_commit(),
        "inputs": {"underlying.parquet": sha(DATA / "underlying.parquet")},
        "rows_with_signal": int(len(s)),
        "drc_windows_4coins": int(len(hit)),
        "p_no_given_drc": float(hit.no_won.mean()),
        "p_no_given_rest": float(rest.no_won.mean()),
        "diff_pp": float(diff),
        "ci_pp": [float(lo), float(hi)],
        "params": {"coins": COINS, "z": Z_THRESHOLD,
                   "vol_lookback": VOL_LOOKBACK, "eps": EPS},
    }
    d.drop(columns=[c for c in ("zb",) if c in d.columns]).to_parquet(
        OUT / "drc_candidates.parquet", index=False)
    (OUT / "drc_reproduction_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {OUT/'drc_candidates.parquet'} and meta")


if __name__ == "__main__":
    main()
