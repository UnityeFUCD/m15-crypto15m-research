"""PTC v3 - the complete historical double test on all 303 LSM orders.

SUPERSEDES research/ptc_adversarial_v2.py and research/results/ptc_v2_report.md.
Those were computed on 220 of 303 orders, where path availability was a proxy
for the outcome (+$153.78 matched vs -$144.40 unmatched). That cohort is now
complete: research/ptc_reconcile.py verifies 303/303 coverage and a ledger that
reconciles to +$9.38.

THE CORRECTION v2 STILL NEEDED - DECISION TIME vs OBSERVATION TIME
  v2 set probe_filled = first_fill <= effective_wait, where effective_wait is
  the timestamp of the first COMPLETE one-minute candle at or after the nominal
  wait. That averaged 99.4s for a 60s design. So a probe filling at 75s was
  scored as filled even though the frozen rule cancels at 60s.

  Those are two different clocks and they must not be merged:

    DECISION  t_cancel = wait + latency
              the moment cancellation is confirmed. A fill at or before this
              instant is a real probe fill (the cancel race). Everything later
              is not - we were already out.

    OBSERVATION t_obs = first candle at or after t_cancel
              the first price we could actually have acted on. Data granularity
              is one minute, so this is later than t_cancel and the IOC is
              priced there. That is a genuine handicap, not an assumption:
              a live implementation reads the book immediately.

  Splitting them moves probes out of the fill branch and into the no-fill
  branch, which CHANGES THE PRIMARY RESULT, so it is applied here and the
  effect is reported explicitly.

ARMS (frozen, not selected on this data)
  CONTROL          existing full-size maker execution at validation quantity
  PROBE_ONLY_60    q1 maker probe, cancel-confirm at 60s, never commit
  PTC_60           q1 probe, cancel-confirm at 60s, then one bounded IOC
  PTC_120          q1 probe, cancel-confirm at 120s, then one bounded IOC

Two live days exist. Nothing here can establish a production mean; the point
is to determine whether the mechanism survives a complete, correctly-clocked
ledger, and to size the prospective trial.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
OUT = ROOT / "research" / "results"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [11, 202608062, 31337, 90210, 4242]
COIN_ORDER = {"BTC": 0, "ETH": 1, "SOL": 2, "XRP": 3, "DOGE": 4, "HYPE": 5}
ASK_FLOOR, ASK_CEIL = 0.60, 0.80
PRIMARY_QTY = 15
LEDGER_PNL = 9.3814


# --------------------------------------------------------------- primitives
def fee_total(qty: int, price: float) -> float:
    """Kalshi taker fee, exact historical rounding: ceil to 4dp of
    0.07 * C * p * (1-p). Maker fee is zero."""
    if qty <= 0:
        return 0.0
    return math.ceil(0.07 * qty * price * (1 - price) * 10_000 - 1e-12) / 10_000


def epoch_seconds(v) -> float:
    """Convert to epoch SECONDS with an explicit unit check.

    pandas stores datetimes at ns/us/ms depending on construction, and dividing
    the integer representation by the wrong power of ten is exactly the bug
    that invalidated PTC v1. Timestamp.timestamp() is unit-agnostic.
    """
    ts = pd.Timestamp(v)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return float(ts.timestamp())


def detect_epoch_unit(value: float) -> str:
    """Name the unit of a raw integer epoch. Used by the timestamp unit tests."""
    a = abs(float(value))
    if a < 1e11:
        return "s"
    if a < 1e14:
        return "ms"
    if a < 1e17:
        return "us"
    return "ns"


def held_quote(side: str, yes_bid: float, yes_ask: float) -> tuple[float, float]:
    if side == "yes":
        return yes_bid, yes_ask
    if side == "no":
        return 1.0 - yes_ask, 1.0 - yes_bid
    raise ValueError(side)


def parse_path(value, close_ts: int, side: str) -> list[dict]:
    try:
        raw = json.loads(value) if isinstance(value, str) else value
    except Exception:
        return []
    pts: list[dict] = []
    for p in raw or []:
        try:
            ml = float(p["ml"])
            yb = float(p.get("yb", p.get("bc")))
            ya = float(p.get("ya", p.get("ac")))
            if not (0 < yb < ya < 1):
                continue
            bid, ask = held_quote(side, yb, ya)
            pts.append({"ml": ml, "ts": float(close_ts) - 60.0 * ml,
                        "bid": bid, "ask": ask})
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(pts, key=lambda p: p["ts"])


# ------------------------------------------------------------------- build
def build() -> pd.DataFrame:
    orders = pd.read_parquet(DATA / "orders_history.parquet")
    fills = pd.read_parquet(DATA / "fills_history.parquet")
    paths = pd.read_parquet(DATA / "paths_full.parquet")

    u = pd.read_parquet(DATA / "underlying.parquet", columns=["ticker", "result"])
    u = u[u.result.isin(["yes", "no"])].drop_duplicates("ticker")
    rec = pd.read_parquet(DATA / "lsm_missing_outcomes.parquet")
    rec = rec[rec.result.isin(["yes", "no"]) & ~rec.ticker.isin(u.ticker)]
    outcomes = pd.concat([u, rec[["ticker", "result"]]],
                         ignore_index=True).drop_duplicates("ticker")

    x = orders[orders.client_order_id.astype(str).str.startswith("lsm")].copy()
    if len(x) != 303:
        raise RuntimeError(f"expected 303 LSM orders, got {len(x)}")

    x["held"] = np.where(x.action.eq("sell"),
                         np.where(x.side.eq("yes"), "no", "yes"), x.side)
    x["created_dt"] = pd.to_datetime(x.created_time, format="mixed", utc=True)
    x["created_ts"] = x.created_dt.map(epoch_seconds)
    x["close_dt"] = (pd.to_datetime(x.ticker.str.split("-").str[1],
                                    format="%y%b%d%H%M", utc=True)
                     + pd.Timedelta(hours=4))
    x["close_ts"] = x.close_dt.map(epoch_seconds)
    x["seconds_to_close"] = x.close_ts - x.created_ts
    if not x.seconds_to_close.between(0, 900).all():
        raise RuntimeError("timestamp-unit failure: order-to-close outside 0-900s")
    if not x.created_ts.between(1.7e9, 1.9e9).all():
        raise RuntimeError("timestamp-unit failure: epoch is not in seconds")

    x["day"] = x.close_dt.dt.date
    x["coin"] = x.ticker.str.extract(r"^KX([A-Z]+)15M")[0]
    x["coin_order"] = x.coin.map(COIN_ORDER).fillna(99).astype(int)
    yp = pd.to_numeric(x.yes_price_dollars, errors="coerce")
    np_ = pd.to_numeric(x.no_price_dollars, errors="coerce")
    x["maker_price"] = np.where(x.held.eq("yes"), yp, np_)
    x["submitted_qty"] = pd.to_numeric(x.initial_count_fp, errors="coerce").fillna(0)
    x["filled_qty"] = pd.to_numeric(x.fill_count_fp, errors="coerce").fillna(0)

    ff = (fills.assign(fd=pd.to_datetime(fills.created_time, format="mixed",
                                         utc=True))
          .groupby("order_id").fd.min().rename("first_fill_dt"))
    x = x.merge(ff, left_on="order_id", right_index=True, how="left")
    x["first_fill_seconds"] = (x.first_fill_dt - x.created_dt).dt.total_seconds()
    bad = x.first_fill_seconds.dropna()
    if len(bad) and (bad.min() < -1 or bad.max() > 900):
        raise RuntimeError(f"fill latency outside 0-900s: {bad.min()}..{bad.max()}")

    x = x.merge(outcomes, on="ticker", how="left")
    if x.result.isna().any():
        raise RuntimeError("unresolved outcomes remain")
    x["won"] = x.held.eq(x.result).astype(int)

    p = paths[["ticker", "close_ts", "path"]].drop_duplicates("ticker")
    x = x.merge(p, on="ticker", how="left", suffixes=("", "_path"))
    x["has_path"] = x.path.notna()
    if not x.has_path.all():
        raise RuntimeError(f"path coverage incomplete: {int((~x.has_path).sum())} missing")

    actual = float((x.filled_qty * (x.won - x.maker_price)).sum())
    if abs(actual - LEDGER_PNL) > 0.02:
        raise RuntimeError(f"ledger mismatch {actual}")

    x["points"] = [parse_path(r.path, int(r.close_ts_path), r.held)
                   for r in x.itertuples()]
    return x.sort_values(["close_dt", "created_ts", "coin_order"]).reset_index(drop=True)


# ------------------------------------------------------------------ replay
def replay(data: pd.DataFrame, wait: float, qty: int = PRIMARY_QTY,
           latency: float = 0.0, slippage_c: float = 0.0,
           fill_fraction: float = 1.0, ceiling: float = ASK_CEIL,
           commit: bool = True) -> pd.DataFrame:
    """One arm. `commit=False` gives PROBE_ONLY."""
    x = data.copy()
    t_cancel = x.created_ts + wait + latency          # DECISION clock
    x["t_cancel"] = t_cancel

    # probe fills only if it filled at or before cancellation confirmation
    x["probe_filled"] = x.first_fill_seconds.notna() & (
        x.first_fill_seconds <= wait + latency)
    x["probe_pnl"] = np.where(x.probe_filled, x.won - x.maker_price, 0.0)
    # capital-minutes for the q1 probe: held from fill to close
    x["probe_capmin"] = np.where(
        x.probe_filled,
        1 * x.maker_price * np.maximum(
            0.0, (x.close_ts - x.created_ts - x.first_fill_seconds.fillna(0))) / 60.0,
        0.0)

    # OBSERVATION clock: first complete candle at or after cancellation
    obs_ask, obs_ts = [], []
    for r in x.itertuples():
        snap = next((p for p in r.points if p["ts"] >= r.t_cancel - 1e-9), None)
        obs_ask.append(snap["ask"] if snap else np.nan)
        obs_ts.append(snap["ts"] if snap else np.nan)
    x["obs_ask"] = obs_ask
    x["obs_ts"] = obs_ts
    x["obs_delay"] = x.obs_ts - x.t_cancel
    x["exec_price"] = x.obs_ask + slippage_c / 100.0

    x["candidate"] = (~x.probe_filled & x.exec_price.notna()
                      & x.exec_price.between(ASK_FLOOR, ceiling, inclusive="both"))
    x["selected"] = False
    if commit:
        c = x[x.candidate]
        if len(c):
            idx = (c.sort_values(["close_dt", "exec_price", "coin_order",
                                  "created_ts"])
                   .groupby("close_dt", as_index=False).head(1).index)
            x.loc[idx, "selected"] = True

    fq = int(math.floor(qty * fill_fraction + 1e-9))
    x["ioc_qty"] = np.where(x.selected, fq, 0)
    x["ioc_fee"] = 0.0
    if fq > 0 and x.selected.any():
        x.loc[x.selected, "ioc_fee"] = [fee_total(fq, p)
                                        for p in x.loc[x.selected, "exec_price"]]
    x["ioc_pnl"] = x.ioc_qty * (x.won - x.exec_price) - x.ioc_fee
    x["ioc_capmin"] = np.where(
        x.selected, x.ioc_qty * x.exec_price
        * np.maximum(0.0, x.close_ts - x.obs_ts) / 60.0, 0.0)

    x["pnl"] = x.probe_pnl + x.ioc_pnl
    x["submitted"] = 1 + x.ioc_qty
    x["filled"] = x.probe_filled.astype(int) + x.ioc_qty
    x["capmin"] = x.probe_capmin + x.ioc_capmin
    return x


def control(data: pd.DataFrame, qty: int = PRIMARY_QTY) -> pd.DataFrame:
    """Standardized full-size maker control at the validation quantity."""
    x = data.copy()
    frac = np.divide(x.filled_qty, x.submitted_qty,
                     out=np.zeros(len(x)), where=x.submitted_qty.to_numpy() > 0)
    x["pnl"] = qty * frac * (x.won - x.maker_price)
    x["submitted"] = qty
    x["filled"] = qty * frac
    x["probe_filled"] = frac > 0
    x["selected"] = False
    x["ioc_qty"] = 0
    x["ioc_fee"] = 0.0
    x["exec_price"] = np.nan
    x["capmin"] = (qty * frac * x.maker_price
                   * np.maximum(0.0, x.close_ts - x.created_ts
                                - x.first_fill_seconds.fillna(0)) / 60.0)
    return x


# ----------------------------------------------------------------- metrics
def worst_block(w: pd.Series, k: int) -> float:
    """Worst contiguous k-window sum (windows are 15 minutes apart)."""
    if len(w) < 1:
        return float("nan")
    v = w.to_numpy(float)
    if len(v) < k:
        return float(v.sum())
    return float(min(v[i:i + k].sum() for i in range(len(v) - k + 1)))


def metrics(x: pd.DataFrame, name: str) -> dict:
    w = x.groupby("close_dt").pnl.sum().sort_index()
    d = x.groupby("day").pnl.sum()
    csum = w.cumsum()
    sel = x[x.selected] if "selected" in x else x.iloc[0:0]
    tot = float(x.pnl.sum())
    return {
        "arm": name,
        "assigned_opportunities": int(len(x)),
        "submitted_contracts": float(x.submitted.sum()),
        "probe_fills": int(x.probe_filled.sum()),
        "no_fills": int((~x.probe_filled).sum()),
        "commitments": int(x.selected.sum()) if "selected" in x else 0,
        "ioc_filled_qty": float(x.ioc_qty.sum()) if "ioc_qty" in x else 0.0,
        "commit_win_rate": float(sel.won.mean()) if len(sel) else float("nan"),
        "mean_commit_ask": float(sel.exec_price.mean()) if len(sel) else float("nan"),
        "fees": float(x.ioc_fee.sum()) if "ioc_fee" in x else 0.0,
        "total_pnl": tot,
        "pnl_per_assigned": tot / len(x) if len(x) else float("nan"),
        "pnl_per_submitted": tot / x.submitted.sum() if x.submitted.sum() else float("nan"),
        "pnl_per_filled": tot / x.filled.sum() if x.filled.sum() else float("nan"),
        "daily_mean": float(d.mean()),
        "daily_sd": float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
        "max_drawdown": float((csum.cummax() - csum).max()),
        "worst_close_window": float(w.min()),
        "worst_30min": worst_block(w, 2),
        "worst_60min": worst_block(w, 4),
        "worst_90min": worst_block(w, 6),
        "positive_day_fraction": float((d > 0).mean()),
        "capital_minutes": float(x.capmin.sum()),
    }


# -------------------------------------------------------------- resampling
def block_boot(diff_w: pd.Series, block: int, seed: int, reps: int = 8000) -> dict:
    """Moving-block bootstrap over 15-minute close windows, within day.

    Reported on the MEAN per close window, not the total. An earlier version
    resampled totals and then rescaled them by observed/mean to re-center; when
    the resampled mean landed near zero that ratio exploded and produced a
    nonsense interval (one comparison returned [-85302, +83404]). The mean is
    scale-stable and needs no such correction.
    """
    df = diff_w.rename("d").reset_index()
    df["day"] = pd.to_datetime(df.close_dt, utc=True).dt.date
    blocks: list[np.ndarray] = []
    for _, g in df.groupby("day"):
        v = g.d.to_numpy(float)
        if len(v) < block:
            blocks.append(v)
            continue
        for s in range(len(v) - block + 1):        # MOVING blocks, overlapping
            blocks.append(v[s:s + block])
    n_draw = max(1, int(round(len(df) / block)))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(blocks), size=(reps, n_draw))
    means = np.array([np.concatenate([blocks[j] for j in row]).mean()
                      for row in idx])
    n_win = len(df)
    return {"observed": float(df.d.sum()),
            "observed_per_window": float(df.d.mean()),
            "ci_lo": float(np.quantile(means, 0.025)) * n_win,
            "ci_hi": float(np.quantile(means, 0.975)) * n_win,
            "p_nonpositive": float((means <= 0).mean())}


def multiseed(diff_w: pd.Series, block: int) -> dict:
    r = [block_boot(diff_w, block, s) for s in SEEDS]
    ps = [z["p_nonpositive"] for z in r]
    return {"observed": r[0]["observed"],
            "ci_lo": float(np.mean([z["ci_lo"] for z in r])),
            "ci_hi": float(np.mean([z["ci_hi"] for z in r])),
            "p_min": min(ps), "p_max": max(ps),
            "p_median": float(np.median(ps))}


def leave_one_out(arms: dict, lhs: str, rhs: str, key: str,
                  data: pd.DataFrame) -> dict:
    """Drop each level of `key` and recompute the total difference."""
    vals = sorted(set(data[key].astype(str)))
    out = {}
    for v in vals:
        m = data[key].astype(str) != v
        out[v] = float(arms[lhs].loc[m.values, "pnl"].sum()
                       - arms[rhs].loc[m.values, "pnl"].sum())
    return out


# ----------------------------------------------------------------- reports
def main() -> None:
    data = build()
    print(f"built {len(data)} orders, path coverage {data.has_path.mean():.4f}, "
          f"ledger {(data.filled_qty*(data.won-data.maker_price)).sum():.4f}")

    arms = {
        "CONTROL": control(data, PRIMARY_QTY),
        "PROBE_ONLY_60": replay(data, 60, commit=False),
        "PTC_60": replay(data, 60),
        "PTC_120": replay(data, 120),
    }
    rows = [metrics(v, k) for k, v in arms.items()]
    M = pd.DataFrame(rows)
    M.to_csv(OUT / "ptc_v3_arms.csv", index=False)

    # decision-vs-observation clock effect (the v2 correction)
    v2style = data.copy()
    obs60 = []
    for r in v2style.itertuples():
        s = next((p for p in r.points if p["ts"] >= r.created_ts + 60 - 1e-9), None)
        obs60.append(s["ts"] - r.created_ts if s else np.nan)
    v2style["eff"] = obs60
    v2_filled = int((v2style.first_fill_seconds.notna()
                     & (v2style.first_fill_seconds <= v2style.eff)).sum())
    v3_filled = int(arms["PTC_60"].probe_filled.sum())
    clock = {"v2_probe_fills_observation_clock": v2_filled,
             "v3_probe_fills_decision_clock": v3_filled,
             "mean_observation_delay_s": float(v2style.eff.mean()),
             "orders_reclassified": v2_filled - v3_filled}

    comps = [("PTC_60", "PROBE_ONLY_60"), ("PTC_120", "PROBE_ONLY_60"),
             ("PTC_60", "CONTROL"), ("PTC_120", "CONTROL")]
    boot_rows = []
    for lhs, rhs in comps:
        dw = (arms[lhs].groupby("close_dt").pnl.sum()
              - arms[rhs].groupby("close_dt").pnl.sum()).sort_index()
        for blk, mins in ((1, 15), (2, 30), (4, 60), (6, 90)):
            z = multiseed(dw, blk)
            z.update({"comparison": f"{lhs} - {rhs}", "block_minutes": mins})
            boot_rows.append(z)
        dd = (arms[lhs].groupby("day").pnl.sum()
              - arms[rhs].groupby("day").pnl.sum())
        boot_rows.append({"comparison": f"{lhs} - {rhs}", "block_minutes": "day",
                          "observed": float(dd.sum()), "ci_lo": float("nan"),
                          "ci_hi": float("nan"), "p_min": float("nan"),
                          "p_max": float("nan"), "p_median": float("nan")})
    B = pd.DataFrame(boot_rows)
    B.to_csv(OUT / "ptc_v3_bootstrap.csv", index=False)

    loo = {}
    for lhs, rhs in comps:
        for key in ("day", "coin"):
            loo[f"{lhs} - {rhs} | drop {key}"] = leave_one_out(arms, lhs, rhs,
                                                              key, data)
    # leave-one-close-window-cluster-out: drop each hour-of-day cluster
    data2 = data.copy()
    data2["wcluster"] = data2.close_dt.dt.floor("h").astype(str)
    for lhs, rhs in comps[:2]:
        loo[f"{lhs} - {rhs} | drop window-cluster"] = leave_one_out(
            arms, lhs, rhs, "wcluster", data2)

    # ---- Part 4 sensitivity ----
    sens = []
    for wait, nm in ((60, "PTC_60"), (120, "PTC_120")):
        for lat in (0.0, 0.25, 0.5, 1.0, 2.0, 3.0):
            m = metrics(replay(data, wait, latency=lat), nm)
            m.update({"axis": "cancel_latency", "value": lat}); sens.append(m)
        for sl in (0.0, 1.0, 2.0, 3.0, 5.0):
            m = metrics(replay(data, wait, slippage_c=sl), nm)
            m.update({"axis": "slippage_c", "value": sl}); sens.append(m)
        for fr in (0.25, 0.50, 0.75, 1.00):
            m = metrics(replay(data, wait, fill_fraction=fr), nm)
            m.update({"axis": "ioc_fill_fraction", "value": fr}); sens.append(m)
        for q in (5, 10, 15, 20):
            m = metrics(replay(data, wait, qty=q), nm)
            m.update({"axis": "commit_qty", "value": q}); sens.append(m)
        for ce in (0.76, 0.78, 0.80, 0.82, 0.85):
            m = metrics(replay(data, wait, ceiling=ce), nm)
            m.update({"axis": "ask_ceiling_SENSITIVITY_ONLY", "value": ce})
            sens.append(m)
    S = pd.DataFrame(sens)
    S.to_csv(OUT / "ptc_v3_sensitivity.csv", index=False)

    # branch economics at the decision clock
    br = []
    for wait in (60, 120):
        x = replay(data, wait)
        for nm, mask in (("filled", x.probe_filled), ("no_fill", ~x.probe_filled)):
            g = x[mask]
            el = g.obs_ask.between(ASK_FLOOR, ASK_CEIL, inclusive="both")
            z = g[el]
            edge = (z.won - z.obs_ask
                    - np.array([fee_total(15, p) / 15 for p in z.obs_ask])
                    ) if len(z) else np.array([])
            br.append({"wait": wait, "branch": nm, "n": int(len(g)),
                       "win_rate": float(g.won.mean()) if len(g) else np.nan,
                       "mean_ask": float(g.obs_ask.mean()) if len(g) else np.nan,
                       "share_60_80": float(el.mean()) if len(g) else np.nan,
                       "eligible_n": int(el.sum()),
                       "taker_edge_c": float(np.mean(edge) * 100) if len(z) else np.nan,
                       "mean_obs_delay_s": float(g.obs_delay.mean()) if len(g) else np.nan})
    BR = pd.DataFrame(br)
    BR.to_csv(OUT / "ptc_v3_branches.csv", index=False)

    summary = {"clock_correction": clock,
               "arms": rows,
               "bootstrap": B.to_dict("records"),
               "leave_one_out": loo,
               "branches": BR.to_dict("records")}
    (OUT / "ptc_v3_summary.json").write_text(json.dumps(summary, indent=2,
                                                        default=str),
                                             encoding="utf-8")
    print("\n=== ARMS ===")
    print(M[["arm", "assigned_opportunities", "submitted_contracts", "probe_fills",
             "no_fills", "commitments", "commit_win_rate", "total_pnl",
             "pnl_per_assigned", "max_drawdown", "worst_close_window"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== CLOCK CORRECTION ===")
    print(json.dumps(clock, indent=2))
    print("\n=== BOOTSTRAP (5 seeds) ===")
    print(B.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== BRANCHES ===")
    print(BR.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== LEAVE-ONE-OUT ===")
    for k, v in loo.items():
        vals = list(v.values())
        print(f"  {k:44s} min {min(vals):+9.2f}  max {max(vals):+9.2f}  "
              f"all>0 {'YES' if min(vals) > 0 else 'NO'}")
    print("\nwrote research/results/ptc_v3_*.csv|json")


if __name__ == "__main__":
    main()
