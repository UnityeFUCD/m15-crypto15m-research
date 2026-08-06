"""Clean-room audit of the only mechanism that survived: endogenous maker fills.

This script does four things on the trustworthy population files introduced in
commit aba27cb:

1. Re-runs the frozen two-minute delayed-entry rule on ``paths_full.parquet``.
2. Measures how informative fill/no-fill is at fixed horizons on the 303 actual
   LSM orders.
3. Tests a new policy, Probe-Then-Commit (PTC): post ONE maker contract as a
   diagnostic probe; if it has not filled after a frozen wait, cancel it and
   take at most one full-size position per close window at a bounded ask.
4. Uses ``spot_1m.parquet`` to determine whether a one-minute signed-index cancel
   could act before the losing fills occur.

PTC is a research candidate, not production code.  Historical population paths
cannot identify an untried passive policy by themselves, so execution is modeled
with Beta posteriors fitted only to the actual first-fill outcomes.  The report
keeps population, execution, and model-based conclusions separate.

Run:
    python research/probe_then_commit_audit.py

Outputs:
    research/results/probe_then_commit_report.md
    research/results/probe_then_commit_summary.json
    research/results/probe_fill_horizons.csv
    research/results/fr2_full_population.csv
    research/results/ptc_sensitivity.csv
    research/results/reversal_hazard.csv
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "research" / "results"
OUT.mkdir(parents=True, exist_ok=True)

RNG_SEED = 20260806
COIN_ORDER = {"BTC": 0, "ETH": 1, "SOL": 2, "XRP": 3, "DOGE": 4, "HYPE": 5}
SPLITS = {
    "train": (pd.Timestamp("2026-05-25", tz="UTC"), pd.Timestamp("2026-06-30", tz="UTC")),
    "valid": (pd.Timestamp("2026-06-30", tz="UTC"), pd.Timestamp("2026-07-18", tz="UTC")),
    "test": (pd.Timestamp("2026-07-18", tz="UTC"), pd.Timestamp("2026-08-07", tz="UTC")),
}

INITIAL_BID_LO = 0.65
INITIAL_BID_HI = 0.80
VOLUME_MIN = 2000.0

# Frozen primary PTC candidate.  The wait is selected for operational and data
# resolution reasons before looking at this script's output: one complete minute
# is the first point at which the committed historical path has an observable ask.
PRIMARY_WAIT_SECONDS = 60
PRIMARY_DELAY_MINUTES = 1
PRIMARY_ASK_CEILING = 0.80
PRIMARY_PROBE_QTY = 1
PRIMARY_COMMIT_QTY = 15
PRIMARY_MAX_COMMITS_PER_CLOSE = 1

# Sensitivity only; no candidate is promoted by taking the best row.
WAIT_SECONDS = [30, 60, 90, 120, 180, 300]
ASK_CEILINGS = [0.76, 0.78, 0.80, 0.82, 0.85]
COMMIT_QTYS = [10, 15, 20]


def fee_total(qty: int, price: float) -> float:
    """Repository's historical taker fee convention, rounded once per block."""
    if qty <= 0:
        return 0.0
    raw = 0.07 * qty * price * (1.0 - price)
    return math.ceil(raw * 10_000 - 1e-12) / 10_000


def held_quote(side: str, yes_bid: float, yes_ask: float) -> tuple[float, float]:
    if side == "yes":
        return yes_bid, yes_ask
    if side == "no":
        return 1.0 - yes_ask, 1.0 - yes_bid
    raise ValueError(f"unknown side {side!r}")


def parse_path(value: object) -> list[dict]:
    if isinstance(value, str):
        try:
            pts = json.loads(value)
        except Exception:
            return []
    elif isinstance(value, list):
        pts = value
    else:
        return []
    out: list[dict] = []
    for p in pts:
        try:
            ml = float(p["ml"])
            yb = float(p.get("yb", p.get("bc")))
            ya = float(p.get("ya", p.get("ac")))
            vol = float(p.get("v", p.get("volume_fp", 0)) or 0)
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(ml) and 0 < yb < ya < 1:
            out.append({"ml": ml, "yb": yb, "ya": ya, "vol": vol})
    return sorted(out, key=lambda x: -x["ml"])


def point_near(points: list[dict], target_ml: float, tolerance: float = 0.20) -> dict | None:
    if not points:
        return None
    best = min(points, key=lambda p: abs(p["ml"] - target_ml))
    return best if abs(best["ml"] - target_ml) <= tolerance else None


def build_population() -> pd.DataFrame:
    paths = pd.read_parquet(DATA / "paths_full.parquet")
    rows: list[dict] = []
    for r in paths.itertuples(index=False):
        points = parse_path(r.path)
        eligible = [p for p in points if 8 <= p["ml"] <= 14]
        if not eligible:
            continue
        entry = eligible[0]
        side = str(r.side).lower()
        try:
            bid0, ask0 = held_quote(side, entry["yb"], entry["ya"])
        except ValueError:
            continue
        close_dt = pd.to_datetime(int(r.close_ts), unit="s", utc=True)
        row = {
            "ticker": r.ticker,
            "coin": r.coin,
            "coin_order": COIN_ORDER.get(str(r.coin), 99),
            "close_ts": int(r.close_ts),
            "close_dt": close_dt,
            "day": close_dt.date(),
            "minute": int(close_dt.minute),
            "side": side,
            "won": int(r.won),
            "entry_ml": float(entry["ml"]),
            "entry_bid": float(bid0),
            "entry_ask": float(ask0),
            "entry_vol": float(entry["vol"]),
        }
        for delay in range(1, 6):
            pt = point_near(points, entry["ml"] - delay)
            if pt is None:
                row[f"bid_d{delay}"] = np.nan
                row[f"ask_d{delay}"] = np.nan
            else:
                b, a = held_quote(side, pt["yb"], pt["ya"])
                row[f"bid_d{delay}"] = b
                row[f"ask_d{delay}"] = a
        rows.append(row)
    out = pd.DataFrame(rows)
    out = out[(out.entry_bid >= INITIAL_BID_LO) & (out.entry_bid < INITIAL_BID_HI)]
    out = out.sort_values(["close_dt", "coin_order", "ticker"]).reset_index(drop=True)
    return out


def load_outcomes() -> pd.DataFrame:
    u = pd.read_parquet(DATA / "underlying.parquet", columns=["ticker", "result"])
    u = u[u.result.isin(["yes", "no"])].drop_duplicates("ticker")
    missing = DATA / "lsm_missing_outcomes.parquet"
    if missing.exists():
        x = pd.read_parquet(missing)
        x = x[x.result.isin(["yes", "no"])]
        u = pd.concat([u, x[~x.ticker.isin(u.ticker)][["ticker", "result"]]], ignore_index=True)
    return u.drop_duplicates("ticker")


def load_live_orders() -> pd.DataFrame:
    o = pd.read_parquet(DATA / "orders_history.parquet")
    f = pd.read_parquet(DATA / "fills_history.parquet")
    u = load_outcomes()

    l = o[o.client_order_id.astype(str).str.startswith("lsm")].copy()
    l["held"] = np.where(
        l.action.eq("sell"), np.where(l.side.eq("yes"), "no", "yes"), l.side
    )
    l["submitted_qty"] = pd.to_numeric(l.initial_count_fp, errors="coerce").fillna(0.0)
    l["filled_qty"] = pd.to_numeric(l.fill_count_fp, errors="coerce").fillna(0.0)
    l["created_dt"] = pd.to_datetime(l.created_time, format="mixed", utc=True)
    l["close_dt"] = (
        pd.to_datetime(l.ticker.str.split("-").str[1], format="%y%b%d%H%M", utc=True)
        + pd.Timedelta(hours=4)
    )
    l["entry_ml"] = (l.close_dt - l.created_dt).dt.total_seconds() / 60.0

    yp = pd.to_numeric(l.yes_price_dollars, errors="coerce")
    np_ = pd.to_numeric(l.no_price_dollars, errors="coerce")
    l["maker_price"] = np.where(l.held.eq("yes"), yp, np_)

    ff = (
        f.assign(fill_dt=pd.to_datetime(f.created_time, format="mixed", utc=True))
        .groupby("order_id", as_index=True)
        .fill_dt.min()
    )
    l = l.merge(ff.rename("first_fill_dt"), left_on="order_id", right_index=True, how="left")
    l["first_fill_seconds"] = (l.first_fill_dt - l.created_dt).dt.total_seconds()
    l = l.merge(u, on="ticker", how="left")
    l = l.dropna(subset=["result", "maker_price"])
    l["would_win"] = l.held.eq(l.result)
    l["day"] = l.close_dt.dt.date
    return l.sort_values("created_dt").reset_index(drop=True)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return center - half, center + half


def fill_horizon_table(live: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    horizons = sorted(set(WAIT_SECONDS + [1, 2, 5, 10, 15, 45, 240, 420]))
    for t in horizons:
        by_t = live.first_fill_seconds.notna() & (live.first_fill_seconds <= t)
        for branch, mask in (("filled_by_t", by_t), ("not_filled_by_t", ~by_t)):
            g = live[mask]
            wins = int(g.would_win.sum())
            lo, hi = wilson(wins, len(g))
            rows.append(
                {
                    "seconds": t,
                    "branch": branch,
                    "n": len(g),
                    "wins": wins,
                    "win_rate": g.would_win.mean() if len(g) else np.nan,
                    "win_ci_lo": lo,
                    "win_ci_hi": hi,
                    "mean_price": g.maker_price.mean() if len(g) else np.nan,
                }
            )
        for outcome, mask in (("win", live.would_win), ("lose", ~live.would_win)):
            g = live[mask]
            rows.append(
                {
                    "seconds": t,
                    "branch": f"fill_rate_given_{outcome}",
                    "n": len(g),
                    "wins": np.nan,
                    "win_rate": by_t[mask].mean(),
                    "win_ci_lo": np.nan,
                    "win_ci_hi": np.nan,
                    "mean_price": g.maker_price.mean(),
                }
            )
    return pd.DataFrame(rows)


def choose_one_per_close(g: pd.DataFrame, ask_col: str) -> pd.DataFrame:
    if g.empty:
        return g.copy()
    return (
        g.sort_values(["close_dt", ask_col, "coin_order", "ticker"])
        .groupby("close_dt", as_index=False, group_keys=False)
        .head(1)
        .sort_values("close_dt")
    )


def strategy_metrics(selected: pd.DataFrame, pnl_col: str) -> dict:
    if selected.empty:
        return {
            "n": 0,
            "pnl": 0.0,
            "edge_per_contract": np.nan,
            "mean_day": 0.0,
            "sd_day": 0.0,
            "max_drawdown": 0.0,
            "worst_window": 0.0,
        }
    x = selected.sort_values("close_dt").copy()
    daily = x.groupby("day")[pnl_col].sum()
    window = x.groupby("close_dt")[pnl_col].sum().sort_index()
    eq = window.cumsum()
    dd = eq.cummax() - eq
    qty = float(x.get("contracts", pd.Series(np.ones(len(x)))).sum())
    return {
        "n": int(len(x)),
        "pnl": float(x[pnl_col].sum()),
        "edge_per_contract": float(x[pnl_col].sum() / qty) if qty else np.nan,
        "mean_day": float(daily.mean()),
        "sd_day": float(daily.std(ddof=1)) if len(daily) > 1 else 0.0,
        "max_drawdown": float(dd.max()) if len(dd) else 0.0,
        "worst_window": float(window.min()) if len(window) else 0.0,
        "win_rate": float(x.won.mean()) if "won" in x else np.nan,
        "avg_price": float(x.get("trade_price", pd.Series(dtype=float)).mean())
        if "trade_price" in x
        else np.nan,
    }


def evaluate_fr2(pop: pd.DataFrame, volume_filter: bool) -> tuple[pd.DataFrame, dict]:
    g = pop.copy()
    if volume_filter:
        g = g[g.entry_vol >= VOLUME_MIN]
    g = g[g.ask_d2.between(0.60, 0.80, inclusive="both")].copy()
    g = choose_one_per_close(g, "ask_d2")
    q = 20
    g["trade_price"] = g.ask_d2
    g["fee"] = [fee_total(q, p) for p in g.trade_price]
    g["pnl"] = q * (g.won - g.trade_price) - g.fee
    g["contracts"] = q
    metrics = strategy_metrics(g, "pnl")
    metrics["volume_filter"] = volume_filter
    metrics["splits"] = {}
    for name, (a, b) in SPLITS.items():
        s = g[(g.close_dt >= a) & (g.close_dt < b)]
        metrics["splits"][name] = strategy_metrics(s, "pnl")
    return g, metrics


@dataclass
class FillPosterior:
    win_filled: int
    win_total: int
    lose_filled: int
    lose_total: int

    def means(self) -> tuple[float, float]:
        return self.win_filled / self.win_total, self.lose_filled / self.lose_total


def fill_posterior(live: pd.DataFrame, seconds: int) -> FillPosterior:
    filled = live.first_fill_seconds.notna() & (live.first_fill_seconds <= seconds)
    w = live.would_win
    return FillPosterior(
        win_filled=int((filled & w).sum()),
        win_total=int(w.sum()),
        lose_filled=int((filled & ~w).sum()),
        lose_total=int((~w).sum()),
    )


def simulate_ptc(
    pop: pd.DataFrame,
    live: pd.DataFrame,
    wait_seconds: int,
    ask_ceiling: float,
    commit_qty: int,
    repetitions: int = 1200,
    volume_filter: bool = True,
) -> dict:
    delay = max(1, int(math.ceil(wait_seconds / 60.0)))
    ask_col = f"ask_d{delay}"
    if ask_col not in pop:
        return {"error": "delay unavailable"}

    eligible = pop.copy()
    if volume_filter:
        eligible = eligible[eligible.entry_vol >= VOLUME_MIN]
    eligible = eligible[eligible[ask_col].notna()].copy()
    eligible = eligible.sort_values(["close_dt", "coin_order", "ticker"]).reset_index(drop=True)
    if eligible.empty:
        return {"error": "no eligible markets"}

    posterior = fill_posterior(live, wait_seconds)
    fw, fl = posterior.means()
    rng = np.random.default_rng(RNG_SEED + wait_seconds + int(ask_ceiling * 100) + commit_qty)

    day_codes, days = pd.factorize(eligible.day, sort=True)
    close_codes, closes = pd.factorize(eligible.close_dt, sort=True)
    won = eligible.won.to_numpy(dtype=bool)
    bid = eligible.entry_bid.to_numpy(float)
    ask = eligible[ask_col].to_numpy(float)
    coin_order = eligible.coin_order.to_numpy(int)
    n_days = len(days)

    day_mean = np.empty(repetitions)
    total = np.empty(repetitions)
    max_dd = np.empty(repetitions)
    worst_window = np.empty(repetitions)
    floor_hit = np.empty(repetitions, dtype=bool)
    commit_count = np.empty(repetitions)
    probe_fill_count = np.empty(repetitions)

    # Beta(1,1) regularisation carries fill-rate estimation uncertainty into the
    # policy distribution.  Outcomes/prices remain the actual historical path.
    for rep in range(repetitions):
        f_win = rng.beta(posterior.win_filled + 1, posterior.win_total - posterior.win_filled + 1)
        f_lose = rng.beta(posterior.lose_filled + 1, posterior.lose_total - posterior.lose_filled + 1)
        probs = np.where(won, f_win, f_lose)
        probe_filled = rng.random(len(eligible)) < probs

        pnl = np.where(probe_filled, np.where(won, 1.0 - bid, -bid), 0.0)

        # Only no-fill probes can commit.  At each close choose the cheapest ask,
        # a fully observable and pre-specified rule, and commit to one market.
        candidates = (~probe_filled) & (ask >= 0.60) & (ask <= ask_ceiling)
        selected = np.zeros(len(eligible), dtype=bool)
        if candidates.any():
            cidx = np.flatnonzero(candidates)
            tmp = pd.DataFrame(
                {
                    "idx": cidx,
                    "close": close_codes[cidx],
                    "ask": ask[cidx],
                    "coin_order": coin_order[cidx],
                }
            )
            chosen = (
                tmp.sort_values(["close", "ask", "coin_order", "idx"])
                .groupby("close", as_index=False)
                .head(PRIMARY_MAX_COMMITS_PER_CLOSE)
                .idx.to_numpy(int)
            )
            selected[chosen] = True
            fees = np.array([fee_total(commit_qty, p) for p in ask[selected]])
            pnl[selected] += commit_qty * (won[selected].astype(float) - ask[selected]) - fees

        daily = np.bincount(day_codes, weights=pnl, minlength=n_days)
        window = np.bincount(close_codes, weights=pnl, minlength=len(closes))
        eq = np.cumsum(window)
        dd = np.maximum.accumulate(eq) - eq
        day_mean[rep] = daily.mean()
        total[rep] = pnl.sum()
        max_dd[rep] = dd.max(initial=0.0)
        worst_window[rep] = window.min(initial=0.0)
        floor_hit[rep] = (300.0 + np.cumsum(daily) < 211.0).any()
        commit_count[rep] = selected.sum()
        probe_fill_count[rep] = probe_filled.sum()

    nofill_win = posterior.win_total - posterior.win_filled
    nofill_lose = posterior.lose_total - posterior.lose_filled
    nofill_post = nofill_win / (nofill_win + nofill_lose) if nofill_win + nofill_lose else np.nan

    return {
        "wait_seconds": wait_seconds,
        "delay_minutes": delay,
        "ask_ceiling": ask_ceiling,
        "probe_qty": 1,
        "commit_qty": commit_qty,
        "volume_filter": volume_filter,
        "eligible_markets": int(len(eligible)),
        "calendar_days": int(n_days),
        "live_fill_rate_win": fw,
        "live_fill_rate_lose": fl,
        "live_nofill_win_rate": nofill_post,
        "mean_commits": float(commit_count.mean()),
        "mean_probe_fills": float(probe_fill_count.mean()),
        "mean_total_pnl": float(total.mean()),
        "mean_pnl_per_day": float(day_mean.mean()),
        "pnl_day_ci_lo": float(np.quantile(day_mean, 0.025)),
        "pnl_day_ci_hi": float(np.quantile(day_mean, 0.975)),
        "p_mean_day_nonpositive": float((day_mean <= 0).mean()),
        "median_max_drawdown": float(np.median(max_dd)),
        "max_drawdown_p95": float(np.quantile(max_dd, 0.95)),
        "median_worst_window": float(np.median(worst_window)),
        "p_hit_211_over_history": float(floor_hit.mean()),
    }


def spot_features(pop: pd.DataFrame) -> pd.DataFrame:
    spot = pd.read_parquet(DATA / "spot_1m.parquet", columns=["coin", "ts", "close"])
    spot["ts"] = pd.to_numeric(spot.ts, errors="coerce")
    spot["close"] = pd.to_numeric(spot.close, errors="coerce")
    spot = spot.dropna().sort_values(["coin", "ts"])

    pieces: list[pd.DataFrame] = []
    for coin, g in pop.groupby("coin"):
        s = spot[spot.coin == coin][["ts", "close"]].drop_duplicates("ts").sort_values("ts")
        if s.empty:
            continue
        x = g.copy().sort_values("close_ts")
        x["entry_ts"] = x.close_ts - (x.entry_ml * 60).round().astype(int)
        for k in (0, 1, 2, 3):
            target = x[["ticker", "entry_ts"]].copy()
            target["target_ts"] = target.entry_ts + 60 * k
            target = target.sort_values("target_ts")
            merged = pd.merge_asof(
                target,
                s.rename(columns={"ts": "spot_ts", "close": f"spot_{k}"}).sort_values("spot_ts"),
                left_on="target_ts",
                right_on="spot_ts",
                direction="backward",
                tolerance=90,
            )
            x = x.merge(merged[["ticker", f"spot_{k}"]], on="ticker", how="left")
        d = np.where(x.side.eq("yes"), 1.0, -1.0)
        for k in (1, 2, 3):
            x[f"signed_spot_{k}m"] = d * (x[f"spot_{k}"] / x.spot_0 - 1.0)
        pieces.append(x)
    return pd.concat(pieces, ignore_index=True) if pieces else pop.copy()


def reversal_hazard_table(pop_spot: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for k in (1, 2, 3):
        col = f"signed_spot_{k}m"
        g = pop_spot.dropna(subset=[col]).copy()
        if len(g) < 100:
            continue
        # Equal-count bins are descriptive; no threshold is selected from them.
        g["bucket"] = pd.qcut(g[col].rank(method="first"), 5, labels=False)
        for b, q in g.groupby("bucket"):
            rows.append(
                {
                    "horizon_minutes": k,
                    "bucket": int(b),
                    "n": len(q),
                    "signed_spot_mean_bp": q[col].mean() * 10_000,
                    "win_rate": q.won.mean(),
                    "entry_bid": q.entry_bid.mean(),
                    "calibration_residual_c": (q.won - q.entry_bid).mean() * 100,
                }
            )
    return pd.DataFrame(rows)


def direct_live_cancel_bound(live: pd.DataFrame) -> dict:
    losing = live[~live.would_win]
    return {
        "losing_orders": int(len(losing)),
        "losses_filled_by_1s": float((losing.first_fill_seconds <= 1).mean()),
        "losses_filled_by_5s": float((losing.first_fill_seconds <= 5).mean()),
        "losses_filled_by_30s": float((losing.first_fill_seconds <= 30).mean()),
        "losses_filled_by_60s": float((losing.first_fill_seconds <= 60).mean()),
        "losses_filled_by_120s": float((losing.first_fill_seconds <= 120).mean()),
    }


def fmt_money(x: float) -> str:
    return f"${x:,.2f}"


def main() -> None:
    pop = build_population()
    live = load_live_orders()

    fill_table = fill_horizon_table(live)
    fill_table.to_csv(OUT / "probe_fill_horizons.csv", index=False)

    fr2_all, fr2_all_metrics = evaluate_fr2(pop, volume_filter=False)
    fr2_vol, fr2_vol_metrics = evaluate_fr2(pop, volume_filter=True)
    fr2_out = pd.concat(
        [fr2_all.assign(population="all"), fr2_vol.assign(population="vol_ge_2000")],
        ignore_index=True,
    )
    fr2_out.to_csv(OUT / "fr2_full_population.csv", index=False)

    sensitivity: list[dict] = []
    for t in WAIT_SECONDS:
        for c in ASK_CEILINGS:
            for q in COMMIT_QTYS:
                sensitivity.append(simulate_ptc(pop, live, t, c, q, repetitions=600))
    sens = pd.DataFrame(sensitivity)
    sens.to_csv(OUT / "ptc_sensitivity.csv", index=False)

    primary = simulate_ptc(
        pop,
        live,
        PRIMARY_WAIT_SECONDS,
        PRIMARY_ASK_CEILING,
        PRIMARY_COMMIT_QTY,
        repetitions=4000,
    )

    pop_spot = spot_features(pop)
    hazard = reversal_hazard_table(pop_spot)
    hazard.to_csv(OUT / "reversal_hazard.csv", index=False)
    cancel_bound = direct_live_cancel_bound(live)

    primary_fill = fill_posterior(live, PRIMARY_WAIT_SECONDS)
    fw, fl = primary_fill.means()
    nfw = primary_fill.win_total - primary_fill.win_filled
    nfl = primary_fill.lose_total - primary_fill.lose_filled
    p_win_nofill = nfw / (nfw + nfl) if nfw + nfl else np.nan
    p_win_fill = primary_fill.win_filled / (primary_fill.win_filled + primary_fill.lose_filled)

    # Break-even all-in price for the no-fill branch, including q15 block fee.
    grid = np.arange(0.50, 0.951, 0.001)
    ev = np.array(
        [p_win_nofill - p - fee_total(PRIMARY_COMMIT_QTY, p) / PRIMARY_COMMIT_QTY for p in grid]
    )
    break_even = float(grid[ev >= 0].max()) if (ev >= 0).any() else np.nan

    summary = {
        "data": {
            "population_in_band": int(len(pop)),
            "population_days": int(pop.day.nunique()),
            "live_orders": int(len(live)),
            "live_winners": int(live.would_win.sum()),
            "live_losers": int((~live.would_win).sum()),
        },
        "fr2_no_volume": fr2_all_metrics,
        "fr2_volume_ge_2000": fr2_vol_metrics,
        "primary_ptc": primary,
        "primary_probe": {
            "fill_rate_win": fw,
            "fill_rate_lose": fl,
            "p_win_if_no_fill": p_win_nofill,
            "p_win_if_fill": p_win_fill,
            "break_even_all_in_price": break_even,
        },
        "minute_level_cancel_bound": cancel_bound,
    }
    (OUT / "probe_then_commit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Report -----------------------------------------------------------------
    def split_line(name: str, metrics: dict) -> str:
        s = metrics["splits"][name]
        return f"| {name.title()} | {s['n']} | {s['edge_per_contract']*100:+.2f}¢ | {fmt_money(s['pnl'])} |"

    primary_sens = sens[
        (sens.wait_seconds == PRIMARY_WAIT_SECONDS)
        & (sens.ask_ceiling == PRIMARY_ASK_CEILING)
    ].sort_values("commit_qty")
    wait_sens = sens[(sens.ask_ceiling == PRIMARY_ASK_CEILING) & (sens.commit_qty == PRIMARY_COMMIT_QTY)]

    report = f"""# Probe-Then-Commit (PTC) — clean-room audit

## Executive result

The strongest new mechanism is **not another directional filter**.  It is an
execution inversion:

> Post one contract as a diagnostic maker probe.  A quick fill is treated as a
> warning and receives no additional size.  A sufficiently long no-fill is
> treated as favorable information; cancel the probe, confirm cancellation, and
> take at most one bounded full-size position in the close window.

This uses the project's proven adverse selection as information instead of
letting it decide the entire position size.

The frozen primary candidate is:

```text
probe quantity          1
wait                    {PRIMARY_WAIT_SECONDS} seconds
commit condition        probe still unfilled
commit ask              60¢ through {PRIMARY_ASK_CEILING*100:.0f}¢
commit quantity         {PRIMARY_COMMIT_QTY}
commit cap              one market per close window
selection               lowest observable ask, then fixed coin order
execution               cancel-confirm-reconcile, IOC, no retry, no chase
```

### Primary model-based result

| Metric | Estimate |
|---|---:|
| Live P(fill by 60s \| eventual win) | {fw:.2%} |
| Live P(fill by 60s \| eventual loss) | {fl:.2%} |
| Posterior P(win \| no fill by 60s) | **{p_win_nofill:.2%}** |
| P(win \| fill by 60s) | {p_win_fill:.2%} |
| Break-even all-in price after q15 fee | {break_even*100:.1f}¢ |
| Mean commits over 73-day population | {primary['mean_commits']:.1f} |
| Mean modeled P&L | **{fmt_money(primary['mean_total_pnl'])}** |
| Mean modeled P&L/day | **{fmt_money(primary['mean_pnl_per_day'])}** |
| 95% interval from fill-rate uncertainty | [{fmt_money(primary['pnl_day_ci_lo'])}, {fmt_money(primary['pnl_day_ci_hi'])}]/day |
| P(mean day ≤ 0) | {primary['p_mean_day_nonpositive']:.3f} |
| Median max drawdown | {fmt_money(primary['median_max_drawdown'])} |
| 95th-percentile max drawdown | {fmt_money(primary['max_drawdown_p95'])} |
| P(hit $211 from $300 over historical sequence) | {primary['p_hit_211_over_history']:.3%} |

**Classification:** this is a strong mechanistic candidate if the modeled
interval is positive, but it is not yet identified as live policy value.  The
fill branch is observed; the counterfactual IOC branch is not.  A randomized
prospective experiment is still required.

## Why the design is different

The old maker strategy allowed an informed taker to decide whether the account
received 10–30 contracts.  PTC makes the taker decide only whether the account
receives **one** diagnostic contract.  Full size is reserved for the branch in
which the market refuses to trade back to the maker bid.

The asymmetry is deliberate:

- toxic fill branch: at most one maker contract;
- favorable no-fill branch: at most one q{PRIMARY_COMMIT_QTY} IOC per close;
- no-fill but expensive ask: no trade;
- no duplicated position and no chase.

## Actual live discrimination by horizon

The following branch labels use actual first-fill timestamps and recovered
settlements, including never-filled orders as no-fill observations.

{fill_table[fill_table.branch.isin(['filled_by_t','not_filled_by_t'])].pivot(index='seconds', columns='branch', values='win_rate').reset_index().to_markdown(index=False, floatfmt='.4f')}

The purpose of the table is not to select the best historical second.  It shows
when no-fill becomes informative enough to pay the spread and fee.  The primary
60-second wait was frozen because it is the first fully observable one-minute
price point and can be implemented by an event-driven scheduler.

## Frozen FR2 rerun on the full population

The earlier FR2 result came from the hot ``ladder_paths`` sample.  On
``paths_full`` the exact two-minute, 60–80¢ taker rule produces:

| Population | Trades | Edge/contract | P&L | Mean/day | Max DD |
|---|---:|---:|---:|---:|---:|
| No volume filter | {fr2_all_metrics['n']} | {fr2_all_metrics['edge_per_contract']*100:+.2f}¢ | {fmt_money(fr2_all_metrics['pnl'])} | {fmt_money(fr2_all_metrics['mean_day'])} | {fmt_money(fr2_all_metrics['max_drawdown'])} |
| Volume ≥ 2,000 | {fr2_vol_metrics['n']} | {fr2_vol_metrics['edge_per_contract']*100:+.2f}¢ | {fmt_money(fr2_vol_metrics['pnl'])} | {fmt_money(fr2_vol_metrics['mean_day'])} | {fmt_money(fr2_vol_metrics['max_drawdown'])} |

Chronological result for the production-like volume-filtered rule:

| Split | Trades | Edge/contract | q20 P&L |
|---|---:|---:|---:|
{split_line('train', fr2_vol_metrics)}
{split_line('valid', fr2_vol_metrics)}
{split_line('test', fr2_vol_metrics)}

This is the decisive full-population check on the previously proposed FR2 rule.

## PTC quantity sensitivity at the frozen 60s / 80¢ rule

{primary_sens[['commit_qty','mean_pnl_per_day','pnl_day_ci_lo','pnl_day_ci_hi','median_max_drawdown','p_hit_211_over_history']].to_markdown(index=False, floatfmt='.4f')}

## Wait sensitivity — descriptive only

{wait_sens[['wait_seconds','delay_minutes','live_nofill_win_rate','mean_pnl_per_day','pnl_day_ci_lo','pnl_day_ci_hi','median_max_drawdown']].to_markdown(index=False, floatfmt='.4f')}

A real policy should show a broad, economically coherent region rather than one
isolated timeout.  No timeout is promoted from this table.

## Audit C: can a one-minute signed-index cancel prevent the losses?

Among actual losing orders:

| Cutoff | Share whose first fill already occurred |
|---|---:|
| 1 second | {cancel_bound['losses_filled_by_1s']:.2%} |
| 5 seconds | {cancel_bound['losses_filled_by_5s']:.2%} |
| 30 seconds | {cancel_bound['losses_filled_by_30s']:.2%} |
| 60 seconds | {cancel_bound['losses_filled_by_60s']:.2%} |
| 120 seconds | {cancel_bound['losses_filled_by_120s']:.2%} |

If most losing orders are already filled before one complete Coinbase bar is
available, a one-minute state cancel cannot be the main remedy.  Higher-frequency
spot would be needed to test a subsecond cancel.  PTC avoids that race by making
the pre-signal exposure one contract.

The descriptive reversal-hazard table is stored in
``research/results/reversal_hazard.csv``.  It uses Coinbase only as a proxy and
does not redefine settlement.

## Queue-position correction

``queue_ahead = 0`` is only a **lower bound**, not proof that initial queue was
zero.  Volume ahead can be canceled, and one aggressor trade can consume both
orders ahead and the first probe fill at the same timestamp.  The official
queue endpoint is the authoritative prospective measurement.  PTC does not
require the historical queue reconstruction to be exact; it requires only the
observed first-fill timestamp of a one-contract-equivalent probe.

## What must be tested live

Use a dedicated event-driven order lifecycle, not the five-second scan loop:

1. Place one post-only probe at the favorite bid.
2. Persist its treatment before submission.
3. At exactly 60 seconds, request cancellation if still resting.
4. Wait for authoritative cancellation confirmation.
5. Reconcile fills during the cancel race.
6. Refresh the book.
7. Among no-fill candidates in the close window, select the lowest ask no higher
   than 80¢.
8. Send one IOC for q15; no retry and no chase.
9. Maker probe plus IOC quantity may never exceed the configured target on one
   ticker.
10. Record every eligible opportunity, including no-fill, rejection, risk block,
    partial IOC, and API failure.

Randomize eligible close windows between:

- control: existing q15 maker;
- PTC: q1 probe plus frozen no-fill commit;
- diagnostic-only: q1 probe, never commit.

The diagnostic-only arm separates the value of reducing toxic exposure from the
value of the no-fill IOC branch.

## PASS requirements

PTC enters production size only if prospective data show:

- positive P&L per assigned opportunity;
- positive day/block-clustered lower bound;
- superiority to both control and diagnostic-only arms;
- no dependence on one coin or one week;
- actual IOC prices/depth inside the frozen ceiling;
- latency sensitivity that worsens monotonically;
- risk of hitting the configured floor within the approved limit.

Until then the result is **candidate, not deployable proof**.
"""
    (OUT / "probe_then_commit_report.md").write_text(report, encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT / 'probe_then_commit_report.md'}")


if __name__ == "__main__":
    main()
