"""PTC adversarial audit v2 — fixes timestamp units and path-selection bias.

The first direct PTC audit accidentally converted pandas microsecond datetimes
as if they were nanoseconds.  That made the effective wait roughly 1.8 billion
seconds and classified eventual no-fills as sixty-second no-fills.  This is the
same class of timestamp-unit bug that previously corrupted floor alignment.

This v2 audit:

* converts every datetime with ``Timestamp.timestamp()``;
* asserts that order-to-close time is 0-15 minutes;
* asserts that a one-minute decision snapshot is within 60-150 seconds of a
  nominal 60-second wait;
* restricts direct policy comparisons to orders with an actual full path;
* reports the outcome/P&L of unmatched orders so missing paths cannot be hidden;
* keeps actual fill time, delayed ask, side and outcome jointly observed.

It remains a two-day counterfactual IOC study, not prospective proof.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "research" / "results"
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(202608062)
COIN_ORDER = {"BTC": 0, "ETH": 1, "SOL": 2, "XRP": 3, "DOGE": 4, "HYPE": 5}
WAITS = [60, 120, 180, 300]
CEILINGS = [0.76, 0.78, 0.80, 0.82, 0.85]
QTYS = [10, 15, 20]
LATENCIES = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0]
SLIPPAGES = [0.0, 1.0, 2.0, 3.0, 5.0]
FILL_FRACS = [0.25, 0.50, 0.75, 1.00]

PRIMARY_WAIT = 60
PRIMARY_CEILING = 0.80
PRIMARY_QTY = 15
ASK_FLOOR = 0.60


def fee_total(qty: int, price: float) -> float:
    if qty <= 0:
        return 0.0
    return math.ceil(0.07 * qty * price * (1 - price) * 10_000 - 1e-12) / 10_000


def epoch_seconds(value: object) -> float:
    return float(pd.Timestamp(value).timestamp())


def held_quote(side: str, yes_bid: float, yes_ask: float) -> tuple[float, float]:
    if side == "yes":
        return yes_bid, yes_ask
    if side == "no":
        return 1.0 - yes_ask, 1.0 - yes_bid
    raise ValueError(side)


def parse_path(value: object, close_ts: int, side: str) -> list[dict]:
    try:
        raw = json.loads(value) if isinstance(value, str) else value
    except Exception:
        return []
    points: list[dict] = []
    for p in raw or []:
        try:
            ml = float(p["ml"])
            yb = float(p.get("yb", p.get("bc")))
            ya = float(p.get("ya", p.get("ac")))
            if not (0 < yb < ya < 1):
                continue
            bid, ask = held_quote(side, yb, ya)
            points.append(
                {
                    "ml": ml,
                    "ts": float(close_ts) - 60.0 * ml,
                    "bid": bid,
                    "ask": ask,
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(points, key=lambda p: p["ts"])


def first_snapshot(points: list[dict], decision_ts: float) -> dict | None:
    for p in points:
        if p["ts"] >= decision_ts - 1e-9:
            return p
    return None


def load_outcomes() -> pd.DataFrame:
    u = pd.read_parquet(DATA / "underlying.parquet", columns=["ticker", "result"])
    u = u[u.result.isin(["yes", "no"])].drop_duplicates("ticker")
    missing = DATA / "lsm_missing_outcomes.parquet"
    if missing.exists():
        x = pd.read_parquet(missing)
        x = x[x.result.isin(["yes", "no"]) & ~x.ticker.isin(u.ticker)]
        u = pd.concat([u, x[["ticker", "result"]]], ignore_index=True)
    return u.drop_duplicates("ticker")


def build() -> pd.DataFrame:
    orders = pd.read_parquet(DATA / "orders_history.parquet")
    fills = pd.read_parquet(DATA / "fills_history.parquet")
    paths = pd.read_parquet(DATA / "paths_full.parquet")
    outcomes = load_outcomes()

    x = orders[orders.client_order_id.astype(str).str.startswith("lsm")].copy()
    if len(x) != 303:
        raise RuntimeError(f"expected 303 LSM orders, got {len(x)}")
    x["held"] = np.where(
        x.action.eq("sell"), np.where(x.side.eq("yes"), "no", "yes"), x.side
    )
    x["created_dt"] = pd.to_datetime(x.created_time, format="mixed", utc=True)
    x["created_ts"] = x.created_dt.map(epoch_seconds)
    x["close_dt"] = (
        pd.to_datetime(x.ticker.str.split("-").str[1], format="%y%b%d%H%M", utc=True)
        + pd.Timedelta(hours=4)
    )
    x["close_ts"] = x.close_dt.map(epoch_seconds)
    x["seconds_to_close"] = x.close_ts - x.created_ts
    if not x.seconds_to_close.between(0, 900).all():
        bad = x.loc[~x.seconds_to_close.between(0, 900), ["ticker", "seconds_to_close"]]
        raise RuntimeError(f"timestamp-unit failure: order-to-close outside 0-900s\n{bad}")

    x["day"] = x.close_dt.dt.date
    x["coin"] = x.ticker.str.extract(r"^KX([A-Z]+)15M")[0]
    x["coin_order"] = x.coin.map(COIN_ORDER).fillna(99).astype(int)
    yp = pd.to_numeric(x.yes_price_dollars, errors="coerce")
    np_ = pd.to_numeric(x.no_price_dollars, errors="coerce")
    x["maker_price"] = np.where(x.held.eq("yes"), yp, np_)
    x["submitted_qty"] = pd.to_numeric(x.initial_count_fp, errors="coerce").fillna(0)
    x["filled_qty"] = pd.to_numeric(x.fill_count_fp, errors="coerce").fillna(0)

    ff = (
        fills.assign(fill_dt=pd.to_datetime(fills.created_time, format="mixed", utc=True))
        .groupby("order_id")
        .fill_dt.min()
        .rename("first_fill_dt")
    )
    x = x.merge(ff, left_on="order_id", right_index=True, how="left")
    x["first_fill_seconds"] = (x.first_fill_dt - x.created_dt).dt.total_seconds()
    x = x.merge(outcomes, on="ticker", how="left")
    if x.result.isna().any():
        raise RuntimeError("unresolved outcomes remain")
    x["won"] = x.held.eq(x.result).astype(int)

    p = paths[["ticker", "close_ts", "path"]].drop_duplicates("ticker")
    x = x.merge(p, on="ticker", how="left", suffixes=("", "_path"))

    rows: list[dict] = []
    for r in x.itertuples(index=False):
        cts = int(r.close_ts_path) if pd.notna(r.close_ts_path) else int(r.close_ts)
        pts = parse_path(r.path, cts, r.held) if isinstance(r.path, str) else []
        row = {
            "order_id": r.order_id,
            "ticker": r.ticker,
            "coin": r.coin,
            "coin_order": int(r.coin_order),
            "close_dt": r.close_dt,
            "day": r.day,
            "held": r.held,
            "won": int(r.won),
            "maker_price": float(r.maker_price),
            "submitted_qty": float(r.submitted_qty),
            "filled_qty": float(r.filled_qty),
            "created_ts": float(r.created_ts),
            "seconds_to_close": float(r.seconds_to_close),
            "first_fill_seconds": float(r.first_fill_seconds)
            if pd.notna(r.first_fill_seconds)
            else np.nan,
            "has_path": bool(pts),
        }
        for wait in WAITS:
            snap = first_snapshot(pts, float(r.created_ts) + wait)
            effective = float(snap["ts"] - float(r.created_ts)) if snap else np.nan
            row[f"effective_wait_{wait}"] = effective
            row[f"ask_{wait}"] = float(snap["ask"]) if snap else np.nan
            row[f"bid_{wait}"] = float(snap["bid"]) if snap else np.nan
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["close_dt", "created_ts", "coin_order"])
    actual = (out.filled_qty * (out.won - out.maker_price)).sum()
    if abs(actual - 9.3814) > 0.02:
        raise RuntimeError(f"ledger mismatch {actual}")

    matched = out[out.has_path]
    for wait in WAITS:
        e = matched[f"effective_wait_{wait}"].dropna()
        # First completed 1m candle should be no earlier than nominal and no more
        # than 90 seconds later.  This catches nanosecond/microsecond confusion.
        if len(e) and (e.min() < wait - 1 or e.max() > wait + 90):
            raise RuntimeError(
                f"effective wait invalid at {wait}s: min={e.min()} max={e.max()}"
            )
    return out.reset_index(drop=True)


def standardized_control(x: pd.DataFrame, qty: int) -> pd.Series:
    frac = np.divide(
        x.filled_qty,
        x.submitted_qty,
        out=np.zeros(len(x), dtype=float),
        where=x.submitted_qty.to_numpy() > 0,
    )
    return qty * frac * (x.won - x.maker_price)


def replay(
    data: pd.DataFrame,
    wait: int = PRIMARY_WAIT,
    ceiling: float = PRIMARY_CEILING,
    qty: int = PRIMARY_QTY,
    latency: float = 0.0,
    slippage_c: float = 0.0,
    fill_fraction: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    # Fair comparison: only orders whose joint path is observed.
    x = data[data.has_path & data[f"effective_wait_{wait}"].notna()].copy()
    cutoff = x[f"effective_wait_{wait}"] + latency
    x["probe_filled"] = x.first_fill_seconds.notna() & (x.first_fill_seconds <= cutoff)
    x["probe_pnl"] = np.where(x.probe_filled, x.won - x.maker_price, 0.0)
    x["exec_price"] = x[f"ask_{wait}"] + slippage_c / 100.0
    x["candidate"] = (
        ~x.probe_filled
        & x.exec_price.between(ASK_FLOOR, ceiling, inclusive="both")
    )
    x["selected"] = False
    c = x[x.candidate]
    if len(c):
        idx = (
            c.sort_values(["close_dt", "exec_price", "coin_order", "created_ts"])
            .groupby("close_dt", as_index=False)
            .head(1)
            .index
        )
        x.loc[idx, "selected"] = True

    fill_qty = int(math.floor(qty * fill_fraction + 1e-9))
    x["ioc_qty"] = np.where(x.selected, fill_qty, 0)
    x["ioc_fee"] = 0.0
    if fill_qty > 0 and x.selected.any():
        x.loc[x.selected, "ioc_fee"] = [
            fee_total(fill_qty, p) for p in x.loc[x.selected, "exec_price"]
        ]
    x["ioc_pnl"] = x.ioc_qty * (x.won - x.exec_price) - x.ioc_fee
    x["pnl_diag"] = x.probe_pnl
    x["pnl_ptc"] = x.probe_pnl + x.ioc_pnl
    x["pnl_control"] = standardized_control(x, qty)

    w = x.groupby("close_dt")[["pnl_diag", "pnl_ptc", "pnl_control"]].sum().sort_index()
    d = x.groupby("day")[["pnl_diag", "pnl_ptc", "pnl_control"]].sum()

    def dd(s: pd.Series) -> float:
        csum = s.cumsum()
        return float((csum.cummax() - csum).max())

    m = {
        "wait": wait,
        "ceiling": ceiling,
        "qty": qty,
        "latency": latency,
        "slippage_c": slippage_c,
        "fill_fraction": fill_fraction,
        "n": int(len(x)),
        "probe_fills": int(x.probe_filled.sum()),
        "no_fills": int((~x.probe_filled).sum()),
        "commits": int(x.selected.sum()),
        "commit_win_rate": float(x.loc[x.selected, "won"].mean()) if x.selected.any() else np.nan,
        "commit_mean_ask": float(x.loc[x.selected, "exec_price"].mean()) if x.selected.any() else np.nan,
        "diag_pnl": float(x.pnl_diag.sum()),
        "ptc_pnl": float(x.pnl_ptc.sum()),
        "control_pnl": float(x.pnl_control.sum()),
        "ptc_minus_diag": float((x.pnl_ptc - x.pnl_diag).sum()),
        "ptc_minus_control": float((x.pnl_ptc - x.pnl_control).sum()),
        "ptc_mean_day": float(d.pnl_ptc.mean()),
        "ptc_max_dd": dd(w.pnl_ptc),
        "ptc_worst_window": float(w.pnl_ptc.min()),
    }
    return x, m


def branches(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for wait in WAITS:
        x = data[data.has_path & data[f"effective_wait_{wait}"].notna()].copy()
        filled = x.first_fill_seconds.notna() & (
            x.first_fill_seconds <= x[f"effective_wait_{wait}"]
        )
        for name, mask in (("filled", filled), ("no_fill", ~filled)):
            g = x[mask]
            eligible = g[f"ask_{wait}"].between(0.60, 0.80, inclusive="both")
            z = g[eligible]
            if len(z):
                fee_pc = np.array([fee_total(15, p) / 15 for p in z[f"ask_{wait}"]])
                edge = z.won.to_numpy() - z[f"ask_{wait}"].to_numpy() - fee_pc
            else:
                edge = np.array([])
            rows.append(
                {
                    "wait": wait,
                    "branch": name,
                    "n": int(len(g)),
                    "win_rate": float(g.won.mean()) if len(g) else np.nan,
                    "mean_ask": float(g[f"ask_{wait}"].mean()) if len(g) else np.nan,
                    "share_ask_60_80": float(eligible.mean()) if len(g) else np.nan,
                    "eligible_n": int(eligible.sum()),
                    "taker_edge_c": float(edge.mean() * 100) if len(edge) else np.nan,
                    "mean_effective_wait": float(g[f"effective_wait_{wait}"].mean()) if len(g) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def bootstrap(primary: pd.DataFrame, reps: int = 12000) -> pd.DataFrame:
    w = primary.groupby("close_dt")[["pnl_ptc", "pnl_diag", "pnl_control"]].sum().sort_index()
    rows: list[dict] = []
    for lhs, rhs in (("pnl_ptc", "pnl_diag"), ("pnl_ptc", "pnl_control")):
        diff = (w[lhs] - w[rhs]).rename("d").reset_index()
        diff["day"] = pd.to_datetime(diff.close_dt, utc=True).dt.date
        for block_len, mins in ((1, 15), (2, 30), (4, 60), (6, 90)):
            blocks: list[float] = []
            for _, g in diff.groupby("day"):
                vals = g.d.to_numpy(float)
                for start in range(0, len(vals), block_len):
                    chunk = vals[start : start + block_len]
                    if len(chunk):
                        blocks.append(float(chunk.sum()))
            b = np.asarray(blocks)
            idx = RNG.integers(0, len(b), size=(reps, len(b)))
            totals = b[idx].sum(axis=1)
            rows.append(
                {
                    "comparison": f"{lhs}_minus_{rhs}",
                    "block_minutes": mins,
                    "observed": float(diff.d.sum()),
                    "ci_lo": float(np.quantile(totals, 0.025)),
                    "ci_hi": float(np.quantile(totals, 0.975)),
                    "p_nonpositive": float((totals <= 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    data = build()
    data.to_csv(OUT / "ptc_v2_orders.csv", index=False)

    matched = data[data.has_path]
    unmatched = data[~data.has_path]
    missing = {
        "n": int(len(unmatched)),
        "win_rate": float(unmatched.won.mean()) if len(unmatched) else np.nan,
        "actual_pnl": float((unmatched.filled_qty * (unmatched.won - unmatched.maker_price)).sum()),
        "filled_fraction": float((unmatched.filled_qty > 0).mean()) if len(unmatched) else np.nan,
        "coins": unmatched.coin.value_counts().to_dict(),
    }

    branch = branches(data)
    branch.to_csv(OUT / "ptc_v2_branches.csv", index=False)
    primary_x, primary = replay(data)
    boot = bootstrap(primary_x)

    sensitivity: list[dict] = []
    for wait in WAITS:
        _, m = replay(data, wait=wait)
        m["axis"] = "wait"
        sensitivity.append(m)
    for c in CEILINGS:
        _, m = replay(data, ceiling=c)
        m["axis"] = "ceiling"
        sensitivity.append(m)
    for q in QTYS:
        _, m = replay(data, qty=q)
        m["axis"] = "qty"
        sensitivity.append(m)
    for lag in LATENCIES:
        _, m = replay(data, latency=lag)
        m["axis"] = "latency"
        sensitivity.append(m)
    for slip in SLIPPAGES:
        _, m = replay(data, slippage_c=slip)
        m["axis"] = "slippage"
        sensitivity.append(m)
    for frac in FILL_FRACS:
        _, m = replay(data, fill_fraction=frac)
        m["axis"] = "fill_fraction"
        sensitivity.append(m)
    sens = pd.DataFrame(sensitivity)
    sens.to_csv(OUT / "ptc_v2_sensitivity.csv", index=False)

    summary = {
        "orders": int(len(data)),
        "paths_matched": int(len(matched)),
        "matched_actual_pnl": float((matched.filled_qty * (matched.won - matched.maker_price)).sum()),
        "unmatched": missing,
        "primary": primary,
        "bootstrap": boot.to_dict("records"),
        "branches": branch.to_dict("records"),
    }
    (OUT / "ptc_v2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    b60 = branch[branch.wait == 60].set_index("branch")
    f = b60.loc["filled"]
    nf = b60.loc["no_fill"]

    def money(v: float) -> str:
        return f"${v:,.2f}"

    report = f"""# Probe-Then-Commit v2 — timestamp-corrected direct audit

## Critical correction

The first adversarial run was invalid.  Pandas stored the parsed datetimes at
microsecond resolution, while the script divided their integer representation
by 1e9 as though it were nanoseconds.  The effective 60-second wait became
about 1.8 billion seconds, so eventual no-fills were misclassified as
60-second no-fills.  This v2 uses ``Timestamp.timestamp()`` and hard assertions
on every time interval.

## Coverage and missingness

| Item | Value |
|---|---:|
| LSM orders | {len(data)} |
| Full-path matched | {len(matched)} |
| Full-path unmatched | {len(unmatched)} |
| Matched actual P&L | {money(summary['matched_actual_pnl'])} |
| Unmatched actual P&L | {money(missing['actual_pnl'])} |
| Unmatched win rate | {missing['win_rate']:.2%} |
| Unmatched order fill rate | {missing['filled_fraction']:.2%} |

Direct PTC comparisons are restricted to the {len(matched)} matched orders.
The unmatched cohort is reported because path availability can be non-random.

## Frozen primary direct replay

```text
q1 diagnostic maker probe
nominal wait 60 seconds
first complete one-minute price snapshot after that wait
cancel-confirm-reconcile
one q15 IOC per close window, ask 60-80c
lowest ask first, fixed coin-order tie break
no retry, no chase
```

| Metric | Result |
|---|---:|
| Effective orders | {primary['n']} |
| Probe fills | {primary['probe_fills']} |
| No-fill branches | {primary['no_fills']} |
| Full-size commitments | {primary['commits']} |
| Commit win rate | {primary['commit_win_rate']:.2%} |
| Mean commit ask | {primary['commit_mean_ask']*100:.2f}c |
| Diagnostic-only P&L | {money(primary['diag_pnl'])} |
| PTC P&L | **{money(primary['ptc_pnl'])}** |
| Standardized q15 maker control | {money(primary['control_pnl'])} |
| PTC minus diagnostic | {money(primary['ptc_minus_diag'])} |
| PTC minus control | {money(primary['ptc_minus_control'])} |
| PTC maximum drawdown | {money(primary['ptc_max_dd'])} |
| PTC worst close window | {money(primary['ptc_worst_window'])} |

## Actual joint branch economics at the corrected 60-second decision

| Branch | n | Win rate | Mean ask | Ask in 60-80c | Direct q15 taker edge | Mean effective wait |
|---|---:|---:|---:|---:|---:|---:|
| Filled | {int(f['n'])} | {f['win_rate']:.2%} | {f['mean_ask']*100:.2f}c | {f['share_ask_60_80']:.2%} | {f['taker_edge_c']:+.2f}c | {f['mean_effective_wait']:.1f}s |
| No fill | {int(nf['n'])} | **{nf['win_rate']:.2%}** | {nf['mean_ask']*100:.2f}c | {nf['share_ask_60_80']:.2%} | **{nf['taker_edge_c']:+.2f}c** | {nf['mean_effective_wait']:.1f}s |

This is the decisive correction to the original model.  Fill state and delayed
ask now come from the same order and the same market path.

## Moving-block concentration check

{boot.to_markdown(index=False, floatfmt='.4f')}

Only two live days exist, so these intervals cannot validate a production
mean.  They test whether one isolated close block carries the result.

## Wait sensitivity

{sens[sens.axis == 'wait'][['wait','n','no_fills','commits','commit_win_rate','commit_mean_ask','ptc_pnl','ptc_max_dd']].to_markdown(index=False, floatfmt='.4f')}

## Cancel-latency sanity check

{sens[sens.axis == 'latency'][['latency','commits','ptc_pnl','ptc_max_dd']].to_markdown(index=False, floatfmt='.4f')}

## Slippage stress

{sens[sens.axis == 'slippage'][['slippage_c','commits','ptc_pnl','ptc_max_dd']].to_markdown(index=False, floatfmt='.4f')}

## IOC-fill stress

{sens[sens.axis == 'fill_fraction'][['fill_fraction','commits','ptc_pnl','ptc_max_dd']].to_markdown(index=False, floatfmt='.4f')}

## Verdict rule

PTC is promoted to a randomized prospective candidate only if:

1. the corrected no-fill branch remains positive after its own ask and fee;
2. PTC beats diagnostic-only, proving the IOC branch adds value;
3. PTC beats the standardized maker control;
4. latency/slippage worsen the result monotonically;
5. unmatched paths do not carry the opposite outcome;
6. no moving time block carries the effect.

Even a pass here is not permission to trade while the account is below its
kill floor.  It only justifies building the randomized control/diagnostic/PTC
experiment.
"""
    (OUT / "ptc_v2_report.md").write_text(report, encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT / 'ptc_v2_report.md'}")


if __name__ == "__main__":
    main()
