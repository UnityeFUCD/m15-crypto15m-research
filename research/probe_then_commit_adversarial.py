"""Adversarial double-test of Probe-Then-Commit (PTC).

The first PTC audit combined two real datasets but still made a strong
conditional-independence assumption: it drew probe fill/no-fill from the live
303-order outcome rates and paired that draw with the delayed ask observed in
the 73-day price-path population.  In reality those variables are coupled.  A
favorite that runs away may be both unfilled and expensive; a favorite that
weakens may both fill and become cheap.

This script removes that assumption wherever history permits.  It joins each
of the 303 actual LSM orders to its own full one-minute price path and uses:

* the actual first-fill timestamp;
* the actual eventual outcome;
* the first completed price observation after each frozen wait;
* the actual maker price of the probe;
* exact historical taker-fee rounding.

It then replays the PTC branches directly:

* first fill before cancel acknowledgement -> one maker probe contract only;
* no fill -> cancel, confirm, then at most one IOC commitment per close window;
* no retry, no chase, and a fixed ask ceiling.

This is still a counterfactual execution study.  It does not know future IOC
depth, quote-to-order latency, or whether changing the visible maker quantity
from 10-30 to one contract changes taker behavior.  Therefore it stress-tests
cancel latency, slippage, IOC fill fractions, and moving time blocks, and it
labels every result as descriptive rather than causal.

Run:
    python research/probe_then_commit_adversarial.py

Outputs:
    research/results/ptc_direct_live.csv
    research/results/ptc_direct_branches.csv
    research/results/ptc_direct_sensitivity.csv
    research/results/ptc_adversarial_summary.json
    research/results/ptc_adversarial_report.md
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

RNG = np.random.default_rng(202608061)
COIN_ORDER = {"BTC": 0, "ETH": 1, "SOL": 2, "XRP": 3, "DOGE": 4, "HYPE": 5}

PRIMARY_WAIT_SECONDS = 60
PRIMARY_ASK_FLOOR = 0.60
PRIMARY_ASK_CEILING = 0.80
PRIMARY_COMMIT_QTY = 15
PRIMARY_CANCEL_LATENCY_SECONDS = 0.0
PRIMARY_SLIPPAGE_CENTS = 0.0
PRIMARY_IOC_FILL_FRACTION = 1.0

WAITS = [60, 120, 180, 300]
ASK_CEILINGS = [0.76, 0.78, 0.80, 0.82, 0.85]
COMMIT_QTYS = [10, 15, 20]
CANCEL_LATENCIES = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0]
SLIPPAGE_CENTS = [0.0, 1.0, 2.0, 3.0, 5.0]
IOC_FILL_FRACTIONS = [0.25, 0.50, 0.75, 1.00]


def fee_total(qty: int, price: float) -> float:
    if qty <= 0:
        return 0.0
    raw = 0.07 * qty * price * (1.0 - price)
    return math.ceil(raw * 10_000 - 1e-12) / 10_000


def held_quote(side: str, yes_bid: float, yes_ask: float) -> tuple[float, float]:
    if side == "yes":
        return yes_bid, yes_ask
    if side == "no":
        return 1.0 - yes_ask, 1.0 - yes_bid
    raise ValueError(side)


def parse_path(value: object, close_ts: int, side: str) -> list[dict]:
    try:
        pts = json.loads(value) if isinstance(value, str) else value
    except Exception:
        return []
    out: list[dict] = []
    for p in pts or []:
        try:
            ml = float(p["ml"])
            yb = float(p.get("yb", p.get("bc")))
            ya = float(p.get("ya", p.get("ac")))
            if not (0 < yb < ya < 1):
                continue
            bid, ask = held_quote(side, yb, ya)
            ts = float(close_ts) - 60.0 * ml
        except (KeyError, TypeError, ValueError):
            continue
        out.append({"ml": ml, "ts": ts, "bid": bid, "ask": ask})
    return sorted(out, key=lambda x: x["ts"])


def first_completed_after(points: list[dict], decision_ts: float) -> dict | None:
    candidates = [p for p in points if p["ts"] >= decision_ts - 1e-9]
    return candidates[0] if candidates else None


def load_outcomes() -> pd.DataFrame:
    u = pd.read_parquet(DATA / "underlying.parquet", columns=["ticker", "result"])
    u = u[u.result.isin(["yes", "no"])].drop_duplicates("ticker")
    p = DATA / "lsm_missing_outcomes.parquet"
    if p.exists():
        x = pd.read_parquet(p)
        x = x[x.result.isin(["yes", "no"])]
        x = x[~x.ticker.isin(u.ticker)]
        u = pd.concat([u, x[["ticker", "result"]]], ignore_index=True)
    return u.drop_duplicates("ticker")


def load_orders_with_paths() -> pd.DataFrame:
    orders = pd.read_parquet(DATA / "orders_history.parquet")
    fills = pd.read_parquet(DATA / "fills_history.parquet")
    paths = pd.read_parquet(DATA / "paths_full.parquet")
    outcomes = load_outcomes()

    lsm = orders[orders.client_order_id.astype(str).str.startswith("lsm")].copy()
    if len(lsm) != 303:
        raise RuntimeError(f"expected 303 LSM orders, found {len(lsm)}")
    if lsm.order_id.duplicated().any():
        raise RuntimeError("duplicate LSM order_id")

    lsm["held"] = np.where(
        lsm.action.eq("sell"),
        np.where(lsm.side.eq("yes"), "no", "yes"),
        lsm.side,
    )
    lsm["created_dt"] = pd.to_datetime(lsm.created_time, format="mixed", utc=True)
    lsm["created_ts"] = lsm.created_dt.astype("int64") / 1e9
    lsm["close_dt"] = (
        pd.to_datetime(lsm.ticker.str.split("-").str[1], format="%y%b%d%H%M", utc=True)
        + pd.Timedelta(hours=4)
    )
    lsm["close_ts"] = lsm.close_dt.astype("int64") / 1e9
    lsm["day"] = lsm.close_dt.dt.date
    lsm["coin"] = lsm.ticker.str.extract(r"^KX([A-Z]+)15M")[0]
    lsm["coin_order"] = lsm.coin.map(COIN_ORDER).fillna(99).astype(int)

    yp = pd.to_numeric(lsm.yes_price_dollars, errors="coerce")
    np_ = pd.to_numeric(lsm.no_price_dollars, errors="coerce")
    lsm["maker_price"] = np.where(lsm.held.eq("yes"), yp, np_)
    lsm["submitted_qty"] = pd.to_numeric(lsm.initial_count_fp, errors="coerce").fillna(0.0)
    lsm["filled_qty"] = pd.to_numeric(lsm.fill_count_fp, errors="coerce").fillna(0.0)

    first_fill = (
        fills.assign(fill_dt=pd.to_datetime(fills.created_time, format="mixed", utc=True))
        .groupby("order_id")
        .fill_dt.min()
        .rename("first_fill_dt")
    )
    lsm = lsm.merge(first_fill, left_on="order_id", right_index=True, how="left")
    lsm["first_fill_seconds"] = (lsm.first_fill_dt - lsm.created_dt).dt.total_seconds()
    lsm = lsm.merge(outcomes, on="ticker", how="left")
    if lsm.result.isna().any():
        raise RuntimeError(f"{lsm.result.isna().sum()} LSM outcomes unresolved")
    lsm["won"] = lsm.held.eq(lsm.result).astype(int)

    pcols = paths[["ticker", "close_ts", "path"]].drop_duplicates("ticker")
    lsm = lsm.merge(pcols, on="ticker", how="left", suffixes=("", "_path"))
    lsm["has_path"] = lsm.path.notna()

    rows: list[dict] = []
    for r in lsm.itertuples(index=False):
        close_ts = int(r.close_ts_path) if pd.notna(r.close_ts_path) else int(r.close_ts)
        points = parse_path(r.path, close_ts, r.held) if r.has_path else []
        row = {
            "order_id": r.order_id,
            "ticker": r.ticker,
            "coin": r.coin,
            "coin_order": r.coin_order,
            "close_dt": r.close_dt,
            "close_ts": int(r.close_ts),
            "day": r.day,
            "held": r.held,
            "won": int(r.won),
            "maker_price": float(r.maker_price),
            "submitted_qty": float(r.submitted_qty),
            "filled_qty": float(r.filled_qty),
            "created_ts": float(r.created_ts),
            "first_fill_seconds": float(r.first_fill_seconds)
            if pd.notna(r.first_fill_seconds)
            else np.nan,
            "has_path": bool(points),
        }
        for wait in WAITS:
            p = first_completed_after(points, float(r.created_ts) + wait)
            row[f"snapshot_wait_{wait}"] = (
                float(p["ts"] - float(r.created_ts)) if p else np.nan
            )
            row[f"ask_{wait}"] = float(p["ask"]) if p else np.nan
            row[f"bid_{wait}"] = float(p["bid"]) if p else np.nan
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["close_dt", "created_ts", "coin_order"])
    # The reconciled ledger identity is a hard guard against the old survivorship bug.
    actual = (out.filled_qty * (out.won - out.maker_price)).sum()
    if abs(actual - 9.38) > 0.02:
        raise RuntimeError(f"actual ledger P&L {actual:.4f} does not reconcile to $9.38")
    return out.reset_index(drop=True)


def standardized_control(data: pd.DataFrame, qty: int = 15) -> pd.DataFrame:
    x = data.copy()
    frac = np.divide(
        x.filled_qty,
        x.submitted_qty,
        out=np.zeros(len(x), dtype=float),
        where=x.submitted_qty.to_numpy() > 0,
    )
    x["pnl_control"] = qty * frac * (x.won - x.maker_price)
    return x


def replay(
    data: pd.DataFrame,
    wait_seconds: int,
    ask_ceiling: float,
    commit_qty: int,
    cancel_latency_seconds: float,
    slippage_cents: float,
    ioc_fill_fraction: float,
) -> tuple[pd.DataFrame, dict]:
    x = data.copy()
    ask_col = f"ask_{wait_seconds}"
    wait_col = f"snapshot_wait_{wait_seconds}"
    if ask_col not in x:
        raise ValueError(wait_seconds)

    # The decision can only occur at the first completed one-minute path point.
    # A fill during the subsequent cancellation race belongs to the toxic probe branch.
    effective_cutoff = x[wait_col] + cancel_latency_seconds
    x["probe_filled"] = (
        x.first_fill_seconds.notna()
        & effective_cutoff.notna()
        & (x.first_fill_seconds <= effective_cutoff)
    )
    x["probe_pnl"] = np.where(x.probe_filled, x.won - x.maker_price, 0.0)

    exec_price = x[ask_col] + slippage_cents / 100.0
    x["exec_price"] = exec_price
    x["commit_candidate"] = (
        ~x.probe_filled
        & exec_price.notna()
        & exec_price.between(PRIMARY_ASK_FLOOR, ask_ceiling, inclusive="both")
    )
    x["commit_selected"] = False
    candidates = x[x.commit_candidate].copy()
    if not candidates.empty:
        chosen = (
            candidates.sort_values(["close_dt", "exec_price", "coin_order", "created_ts"])
            .groupby("close_dt", as_index=False)
            .head(1)
            .index
        )
        x.loc[chosen, "commit_selected"] = True

    fill_qty = int(math.floor(commit_qty * ioc_fill_fraction + 1e-9))
    x["ioc_filled_qty"] = np.where(x.commit_selected, fill_qty, 0)
    fees = np.zeros(len(x))
    if fill_qty > 0:
        mask = x.commit_selected.to_numpy()
        fees[mask] = [fee_total(fill_qty, p) for p in x.loc[mask, "exec_price"]]
    x["ioc_fee"] = fees
    x["commit_pnl"] = x.ioc_filled_qty * (x.won - x.exec_price) - x.ioc_fee
    x["pnl_diagnostic"] = x.probe_pnl
    x["pnl_ptc"] = x.probe_pnl + x.commit_pnl

    control = standardized_control(x, commit_qty)
    x["pnl_control"] = control.pnl_control
    metrics = aggregate_metrics(x)
    metrics.update(
        {
            "wait_seconds": wait_seconds,
            "ask_ceiling": ask_ceiling,
            "commit_qty": commit_qty,
            "cancel_latency_seconds": cancel_latency_seconds,
            "slippage_cents": slippage_cents,
            "ioc_fill_fraction": ioc_fill_fraction,
            "path_matched": int(x.has_path.sum()),
            "probe_fills": int(x.probe_filled.sum()),
            "commits": int(x.commit_selected.sum()),
            "commit_win_rate": float(x.loc[x.commit_selected, "won"].mean())
            if x.commit_selected.any()
            else np.nan,
            "commit_mean_ask": float(x.loc[x.commit_selected, "exec_price"].mean())
            if x.commit_selected.any()
            else np.nan,
        }
    )
    return x, metrics


def drawdown(values: pd.Series) -> float:
    c = values.cumsum()
    return float((c.cummax() - c).max()) if len(c) else 0.0


def aggregate_metrics(x: pd.DataFrame) -> dict:
    window = x.groupby("close_dt")[["pnl_ptc", "pnl_diagnostic", "pnl_control"]].sum().sort_index()
    daily = x.groupby("day")[["pnl_ptc", "pnl_diagnostic", "pnl_control"]].sum()
    return {
        "ptc_pnl": float(x.pnl_ptc.sum()),
        "diagnostic_pnl": float(x.pnl_diagnostic.sum()),
        "control_pnl": float(x.pnl_control.sum()),
        "ptc_minus_diagnostic": float((x.pnl_ptc - x.pnl_diagnostic).sum()),
        "ptc_minus_control": float((x.pnl_ptc - x.pnl_control).sum()),
        "ptc_mean_day": float(daily.pnl_ptc.mean()),
        "diagnostic_mean_day": float(daily.pnl_diagnostic.mean()),
        "control_mean_day": float(daily.pnl_control.mean()),
        "ptc_max_drawdown": drawdown(window.pnl_ptc),
        "diagnostic_max_drawdown": drawdown(window.pnl_diagnostic),
        "control_max_drawdown": drawdown(window.pnl_control),
        "ptc_worst_window": float(window.pnl_ptc.min()),
        "diagnostic_worst_window": float(window.pnl_diagnostic.min()),
        "control_worst_window": float(window.pnl_control.min()),
    }


def branch_table(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for wait in WAITS:
        ask_col = f"ask_{wait}"
        wait_col = f"snapshot_wait_{wait}"
        effective = data[wait_col]
        filled = (
            data.first_fill_seconds.notna()
            & effective.notna()
            & (data.first_fill_seconds <= effective)
        )
        for label, mask in (("filled", filled), ("no_fill", ~filled & effective.notna())):
            g = data[mask]
            eligible = g[ask_col].between(0.60, 0.80, inclusive="both")
            q = g[eligible]
            if len(q):
                fee_pc = np.array([fee_total(15, p) / 15 for p in q[ask_col]])
                ev = q.won.to_numpy() - q[ask_col].to_numpy() - fee_pc
            else:
                ev = np.array([], dtype=float)
            rows.append(
                {
                    "wait_seconds": wait,
                    "branch": label,
                    "n": len(g),
                    "win_rate": float(g.won.mean()) if len(g) else np.nan,
                    "mean_ask": float(g[ask_col].mean()) if len(g) else np.nan,
                    "share_ask_60_80": float(eligible.mean()) if len(g) else np.nan,
                    "eligible_n": int(eligible.sum()),
                    "direct_taker_edge_c": float(ev.mean() * 100) if len(ev) else np.nan,
                    "mean_effective_wait": float(g[wait_col].mean()) if len(g) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def nonoverlap_blocks(window_diff: pd.Series, block_len: int) -> np.ndarray:
    s = window_diff.sort_index()
    frame = s.rename("d").reset_index()
    frame["day"] = pd.to_datetime(frame.close_dt, utc=True).dt.date
    blocks: list[float] = []
    for _, g in frame.groupby("day", sort=True):
        vals = g.d.to_numpy(float)
        for start in range(0, len(vals), block_len):
            chunk = vals[start : start + block_len]
            if len(chunk):
                blocks.append(float(chunk.sum()))
    return np.asarray(blocks, dtype=float)


def block_bootstrap(x: pd.DataFrame, lhs: str, rhs: str, reps: int = 12000) -> list[dict]:
    window = x.groupby("close_dt")[[lhs, rhs]].sum().sort_index()
    diff = window[lhs] - window[rhs]
    rows: list[dict] = []
    for block_len, minutes in ((1, 15), (2, 30), (4, 60), (6, 90)):
        blocks = nonoverlap_blocks(diff, block_len)
        if len(blocks) < 2:
            continue
        draws = RNG.integers(0, len(blocks), size=(reps, len(blocks)))
        totals = blocks[draws].sum(axis=1)
        rows.append(
            {
                "comparison": f"{lhs}_minus_{rhs}",
                "block_minutes": minutes,
                "observed": float(diff.sum()),
                "ci_lo": float(np.quantile(totals, 0.025)),
                "ci_hi": float(np.quantile(totals, 0.975)),
                "p_nonpositive": float((totals <= 0).mean()),
                "n_blocks": int(len(blocks)),
            }
        )
    return rows


def main() -> None:
    data = load_orders_with_paths()
    data.to_csv(OUT / "ptc_direct_live.csv", index=False)

    branches = branch_table(data)
    branches.to_csv(OUT / "ptc_direct_branches.csv", index=False)

    primary_x, primary = replay(
        data,
        PRIMARY_WAIT_SECONDS,
        PRIMARY_ASK_CEILING,
        PRIMARY_COMMIT_QTY,
        PRIMARY_CANCEL_LATENCY_SECONDS,
        PRIMARY_SLIPPAGE_CENTS,
        PRIMARY_IOC_FILL_FRACTION,
    )

    sensitivity: list[dict] = []
    # One-factor-at-a-time stress around the frozen primary policy.
    for wait in WAITS:
        _, m = replay(data, wait, 0.80, 15, 0.0, 0.0, 1.0)
        m["stress_axis"] = "wait"
        sensitivity.append(m)
    for ceiling in ASK_CEILINGS:
        _, m = replay(data, 60, ceiling, 15, 0.0, 0.0, 1.0)
        m["stress_axis"] = "ask_ceiling"
        sensitivity.append(m)
    for qty in COMMIT_QTYS:
        _, m = replay(data, 60, 0.80, qty, 0.0, 0.0, 1.0)
        m["stress_axis"] = "quantity"
        sensitivity.append(m)
    for latency in CANCEL_LATENCIES:
        _, m = replay(data, 60, 0.80, 15, latency, 0.0, 1.0)
        m["stress_axis"] = "cancel_latency"
        sensitivity.append(m)
    for slip in SLIPPAGE_CENTS:
        _, m = replay(data, 60, 0.80, 15, 0.0, slip, 1.0)
        m["stress_axis"] = "slippage"
        sensitivity.append(m)
    for frac in IOC_FILL_FRACTIONS:
        _, m = replay(data, 60, 0.80, 15, 0.0, 0.0, frac)
        m["stress_axis"] = "ioc_fill_fraction"
        sensitivity.append(m)
    sens = pd.DataFrame(sensitivity)
    sens.to_csv(OUT / "ptc_direct_sensitivity.csv", index=False)

    bootstrap = block_bootstrap(primary_x, "pnl_ptc", "pnl_diagnostic")
    bootstrap += block_bootstrap(primary_x, "pnl_ptc", "pnl_control")

    summary = {
        "data": {
            "orders": int(len(data)),
            "paths_matched": int(data.has_path.sum()),
            "days": int(data.day.nunique()),
            "close_windows": int(data.close_dt.nunique()),
            "actual_reconciled_pnl": float((data.filled_qty * (data.won - data.maker_price)).sum()),
        },
        "primary": primary,
        "block_bootstrap": bootstrap,
        "branches": branches.to_dict("records"),
    }
    (OUT / "ptc_adversarial_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    b60 = branches[branches.wait_seconds == 60].set_index("branch")
    nofill = b60.loc["no_fill"]
    filled = b60.loc["filled"]
    boot = pd.DataFrame(bootstrap)

    def money(x: float) -> str:
        return f"${x:,.2f}"

    report = f"""# Probe-Then-Commit — adversarial direct-live double test

## Why this rerun was necessary

The first PTC audit was promising, but it paired simulated fill/no-fill with
population delayed asks conditional only on the eventual outcome.  That can
invent cheap no-fill opportunities.  This audit instead joins each actual LSM
order to its own committed full path and keeps fill time, delayed ask, outcome,
and maker price together.

## Data integrity

| Item | Value |
|---|---:|
| LSM orders | {len(data)} |
| Orders matched to a full path | {int(data.has_path.sum())} |
| Close windows | {data.close_dt.nunique()} |
| Calendar days | {data.day.nunique()} |
| Reconciled actual P&L | {money(summary['data']['actual_reconciled_pnl'])} |

## Frozen primary direct replay

```text
probe quantity          1
nominal wait            60 seconds
actual decision point   first completed one-minute path point after 60s
cancel latency stress   0s in primary; 0.25-3s reported separately
commit condition        probe still unfilled at effective cancel time
commit ask              60-80c
commit quantity         15
commit cap              one per close window
IOC                     displayed ask, no retry, no chase
```

| Metric | Direct two-day counterfactual |
|---|---:|
| Probe fills | {primary['probe_fills']} |
| Full-size commitments | {primary['commits']} |
| Commit win rate | {primary['commit_win_rate']:.2%} |
| Mean commit ask | {primary['commit_mean_ask']*100:.2f}c |
| Diagnostic q1 P&L | {money(primary['diagnostic_pnl'])} |
| PTC P&L | **{money(primary['ptc_pnl'])}** |
| Standardized q15 maker control | {money(primary['control_pnl'])} |
| PTC minus diagnostic | {money(primary['ptc_minus_diagnostic'])} |
| PTC minus standardized control | {money(primary['ptc_minus_control'])} |
| PTC max drawdown | {money(primary['ptc_max_drawdown'])} |
| PTC worst close window | {money(primary['ptc_worst_window'])} |

### The actual 60-second branches

| Branch | n | Win rate | Mean observed ask | Ask in 60-80c | Direct q15 taker edge |
|---|---:|---:|---:|---:|---:|
| Filled before decision | {int(filled['n'])} | {filled['win_rate']:.2%} | {filled['mean_ask']*100:.2f}c | {filled['share_ask_60_80']:.2%} | {filled['direct_taker_edge_c']:+.2f}c |
| Still unfilled | {int(nofill['n'])} | **{nofill['win_rate']:.2%}** | {nofill['mean_ask']*100:.2f}c | {nofill['share_ask_60_80']:.2%} | **{nofill['direct_taker_edge_c']:+.2f}c** |

This table is the central test.  If the no-fill branch remains profitable after
its own observed ask is included, the mechanism is not an artifact of pairing
no-fill with unrelated cheap quotes.

## Moving-block uncertainty

{boot.to_markdown(index=False, floatfmt='.4f')}

With only two live days, none of these intervals should be read as a durable
performance estimate.  They are a hostile concentration check, not prospective
proof.

## Sensitivity

### Wait

{sens[sens.stress_axis == 'wait'][['wait_seconds','commits','commit_win_rate','commit_mean_ask','ptc_pnl','ptc_max_drawdown']].to_markdown(index=False, floatfmt='.4f')}

### Cancel latency

{sens[sens.stress_axis == 'cancel_latency'][['cancel_latency_seconds','commits','ptc_pnl','ptc_max_drawdown']].to_markdown(index=False, floatfmt='.4f')}

### Slippage

{sens[sens.stress_axis == 'slippage'][['slippage_cents','commits','ptc_pnl','ptc_max_drawdown']].to_markdown(index=False, floatfmt='.4f')}

### IOC fill fraction

{sens[sens.stress_axis == 'ioc_fill_fraction'][['ioc_fill_fraction','commits','ptc_pnl','ptc_max_drawdown']].to_markdown(index=False, floatfmt='.4f')}

A credible effect should deteriorate monotonically with latency and slippage.
Failure of that sanity check is a rejection signal, as it was for RACE.

## Interpretation limits

1. The q1 probe is inferred from first-fill time of original q10-30 orders.
   Visible size may affect taker behavior.
2. The first completed one-minute path point can be almost a minute later than
   the nominal wait; the effective wait is reported in the branch CSV.
3. IOC depth and partial fills were not historically observed.  Fill-fraction
   and slippage rows are stress tests, not estimates.
4. The sample has 303 orders and two days.  It cannot identify long-run P&L or
   ruin probability.
5. A real trial must randomize by close window between control, diagnostic-only,
   and PTC.  The account remains below its kill floor, so this audit does not
   authorize live orders.

## Decision rule

PTC deserves a prospective trial only when all of the following are true in
this direct audit:

- no-fill branch edge remains positive after its own observed ask and fee;
- PTC beats diagnostic-only, showing the commit branch adds value;
- PTC beats the standardized maker control;
- latency and slippage sensitivity are monotone;
- no single close block carries the result.

Passing those conditions makes PTC a strong candidate, not a deployment PASS.
"""
    (OUT / "ptc_adversarial_report.md").write_text(report, encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT / 'ptc_adversarial_report.md'}")


if __name__ == "__main__":
    main()
