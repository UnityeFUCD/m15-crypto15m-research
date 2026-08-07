from __future__ import annotations
import json, math, os, time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BASE = "https://api.elections.kalshi.com/trade-api/v2"
COINS = {"BTC", "ETH", "SOL", "XRP", "DOGE", "HYPE"}
QGRID = [10, 20, 30, 40, 50, 60, 75, 100]
LGRID_MS = [0, 50, 100, 250, 500, 1000]


def pnum(s):
    return pd.to_numeric(s, errors="coerce")


def true_held(o):
    if str(o.action).lower() == "sell":
        return "no" if str(o.side).lower() == "yes" else "yes"
    return str(o.side).lower()


def coin_of(t):
    for c in COINS:
        if t.startswith("KX" + c + "15M") or (c == "XRP" and t.startswith("KXXRP15M")):
            return c
    return None


def load():
    O = pd.read_parquet(DATA / "orders_history.parquet")
    F = pd.read_parquet(DATA / "fills_history.parquet")
    U = pd.read_parquet(DATA / "underlying.parquet", columns=["ticker", "result"])
    U = U[U.result.isin(["yes", "no"])].drop_duplicates("ticker")
    mx = DATA / "lsm_missing_outcomes.parquet"
    if mx.exists():
        X = pd.read_parquet(mx)
        U = pd.concat([U, X[["ticker", "result"]]], ignore_index=True).drop_duplicates("ticker", keep="last")

    l = O[O.client_order_id.astype(str).str.startswith("lsm")].copy()
    l["held"] = [true_held(r) for r in l.itertuples()]
    l["coin"] = l.ticker.map(coin_of)
    l = l[l.coin.isin(COINS)].copy()
    l["t0"] = pd.to_datetime(l.created_time, format="mixed", utc=True)
    l["t_end"] = pd.to_datetime(l.last_update_time, format="mixed", utc=True)
    l["close_key"] = l.ticker.str.split("-", n=1).str[1]
    l["req"] = pnum(l.initial_count_fp)
    yp = pnum(l.yes_price_dollars)
    np_ = pnum(l.no_price_dollars)
    l["held_px"] = np.where(l.held.eq("yes"), yp, np_)
    l["yes_px"] = np.where(l.held.eq("yes"), yp, 1.0 - np_)
    l = l.merge(U, on="ticker", how="left")
    assert len(l) == 303, len(l)
    assert l.result.isin(["yes", "no"]).all(), l[l.result.isna()].ticker.tolist()
    l["win"] = l.held.eq(l.result)

    f = F[F.order_id.isin(l.order_id)].copy()
    f["ft"] = pd.to_datetime(f.created_time, format="mixed", utc=True)
    f["qty"] = pnum(f.count_fp)
    f["held"] = f.outcome_side.astype(str).str.lower()
    fy = pnum(f.yes_price_dollars)
    fn = pnum(f.no_price_dollars)
    f["px"] = np.where(f.held.eq("yes"), fy, fn)
    f = f.merge(
        l[["order_id", "ticker", "held", "held_px", "yes_px", "t0", "t_end", "close_key", "req", "result", "win", "coin"]],
        on=["order_id", "ticker"], how="left", suffixes=("", "_o"))
    assert (f.held == f.held_o).all()
    assert (~f.is_taker.astype(bool)).all()
    return l, f


def actual_ffcb(l, f, lat_ms):
    rows = []
    for ck, og in l.groupby("close_key"):
        fg = f[f.close_key.eq(ck)].sort_values("ft")
        if fg.empty:
            continue
        tstar = fg.ft.min()
        deadline = tstar + pd.Timedelta(milliseconds=lat_ms)
        submitted = set(og.loc[og.t0 <= tstar, "order_id"])
        h = fg[fg.order_id.isin(submitted) & (fg.ft <= deadline)].copy()
        if not h.empty:
            rows.append(h)
    H = pd.concat(rows, ignore_index=True) if rows else f.iloc[0:0].copy()
    H["pnl"] = H.qty * (H.win.astype(float) - H.px)
    return H


def fetch_raw(tickers, tmin_by_ticker, tmax_by_ticker):
    s = requests.Session()
    rows, errs = [], []
    for i, tk in enumerate(sorted(tickers), 1):
        cursor, pages = None, 0
        while pages < 10:
            p = {"ticker": tk, "limit": 1000, "min_ts": int(tmin_by_ticker[tk]), "max_ts": int(tmax_by_ticker[tk])}
            if cursor:
                p["cursor"] = cursor
            try:
                r = s.get(BASE + "/markets/trades", params=p, timeout=20)
            except Exception as e:
                errs.append((tk, "EXC", repr(e)))
                break
            if r.status_code != 200:
                errs.append((tk, r.status_code, r.text[:200]))
                break
            b = r.json()
            batch = b.get("trades") or []
            for x in batch:
                rows.append(dict(
                    ticker=tk, trade_id=x.get("trade_id"), created_time=x.get("created_time"),
                    count=float(x.get("count_fp") or x.get("count") or 0),
                    yes_px=float(x.get("yes_price_dollars") or 0),
                    no_px=float(x.get("no_price_dollars") or 0),
                    taker_book_side=x.get("taker_book_side"), taker_outcome_side=x.get("taker_outcome_side")))
            cursor = b.get("cursor")
            pages += 1
            if not cursor or not batch:
                break
        if i % 25 == 0:
            print("FETCH", i, "/", len(tickers), "rows", len(rows), "errs", len(errs), flush=True)
        time.sleep(0.03)
    D = pd.DataFrame(rows)
    if len(D):
        D = D.drop_duplicates("trade_id")
        D["t"] = pd.to_datetime(D.created_time, format="mixed", utc=True)
    return D, errs


def anchored_scaled(l, f, raw, q, lat_ms):
    byraw = {k: g.sort_values("t") for k, g in raw.groupby("ticker")} if len(raw) else {}
    out, caps = [], []
    for ck, og in l.groupby("close_key"):
        fg = f[f.close_key.eq(ck)].sort_values("ft")
        if fg.empty:
            continue
        tstar = fg.ft.min()
        deadline = tstar + pd.Timedelta(milliseconds=lat_ms)
        og = og[og.t0 <= tstar].copy()
        for r in og.itertuples():
            of = fg[(fg.order_id == r.order_id) & (fg.ft <= deadline)].sort_values("ft")
            if of.empty:
                continue
            obs_qty = float(of.qty.sum())
            qty = min(float(q), obs_qty)
            source = "private"
            all_of = f[f.order_id.eq(r.order_id)].sort_values("ft").copy()
            all_of["cum"] = all_of.qty.cumsum()
            exh = all_of[all_of.cum >= float(r.req) - 1e-9]
            texh = exh.ft.iloc[0] if len(exh) else pd.NaT
            extra = 0.0
            if qty < q and pd.notna(texh) and texh <= deadline and r.ticker in byraw:
                tr = byraw[r.ticker]
                m = (tr.t > texh) & (tr.t <= deadline)
                m &= tr.taker_outcome_side.eq(r.held)
                m &= tr.taker_book_side.eq("bid")
                m &= (tr.yes_px - float(r.yes_px)).abs() < 1e-9
                extra = float(tr.loc[m, "count"].sum())
                add = min(q - qty, extra)
                qty += add
                if add > 0:
                    source = "private+post_exhaust_public"
            if qty <= 0:
                continue
            pnl = qty * ((1.0 if r.win else 0.0) - float(r.held_px))
            out.append(dict(close_key=ck, ticker=r.ticker, coin=r.coin, order_id=r.order_id,
                            tstar=tstar, deadline=deadline, qty=qty, pnl=pnl, win=bool(r.win),
                            px=float(r.held_px), source=source, extra_public=extra))
        capreq = float((og.held_px.astype(float) * q).sum())
        caps.append((ck, tstar, capreq, len(og)))
    H = pd.DataFrame(out)
    C = pd.DataFrame(caps, columns=["close_key", "tstar", "capital_required", "orders_live"])
    return H, C


def same_trade_upper(l, f, raw, q):
    rid = raw.set_index("trade_id") if len(raw) and raw.trade_id.notna().any() else None
    rows = []
    for ck, fg in f.groupby("close_key"):
        fg = fg.sort_values("ft")
        first = fg.iloc[0]
        pub = np.nan
        if rid is not None and first.trade_id in rid.index:
            rr = rid.loc[first.trade_id]
            if isinstance(rr, pd.DataFrame):
                rr = rr.iloc[0]
            pub = float(rr["count"])
        cap = max(float(first.qty), pub if np.isfinite(pub) else 0.0)
        qty = min(float(q), cap)
        rows.append(dict(close_key=ck, qty=qty,
                         pnl=qty * ((1.0 if first.win else 0.0) - float(first.px)),
                         pub=pub, private=float(first.qty)))
    return pd.DataFrame(rows)


def summarize(H, C, days=2):
    if H.empty:
        return {}
    close = H.groupby(["close_key", "tstar"], as_index=False).agg(pnl=("pnl", "sum"), qty=("qty", "sum"), nlegs=("ticker", "size"))
    close = close.sort_values("tstar")
    close["day"] = close.tstar.dt.floor("D")
    daily = close.groupby("day").pnl.sum()
    eq = close.pnl.cumsum()
    peak = eq.cummax()
    dd = peak - eq
    return dict(
        closes=int(len(close)), contracts=float(H.qty.sum()), pnl=float(H.pnl.sum()),
        dollars_day=float(H.pnl.sum() / days), c_per_filled=float(100 * H.pnl.sum() / H.qty.sum()),
        positive_close=float((close.pnl > 0).mean()), worst_close=float(close.pnl.min()),
        max_additive_dd=float(dd.max()), worst_day=float(daily.min()), positive_days=int((daily > 0).sum()),
        max_capital=float(C.capital_required.max()) if len(C) else None,
        median_capital=float(C.capital_required.median()) if len(C) else None,
        max_orders_live=int(C.orders_live.max()) if len(C) else None)


def main():
    l, f = load()
    print("CERT 303 orders", len(f), "fills", l.result.value_counts().to_dict())
    print("\nPRIVATE CAUSAL REPLAY")
    for L in LGRID_MS:
        H = actual_ffcb(l, f, L)
        print(json.dumps({"lat_ms": L, "contracts": round(H.qty.sum(), 4), "pnl": round(H.pnl.sum(), 6),
                          "c_per_contract": round(100 * H.pnl.sum() / H.qty.sum(), 4),
                          "closes": H.close_key.nunique()}, sort_keys=True))

    first = f.sort_values("ft").groupby("close_key", as_index=False).first()
    tmin, tmax = {}, {}
    for tk, g in first.groupby("ticker"):
        tmin[tk] = g.ft.min().timestamp() - 2
        tmax[tk] = g.ft.max().timestamp() + 4
    raw, errs = fetch_raw(set(first.ticker), tmin, tmax)
    print("\nRAW_FETCH rows", len(raw), "tickers", raw.ticker.nunique() if len(raw) else 0, "errors", len(errs))
    if errs:
        print("FETCH_ERRORS", errs[:20])
    if len(raw):
        rf = raw[["trade_id", "count", "ticker", "t"]].drop_duplicates("trade_id")
        z = f.merge(rf, on=["trade_id", "ticker"], how="left")
        print("TRADE_ID_MATCH", z["count"].notna().mean(), "public>=private", ((z["count"] + 1e-9) >= z.qty).mean(), "matched", z["count"].notna().sum(), "of", len(z))
        if z["count"].notna().sum():
            ratio = (z.loc[z["count"].notna(), "count"] / z.loc[z["count"].notna(), "qty"]).replace([np.inf, -np.inf], np.nan)
            print("PUBLIC_PRIVATE_COUNT_RATIO", ratio.describe().to_dict())

    print("\nANCHORED LOWER-BOUND CAPACITY")
    results = []
    for L in [50, 100, 250, 500, 1000]:
        for q in QGRID:
            H, C = anchored_scaled(l, f, raw, q, L)
            s = summarize(H, C, days=2)
            s.update(q=q, lat_ms=L)
            results.append(s)
            print(json.dumps(s, sort_keys=True))

    print("\nSAME-TRADE OPTIMISTIC UPPER DIAGNOSTIC")
    for q in QGRID:
        H = same_trade_upper(l, f, raw, q)
        print(json.dumps({"q": q, "pnl": round(H.pnl.sum(), 6), "dollars_day": round(H.pnl.sum()/2, 4),
                          "contracts": round(H.qty.sum(), 2), "match_count": int(H.pub.notna().sum())}, sort_keys=True))
    Path("ffcb_capacity_results.json").write_text(json.dumps({"results": results, "fetch_errors": errs}, default=str, indent=2))


if __name__ == "__main__":
    main()
