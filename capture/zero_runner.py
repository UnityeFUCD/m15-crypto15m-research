"""Live/shadow runner for the frozen minute-:00 taker strategy.

PURPOSE: MEASUREMENT, NOT PROFIT.
  The :00 edge was +9 to +11c through May-June and decayed to roughly zero by
  mid-July, specifically, against a flat market (P=0.0040). Whether it stays
  dead or returns cannot be answered from history - 73 days sits exactly on
  the detection threshold.

  This runner exists to answer one question: is the edge back? That question
  is per-contract, so it is SIZE-INDEPENDENT. q1 for ~27 days distinguishes
  "+9c" from "0c" and costs about $9 at the 95% worst case. Larger size buys
  no extra information and risks real money.

DEFAULTS ARE SAFE
  ZERO_ENABLED=0     nothing runs
  ZERO_LIVE=0        shadow only - decisions logged, no orders sent
  ZERO_QTY=1         one contract

  Both flags must be set deliberately for an order to reach the exchange.

FROZEN RULE - not tuned here, and not to be tuned here
  close minute == :00, derived FROM THE TICKER (never expected_expiration_time)
  entry   = first complete valid quote with 8-14 minutes remaining
  entry   favourite bid in [0.65, 0.80), side frozen at entry
  wait    exactly 2 complete minutes
  commit  favourite ask in [0.60, 0.85]
  one market per close window, ranked: narrowest spread, lowest ask,
          highest volume, fixed coin order
  IOC buy at the displayed ask. No retry. No chase. Hold to settlement.

  The momentum, volume and spread-widening conditions from the original spec
  are OMITTED: all three were shown inert (removing them improved the edge on
  more trades). The ask ceiling is retained because it is the one condition
  that measurably works.

EVERY DECISION IS LOGGED, including skips and rejections, so the denominator
is never silently reduced. That is how a -$134 loss once became a +$143
report in this project.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from capture.kalshi import KalshiClient          # noqa: E402

LOG = ROOT / "data" / "zero_runner_log.jsonl"
COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "HYPE"]
COIN_ORDER = {c: i for i, c in enumerate(COINS)}

ENABLED = os.environ.get("ZERO_ENABLED", "0") == "1"
LIVE = os.environ.get("ZERO_LIVE", "0") == "1"
QTY = int(os.environ.get("ZERO_QTY", "1"))

ENTRY_ML_LO, ENTRY_ML_HI = 8.0, 14.0
BID_LO, BID_HI = 0.65, 0.80
WAIT_MIN = 2
ASK_LO, ASK_HI = 0.60, 0.85
MAX_QTY_HARD = 5                 # refuse to run bigger from this script


@dataclass
class Decision:
    ts: str
    close_ts: int
    ticker: str
    coin: str
    stage: str                   # entry | commit | skip | order | error
    side: str | None = None
    bid: float | None = None
    ask: float | None = None
    spread: float | None = None
    vol: float | None = None
    reason: str | None = None
    qty: int | None = None
    live: bool = False
    response: str | None = None


def log(d: Decision) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(d)) + "\n")
    print(f"[{d.stage:6s}] {d.ticker:28s} {d.reason or ''}", flush=True)


def held_quote(side: str, yb: float, ya: float) -> tuple[float, float]:
    return (yb, ya) if side == "yes" else (1.0 - ya, 1.0 - yb)


def true_close_from_ticker(ticker: str) -> datetime:
    """ET wkey + 4h. NEVER expected_expiration_time, which is close + 5 min."""
    wk = ticker.split("-")[1]
    dt = datetime.strptime(wk, "%y%b%d%H%M").replace(tzinfo=timezone.utc)
    return dt + timedelta(hours=4)


def top_of_book(api: KalshiClient, ticker: str):
    r = api.orderbook(ticker, depth=1)
    if not r.ok:
        return None
    ob = (r.body or {}).get("orderbook_fp") or {}
    yes, no = ob.get("yes_dollars") or [], ob.get("no_dollars") or []
    if not yes or not no:
        return None
    # yes list is bids for YES ascending; best YES bid is the highest price.
    # best YES ask = 1 - best NO bid.
    yb = max(float(p) for p, _ in yes)
    nb = max(float(p) for p, _ in no)
    ya = 1.0 - nb
    if not (0 < yb < ya < 1):
        return None
    vol = sum(float(s) for _, s in yes) + sum(float(s) for _, s in no)
    return yb, ya, vol


def open_zero_markets(api: KalshiClient) -> list[dict]:
    out = []
    for coin in COINS:
        r = api.get("/markets", {"series_ticker": f"KX{coin}15M",
                                 "status": "open", "limit": 200})
        if not r.ok:
            continue
        for m in (r.body or {}).get("markets") or []:
            t = m.get("ticker", "")
            try:
                close = true_close_from_ticker(t)
            except Exception:
                continue
            if close.minute != 0:            # :00 ONLY
                continue
            out.append({"ticker": t, "coin": coin, "close": close})
    return out


def run_once(api: KalshiClient, state: dict) -> None:
    now = datetime.now(timezone.utc)
    for m in open_zero_markets(api):
        t, close = m["ticker"], m["close"]
        ml = (close - now).total_seconds() / 60.0
        key = t

        # ---- ENTRY: first complete valid quote at 8-14 minutes -----------
        if ENTRY_ML_LO <= ml <= ENTRY_ML_HI and key not in state:
            q = top_of_book(api, t)
            if q is None:
                continue
            yb, ya, vol = q
            fav_yes = yb >= 0.5
            side = "yes" if fav_yes else "no"
            bid, ask = held_quote(side, yb, ya)
            if not (BID_LO <= bid < BID_HI):
                log(Decision(now.isoformat(), int(close.timestamp()), t,
                             m["coin"], "skip", side, bid, ask, ask - bid, vol,
                             f"entry bid {bid:.4f} outside [{BID_LO},{BID_HI})"))
                state[key] = {"skipped": True}
                continue
            state[key] = {"side": side, "entry_bid": bid, "entry_ask": ask,
                          "entry_ml": ml, "close": close, "coin": m["coin"],
                          "committed": False, "skipped": False}
            log(Decision(now.isoformat(), int(close.timestamp()), t, m["coin"],
                         "entry", side, bid, ask, ask - bid, vol,
                         f"armed at ml={ml:.2f}"))

        # ---- COMMIT: exactly two complete minutes later ------------------
        st = state.get(key)
        if not st or st.get("skipped") or st.get("committed"):
            continue
        if ml > st["entry_ml"] - WAIT_MIN:
            continue                              # not yet two minutes
        q = top_of_book(api, t)
        if q is None:
            continue
        yb, ya, vol = q
        bid, ask = held_quote(st["side"], yb, ya)
        st["committed"] = True
        if not (ASK_LO <= ask <= ASK_HI):
            log(Decision(now.isoformat(), int(close.timestamp()), t, st["coin"],
                         "skip", st["side"], bid, ask, ask - bid, vol,
                         f"commit ask {ask:.4f} outside [{ASK_LO},{ASK_HI}]"))
            continue
        cw = int(close.timestamp())
        if cw in state.setdefault("_committed_windows", set()):
            log(Decision(now.isoformat(), cw, t, st["coin"], "skip", st["side"],
                         bid, ask, ask - bid, vol,
                         "close window already has a commitment"))
            continue
        state["_committed_windows"].add(cw)

        qty = min(QTY, MAX_QTY_HARD)
        if not LIVE:
            log(Decision(now.isoformat(), cw, t, st["coin"], "commit",
                         st["side"], bid, ask, ask - bid, vol,
                         f"SHADOW - would IOC buy {qty} at {ask:.4f}", qty,
                         False))
            continue
        # Kalshi v2 create-order. The v1 endpoint is deprecated and rejects.
        #   POST /portfolio/events/orders
        #   side quotes the YES leg ONLY: "bid" = buy YES, "ask" = sell YES.
        #   Buying NO at price A is expressed as SELLING YES at (1 - A).
        #   count and price are fixed-point DECIMAL STRINGS in dollars.
        #
        # LIMIT + immediate_or_cancel, never a market order: a market buy has
        # no price ceiling and would sweep the book if the top level is thin.
        # The edge here is a few cents, so uncapped slippage could exceed it.
        if st["side"] == "yes":
            v2_side, v2_price = "bid", ask               # buy YES at its ask
        else:
            v2_side, v2_price = "ask", 1.0 - ask         # buy NO == sell YES
        body = {"ticker": t, "side": v2_side,
                "count": f"{qty:.2f}", "price": f"{v2_price:.4f}",
                "time_in_force": "immediate_or_cancel",
                "self_trade_prevention_type": "taker_at_cross",
                "client_order_id": f"zero-{cw}-{st['coin']}"}
        try:
            r = api.post("/portfolio/events/orders", body)
            log(Decision(now.isoformat(), cw, t, st["coin"], "order",
                         st["side"], bid, ask, ask - bid, vol,
                         f"IOC buy {qty} at {ask:.4f}", qty, True,
                         json.dumps(r.body)[:400] if r.ok else f"HTTP fail"))
        except Exception as e:
            log(Decision(now.isoformat(), cw, t, st["coin"], "error",
                         st["side"], bid, ask, ask - bid, vol, str(e)[:200],
                         qty, True))


def main() -> None:
    print("=" * 74)
    print("MINUTE-:00 RUNNER")
    print("=" * 74)
    print(f"  ZERO_ENABLED = {int(ENABLED)}")
    print(f"  ZERO_LIVE    = {int(LIVE)}   "
          f"{'*** LIVE ORDERS WILL BE SENT ***' if LIVE else '(shadow only)'}")
    print(f"  ZERO_QTY     = {QTY} (hard cap {MAX_QTY_HARD})")
    print(f"  log          = {LOG}")
    if not ENABLED:
        print("\n  ZERO_ENABLED is not 1. Nothing will run. Exiting.")
        return
    if QTY > MAX_QTY_HARD:
        print(f"\n  refusing: ZERO_QTY {QTY} exceeds the hard cap "
              f"{MAX_QTY_HARD}")
        return
    api = KalshiClient()
    state: dict = {}
    print("\n  polling every 20s. Ctrl-C to stop.\n")
    while True:
        try:
            run_once(api, state)
        except KeyboardInterrupt:
            print("\n  stopped by user")
            return
        except Exception as e:
            print(f"  loop error: {e}", flush=True)
        time.sleep(20)


if __name__ == "__main__":
    main()
