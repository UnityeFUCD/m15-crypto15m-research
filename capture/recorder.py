"""Passive book + tape recorder with virtual order simulation.

WHAT THIS SOLVES
  The entire live history used ONE execution policy: join the bid, 608 times.
  You cannot estimate P(fill | action) when the action never varied - no
  quantity of additional join-the-bid data helps.

  The usual fix is to place real orders at different prices. This does it
  WITHOUT TRADING. For any candidate price we open a VIRTUAL order, record the
  displayed size that would have been ahead of it, and then follow how much
  volume actually trades at that exact price. When cumulative traded volume
  exceeds the size that was ahead, the virtual order would have filled.

  Zero capital. Zero orders. Zero risk. No interference with the live runner.

WHAT IS OBSERVABLE (verified 2026-08-06)
  GET /markets/{t}/orderbook  -> orderbook_fp.yes_dollars / .no_dollars
                                 [[price, size], ...] full depth, ~20 levels
  GET /markets/trades         -> price, size, timestamp, taker_side,
                                 taker_book_side

  Note the response key is `orderbook_fp` with `*_dollars` arrays, NOT
  `orderbook` with `yes`/`no`. Getting that wrong silently yields an empty
  book, which is how it first looked unavailable.

THE BOUNDS, AND WHY BOTH ARE KEPT
  optimistic  any trade at our price would have filled us (front of queue)
  pessimistic we fill only after cumulative volume at our price exceeds the
              size that was displayed ahead of us at join time

  Truth lies between. The optimistic bound is what the earlier ladder analysis
  assumed, which is why it overstated the value of improving the quote. The
  pessimistic bound never advances the queue on cancellations - orders ahead
  that are pulled do move us up, but L2 cannot tell whether a size decrease
  happened ahead of us or behind. Reporting both brackets the answer honestly.
"""
from __future__ import annotations

import argparse
import signal
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .kalshi import KalshiClient
from .money import qty_to_hundredths, to_micros, tick_micros
from .store import ProcessLock, Store

SERIES = ["KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M", "KXDOGE15M",
          "KXHYPE15M"]

# virtual prices to track, as tick offsets from the favourite's bid
OFFSETS = {"back_one_tick": -1, "join": 0, "improve_one_tick": +1,
           "improve_two_ticks": +2}


def now_ms() -> int:
    return int(time.time() * 1000)


def parse_book(body: dict) -> tuple[list, list]:
    """Return (yes_levels, no_levels) as [(price_micros, qty_hundredths)].

    Kalshi returns ASCENDING price. The best YES bid is the LAST yes entry;
    the best NO bid is the last no entry. A YES ask is 1 - (best NO bid).
    """
    ob = (body or {}).get("orderbook_fp") or {}
    out = []
    for key in ("yes_dollars", "no_dollars"):
        lv = []
        for row in (ob.get(key) or []):
            try:
                lv.append((to_micros(row[0]), qty_to_hundredths(row[1])))
            except Exception:
                continue
        lv.sort()
        out.append(lv)
    return out[0], out[1]


def best_quotes(yes_lv, no_lv):
    """Top of book in YES terms. Returns (yes_bid, yes_ask, size_at_bid)."""
    yes_bid = yes_lv[-1][0] if yes_lv else None
    yes_bid_sz = yes_lv[-1][1] if yes_lv else 0
    no_bid = no_lv[-1][0] if no_lv else None
    yes_ask = (1_000_000 - no_bid) if no_bid is not None else None
    return yes_bid, yes_ask, yes_bid_sz


@dataclass
class VirtualOrder:
    vo_id: str
    ticker: str
    policy: str
    side: str                  # "yes" | "no" - the favourite side
    price_micros: int
    opened_ms: int
    ahead_at_open: int         # displayed size ahead, hundredths
    traded_at_px: int = 0      # cumulative, hundredths
    filled_pess_ms: int | None = None
    filled_opt_ms: int | None = None
    closed_ms: int | None = None
    seen_trade_ids: set = field(default_factory=set)

    def observe(self, trades: list[dict], ts_ms: int):
        """Consume tape. A trade fills our side when the aggressor takes the
        opposite side at our price."""
        for tr in trades:
            tid = tr.get("trade_id")
            if not tid or tid in self.seen_trade_ids:
                continue
            try:
                yes_px = to_micros(tr["yes_price_dollars"])
                size = qty_to_hundredths(tr["count_fp"])
            except Exception:
                continue
            our_px = yes_px if self.side == "yes" else 1_000_000 - yes_px
            if our_px != self.price_micros:
                continue
            self.seen_trade_ids.add(tid)
            # a resting BID is hit by a seller: taker sells our side
            if tr.get("taker_side") == self.side:
                continue                      # taker bought our side; not our fill
            if self.filled_opt_ms is None:
                self.filled_opt_ms = ts_ms    # any trade at our price
            self.traded_at_px += size
            if (self.filled_pess_ms is None
                    and self.traded_at_px > self.ahead_at_open):
                self.filled_pess_ms = ts_ms


class Recorder:
    def __init__(self, store: Store, client: KalshiClient,
                 series=None, poll_s: float = 5.0):
        self.st = store
        self.api = client
        self.series = series or SERIES
        self.poll_s = poll_s
        self.vos: dict[str, list[VirtualOrder]] = {}
        self.last_trade_seen: dict[str, str] = {}
        self.stop = False

    def open_markets(self) -> list[dict]:
        out = []
        for s in self.series:
            out += list(self.api.paginate("/markets", "markets",
                                          {"series_ticker": s,
                                           "status": "open"}, max_pages=3))
        return out

    def snapshot(self, m: dict):
        ticker = m["ticker"]
        ts0 = now_ms()
        res = self.api.orderbook(ticker, depth=50)
        if not res.ok:
            self.st.put("api_errors", {
                "ts_ms": ts0, "path": res.path, "seq_hint": 0,
                "status": res.status, "body": str(res.body)[:500],
                "error": res.error})
            return
        yes_lv, no_lv = parse_book(res.body)
        yes_bid, yes_ask, _ = best_quotes(yes_lv, no_lv)
        if yes_bid is None or yes_ask is None:
            return
        self.st.put("book_snapshots", {
            "ticker": ticker, "ts_ms": ts0, "reason": "poll",
            "yes_levels": yes_lv, "no_levels": no_lv,
            "yes_bid_micros": yes_bid, "yes_ask_micros": yes_ask,
            "ts_request_ms": res.ts_request_ms,
            "ts_response_ms": res.ts_response_ms})

        # tape
        trades = self.api.public_trades(ticker, max_pages=2)
        new = []
        seen = self.last_trade_seen.get(ticker)
        for tr in trades:
            if tr.get("trade_id") == seen:
                break
            new.append(tr)
        if trades:
            self.last_trade_seen[ticker] = trades[0].get("trade_id")
        if new:
            self.st.put_many("public_trades", [{
                "trade_id": t.get("trade_id"), "ticker": ticker,
                "ts_iso": t.get("created_time"),
                "yes_price_micros": to_micros(t["yes_price_dollars"]),
                "size_hundredths": qty_to_hundredths(t["count_fp"]),
                "taker_side": t.get("taker_side"),
                "taker_book_side": t.get("taker_book_side"),
            } for t in new if t.get("trade_id")])

        # open virtual orders once per market, at the favourite's bid ladder
        if ticker not in self.vos:
            fav_yes = yes_bid >= 500_000
            fav_bid = yes_bid if fav_yes else 1_000_000 - yes_ask
            fav_ask = yes_ask if fav_yes else 1_000_000 - yes_bid
            if not (650_000 <= fav_bid < 800_000):
                return
            side = "yes" if fav_yes else "no"
            lv = yes_lv if fav_yes else no_lv
            depth = {p: q for p, q in lv}
            t = tick_micros(fav_bid)
            made = []
            for pol, k in OFFSETS.items():
                px = fav_bid + k * t
                if px >= fav_ask or px <= 0:
                    continue          # never cross; that would be taking
                made.append(VirtualOrder(
                    vo_id=str(uuid.uuid4()), ticker=ticker, policy=pol,
                    side=side, price_micros=px, opened_ms=ts0,
                    ahead_at_open=depth.get(px, 0)))
            self.vos[ticker] = made
            for v in made:
                self.st.put("virtual_orders", {
                    "vo_id": v.vo_id, "ticker": ticker, "policy": v.policy,
                    "side": side, "price_micros": v.price_micros,
                    "opened_ms": v.opened_ms,
                    "ahead_at_open_hundredths": v.ahead_at_open,
                    "fav_bid_micros": fav_bid, "fav_ask_micros": fav_ask,
                    "spread_micros": fav_ask - fav_bid})

        for v in self.vos.get(ticker, []):
            before_p, before_o = v.filled_pess_ms, v.filled_opt_ms
            v.observe(new, ts0)
            cur_depth = {p: q for p, q in (yes_lv if v.side == "yes" else no_lv)}
            self.st.put("virtual_queue", {
                "vo_id": v.vo_id, "ts_ms": ts0,
                "traded_at_px_hundredths": v.traded_at_px,
                "displayed_at_px_hundredths": cur_depth.get(v.price_micros, 0),
                "ahead_at_open_hundredths": v.ahead_at_open,
                "filled_optimistic": v.filled_opt_ms is not None,
                "filled_pessimistic": v.filled_pess_ms is not None,
                "newly_filled_opt": before_o is None and v.filled_opt_ms is not None,
                "newly_filled_pess": before_p is None and v.filled_pess_ms is not None})

    def run(self, max_seconds: float | None = None):
        run_id = self.st.mark_start("recorder")
        t_end = time.time() + max_seconds if max_seconds else None
        cycles = 0
        try:
            while not self.stop:
                t0 = time.time()
                try:
                    mkts = self.open_markets()
                    for m in mkts:
                        if self.stop:
                            break
                        self.snapshot(m)
                    self.st.put("capture_heartbeats", {
                        "ts_ms": now_ms(), "component": "recorder",
                        "run_id": run_id, "markets": len(mkts),
                        "virtual_orders": sum(len(v) for v in self.vos.values()),
                        "cycle": cycles})
                except Exception as e:
                    self.st.put("api_errors", {
                        "ts_ms": now_ms(), "path": "recorder_cycle",
                        "seq_hint": cycles, "status": -1,
                        "body": f"{type(e).__name__}: {e}"[:500]})
                cycles += 1
                if t_end and time.time() >= t_end:
                    break
                time.sleep(max(0.0, self.poll_s - (time.time() - t0)))
        finally:
            self.st.mark_clean_shutdown("recorder")
        return cycles


def pol_of(v: VirtualOrder) -> str:
    return v.policy


def main():
    ap = argparse.ArgumentParser(
        description="Passive book+tape recorder. Places NO orders.")
    ap.add_argument("--seconds", type=float, default=None,
                    help="stop after N seconds (default: run forever)")
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--db", default=None)
    a = ap.parse_args()
    st = Store(a.db) if a.db else Store()
    with ProcessLock("recorder"):
        rec = Recorder(st, KalshiClient(), poll_s=a.poll)

        def _sig(*_):
            rec.stop = True
        for s in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(s, _sig)
            except Exception:
                pass
        print(f"recorder starting {datetime.now(timezone.utc):%H:%M:%S}Z "
              f"(passive, no orders)")
        n = rec.run(a.seconds)
        for t in ("book_snapshots", "public_trades", "virtual_orders",
                  "virtual_queue"):
            try:
                print(f"  {t:18} {st.count(t):>7} rows")
            except Exception:
                pass
        print(f"cycles {n}")
    st.close()


if __name__ == "__main__":
    main()
