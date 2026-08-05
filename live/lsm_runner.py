"""LONGSHOT-SELLER MAKER (LSM) - deploy of the validated maker edge.

WHAT THE RESEARCH ACTUALLY SAYS
  `research/final_minute_favorite_maker_validation/` established, on 6,238,047
  trades / 375.8M contracts / 4,480 windows / 10 days, in exact integer
  arithmetic with settlement validated against official results:

    a MAKER who is filled on the favourite side at 65-90c earns roughly
    +9 to +13 cents per contract, in ALL FIVE COINS, across the ENTIRE
    15-minute window, largest at 8-14 minutes remaining.

  Economically this is NOT settlement prediction. maker_px > 50c means the
  TAKER bought the cheap side, so we are selling longshots to people who
  insist on buying them. That is why the edge is ~10c here and only ~1.25c
  when measured at the quoted mid: crossing a spread and paying a fee to buy a
  25c longshot is a strongly self-selected act.

THE ONE RULE THAT CANNOT BE BROKEN
  **NEVER CROSS.** At 65-85c the taker side is -1.68c/contract (measured).
  Chasing an unfilled order converts a +10c edge into a loss. This runner has
  no taker path at all: it posts, it waits, it cancels. Nothing else.

THE ONE THING STILL UNMEASURED
  P(fill) for our own resting order. It cannot be obtained from history -
  Kalshi never emits L3, so queue position is unobservable to everyone. This
  runner measures it live: every submission is logged whether or not it fills,
  so `net P&L per SUBMITTED order` (the number that actually matters) is
  computable from its own log from day one.

SIZING
  Deliberately small. The edge is large but P(fill) is unknown, and an unknown
  fill rate on a known edge is exactly the situation where you find out cheaply.
"""
from __future__ import annotations
import base64, json, math, os, random, signal, sys, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import risk                      # unified, persistent, fractional risk control

CRED = Path(os.environ.get("KALSHI_CRED", "C:/Users/fycin/kalshi_weather_trading"))
BASE = "https://api.elections.kalshi.com/trade-api/v2"
PREFIX = "/trade-api/v2"
HERE = Path(__file__).resolve().parent
KILL = (HERE / "LSM_KILL").resolve()
LOG = HERE / f"lsm_{datetime.now(timezone.utc):%Y%m%d}.jsonl"

DRY_RUN   = os.environ.get("LSM_LIVE", "0") != "1"
QTY       = int(os.environ.get("LSM_QTY", "10"))
MAX_QTY   = 35          # hard ceiling; raised 25 -> 35 to permit qty 30
# SIZING RATIONALE, 2026-08-04 after 56 orders / 49 settlements live:
#   breakeven = price paid = 68.64c   predicted win 78-80%   ACTUAL win 82.35%
#   margin +13.59c/contract, F_Q 0.888, zero taker fills
#   The unknown that blocked deployment - fill rate at the back of the queue -
#   is now measured across 497 filled contracts and did not collapse.
# The binding risk is NOT capital (peak use ~$41 at qty 30 on a $281 account).
# It is the 15-minute WINDOW: coins inside one window settle together
# (correlation 0.41), so P&L per window has sd $5.78 against a mean of $2.54
# at qty 10 - a bad window loses everything at once. Worst observed -$13.10,
# which becomes -$39.30 at qty 30, from a sample of only 27 windows. The true
# worst is worse than anything yet seen, so stops are scaled by that, not by
# the size multiple.
BAND      = (0.65, 0.85)          # favourite BID must sit in here
# Live orders stop below LIVE_HI; the band between LIVE_HI and BAND[1] is
# SHADOWED - evaluated and logged, never traded. See the shadow_band block.
LIVE_HI   = float(os.environ.get("LSM_LIVE_HI", "0.80"))
# EDGE-WEIGHTED SIZING. Coefficients are fitted OFFLINE (54_edge_weighted_sizing
# .py) and loaded here, so the runner never fits anything live - it only
# evaluates a fixed linear score. Bounds are deliberately tight: the
# unconstrained fit earned more but pushed the worst window from -$34.50 to
# -$56.82 and the largest position to $23.70.
# AGREEMENT RULE. Within one window, only add positions whose favourite points
# the SAME way as the first position taken there.
#
# Direction-MIXED windows are unprofitable: post-shift they earn -11.08c
# against +6.51c for aligned windows, a 17.58c gap with 95% CI [+16.06,+19.22].
# The mechanism is that a cross-section pointing different ways is a CHOPPING
# market, and chop is exactly what overturns a favourite in the last eight
# minutes. Each coin's own book cannot see the disagreement.
#
# Verified on BOTH regimes, which is why it is deployed where other findings
# were not:
#   pre-shift  (10 days): $206.07 -> $220.70/day, better on 7/10, CI
#                         [+1.39,+31.62], P(worse) 0.011
#   post-shift (3 days) : $133.95 -> $141.55/day
# The discarded entries carry edge -14.55c with CI [-28.56,-1.45] - genuinely
# negative, not merely marginal, which is the bar every filter must clear here.
#
# CRITICAL DETAIL: the slot is BURNED, not backfilled. Replacing a disagreeing
# candidate with an agreeing one turned a +$7.60/day gain into -$16.95/day
# post-shift, because backfilling concentrates the book into aligned windows,
# which wipe out far more often (8.44% vs 1.52%).
AGREE_ON = os.environ.get("LSM_AGREE_RULE", "0") == "1"
SIZE_ON = os.environ.get("LSM_EDGE_SIZE", "0") == "1"
SIZE_CLIP_LO = float(os.environ.get("LSM_SIZE_CLIP_LO", "0.5"))
SIZE_CLIP_HI = float(os.environ.get("LSM_SIZE_CLIP_HI", "1.5"))
SIZE_QTY_LO = int(os.environ.get("LSM_SIZE_QTY_LO", "8"))
SIZE_QTY_HI = int(os.environ.get("LSM_SIZE_QTY_HI", "22"))
SIZE_MODEL = None
if SIZE_ON:
    try:
        _sm = json.loads((HERE / "size_model.json").read_text())
        if (len(_sm.get("mu", [])) == 4 and len(_sm.get("sd", [])) == 4
                and len(_sm.get("w", [])) == 5 and _sm.get("ec_mean")):
            SIZE_MODEL = _sm
        else:
            print("size_model.json malformed - edge sizing DISABLED", flush=True)
    except Exception as _e:
        print(f"size_model.json unreadable ({_e}) - edge sizing DISABLED",
              flush=True)
MIN_LEFT  = (float(os.environ.get("LSM_MIN_LEFT_LO", "8")),
             float(os.environ.get("LSM_MIN_LEFT_HI", "14")))   # minutes remaining
MAX_GROSS = float(os.environ.get("LSM_MAX_GROSS", "80"))
MAX_LOSS  = float(os.environ.get("LSM_MAX_LOSS", "50"))
MAX_ORDERS_DAY = 400
# Maximum tolerated drawdown from the running PEAK of realised equity, enforced
# as a hard pre-submit reservation (see the envelope check in the scan loop).
MAX_DD_FRAC = float(os.environ.get("LSM_MAX_DD_FRAC", "0.30"))
# Minimum contracts already traded in the market before we will post into it.
MIN_VOLUME = float(os.environ.get("LSM_MIN_VOLUME", "500"))
MAX_PER_WINDOW = int(os.environ.get("LSM_PER_WINDOW", "2"))
# The binding constraint is OPPORTUNITY, not capital. With equity ~$462 and a
# $110 gross cap, the runner deploys $20-60 and logs zero gross_cap or drawdown
# blocks - it simply cannot find enough markets. (An earlier reading of "177
# refusals per 36 fills" was wrong: those counts were inflated by the same
# market being re-evaluated every 5 seconds.)
#
# So throughput comes from more SERIES, not better selection. Kalshi lists nine
# further 15-minute crypto series; these two are the ones currently carrying
# enough volume to clear the 2,000-contract floor:
#     KXBNB15M   28,883 contracts  (more than ETH's 21,636)
#     KXHYPE15M   7,387 contracts  (comparable to XRP 9,390, DOGE 9,534)
# NEAR (567) and ZEC (803) are too thin; ADA/BCH/TON have no open markets.
#
# Verified mechanically identical before adding: strike_type greater_or_equal,
# settling on "the simple average of the sixty seconds of CF Benchmarks'
# <X>USDRTI" against the same measure at the window open - the same A1 >= A0
# contract with ties paying YES, just a different underlying.
#
# The edge itself is UNVERIFIED on these two, exactly as it is on BTC and ETH
# (the backtest grid is essentially DOGE/SOL/XRP). The volume floor is the
# protection: a thin window is rejected automatically.
SERIES = os.environ.get(
    "LSM_SERIES",
    "KXBTC15M,KXETH15M,KXSOL15M,KXXRP15M,KXDOGE15M,KXBNB15M,KXHYPE15M"
).split(",")

_cfg = {}
for line in (CRED / ".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        _cfg[k.strip()] = v.strip().strip('"').strip("'")
_priv = serialization.load_pem_private_key((CRED / "kalshi_private_key.pem").read_bytes(),
                                           password=None)
_KEY = _cfg.get("KALSHI_API_KEY_ID")
_S = requests.Session()


def log(ev, **kw):
    rec = {"ev": ev, "utc": datetime.now(timezone.utc).isoformat(), **kw}
    with LOG.open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    print(json.dumps(rec, default=str), flush=True)
    return rec


def api(method, path, params=None, body=None, timeout=12):
    ts = str(int(time.time() * 1000))
    sig = base64.b64encode(_priv.sign((ts + method + PREFIX + path).encode(),
          padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
          hashes.SHA256())).decode()
    h = {"KALSHI-ACCESS-KEY": _KEY, "KALSHI-ACCESS-SIGNATURE": sig,
         "KALSHI-ACCESS-TIMESTAMP": ts, "Content-Type": "application/json"}
    try:
        r = _S.request(method, BASE + path, params=params, json=body, headers=h,
                       timeout=timeout)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"_raw": r.text[:300]}
    except Exception as e:
        return 0, {"_err": str(e)}


fee = lambda p, c=1: math.ceil(0.07 * c * p * (1 - p) * 1e4 - 1e-12) / 1e4
tick_of = lambda p: 0.001 if (p > 0.90 or p < 0.10) else 0.01
snap = lambda p: round(round(p / tick_of(p), 0) * tick_of(p), 4)

STATE = {"orders": 0, "realised": 0.0, "start_balance": None, "entered": set(),
         "per_window": {}, "resting": {}, "deployed": 0.0, "open_cost": [],
         "win_dir": {}}
ENT_F = HERE / f"lsm_entered_{datetime.now(timezone.utc):%Y%m%d}.json"
if ENT_F.exists():
    try:
        d = json.loads(ENT_F.read_text())
        STATE["entered"] = set(d.get("entered", []))
        STATE["per_window"] = {k: int(v) for k, v in (d.get("per_window") or {}).items()}
        STATE["deployed"] = float(d.get("deployed", 0.0))
        STATE["open_cost"] = d.get("open_cost", [])
        print(f"resumed: {len(STATE['entered'])} markets already entered, "
              f"${STATE['deployed']:.2f} deployed")
    except Exception as e:
        print(f"WARNING could not read {ENT_F}: {e}")


def seed_entered_from_exchange():
    """Seed the 'already entered' set from OPEN POSITIONS on the exchange.

    THE BUG THIS FIXES: ENT_F is date-stamped at module load, so a runner that
    starts before midnight UTC keeps writing to yesterday's filename all night,
    and the next restart after midnight looks for a file that was never
    created. The entered-markets guard silently vanishes at the date rollover -
    and that guard is the only thing preventing a market we already hold from
    being entered a SECOND time. Losing it is exactly the failure that cost
    $34 on 2026-08-04 via a different route (process restart).

    A local file can always disagree with reality. Open positions cannot: if
    the exchange says we hold it, we entered it. Seeding from there makes the
    guard independent of filenames, clocks and restarts.
    """
    if DRY_RUN:
        return
    c, j = api("GET", "/portfolio/positions", {"limit": 200})
    if c != 200 or not j:
        log("seed_failed", status=c, note="could not read positions; "
            "duplicate-entry guard is NOT seeded")
        return
    n = 0
    for p in (j.get("market_positions") or []):
        try:
            if float(p.get("position_fp") or 0) != 0:
                tk = p.get("ticker")
                if tk and tk not in STATE["entered"]:
                    STATE["entered"].add(tk)
                    n += 1
        except Exception:
            pass
    log("seeded_entered_from_exchange", added=n,
        total=len(STATE["entered"]))


def persist():
    try:
        ENT_F.write_text(json.dumps({"entered": sorted(STATE["entered"]),
                                     "per_window": STATE["per_window"],
                                     "deployed": STATE["deployed"],
                                     "open_cost": STATE["open_cost"]}))
    except Exception as e:
        log("persist_error", err=str(e)[:120])


def book(tk):
    c, j = api("GET", f"/markets/{tk}/orderbook", {"depth": 10})
    if c != 200 or not j:
        return None
    ob = j.get("orderbook_fp") or {}
    yb = sorted([(float(p), float(s)) for p, s in (ob.get("yes_dollars") or [])],
                key=lambda x: -x[0])
    nb = sorted([(float(p), float(s)) for p, s in (ob.get("no_dollars") or [])],
                key=lambda x: -x[0])
    if not yb or not nb:
        return None
    ybid, yask = yb[0][0], 1.0 - nb[0][0]
    if not (0 < ybid < yask < 1):
        return None
    return dict(yes_bid=ybid, yes_ask=yask, mid=(ybid + yask) / 2,
                yes_bid_sz=yb[0][1], no_bid_sz=nb[0][1])


# ---------------------------------------------------------------------------
# DEAD-ENTRY FILTER
#
# Two signals, each computable before we commit, jointly identify entries that
# earn nothing. Verified on 826 historical entries, rolling walk-forward with
# thresholds taken only from prior days:
#
#     both LOW : +2.17c/contract   (18.3% of entries)
#     otherwise: +15.59c/contract
#     difference -13.41c, day-clustered 95% CI [-15.29, -11.31]
#     worse on 7 of 7 out-of-sample days
#     permutation p = 0.0005 (1 of 2,000 shuffles this extreme, outcomes
#     shuffled within coin x price bucket, so it is not "worst of four cells")
#     present in every coin: DOGE -12.05, SOL -7.50, XRP -20.87
#
# NEITHER SIGNAL ALONE IS ENOUGH - others_px alone -6.68c, frac_long alone
# -5.81c. It is the conjunction that isolates dead entries, which is why both
# are computed.
#
# others_px  mean favourite price of the OTHER coins in the same window. Low
#            means the whole complex is near a coin-flip, so nothing is
#            trending decisively and favourites are fragile.
# frac_long  share of traded volume where the taker bought the CHEAP side.
#            Low means the impatient longshot demand we are paid to absorb is
#            simply not present.
#
# Skipping them moved (stressed regime) Sharpe 1.20 -> 1.42, 95% drawdown
# $332 -> $253 and P(ruin) 1.52% -> 0.52%. The P&L gain was inside noise; this
# is a RISK filter, not a profit source, and is deployed as such.
#
# SAFETY: this can only ever SKIP. It cannot raise size or exposure. If either
# signal is unavailable the entry is TAKEN as before - we only decline on
# positive evidence, never on missing data.
FILTER_ON = os.environ.get("LSM_DEAD_FILTER", "1") == "1"
OTHERS_MAX = float(os.environ.get("LSM_OTHERS_MAX", "0.71"))
FLONG_MAX = float(os.environ.get("LSM_FLONG_MAX", "0.5046"))
_CYCLE = {"markets": {}, "book": {}}


def markets_cached(ser):
    if ser not in _CYCLE["markets"]:
        c, j = api("GET", "/markets", {"series_ticker": ser, "status": "open",
                                       "limit": 20})
        _CYCLE["markets"][ser] = (j.get("markets") or []) if (c == 200 and j) else []
    return _CYCLE["markets"][ser]


def book_cached(tk):
    if tk not in _CYCLE["book"]:
        _CYCLE["book"][tk] = book(tk)
    return _CYCLE["book"][tk]


def frac_long(tk):
    """Share of traded volume where the TAKER bought the cheap side.

    Returns None on any failure, which the caller treats as "do not skip"."""
    c, j = api("GET", "/markets/trades", {"ticker": tk, "limit": 1000})
    if c != 200 or not j:
        return None
    tot = lon = 0.0
    for t in (j.get("trades") or []):
        try:
            yp = float(t["yes_price_dollars"])
            n = float(t["count_fp"])
            sd = t.get("taker_side")
        except Exception:
            continue
        if sd not in ("yes", "no") or n <= 0:
            continue
        tp = yp if sd == "yes" else 1.0 - yp      # price the TAKER paid
        tot += n
        if tp < 0.50:
            lon += n
    return (lon / tot) if tot > 0 else None


def place_maker(tk, side, price, qty):
    """post_only ALWAYS. If it would cross, the exchange rejects it - which is
    the desired behaviour, not an error to work around."""
    v2_side, v2_price = ("bid", price) if side == "yes" else ("ask", round(1 - price, 4))
    body = {"ticker": tk, "side": v2_side, "count": f"{qty:.2f}",
            "price": f"{v2_price:.4f}",
            "client_order_id": f"lsm{int(time.time()*1000)}",
            "time_in_force": "good_till_canceled",
            "self_trade_prevention_type": "taker_at_cross",
            "post_only": True}
    if DRY_RUN:
        log("DRY_post", ticker=tk, side=side, price=price, qty=qty)
        return None
    c, j = api("POST", "/portfolio/events/orders", body=body)
    o = j or {}
    log("post", ticker=tk, side=side, price=price, qty=qty, status=c,
        immediate_fill=o.get("fill_count"), resp=str(j)[:200])
    return o.get("order_id") if c in (200, 201) else None


def order_filled_qty(oid):
    """How many contracts of this order actually traded. Ground truth is the
    fills endpoint, filtered client-side (the order_id query param is not
    reliably honoured). Returns (filled, n_fill_records)."""
    c, j = api("GET", "/portfolio/fills", {"order_id": oid, "limit": 200})
    fl = [f for f in ((j or {}).get("fills") or []) if f.get("order_id") == oid]
    return sum(int(float(f.get("count_fp") or 0)) for f in fl), len(fl)


def cancel(oid, reason, info=None):
    """Cancel, then establish what ACTUALLY filled.

    THE BUG THIS FIXES: the first version cancelled and logged the result
    without ever asking whether the order had filled. Both of the first two
    live orders had already filled completely, so the cancel was a no-op and
    got logged as if nothing had happened. The runner held two winning
    positions it did not know about, and P(fill) - the entire reason this
    program exists - was being recorded as zero.
    """
    if DRY_RUN:
        return 0
    c, j = api("DELETE", f"/portfolio/events/orders/{oid}")
    red = (j or {}).get("reduced_by")
    tgt = (info or {}).get("qty", 0)
    unfilled = int(float(red)) if red is not None else None
    filled, nf = order_filled_qty(oid)
    if unfilled is None:
        unfilled = max(0, tgt - filled)
    # trust the fills endpoint over the cancel ack when they disagree
    if nf and filled + unfilled != tgt:
        unfilled = max(0, tgt - filled)
    log("terminal", oid=oid, ticker=(info or {}).get("ticker"),
        cancel_status=c, reason=reason, target=tgt,
        FILLED=filled, unfilled=unfilled,
        fill_records=nf,
        queue_ahead=(info or {}).get("queue_ahead"),
        price=(info or {}).get("price"),
        F_Q=round(filled / tgt, 4) if tgt else None)
    if unfilled > 0:
        # capital for the unfilled portion was never spent - release it now,
        # do not wait for the window to settle
        back = unfilled * (info or {}).get("price", 0.0)
        STATE["deployed"] = max(0.0, STATE["deployed"] - back)
        for i, (ct, cost) in enumerate(STATE["open_cost"]):
            if ct == (info or {}).get("window"):
                STATE["open_cost"][i] = [ct, max(0.0, cost - back)]
                break
        persist()
    return filled


def cancel_all(reason):
    for oid, info in list(STATE["resting"].items()):
        cancel(oid, reason, info)
        STATE["resting"].pop(oid, None)


def on_sig(s, f):
    log("sigexit", sig=int(s))
    cancel_all("signal")
    sys.exit(0)


for s in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(s, on_sig)
    except Exception:
        pass

seed_entered_from_exchange()
print(f"KILL FILE: {KILL}")
print(f"MODE: {'DRY RUN' if DRY_RUN else '*** LIVE ***'}  qty={QTY}  band={BAND}  "
      f"minutes_left={MIN_LEFT}  MAX_GROSS=${MAX_GROSS}  "
      f"stop=FLOOR ${risk.FLOOR:.2f} "
      f"(fractional {risk.STOP_FRAC:.0%} of peak is inert while above it)")
print(f"VOLUME FILTER: market must have traded >= {MIN_VOLUME:.0f} contracts")
print("NEVER CROSSES - post_only only, no taker path exists in this program")
log("start", dry_run=DRY_RUN, qty=QTY, band=BAND, min_left=MIN_LEFT,
    max_gross=MAX_GROSS, max_loss=MAX_LOSS, per_window=MAX_PER_WINDOW,
    min_volume=MIN_VOLUME, max_dd_frac=MAX_DD_FRAC)
# Publish the live config so a watchdog restart reproduces THIS configuration.
# Without this the watchdog fell back to its own hardcoded defaults (qty 10,
# gross 80, no volume floor), silently reverting the tuned settings mid-session.
if not DRY_RUN:
    try:
        (HERE / "lsm_config.json").write_text(json.dumps({
            "LSM_LIVE": "1", "LSM_QTY": str(QTY),
            "LSM_MAX_GROSS": str(MAX_GROSS), "LSM_MAX_LOSS": str(MAX_LOSS),
            "LSM_MAX_DD_FRAC": str(MAX_DD_FRAC),
            "LSM_MIN_VOLUME": str(MIN_VOLUME),
            "LSM_PER_WINDOW": str(MAX_PER_WINDOW),
            "LSM_DEAD_FILTER": "1" if FILTER_ON else "0",
            "LSM_OTHERS_MAX": str(OTHERS_MAX),
            "LSM_FLONG_MAX": str(FLONG_MAX),
            "LSM_SERIES": ",".join(SERIES),
            # Risk fractions must be published too. If they are absent the
            # watchdog's restart falls back to risk.py defaults, silently
            # reverting any tuned value - the same class of bug that put a
            # differently-configured runner live at 03:00:22Z on 2026-08-05.
            "LSM_ENVELOPE_FRAC": str(risk.ENVELOPE_FRAC),
            "LSM_STOP_FRAC": str(risk.STOP_FRAC),
            "LSM_LIVE_HI": str(LIVE_HI),
            "LSM_EDGE_SIZE": "1" if SIZE_ON else "0",
            "LSM_SIZE_QTY_LO": str(SIZE_QTY_LO),
            "LSM_SIZE_QTY_HI": str(SIZE_QTY_HI),
            "LSM_AGREE_RULE": "1" if AGREE_ON else "0"}, indent=1))
    except Exception as e:
        log("config_publish_error", err=str(e)[:120])
if QTY > MAX_QTY:
    sys.exit("QTY exceeds MAX_QTY")

while True:
    if KILL.exists():
        log("halted"); cancel_all("kill"); break
    if STATE["orders"] >= MAX_ORDERS_DAY:
        log("cap_orders"); cancel_all("cap"); break
    now = datetime.now(timezone.utc)

    # release capital from settled windows
    live, freed = [], 0.0
    for ct, cost in STATE["open_cost"]:
        try:
            cl = datetime.fromisoformat(str(ct).replace("Z", "+00:00"))
        except Exception:
            live.append([ct, cost])      # unparseable: keep it counted, never free
            continue
        if now > cl + timedelta(seconds=90):
            freed += cost                # window has settled, capital is back
        else:
            live.append([ct, cost])
    if freed:
        STATE["open_cost"] = live
        STATE["deployed"] = max(0.0, STATE["deployed"] - freed)
        log("capital_released", freed=round(freed, 2),
            deployed=round(STATE["deployed"], 2))
        persist()

    if not DRY_RUN:
        c, j = api("GET", "/portfolio/balance")
        if c == 200 and j:
            bal = float((j or {}).get("balance_dollars") or 0)
            # DEPLOYED CAPITAL IS READ FROM THE EXCHANGE, NOT TRACKED INTERNALLY.
            # The incremental version drifted: it recorded cost at post time and
            # released it on cancel/settlement, so any partial fill, missed
            # terminal record, or order surviving a restart left it wrong. It
            # was found understating true exposure by $6.60 against the
            # exchange. Internal tallies of external state always drift; the
            # exchange is the only authority, so ask it.
            c2, j2 = api("GET", "/portfolio/positions", {"limit": 200})
            if c2 == 200 and j2:
                real_dep, npos = 0.0, 0
                for p in (j2.get("market_positions") or []):
                    try:
                        if float(p.get("position_fp") or 0) != 0:
                            npos += 1
                            real_dep += abs(float(p.get("total_traded_dollars") or 0))
                    except Exception:
                        pass
                if abs(real_dep - STATE["deployed"]) > 0.01:
                    log("deployed_reconciled", internal=round(STATE["deployed"], 2),
                        exchange=round(real_dep, 2),
                        drift=round(real_dep - STATE["deployed"], 2),
                        open_positions=npos)
                STATE["deployed"] = real_dep
                # An order that has FILLED is already inside real_dep. It also
                # stays in STATE["resting"] until its eligibility window
                # expires, so `gross` below counted the same money twice for
                # several minutes after every fill - overstating deployment and
                # blocking entries the account could afford. Errs safe, but it
                # throttles the strategy, so reconcile against the exchange's
                # own resting list rather than inferring from fills.
                if STATE["resting"]:
                    co, jo = api("GET", "/portfolio/orders",
                                 {"status": "resting", "limit": 200})
                    orders = (jo or {}).get("orders") or []
                    # Schema guard: if the id field were ever named something
                    # else, every id would read None, every order would look
                    # stale, gross would be UNDERSTATED and the runner would
                    # over-deploy. Only trust the response when the ids are
                    # actually there; an empty list is a genuine "none resting".
                    ok_schema = all(o.get("order_id") for o in orders)
                    if co == 200 and jo is not None and ok_schema:
                        live_oids = {o.get("order_id") for o in orders}
                        stale = [o for o in STATE["resting"] if o not in live_oids]
                        for oid in stale:
                            STATE["resting"].pop(oid, None)
                        if stale:
                            log("resting_reconciled", removed=len(stale),
                                still_resting=len(STATE["resting"]),
                                note="filled or cancelled; was double-counted")
            # BASELINE IS EQUITY, NOT CASH. Capturing only cash at startup
            # while positions were already open made `realised` read +$13.90
            # of pure fiction the instant the runner restarted - and it errs
            # in the UNSAFE direction, delaying the loss stop by exactly the
            # open-position cost. Baseline must be cash + positions, captured
            # after the reconciliation above so both sides are the same clock.
            if STATE["start_balance"] is None:
                STATE["start_balance"] = bal + STATE["deployed"]
                log("start_equity", equity=round(STATE["start_balance"], 4),
                    cash=bal, positions=round(STATE["deployed"], 2))
            STATE["realised"] = (bal + STATE["deployed"]) - STATE["start_balance"]
            # realised equity = cash + open positions AT COST. Peak is tracked
            # here so the pre-submit envelope has a reference that only moves
            # on settlement, never on a transient mark.
            STATE["cash"] = bal
            eq_now = bal + STATE["deployed"]
            # PERSISTENT peak, not a per-process one. The old in-memory peak
            # was discarded on every restart (17 on 2026-08-05), so no stop
            # ever accumulated a real measurement window and a decline after a
            # restart was measured from a lower reference - reading as a
            # smaller drawdown than it was.
            if not risk.hwm_ok():
                # Unreadable mark means dd reads 0 and NO stop can fire, which
                # is the unsafe direction. Re-seed from current equity so the
                # controls are live again from this cycle onward, and say so.
                log("hwm_reseeded", equity=round(eq_now, 2),
                    note="persistent peak was unreadable; history lost")
            # observe_CONFIRMED, never observe(). At settlement the payout is
            # credited to cash while the settled position is still returned by
            # /portfolio/positions, so one sample reads high by the cost basis.
            # On 2026-08-05 that wrote a $567.31 peak when true equity went
            # $477.31 -> $507.31. The mark only ever rises, so such a blip
            # poisons it permanently and every later stop fires early.
            STATE["peak_equity"] = risk.observe_confirmed(eq_now, who="runner")
            STATE["equity"] = eq_now
    # HARD STOP, fractional and persistent.
    # Was: realised <= -$150, measured from a session baseline that reset to
    # zero on every restart. A fixed dollar stop is also not a constant risk
    # policy - $75 was 22% of the bankroll at $341 and would be 3.8% at $2,000,
    # firing constantly on ordinary noise as the account grows. Simulated, a
    # $75 stop fires with probability 1.0000 within 20 days at EVERY bankroll
    # with the edge fully alive.
    _eq = STATE.get("equity")
    if _eq is not None and risk.stop_breached(_eq):
        log("stop_loss", **risk.summary(_eq))
        cancel_all("stoploss"); break

    # cancel any resting order whose eligibility window has passed
    for oid, info in list(STATE["resting"].items()):
        cl = datetime.fromisoformat(info["window"].replace("Z", "+00:00"))
        if (cl - now).total_seconds() / 60.0 < MIN_LEFT[0] - 1.0:
            cancel(oid, "past_eligible_window", info)
            STATE["resting"].pop(oid, None)

    gross = STATE["deployed"] + sum(i["qty"] * i["price"]
                                    for i in STATE["resting"].values())
    cash_now = STATE.get("cash") or 0.0
    _CYCLE["markets"].clear()
    _CYCLE["book"].clear()
    # PRE-PASS: the cross-sectional state of each open window. The main loop
    # below is per-series, so when it reaches a candidate it has not yet seen
    # the other coins in that window - and others_px is defined by exactly
    # those. Both the /markets and orderbook results are cached for the cycle,
    # so this pre-pass adds no API calls the main loop was not already making.
    XSEC = {}
    if FILTER_ON:
        try:
            for ser in SERIES:
                for mk in markets_cached(ser):
                    tk2, ct2 = mk.get("ticker"), mk.get("close_time")
                    if not tk2 or not ct2:
                        continue
                    close2 = datetime.fromisoformat(ct2.replace("Z", "+00:00"))
                    lf = (close2 - now).total_seconds() / 60.0
                    if not (MIN_LEFT[0] - 1 <= lf <= MIN_LEFT[1] + 1):
                        continue
                    b2 = book_cached(tk2)
                    if not b2:
                        continue
                    fb = b2["yes_bid"] if b2["mid"] > 0.5 else 1.0 - b2["yes_ask"]
                    if 0.50 <= fb < 0.95:      # matches the research extraction
                        XSEC.setdefault(ct2, {})[tk2] = fb
        except Exception as e:
            log("xsec_error", err=str(e)[:160])
            XSEC = {}
    try:
        for ser in SERIES:
            for mk in markets_cached(ser):
                tk, ct = mk.get("ticker"), mk.get("close_time")
                if not tk or not ct or tk in STATE["entered"]:
                    continue
                close = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                left = (close - now).total_seconds() / 60.0
                if not (MIN_LEFT[0] <= left <= MIN_LEFT[1]):
                    continue
                if STATE["per_window"].get(ct, 0) >= MAX_PER_WINDOW:
                    continue

                # VOLUME FILTER - the strongest conditioning variable found.
                # Our edge is selling longshots to people who insist on buying
                # them. In a market nobody has traded, there is no such demand:
                # the quote is a market maker's guess, not a consensus, and
                # there is no premium to collect. Measured on 2,106 historical
                # entries, window-equal-weighted, at a FLAT entry price of
                # 66.7-66.9c across every bucket:
                #
                #   contracts traded    win rate    edge
                #        0                0.6439   -2.51 pts   <- we LOSE here
                #      205                0.6357   -3.12 pts   <- and here
                #      546                0.6929   +2.42
                #      972                0.7262   +5.85
                #    2,079                0.7986  +13.05
                #
                # Survives every test that killed the z-score: works within all
                # three coins (+10.05/+13.38/+6.76), correlates NEGATIVELY with
                # price (-0.23, so it is not a price proxy), holds within coin
                # AND price (+10.77, CI [+7.35,+14.74]), and is stronger out of
                # sample (H1 +6.46, H2 +16.68). Max-statistic p = 0.0017 across
                # six features tested.
                #
                # Threshold 500 maximises TOTAL edge (+8,786 vs +6,605 pts
                # unfiltered) while keeping ~50% of opportunities - the half we
                # drop has negative expected value, so this raises profit and
                # frees capital at the same time.
                try:
                    mvol = float(mk.get("volume_fp") or 0)
                except Exception:
                    mvol = 0.0
                if mvol < MIN_VOLUME:
                    log("volume_filter_skip", ticker=tk, volume=mvol,
                        required=MIN_VOLUME, min_left=round(left, 2))
                    continue
                b = book_cached(tk)
                if b is None:
                    continue
                fav_yes = b["mid"] > 0.5
                side = "yes" if fav_yes else "no"
                fav_bid = b["yes_bid"] if fav_yes else 1.0 - b["yes_ask"]
                fav_ask = b["yes_ask"] if fav_yes else 1.0 - b["yes_bid"]
                if not (BAND[0] <= fav_bid <= BAND[1]):
                    continue
                # AGREEMENT RULE (see the block near AGREE_ON above). Applied
                # here because it needs `side`, which is only known once the
                # book has been read.
                if AGREE_ON:
                    fd = STATE["win_dir"].get(ct)
                    if fd is not None and fd != side:
                        # BURN the slot - do not look for an agreeing
                        # replacement. Backfilling concentrates the book into
                        # aligned windows and measured strictly worse.
                        STATE["per_window"][ct] = STATE["per_window"].get(ct, 0) + 1
                        log("direction_disagree_skip", ticker=tk, side=side,
                            window_dir=fd, window=ct, fav_bid=fav_bid,
                            min_left=round(left, 2),
                            note="slot burned, not backfilled")
                        continue

                # SHADOW BAND. Candidates at or above LIVE_HI are evaluated and
                # logged in full but NOT traded, so the band keeps producing
                # research data without risking money on it.
                #
                # Why 80c is the cut. Every band must clear its own break-even
                # (a contract at price p needs a win rate of p), so what matters
                # is SURPLUS = realised win rate - required win rate:
                #     65-70c  +11.3pp surplus, 16.6% return on capital
                #     75-80c  +13.9pp surplus, 17.9% return on capital
                #     80-85c   +6.6pp surplus,  7.95% return on capital
                #     85-95c   +5.5pp surplus,  6.09% return on capital
                # 80-85c posts a fine-looking 89.6% win rate but needs 82% just
                # to break even. It consumes 11.1% of deployed capital and
                # returns 6.7% of P&L - it dilutes the book. Dropping it lifts
                # book return on capital from 14.03% to 15.14%.
                #
                # HONESTY NOTE: the LIVE evidence that this band is losing is
                # NOT statistically established - 7 positions, SE 17.1pp, the
                # deficit is 0.62 standard errors and P(<=5 wins in 7) under the
                # backtest rate is 0.145. It is shadowed rather than deleted
                # precisely because the question is unresolved and the shadow
                # log is what will resolve it.
                if fav_bid >= LIVE_HI:
                    log("shadow_band", ticker=tk, side=side, fav_bid=fav_bid,
                        fav_ask=fav_ask, volume=mvol, min_left=round(left, 2),
                        window=ct, live_hi=LIVE_HI,
                        note="evaluated, logged, NOT traded")
                    continue
                px = snap(fav_bid)                 # JOIN the bid, back of queue
                if px >= fav_ask:                  # would cross -> refuse
                    log("would_cross_skip", ticker=tk, px=px, ask=fav_ask)
                    continue
                if gross + QTY * px > MAX_GROSS:
                    # Record WHICH market was refused, not just the gross. The
                    # capital caps block ~177 orders per 36 taken, so they are
                    # the dominant constraint on this strategy - but without the
                    # ticker there is no way to settle-match them later and
                    # measure what the refusals actually cost. That measurement
                    # is the only honest basis for changing the cap.
                    log("gross_cap", gross=round(gross, 2), ticker=tk,
                        price=px, qty=QTY, cost=round(QTY * px, 2),
                        window=ct, min_left=round(left, 2))
                    continue

                # HARD PRE-SUBMIT DRAWDOWN ENVELOPE.
                # The watchdog is a 60-second POLL - an alarm after the fact.
                # Between polls the runner could submit orders that push total
                # exposure past any drawdown limit, and a cancel/fill race can
                # turn a resting order into a position regardless. So the limit
                # has to be enforced BEFORE the order is sent, not observed
                # afterwards.
                #
                # Kalshi reserves cash the moment an order rests AND when it
                # fills, so if every open position and every resting order went
                # to zero, what remains is exactly the cash balance. That makes
                # the worst case trivial to state:
                #
                #     worst_case_equity = cash - cost_of_proposed_order
                #
                # and the envelope is that this must not breach MAX_DD_FRAC of
                # the running peak. Resting orders stay reserved until the
                # exchange confirms the cancel, so a cancel losing a race to a
                # fill cannot breach it either.
                # Peak comes from the PERSISTENT mark so the envelope cannot be
                # reset by a restart. Same fraction as before (30%); only the
                # reference is now durable.
                peak = risk.peak()
                worst_case = cash_now - QTY * px
                floor = risk.envelope_floor()
                if peak > 0 and worst_case < floor:
                    log("drawdown_envelope_block", ticker=tk, price=px,
                        qty=QTY, window=ct, min_left=round(left, 2),
                        peak=round(peak, 2),
                        floor=round(floor, 2), cash=round(cash_now, 2),
                        proposed=round(QTY * px, 2),
                        worst_case=round(worst_case, 2),
                        headroom=round(cash_now - floor, 2))
                    continue
                # SIGNALS. Both are needed now on EVERY candidate, not only on
                # weak ones, because they drive SIZE as well as the skip.
                opx = fl = None
                peers = [v for k, v in (XSEC.get(ct) or {}).items() if k != tk]
                if len(peers) >= 2:
                    opx = sum(peers) / len(peers)
                if opx is not None:
                    fl = frac_long(tk)

                # DEAD-ENTRY FILTER - unchanged behaviour, only reordered.
                if FILTER_ON and opx is not None and fl is not None:
                    if opx <= OTHERS_MAX and fl <= FLONG_MAX:
                        log("dead_entry_skip", ticker=tk,
                            others_px=round(opx, 4), n_peers=len(peers),
                            frac_long=round(fl, 4),
                            others_max=OTHERS_MAX, flong_max=FLONG_MAX,
                            min_left=round(left, 2))
                        continue

                # EDGE-WEIGHTED SIZING.
                #
                # Four signals genuinely predict edge, but every attempt to use
                # them as GATES lost money: discarding an entry buys nothing
                # back when opportunity, not capital, is the binding constraint.
                # Quote-price control was the obvious alternative and is mostly
                # impossible here - 84% of observed spreads are exactly one
                # tick, so quoting better would cross and be rejected.
                #
                # That leaves SIZE, which has no capacity limit and no
                # opportunity cost: every entry is still taken, only the
                # distribution changes. Walk-forward, at MATCHED contracts per
                # day (0.995x), this earned +$21.93/day on a $186.63 base -
                # about +12% - better on 5 of 7 held-out days, 95% CI
                # [-0.96, +37.99], P(worse) 0.031, and stable across every
                # regularisation, clip range and quantity band tried.
                #
                # THE COST, stated plainly: concentrating contracts on
                # higher-edge entries WORSENS the tail. Worst window goes from
                # -$34.50 to -$47.44 and the largest single position from
                # $11.85 to $17.38, even though overall window sd improves
                # slightly (8.35 -> 8.20). The clip and quantity bounds below
                # are deliberately conservative for that reason - the
                # unconstrained fit earned more and had a materially worse tail.
                q_use = QTY
                if SIZE_MODEL and opx is not None and fl is not None:
                    try:
                        rank_i = STATE["per_window"].get(ct, 0) + 1
                        xs = [rank_i, mvol, fl, opx]
                        z = [(xs[i] - SIZE_MODEL["mu"][i]) / SIZE_MODEL["sd"][i]
                             for i in range(4)]
                        pe = SIZE_MODEL["w"][0] + sum(
                            SIZE_MODEL["w"][i + 1] * z[i] for i in range(4))
                        r = pe / SIZE_MODEL["ec_mean"]
                        r = max(SIZE_CLIP_LO, min(SIZE_CLIP_HI, r))
                        q_use = int(max(SIZE_QTY_LO,
                                        min(SIZE_QTY_HI, round(QTY * r))))
                    except Exception as e:
                        log("size_model_error", err=str(e)[:120])
                        q_use = QTY
                if q_use > MAX_QTY:
                    q_use = MAX_QTY

                # The capital checks must use the SIZE WE WILL ACTUALLY SEND,
                # not the base quantity, or a sized-up order could slip past a
                # limit that was verified against a smaller number.
                if gross + q_use * px > MAX_GROSS:
                    log("gross_cap", gross=round(gross, 2), ticker=tk,
                        price=px, qty=q_use, cost=round(q_use * px, 2),
                        window=ct, min_left=round(left, 2))
                    continue
                if peak > 0 and (cash_now - q_use * px) < floor:
                    log("drawdown_envelope_block", ticker=tk, price=px,
                        qty=q_use, window=ct, min_left=round(left, 2),
                        peak=round(peak, 2), floor=round(floor, 2),
                        cash=round(cash_now, 2),
                        proposed=round(q_use * px, 2),
                        worst_case=round(cash_now - q_use * px, 2),
                        headroom=round(cash_now - floor, 2))
                    continue

                qa = b["yes_bid_sz"] if fav_yes else b["no_bid_sz"]
                log("opportunity", ticker=tk, side=side, min_left=round(left, 2),
                    volume=mvol,
                    fav_bid=fav_bid, fav_ask=fav_ask,
                    spread_c=round((fav_ask - fav_bid) * 100, 2),
                    post_at=px, queue_ahead=qa, qty=q_use, base_qty=QTY,
                    others_px=round(opx, 4) if opx is not None else None,
                    frac_long=round(fl, 4) if fl is not None else None,
                    expected_c=None)
                oid = place_maker(tk, side, px, q_use)
                STATE["orders"] += 1
                STATE["entered"].add(tk)
                STATE["per_window"][ct] = STATE["per_window"].get(ct, 0) + 1
                STATE["win_dir"].setdefault(ct, side)
                STATE["deployed"] += q_use * px
                STATE["open_cost"].append([ct, q_use * px])
                gross += q_use * px
                QTY_SENT = q_use
                persist()
                if oid:
                    STATE["resting"][oid] = {"ticker": tk, "side": side, "qty": q_use,
                                             "price": px, "window": ct,
                                             "queue_ahead": qa, "placed": now}
    except Exception as e:
        log("scan_error", err=str(e)[:200])
    log("heartbeat", resting=len(STATE["resting"]), orders=STATE["orders"],
        entered=len(STATE["entered"]), gross=round(gross, 2),
        deployed=round(STATE["deployed"], 2),
        realised=round(STATE["realised"], 4))
    time.sleep(5)
