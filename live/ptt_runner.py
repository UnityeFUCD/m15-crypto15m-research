"""POST-THEN-TAKE runner + randomised execution experiment (PTT-X).

STRATEGY
  For each 15-minute window, each coin, at decision minutes 5,7,9,11,13:
    - if the favourite MID is in 90-98c and we have no entry in that market,
      acquire QTY contracts of the favourite.
  The ACQUISITION POLICY is randomised per opportunity (see ARMS).
  Hold to settlement, one entry per market, max 3 per close window.

EXPERIMENT
  Primary observable is implementation shortfall, which needs NO settlement:

      IS = C_ptt - C_immediate_taker            (per contract, negative = cheaper)

  C_immediate_taker is the counterfactual all-in cost of crossing the whole
  target quantity at the decision instant, computed by WALKING THE BOOK DEPTH
  (not top-of-book: at size the ask has finite depth) with the exact fee at
  each level.

  Arm 0 fires that immediate IOC for real. It is both the control arm and the
  validator of the counterfactual model: for arm 0, IS should be ~0, and any
  systematic deviation is depth-model error that biases every other arm.

  Selection only cancels if the ORIGINAL TARGET QUANTITY is ultimately
  acquired. Every abandoned residual is logged as `shortfall_qty` so the
  same-population condition is measured, not assumed.

  Fill is measured QUANTITY-weighted (F_Q), never order-count: at QTY>1 an
  order that fills 1 of 11 is not "a fill".

IDENTIFICATION (why this design and not a cheaper one)
  Maker-only value is NOT point-identified from historical fills: two latent
  models with F=Y and F=1-Y produce byte-identical records and values of
  +0.25 and -0.25. Post-then-take escapes this ONLY because the residual is
  acquired regardless of fill, so R = Z - [F*pL + (1-F)*(pT+c)] and the
  fill/settlement association never enters the value. Verified numerically:
  maker-only diverges by 0.50 across the two latent models, PTT by 0.0000.
  That escape is CONDITIONAL - abandoning 10% of the unfilled leg reopens a
  0.05 gap, abandoning 50% reopens 0.25. Hence the retry loop below.

SAFETY
  - the fallback IOC is never sent until the passive order is confirmed
    TERMINAL (cancel acked, or explicitly re-queried) - otherwise a fill during
    the cancel race doubles the position
  - DRY_RUN default: logs intended actions, places nothing
  - kill file checked EVERY loop, at an absolute path echoed at startup
  - hard caps on qty, orders/day, daily loss, concurrent exposure
  - every resting order is force-cancelled on exit
"""
from __future__ import annotations
import base64, json, math, os, random, signal, sys, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

CRED = Path(os.environ.get("KALSHI_CRED", "C:/Users/fycin/kalshi_weather_trading"))
BASE = "https://api.elections.kalshi.com/trade-api/v2"
PREFIX = "/trade-api/v2"
HERE = Path(__file__).resolve().parent
KILL = (HERE / "PTT_KILL").resolve()
LOG = HERE / f"ptt_{datetime.now(timezone.utc):%Y%m%d}.jsonl"

DRY_RUN        = os.environ.get("PTT_LIVE", "0") != "1"
QTY            = int(os.environ.get("PTT_QTY", "11"))
MAX_QTY        = 45
BAND           = (0.90, 0.98)
DEC_MIN        = (5, 7, 9, 11, 13)
MAX_PER_WINDOW = int(os.environ.get("PTT_PER_WINDOW", "1"))
# ^ 1 during the size-ladder experiment. Coins inside a 15-minute window are
# correlated (rho=0.4111) and compete for the gross cap, so drawing an
# independent arm/size per coin is NOT window-level randomisation. One
# experimental market per window makes the window the clean unit.
MAX_ORDERS_DAY = 250
MAX_DAILY_LOSS = float(os.environ.get("PTT_MAX_LOSS", "80.00"))
MAX_GROSS      = float(os.environ.get("PTT_MAX_GROSS", "150.00"))
CAP_PX         = 0.985          # never pay above this when walking the book
DEPTH          = 20
SERIES = ["KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M", "KXDOGE15M"]

# --- randomised policy arms -------------------------------------------------
# timeout seconds before crossing for the unfilled residual.
# 0 = immediate IOC control (also validates the depth-walk counterfactual).
# Live fills so far arrive in 1.7-5.4s, so the short arms are the informative
# ones; 120 is kept because it is the configuration the backtest priced.
ARMS = (0, 15, 30, 60, 120)
# SIZE LADDER - the one measurement that decides the whole timeline.
# F_Q is measured at 1.0000 (q=1, 58 contracts) and 0.5704 (q=11, 270 contracts),
# with ZERO partial fills at q=11 - fills are all-or-nothing on price, so 11 is
# nowhere near the size threshold. Where that threshold IS decides whether the
# capacity ceiling is ~$556/day or ~$16/day. Randomised per opportunity so size
# is not confounded with time of day, which is exactly what the q=1 vs q=11
# comparison suffers from.
QTY_LADDER = tuple(int(x) for x in
                   os.environ.get("PTT_LADDER", "3,8,20,45").split(","))
# where to rest, when the spread is wide enough for the two to differ
PLACEMENTS = ("bid_plus_1", "ask_minus_1")
RNG = random.Random(int(os.environ.get("PTT_SEED", "20260804")))

_cfg = {}
for line in (CRED / ".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        _cfg[k.strip()] = v.strip().strip('"').strip("'")
_priv = serialization.load_pem_private_key((CRED / "kalshi_private_key.pem").read_bytes(), password=None)
_KEY = _cfg.get("KALSHI_API_KEY_ID") or os.environ.get("KALSHI_API_KEY_ID")
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
        r = _S.request(method, BASE + path, params=params, json=body, headers=h, timeout=timeout)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"_raw": r.text[:300]}
    except Exception as e:
        return 0, {"_err": str(e)}


def halted():
    return KILL.exists()


STATE = {"orders_today": 0, "realised": 0.0, "start_balance": None, "entered": set(),
         "per_window": {}, "resting": {}, "probed": False, "deployed": 0.0}

# FIX 1 of 3. The "one entry per market" guard used to live only in process
# memory. Restarting the runner forgot every market it already held, so three
# restarts in twenty minutes re-entered seven markets and DOUBLED the position
# on each. That cost $34 of the $76 lost on 2026-08-04. It is now on disk.
ENTERED_F = HERE / f"entered_{datetime.now(timezone.utc):%Y%m%d}.json"
if ENTERED_F.exists():
    try:
        _d = json.loads(ENTERED_F.read_text())
        STATE["entered"] = set(_d.get("entered", []))
        STATE["per_window"] = {k: int(v) for k, v in (_d.get("per_window") or {}).items()}
        STATE["deployed"] = float(_d.get("deployed", 0.0))
        print(f"resumed: {len(STATE['entered'])} markets already entered today, "
              f"${STATE['deployed']:.2f} deployed")
    except Exception as e:
        print(f"WARNING could not read {ENTERED_F}: {e}")


def persist():
    try:
        ENTERED_F.write_text(json.dumps({"entered": sorted(STATE["entered"]),
                                         "per_window": STATE["per_window"],
                                         "deployed": STATE["deployed"]}))
    except Exception as e:
        log("persist_error", err=str(e)[:120])


def fee(p, c=1):
    """Verified against 40 own taker fills: 0.07*C*P*(1-P) rounded UP to the
    CENTICENT. The round-up applies ONCE PER FILL BLOCK, not per contract, so
    at size this is strictly cheaper than c*fee(p,1) - 0.0045c/contract at
    P=0.92, C=11. Small, but it is the number the whole experiment measures."""
    return math.ceil(0.07 * c * p * (1 - p) * 1e4 - 1e-12) / 1e4


def tick_of(price):
    """tapered grid: 0.1c below 10c or above 90c, else 1c"""
    return 0.001 if (price > 0.90 or price < 0.10) else 0.01


def snap(p):
    t = tick_of(p)
    return round(round(p / t) * t, 4)


def book(ticker):
    """Full-depth book. `yes_levels`/`no_levels` are resting BIDS on each side,
    descending. To buy side S you consume the OPPOSITE side's bids: buying NO at
    (1 - yes_bid), buying YES at (1 - no_bid)."""
    c, j = api("GET", f"/markets/{ticker}/orderbook", {"depth": DEPTH})
    if c != 200 or not j:
        return None
    ob = j.get("orderbook_fp") or {}
    yb = sorted([(float(p), float(s)) for p, s in (ob.get("yes_dollars") or [])], key=lambda x: -x[0])
    nb = sorted([(float(p), float(s)) for p, s in (ob.get("no_dollars") or [])], key=lambda x: -x[0])
    if not yb or not nb:
        return None
    yes_bid, yes_ask = yb[0][0], 1.0 - nb[0][0]
    if not (0 < yes_bid < yes_ask < 1):
        return None
    return dict(yes_bid=yes_bid, yes_ask=yes_ask, mid=(yes_bid + yes_ask) / 2,
                yes_levels=yb, no_levels=nb)


def walk(b, side, qty, cap=CAP_PX):
    """All-in cost of crossing `qty` of `side` right now, walking real depth.
    Returns (total_cost_incl_fee, qty_available, worst_price_touched)."""
    levels = b["yes_levels"] if side == "no" else b["no_levels"]
    cost, got, worst = 0.0, 0, None
    for opp_px, sz in levels:
        px = round(1.0 - opp_px, 4)
        if px > cap:
            break
        take = min(int(sz), qty - got)
        if take <= 0:
            break
        cost += take * px + fee(px, take)     # block fee, rounded up once
        got += take
        worst = px
        if got >= qty:
            break
    return cost, got, worst


def actual_cost(oid, since_ms=None):
    """All-in cost of what an order REALLY filled, from the fills endpoint.
    The control arm is worthless if both sides of the comparison come from the
    same model - it would make IS identically zero by construction.

    FIX 3 of 3. This used to trust `order_id` as a server-side filter. If the
    endpoint ignores it, EVERY recent fill comes back and gets attributed to
    this one order - which is how the runner reported passive_qty=0 on orders
    the exchange had recorded as maker fills, and why F_Q looked unmeasured all
    day when it was 0.5704. Always filter client-side."""
    c, j = api("GET", "/portfolio/fills", {"order_id": oid, "limit": 200})
    fl = ((j or {}).get("fills") or []) if c == 200 else []
    fl = [f for f in fl if f.get("order_id") == oid]
    tot, qty = 0.0, 0
    for f in fl:
        # V2 reports size in count_fp; plain `count` is always null
        n = int(float(f.get("count_fp") or f.get("count") or 0))
        if n <= 0:
            continue
        # fills report both legs; our economic price is the side we bought
        yp = f.get("yes_price_dollars")
        np_ = f.get("no_price_dollars")
        px = float(np_ if (f.get("side") == "no") else yp)
        taker = f.get("is_taker")
        tot += n * px + (fee(px, n) if taker else 0.0)
        qty += n
    return (tot, qty, len(fl)) if fl else (None, 0, 0)


def cancel_all(reason="shutdown"):
    for oid, info in list(STATE["resting"].items()):
        c, j = (200, {"_dry": True}) if DRY_RUN else api("DELETE", f"/portfolio/events/orders/{oid}")
        log("cancel", oid=oid, ticker=info.get("ticker"), status=c, reason=reason)
        STATE["resting"].pop(oid, None)


def on_signal(sig, frm):
    log("sigexit", sig=int(sig))
    cancel_all("signal")
    sys.exit(0)


for s in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(s, on_signal)
    except Exception:
        pass


def place(ticker, side, price, qty, post_only, tag):
    """V2 order API. `side` is our economic side ('yes'/'no'); V2 speaks only in
    YES terms: 'bid' = buy YES, 'ask' = sell YES. Buying the NO favourite at X
    is selling YES at (1 - X). Returns (order_id, filled_now, ack_ms)."""
    if side == "yes":
        v2_side, v2_price = "bid", price
    else:
        v2_side, v2_price = "ask", round(1.0 - price, 4)
    body = {"ticker": ticker, "side": v2_side,
            "count": f"{qty:.2f}", "price": f"{v2_price:.4f}",
            "client_order_id": f"ptt{int(time.time()*1000)}",
            "time_in_force": "good_till_canceled" if post_only else "immediate_or_cancel",
            "self_trade_prevention_type": "taker_at_cross",
            "post_only": bool(post_only)}
    if DRY_RUN:
        log("DRY_order", ticker=ticker, side=side, price=price, qty=qty,
            post_only=post_only, tag=tag)
        return None, 0, None
    c, j = api("POST", "/portfolio/events/orders", body=body)
    o = j or {}
    filled = int(float(o.get("fill_count") or 0))
    log("order", ticker=ticker, side=side, v2_side=v2_side, price=price,
        v2_price=v2_price, qty=qty, post_only=post_only, tag=tag, status=c,
        immediate_fill=filled, remaining=o.get("remaining_count"),
        ack_ms=o.get("ts_ms"), resp=str(j)[:200])
    if c in (200, 201):
        oid = o.get("order_id") or (o.get("order") or {}).get("order_id")
        return oid, filled, o.get("ts_ms")
    return None, 0, None


def terminal_unfilled(oid, target):
    """Drive a resting order to a TERMINAL state and return the quantity that
    never filled. Never guess from a status code: a cancel that does not return
    an explicit reduced_by is re-queried before we are willing to cross."""
    c, j = api("DELETE", f"/portfolio/events/orders/{oid}")
    j = j or {}
    if c in (200, 201) and j.get("reduced_by") is not None:
        return int(float(j["reduced_by"])), c, "cancel_acked"
    # ambiguous (404 / no reduced_by): ask the exchange what actually happened
    c2, j2 = api("GET", f"/portfolio/orders/{oid}")
    o = ((j2 or {}).get("order") or j2 or {})
    rem = o.get("remaining_count")
    st = o.get("status")
    if rem is not None:
        return int(float(rem)), c, f"requeried:{st}"
    fc = o.get("fill_count")
    if fc is not None:
        return max(0, target - int(float(fc))), c, f"requeried_fc:{st}"
    # last resort: the fills themselves. Never guess - a wrong guess here does
    # not just misprice a row, it silently fabricates the F_Q we are measuring.
    _, fq, nf = actual_cost(oid)
    if nf:
        return max(0, target - fq), c, "resolved_by_fills"
    return None, c, f"UNRESOLVED:{c2}"


def shortfall(info, passive_qty, passive_px, taker_qty, taker_cost, abandoned, src="mixed"):
    """One row per opportunity. IS is per contract ACQUIRED; negative = cheaper
    than crossing immediately. Settlement never enters this calculation."""
    acquired = passive_qty + taker_qty
    c_ptt = passive_qty * passive_px + taker_cost      # maker fee is zero (verified)
    c_imm_per = info["c0_per"] if info["c0_per"] else float("nan")
    is_tot = c_ptt - acquired * c_imm_per if acquired else 0.0
    log("shortfall",
        ticker=info["ticker"], window=info["window"], minute=info["minute"],
        arm=info["arm"], placement=info["placement"],
        target=info["target"], passive_qty=passive_qty, taker_qty=taker_qty,
        abandoned=abandoned, acquired=acquired,
        F_Q=round(passive_qty / info["target"], 4) if info["target"] else None,
        completed=(acquired == info["target"]),
        rest_px=passive_px, c_imm_per=round(c_imm_per, 6),
        c_ptt=round(c_ptt, 6), c_imm=round(acquired * c_imm_per, 6),
        IS_total=round(is_tot, 6),
        IS_per_contract_c=round(is_tot / acquired * 100, 4) if acquired else None,
        cost_src=src, depth_at_decision=info["depth0"],
        spread0_c=round(info["spread0"] * 100, 2))


print(f"KILL FILE (touch to halt): {KILL}")
print(f"MODE: {'DRY RUN - places nothing' if DRY_RUN else '*** LIVE ***'}   qty={QTY}")
log("start", dry_run=DRY_RUN, qty=QTY, kill=str(KILL), arms=list(ARMS),
    placements=list(PLACEMENTS), max_loss=MAX_DAILY_LOSS, max_gross=MAX_GROSS)
if max(QTY_LADDER) > MAX_QTY:
    sys.exit(f"ladder {QTY_LADDER} exceeds MAX_QTY {MAX_QTY}")
print(f"SIZE LADDER: {QTY_LADDER}   MAX_GROSS ${MAX_GROSS}   "
      f"stop ${MAX_DAILY_LOSS}")

while True:
    if halted():
        log("halted", path=str(KILL)); cancel_all("kill"); break
    if STATE["orders_today"] >= MAX_ORDERS_DAY:
        log("cap_orders"); cancel_all("cap"); break
    if not DRY_RUN:
        c, j = api("GET", "/portfolio/balance")
        if c == 200 and j:
            bal = float((j or {}).get("balance_dollars") or 0)
            if STATE.get("start_balance") is None:
                STATE["start_balance"] = bal
                log("start_balance", balance=bal)
            open_cost = sum(i["qty"] * i["price"] for i in STATE["resting"].values())
            STATE["realised"] = bal - STATE["start_balance"] + open_cost
    if STATE["realised"] <= -MAX_DAILY_LOSS:
        log("stop_loss", realised=round(STATE["realised"], 4)); cancel_all("stoploss"); break
    now = datetime.now(timezone.utc)

    # ---- release capital from windows that have settled
    oc = STATE.setdefault("open_cost", [])
    if oc:
        live_oc, freed = [], 0.0
        for ctime, cost in oc:
            try:
                cl = datetime.fromisoformat(str(ctime).replace("Z", "+00:00"))
            except Exception:
                continue
            if now > cl + timedelta(seconds=90):
                freed += cost
            else:
                live_oc.append([ctime, cost])
        if freed:
            STATE["open_cost"] = live_oc
            STATE["deployed"] = max(0.0, STATE["deployed"] - freed)
            log("capital_released", freed=round(freed, 2),
                deployed=round(STATE["deployed"], 2))
            persist()

    # ---- resting orders that have reached their arm's timeout
    for oid, info in list(STATE["resting"].items()):
        if (now - info["placed"]).total_seconds() < info["arm"]:
            continue
        if DRY_RUN:
            unfilled, st, how = info["qty"], 200, "dry"
        else:
            unfilled, st, how = terminal_unfilled(oid, info["qty"])
        STATE["resting"].pop(oid, None)
        log("terminal", oid=oid, ticker=info["ticker"], arm=info["arm"], status=st,
            how=how, unfilled=unfilled,
            passive_filled=None if unfilled is None else info["qty"] - unfilled,
            waited_s=round((now - info["placed"]).total_seconds(), 1))
        if unfilled is None:
            # terminal state genuinely unknown: do NOT cross (a fill during the
            # race would double the position) and do NOT record a row
            log("dropped_row", ticker=info["ticker"], reason="unresolved_terminal")
            continue
        passive_qty = info["qty"] - unfilled
        taker_qty, taker_cost, abandoned = 0, 0.0, unfilled
        # RETRY UNTIL ACQUIRED. Post-then-take is identified only while the
        # residual is actually taken: an IOC that partially fills and silently
        # drops the rest reopens exactly the ambiguity this design exists to
        # close (10% abandonment -> 0.05 gap, 50% -> 0.25). Keep chasing until
        # the target is met or the window is too close to settlement.
        want = unfilled
        for attempt in range(6):
            if want <= 0:
                break
            try:
                cl = datetime.fromisoformat(str(info["window"]).replace("Z", "+00:00"))
                if now > cl - timedelta(seconds=20):
                    log("chase_abandoned_late", ticker=info["ticker"], want=want)
                    break
            except Exception:
                pass
            b = book(info["ticker"])
            if not b:
                break
            mcost, avail, worst = walk(b, info["side"], want)
            if avail <= 0 or not worst:
                time.sleep(0.4)
                continue
            oid2, tf, _ = place(info["ticker"], info["side"], snap(worst),
                                min(avail, want), post_only=False,
                                tag=f"promote{attempt}")
            STATE["orders_today"] += 1
            ac, aq, nf = actual_cost(oid2) if oid2 else (None, 0, 0)
            got = aq if nf else tf
            taker_qty += got
            taker_cost += (ac if ac is not None else
                           (walk(b, info["side"], got)[0] if got > 0 else 0.0))
            want -= got
            log("promote", ticker=info["ticker"], attempt=attempt, want=want + got,
                avail=avail, got=got, worst_px=worst,
                cost_src="fills" if nf else "model", still_needed=want)
            if got == 0:
                time.sleep(0.5)
        abandoned = want
        if abandoned:
            log("ABANDONED", ticker=info["ticker"], qty=abandoned,
                target=info["qty"],
                note="identification degrades with abandonment rate")
        shortfall(info, passive_qty, info["price"], taker_qty, taker_cost, abandoned)

    # ---- scan open markets for eligible entries
    # FIX 2 of 3. gross used to count RESTING orders only. The instant an order
    # filled it left the dict and stopped counting, so there was no cap at all
    # on accumulated positions - $270 of a $287 account ended up deployed on
    # 2026-08-04. Filled cost now stays counted until the window settles.
    gross = (sum(i["qty"] * i["price"] for i in STATE["resting"].values())
             + STATE["deployed"])
    try:
        for ser in SERIES:
            c, j = api("GET", "/markets", {"series_ticker": ser, "status": "open", "limit": 20})
            if c != 200 or not j:
                continue
            for mk in (j.get("markets") or []):
                tk = mk.get("ticker")
                ct = mk.get("close_time")
                if not tk or not ct or tk in STATE["entered"]:
                    continue
                close = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                elapsed = (now - (close - timedelta(minutes=15))).total_seconds()
                if elapsed < 0 or now >= close - timedelta(seconds=30):
                    continue
                minute = int(elapsed // 60)
                if minute not in DEC_MIN:
                    continue
                if STATE["per_window"].get(ct, 0) >= MAX_PER_WINDOW:
                    continue
                b = book(tk)
                if b is None:
                    continue
                fav_yes = b["mid"] > 0.5
                fav_mid = b["mid"] if fav_yes else 1.0 - b["mid"]
                if not (BAND[0] <= fav_mid <= BAND[1]):
                    continue
                side = "yes" if fav_yes else "no"
                fav_bid = b["yes_bid"] if fav_yes else 1.0 - b["yes_ask"]
                fav_ask = b["yes_ask"] if fav_yes else 1.0 - b["yes_bid"]

                # size is drawn UNCONDITIONALLY, before looking at capital.
                # Conditioning the draw on remaining room would tie the
                # treatment to earlier outcomes and break the randomisation.
                QTY = RNG.choice(QTY_LADDER)
                if gross + QTY * fav_ask > MAX_GROSS:
                    log("size_skipped", ticker=tk, drawn=QTY,
                        gross=round(gross, 2), room=round(MAX_GROSS - gross, 2))
                    continue
                # counterfactual: cross the WHOLE target now, walking real depth
                c0_tot, c0_avail, c0_worst = walk(b, side, QTY)
                arm = RNG.choice(ARMS)
                # SIZE-SELECTION DEFECT, fixed. This used to skip whenever the
                # book could not absorb QTY by CROSSING. But a big draw is far
                # more likely to fail that test than a small one, so the sizes
                # that survived were biased toward deep-book markets - which
                # corrupts the very F_Q(q) curve this experiment exists to
                # measure. Depth is only genuinely required for arm 0, which
                # actually crosses. Every other arm POSTS: it supplies depth
                # rather than consuming it, and its residual is chased later
                # with retries. So only arm 0 needs the book to be deep enough.
                if c0_avail < QTY:
                    if arm == 0:
                        log("thin_book", ticker=tk, want=QTY, avail=c0_avail,
                            arm=arm, note="control arm needs crossable depth")
                        continue
                    log("thin_at_entry", ticker=tk, want=QTY, avail=c0_avail,
                        arm=arm, note="posting anyway; counterfactual is partial")
                c0_per = c0_tot / c0_avail if c0_avail else fav_ask + fee(fav_ask)
                spread = fav_ask - fav_bid
                two_ticks = spread > 1.5 * tick_of(fav_bid)
                placement = RNG.choice(PLACEMENTS) if two_ticks else "bid_plus_1"
                px = snap(fav_bid + tick_of(fav_bid)) if placement == "bid_plus_1" \
                    else snap(fav_ask - tick_of(fav_ask))
                if px >= fav_ask:                       # never cross when resting
                    px = snap(fav_bid + tick_of(fav_bid))
                    placement = "bid_plus_1"
                if px >= fav_ask:
                    continue
                if gross + QTY * px > MAX_GROSS:
                    log("gross_cap", gross=round(gross, 2))
                    continue

                info = dict(ticker=tk, side=side, qty=QTY, target=QTY, price=px,
                            placed=now, arm=arm, placement=placement, window=ct,
                            minute=minute, c0_per=c0_per, depth0=c0_avail,
                            spread0=spread, ask0=fav_ask, bid0=fav_bid)
                log("opportunity", ticker=tk, minute=minute, side=side, arm=arm,
                    placement=placement, fav_mid=round(fav_mid, 4),
                    bid=round(fav_bid, 4), ask=round(fav_ask, 4), rest_at=px,
                    c_imm_per=round(c0_per, 6),
                    c_imm_per_c=round(c0_per * 100, 3),
                    top_ask_c=round((fav_ask + fee(fav_ask)) * 100, 3),
                    depth_avail=c0_avail, worst_px=c0_worst)
                STATE["entered"].add(tk)
                STATE["per_window"][ct] = STATE["per_window"].get(ct, 0) + 1
                # capital committed the moment we commit to acquiring, released
                # when the window settles (pruned at the top of the loop)
                STATE["deployed"] += QTY * px
                STATE.setdefault("open_cost", []).append([ct, QTY * px])
                persist()

                if arm == 0:
                    # CONTROL: cross immediately. Its cost MUST come from the
                    # fills, not the walk - scoring it with the same model that
                    # produced c_imm would force IS to zero by construction and
                    # the control would validate nothing.
                    oid0, tf, _ = place(tk, side, snap(c0_worst), QTY,
                                        post_only=False, tag="control_ioc")
                    STATE["orders_today"] += 1
                    ac, aq, nf = actual_cost(oid0) if oid0 else (None, 0, 0)
                    if not nf:
                        log("dropped_row", ticker=tk, reason="control_fills_unavailable")
                        continue
                    shortfall(info, 0, px, aq, ac, QTY - aq, src="fills")
                    continue

                oid, imm, ack = place(tk, side, px, QTY, post_only=True, tag="rest")
                STATE["orders_today"] += 1
                gross += QTY * px
                if oid:
                    info["ack_ms"] = ack
                    STATE["resting"][oid] = info
                    if not STATE["probed"] and not DRY_RUN:
                        # one-shot discovery: which GET path resolves an order,
                        # and does the exchange expose queue position under
                        # price-time priority? terminal_unfilled depends on this.
                        for p in (f"/portfolio/orders/{oid}",
                                  f"/portfolio/events/orders/{oid}",
                                  f"/portfolio/orders/{oid}/queue_position"):
                            pc, pj = api("GET", p)
                            log("order_probe", path=p, status=pc, fields=str(pj)[:350])
                        STATE["probed"] = True
                elif imm == 0:
                    shortfall(info, 0, px, 0, 0.0, QTY)
    except Exception as e:
        log("scan_error", err=str(e)[:200])
    log("heartbeat", resting=len(STATE["resting"]), orders=STATE["orders_today"],
        entered=len(STATE["entered"]), gross=round(gross, 2),
        realised=round(STATE["realised"], 4))
    time.sleep(5)
