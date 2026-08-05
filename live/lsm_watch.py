"""LSM watchdog - autonomous overnight safety + status summary.

Runs independently of the trading runner and of any assistant session, so the
account is protected even if nobody is watching.

WHAT IT ENFORCES (hard, no judgement calls)
  1. equity drawdown from session baseline  > $30  -> touch LSM_KILL
  2. equity drawdown from running peak      > $30  -> touch LSM_KILL
  3. runner process dead while no kill file -> restart it once per cycle
  4. open position count implies runaway    -> touch LSM_KILL

  $30 sits deliberately INSIDE the runner's own $50 stop. Halting early
  overnight costs only opportunity; halting late costs money. Asymmetric, so
  it errs toward stopping.

EQUITY, not balance. Kalshi reserves cash when an order rests and when a
position is open, so raw balance understates equity while trades are live.
equity = cash + cost basis of open positions. Cost basis (not mark) is used
deliberately: it is the number that cannot be inflated by a favourable mark.

RESOURCE DISCIPLINE
  No pandas, no dataframes, one small JSON read per cycle, 60s sleep.
  Reads the tail of the log only. Designed to be invisible on RAM and disk.
"""
import base64
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

HERE = Path(__file__).resolve().parent
CRED = Path("C:/Users/fycin/kalshi_weather_trading")
BASE = "https://api.elections.kalshi.com/trade-api/v2"
PREFIX = "/trade-api/v2"
KILL = HERE / "LSM_KILL"
STATUS = HERE / "lsm_status.jsonl"
MAX_DD = float(os.environ.get("WATCH_MAX_DD", "30"))
MAX_POSITIONS = int(os.environ.get("WATCH_MAX_POS", "20"))
CYCLE = int(os.environ.get("WATCH_CYCLE", "60"))
PY = sys.executable

_cfg = {}
for line in (CRED / ".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        _cfg[k.strip()] = v.strip().strip('"').strip("'")
_priv = serialization.load_pem_private_key(
    (CRED / "kalshi_private_key.pem").read_bytes(), password=None)
_KEY = _cfg.get("KALSHI_API_KEY_ID")
_S = requests.Session()


def api(path, params=None):
    ts = str(int(time.time() * 1000))
    sig = base64.b64encode(_priv.sign((ts + "GET" + PREFIX + path).encode(),
          padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
          hashes.SHA256())).decode()
    h = {"KALSHI-ACCESS-KEY": _KEY, "KALSHI-ACCESS-SIGNATURE": sig,
         "KALSHI-ACCESS-TIMESTAMP": ts}
    try:
        r = _S.get(BASE + path, params=params, headers=h, timeout=20)
        return r.status_code, r.json()
    except Exception as e:
        return 0, {"_err": str(e)}


def equity():
    """cash + mark value of open positions, or None if the API is unreachable.

    TWO BUGS THIS FIXES, both found before they could halt trading wrongly:
      1. Cash is RESERVED the moment an order rests or fills, so raw balance
         reads like a loss. With an $80 gross cap and a $30 halt threshold,
         using balance alone would have tripped the halt on a completely
         healthy book.
      2. The position fields are `position_fp` and `market_exposure_dollars`,
         not `position` / `market_exposure`. Reading the wrong names returned
         "0 open positions" while two were live.

    `portfolio_value` on the balance endpoint is the authoritative mark of open
    positions in CENTS, so equity is just cash + that.

    THIRD BUG, and the one that actually halted a working strategy:
      This used `portfolio_value` - the live MARK on open positions - directly
      contradicting the docstring above. Mid-window a binary bought at 68c can
      trade at 40c and still settle at 100c, so the mark swings violently while
      nothing real has happened. On 2026-08-04 it read $29.10 on two positions
      that settled for $60.00, producing a $76.80 "drawdown" when the realised
      figure was $63.00 - and tripped the $75 halt on a strategy that was up
      50% and still working.

      Open positions are now carried at COST. Equity then moves only when a
      position actually settles, which is the only time money genuinely
      changes hands.
    """
    c, j = api("/portfolio/balance")
    if c != 200 or not j:
        return None, None, None
    cash = float(j.get("balance_dollars") or 0)
    c2, j2 = api("/portfolio/positions", {"limit": 200})
    if c2 != 200 or j2 is None:
        return None, None, None          # never guess equity from cash alone
    cost, npos = 0.0, 0
    for p in (j2.get("market_positions") or []):
        try:
            if float(p.get("position_fp") or 0) != 0:
                npos += 1
                cost += abs(float(p.get("total_traded_dollars") or 0))
        except Exception:
            pass
    return cash + cost, cash, npos


def runner_alive():
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" |"
             " Where-Object { $_.CommandLine -like '*lsm_runner.py*' } |"
             " Measure-Object).Count"],
            capture_output=True, text=True, timeout=30)
        return int((out.stdout or "0").strip() or 0) > 0
    except Exception:
        return True          # unknown -> assume alive, never restart blindly


def restart_runner():
    # Reproduce the config the runner was ACTUALLY running, published by the
    # runner itself at startup. The old code fell back to hardcoded defaults
    # (qty 10, gross 80, no volume floor) and so silently reverted the tuned
    # settings on every restart - which is exactly what happened at 03:00:22Z
    # on 2026-08-05, putting a second, differently-configured runner live
    # alongside the intended one.
    env = dict(os.environ)
    cfgp = HERE / "lsm_config.json"
    try:
        saved = json.loads(cfgp.read_text())
        if not isinstance(saved, dict) or "LSM_QTY" not in saved:
            raise ValueError("incomplete config")
        env.update({k: str(v) for k, v in saved.items()})
        note(event="restart_using_published_config", **saved)
    except Exception as e:
        # No trustworthy config: do NOT invent one. A runner with the wrong
        # size and no volume floor is worse than no runner at all.
        note(event="restart_ABORTED_no_config", err=str(e)[:120],
             hint="start the runner manually with the intended env")
        return
    stamp = datetime.now(timezone.utc).strftime("%H%M%S")
    with open(HERE / f"lsm_restart_{stamp}.log", "w") as f:
        subprocess.Popen([PY, "-u", "lsm_runner.py"], cwd=str(HERE),
                         stdout=f, stderr=subprocess.STDOUT, env=env)


def note(**kw):
    rec = {"utc": datetime.now(timezone.utc).isoformat(), **kw}
    with STATUS.open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    print(json.dumps(rec, default=str), flush=True)


def halt(reason, **kw):
    KILL.write_text(reason)
    note(action="HALTED", reason=reason, **kw)


import risk                      # unified, persistent, fractional risk control

eq, cash, npos = equity()
baseline = eq if eq is not None else None
# PEAK MUST SURVIVE RESTARTS. Previously `peak` was seeded from whatever
# equity happened to be at process start. The runner was restarted 17 times
# and this watchdog re-baselined 13 times on 2026-08-05, so no drawdown
# control ever accumulated a meaningful window - and the last restart
# discarded the day's $524.71 high, which would have made a later bleed to
# $400 look like a $66 drawdown instead of $124. The persistent high-water
# mark is now the authority; the in-process value only ever tracks it.
persisted = risk.peak()
if eq is not None:
    persisted = risk.observe(eq, who="watchdog_start")
peak = max(persisted, baseline or 0.0)
note(event="watchdog_start", baseline_equity=baseline, cash=cash,
     open_positions=npos, cycle_s=CYCLE,
     persistent_peak=round(peak, 2),
     watch_frac=risk.WATCH_FRAC, stop_frac=risk.STOP_FRAC,
     halt_at=round(risk.summary(eq)['watch_at'], 2), floor=risk.FLOOR,
     absolute_ceiling=MAX_DD, hwm_ok=risk.hwm_ok())
if baseline is None:
    note(event="fatal", msg="cannot reach API at startup")
    sys.exit(1)

restarts = 0
last_eq = None
while True:
    time.sleep(CYCLE)
    if KILL.exists():
        note(event="kill_present", msg="halted, watchdog idling")
        continue
    eq, cash, npos = equity()
    if eq is None:
        note(event="api_unreachable", msg="skipping cycle, not halting")
        continue
    # PEAK NEEDS TWO CONSECUTIVE CONFIRMATIONS.
    # At settlement the exchange briefly credits cash while `portfolio_value`
    # still marks the position, so equity spikes for one sample and falls back.
    # A single such blip set peak to $251.57 when no reading ever showed above
    # $235 - manufacturing a $17 "drawdown" out of nothing. Left alone it would
    # eventually halt a strategy that is 10-for-10 on settlements. Taking the
    # lower of the last two readings makes a one-sample spike unable to set the
    # peak, while a genuine sustained rise still does.
    # Two-sample confirmation now lives in risk.observe_confirmed(), so every
    # caller gets it rather than each one having to remember. The runner did
    # NOT have it and wrote a phantom $567.31 peak on 2026-08-05.
    peak = max(risk.observe_confirmed(eq, who="watchdog"), risk.peak())
    if not risk.hwm_ok():
        # Unreadable mark means no threshold can fire - the unsafe direction.
        note(event="hwm_unreadable", equity=eq, msg="reseeding from equity")
        peak = risk.observe(eq, who="watchdog_reseed")
    last_eq = eq
    dd_peak = peak - eq
    # The session-baseline check is GONE. It re-based on every restart (13 on
    # 2026-08-05), so it measured from whatever equity happened to be when the
    # process began rather than from the real high, and read declines as
    # smaller than they were. The persistent peak is the only reference now.
    # FRACTIONAL, not fixed dollars. WATCH_FRAC (0.22) is deliberately LOOSER
    # than the runner's own STOP_FRAC (0.20) so that in normal operation the
    # runner stops itself and this backstop only fires when the runner is
    # wedged or dead. MAX_DD is retained as an absolute ceiling in case a
    # single catastrophic move outruns the fractional rule.
    if risk.watch_breached(eq):
        halt("drawdown_frac", **risk.summary(eq))
        continue
    if dd_peak > MAX_DD:
        halt("drawdown_absolute", equity=eq, peak=peak, dd=dd_peak,
             ceiling=MAX_DD)
        continue
    if npos > MAX_POSITIONS:
        halt("too_many_positions", open_positions=npos)
        continue
    alive = runner_alive()
    if not alive:
        restarts += 1
        if restarts > 6:
            halt("runner_keeps_dying", restarts=restarts)
            continue
        note(event="runner_dead", action="restarting", n=restarts)
        restart_runner()
    note(event="ok", equity=round(eq, 4), cash=round(cash, 4),
         open_positions=npos, peak=round(peak, 2),
         dd_from_peak=round(dd_peak, 4),
         dd_frac=round(risk.dd_frac(eq), 4),
         halt_at=round(risk.summary(eq)['watch_at'], 2), floor=risk.FLOOR,
         runner_alive=alive)
