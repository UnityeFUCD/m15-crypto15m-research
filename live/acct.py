"""Standalone account snapshot.

Deliberately does NOT import lsm_runner: that module is a bare module-level
`while True` trading loop, so importing it starts a SECOND runner sharing the
same date-stamped state file.
"""
import base64
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

CRED = Path("C:/Users/fycin/kalshi_weather_trading")
BASE = "https://api.elections.kalshi.com/trade-api/v2"
cfg = {}
for line in (CRED / ".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        cfg[k.strip()] = v.strip().strip('"').strip("'")
KEY = cfg.get("KALSHI_API_KEY_ID")
PRIV = serialization.load_pem_private_key(
    (CRED / "kalshi_private_key.pem").read_bytes(), password=None)
S = requests.Session()


def api(path, params=None):
    ts = str(int(time.time() * 1000))
    msg = f"{ts}GET/trade-api/v2{path}".encode()
    sig = base64.b64encode(PRIV.sign(
        msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                         salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256())).decode()
    r = S.get(BASE + path, params=params or {}, timeout=15, headers={
        "KALSHI-ACCESS-KEY": KEY, "KALSHI-ACCESS-SIGNATURE": sig,
        "KALSHI-ACCESS-TIMESTAMP": ts})
    return r.status_code, (r.json() if r.status_code == 200 else None)


c, j = api("/portfolio/balance")
bal = float(j.get("balance_dollars") or 0) if j else 0.0
c2, j2 = api("/portfolio/positions", {"limit": 200})
cost, n, detail = 0.0, 0, []
for p in ((j2 or {}).get("market_positions") or []):
    pos = float(p.get("position_fp") or 0)
    if pos != 0:
        tt = abs(float(p.get("total_traded_dollars") or 0))
        n += 1
        cost += tt
        detail.append((p.get("ticker"), pos, tt))
print(f"utc        {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z")
print(f"cash       ${bal:.2f}")
print(f"open pos   {n}   cost basis ${cost:.2f}")
print(f"EQUITY     ${bal + cost:.2f}")
for t, pos, tt in detail:
    print(f"    {t:<34} {pos:>+7.0f}  ${tt:.2f}")

# settlements: `revenue` is an INTEGER IN CENTS (3000 = $30.00); cost is split
# across yes_/no_total_cost_dollars. There is no revenue_dollars field.
s, cur = [], None
while True:
    p = {"limit": 200}
    if cur:
        p["cursor"] = cur
    c3, j3 = api("/portfolio/settlements", p)
    if c3 != 200 or not j3:
        break
    b = j3.get("settlements") or []
    s += b
    cur = j3.get("cursor")
    if not cur or not b or len(s) >= 2000:
        break
mine = [x for x in s if x.get("ticker", "").startswith(
    ("KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M", "KXDOGE15M"))]
if mine:
    rev = sum(float(x.get("revenue") or 0) / 100.0 for x in mine)
    cst = sum(float(x.get("yes_total_cost_dollars") or 0) +
              float(x.get("no_total_cost_dollars") or 0) for x in mine)
    fee = sum(float(x.get("fee_cost") or 0) for x in mine)
    won = sum(1 for x in mine if float(x.get("revenue") or 0) > 0)
    ncon = sum(abs(float(x.get("yes_count_fp") or 0)) +
               abs(float(x.get("no_count_fp") or 0)) for x in mine)
    pnl = rev - cst - fee
    print(f"\nsettled    {len(mine)}   won {won} ({won/len(mine):.4f})   "
          f"contracts {ncon:,.0f}   fees ${fee:.2f}")
    print(f"realised   ${pnl:+.2f}   "
          f"per-contract {pnl / max(ncon, 1) * 100:+.2f}c")
