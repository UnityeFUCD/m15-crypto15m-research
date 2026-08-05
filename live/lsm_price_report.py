"""Per-price-bucket economics, from the exchange's own settlement records.

WHY THIS EXISTS
  The price we pay IS our breakeven. At a 67c fill we need 67% to break even;
  at 83c we need 83%. Our overall win rate is ~82%, so a book of 83c fills is
  roughly break-even while a book of 67c fills earns ~10c/contract. Same
  strategy, same band, completely different economics - so the average hides
  the thing that matters.

  Research across 6.2M historical trades put the sweet spot at 70-80c and said
  the edge roughly HALVES in 80-85c (win rate barely rises, price paid jumps).
  This checks that live, bucket by bucket, so a drift toward expensive fills is
  visible before it costs anything - which matters more at qty 30, where each
  83c fill commits ~$25 for half the edge.

Read-only. One API call. No dataframes.
"""
import base64
import time
from collections import defaultdict
from pathlib import Path

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

CRED = Path("C:/Users/fycin/kalshi_weather_trading")
BASE = "https://api.elections.kalshi.com/trade-api/v2"
PREFIX = "/trade-api/v2"
SINCE = "2026-08-04T13:0"

# research prediction, 6.2M trades: (win rate, edge in cents)
PRED = {(0.65, 0.70): (0.720, 5.1), (0.70, 0.75): (0.803, 8.3),
        (0.75, 0.80): (0.879, 10.8), (0.80, 0.85): (0.880, 5.9),
        (0.85, 1.00): (0.922, 5.1)}

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
    r = _S.get(BASE + path, params=params, headers=h, timeout=25)
    return r.status_code, r.json()


c, j = api("/portfolio/settlements", {"limit": 200})
rows = []
for x in (j.get("settlements") or []):
    if (x.get("settled_time") or "") < SINCE:
        continue
    rev = float(x.get("revenue") or 0) / 100.0
    cost = (float(x.get("yes_total_cost_dollars") or 0)
            + float(x.get("no_total_cost_dollars") or 0))
    n = abs(float(x.get("yes_count_fp") or 0)) + abs(float(x.get("no_count_fp") or 0))
    if n <= 0:
        continue
    rows.append((cost / n, rev - cost, n, rev > cost))

if not rows:
    raise SystemExit("no settled positions yet")

tot_ct = sum(r[2] for r in rows)
tot_pnl = sum(r[1] for r in rows)
print(f"settled positions {len(rows)}   contracts {tot_ct:.0f}   "
      f"net ${tot_pnl:+.2f}   {tot_pnl/tot_ct*100:+.2f}c/contract")
print(f"\n{'price band':>12} {'pos':>5} {'contracts':>10} {'win':>7} "
      f"{'predicted':>10} {'net c/ct':>9} {'predicted':>10} {'verdict':>12}")
for (lo, hi), (pw, pe) in PRED.items():
    k = [r for r in rows if lo <= r[0] < hi]
    if not k:
        print(f"{lo*100:>5.0f}-{hi*100:<6.0f} {'-':>5} {'-':>10} {'-':>7} "
              f"{pw:>10.3f} {'-':>9} {pe:>+9.1f}c {'no fills':>12}")
        continue
    ct = sum(r[2] for r in k)
    pnl = sum(r[1] for r in k)
    wr = sum(r[3] for r in k) / len(k)
    per = pnl / ct * 100
    if len(k) < 5:
        v = "too few"
    elif per > pe:
        v = "beating"
    elif per > 0:
        v = "below pred"
    else:
        v = "LOSING"
    print(f"{lo*100:>5.0f}-{hi*100:<6.0f} {len(k):>5} {ct:>10.0f} {wr:>7.3f} "
          f"{pw:>10.3f} {per:>+9.2f} {pe:>+9.1f}c {v:>12}")

exp = [r for r in rows if r[0] >= 0.80]
cheap = [r for r in rows if r[0] < 0.75]
print(f"\nCOMPOSITION  fills >=80c: {len(exp)}/{len(rows)} "
      f"({len(exp)/len(rows):.1%})   fills <75c: {len(cheap)}/{len(rows)} "
      f"({len(cheap)/len(rows):.1%})")
if exp:
    ct = sum(r[2] for r in exp)
    print(f"  expensive fills earn {sum(r[1] for r in exp)/ct*100:+.2f}c/contract "
          f"and tie up ${sum(r[0]*r[2] for r in exp):.2f}")
if cheap:
    ct = sum(r[2] for r in cheap)
    print(f"  cheap     fills earn {sum(r[1] for r in cheap)/ct*100:+.2f}c/contract "
          f"and tie up ${sum(r[0]*r[2] for r in cheap):.2f}")
print("\nWATCH FOR: expensive share climbing, or 80-85c turning negative.")
print("At qty 30 an 83c fill commits ~$25 for roughly half the edge of a 75c fill.")
