"""STAGE 2 - independently reconstruct every trade, in exact integer arithmetic.

Nothing here reuses any earlier profit column. Prices are converted to integer
MICROS (1e-6 dollars) and counts to integer HUNDREDTHS of a contract, so every
identity is checked exactly rather than to a float tolerance.

SEMANTICS (Kalshi V2 trade record)
  taker_side       the OUTCOME side the aggressor bought ("yes" or "no")
  yes_price_dollars / no_price_dollars   the two legs; must sum to $1
  The MAKER necessarily holds the opposite outcome side.

  taker buys YES at y  ->  maker buys NO  at n = 1-y
  taker buys NO  at n  ->  maker buys YES at y = 1-n

SETTLEMENT
  YES settles -> YES contract pays 1, NO pays 0, and vice versa.

FEES
  maker fee is ZERO (verified on 25 own maker fills)
  taker fee = ceil(0.07 * C * P * (1-P) * 1e4) / 1e4  with P the taker's price,
  rounded UP once per fill block to the centicent (verified on 40 own fills).

IDENTITIES REQUIRED (exact, integer)
  maker_gross + taker_gross == 0
  maker_net + taker_net == -(maker_fee + taker_fee)
Failure of either stops the run.
"""
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("C:/Users/fycin/m15-claude-independent")
OUT = ROOT / "research/final_minute_favorite_maker_validation"
TRADES = ROOT / "data/sare/trades"
MICRO = 1_000_000            # $1 in micros
CENTICENT = 100              # $0.0001 in micros


def to_micros(s):
    """'0.9990' -> 999000 exactly, via integer string parsing (no binary float)."""
    s = str(s).strip()
    neg = s.startswith("-")
    if neg:
        s = s[1:]
    if "." in s:
        whole, frac = s.split(".", 1)
    else:
        whole, frac = s, ""
    frac = (frac + "000000")[:6]
    v = int(whole or "0") * MICRO + int(frac or "0")
    return -v if neg else v


def to_hundredths(s):
    """'250.00' -> 25000 ; '1.62' -> 162 (hundredths of a contract)."""
    s = str(s).strip()
    if "." in s:
        whole, frac = s.split(".", 1)
    else:
        whole, frac = s, ""
    frac = (frac + "00")[:2]
    return int(whole or "0") * 100 + int(frac or "0")


def taker_fee_micros(price_micros, count_hundredths):
    """ceil(0.07*C*P*(1-P)) to the centicent, C in whole-contract units."""
    p = price_micros / MICRO
    c = count_hundredths / 100.0
    raw = 0.07 * c * p * (1.0 - p)                    # dollars
    return int(math.ceil(raw * 10_000 - 1e-9)) * CENTICENT


def reconstruct(t):
    """Return a dict of exact integer quantities, or ('SKIP', reason)."""
    ypx = to_micros(t["yes_price_dollars"])
    npx = to_micros(t["no_price_dollars"])
    if ypx + npx != MICRO:
        return None, f"yes+no != $1 ({ypx}+{npx})"
    if not (0 < ypx < MICRO):
        return None, f"yes price out of range ({ypx})"
    cnt = to_hundredths(t["count_fp"])
    if cnt <= 0:
        return None, f"non-positive count ({cnt})"
    ts = t.get("taker_side")
    if ts not in ("yes", "no"):
        return None, f"bad taker_side ({ts!r})"
    taker_px = ypx if ts == "yes" else npx
    maker_side = "no" if ts == "yes" else "yes"
    maker_px = MICRO - taker_px
    return dict(yes_px=ypx, no_px=npx, cnt=cnt, taker_side=ts,
                maker_side=maker_side, taker_px=taker_px, maker_px=maker_px,
                taker_fee=taker_fee_micros(taker_px, cnt), maker_fee=0), None


def pnl(rec, yes_settles):
    """Gross and net P&L in micros for the whole fill block."""
    c = rec["cnt"]                                    # hundredths of a contract
    taker_wins = (rec["taker_side"] == "yes") == bool(yes_settles)
    tg = ((MICRO - rec["taker_px"]) if taker_wins else -rec["taker_px"]) * c // 100
    mg = ((MICRO - rec["maker_px"]) if not taker_wins else -rec["maker_px"]) * c // 100
    return tg, mg, tg - rec["taker_fee"], mg - rec["maker_fee"]


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    files = sorted(TRADES.glob("*.json"))
    if limit:
        files = files[:limit]
    n_tr = 0
    skipped = Counter()
    id_fail_gross = []
    id_fail_net = []
    dup = Counter()
    px_hist = Counter()
    print(f"stage 2: reconstructing {len(files)} files", flush=True)
    for i, f in enumerate(files):
        for t in json.loads(f.read_text()):
            n_tr += 1
            dup[t.get("trade_id")] += 1
            rec, why = reconstruct(t)
            if rec is None:
                skipped[why.split("(")[0].strip()] += 1
                continue
            px_hist[rec["yes_px"] // 10000] += 1
            # the identity must hold for BOTH settlement outcomes
            for ys in (0, 1):
                tg, mg, tn, mn = pnl(rec, ys)
                if tg + mg != 0:
                    if len(id_fail_gross) < 5:
                        id_fail_gross.append((t.get("trade_id"), ys, tg, mg))
                if tn + mn != -(rec["taker_fee"] + rec["maker_fee"]):
                    if len(id_fail_net) < 5:
                        id_fail_net.append((t.get("trade_id"), ys, tn, mn,
                                            rec["taker_fee"]))
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(files)}  {n_tr:,} trades", flush=True)

    dups = sum(1 for k, v in dup.items() if v > 1)
    print("\n=== STAGE 2 RESULT ===")
    print(f"  files                {len(files):,}")
    print(f"  trades read          {n_tr:,}")
    print(f"  duplicate trade_ids  {dups:,}")
    print(f"  skipped              {sum(skipped.values()):,}  {dict(skipped)}")
    print(f"  GROSS identity failures  {len(id_fail_gross)}")
    for x in id_fail_gross:
        print("     ", x)
    print(f"  NET   identity failures  {len(id_fail_net)}")
    for x in id_fail_net:
        print("     ", x)
    res = {"files": len(files), "trades": n_tr, "duplicate_trade_ids": dups,
           "skipped": dict(skipped),
           "gross_identity_failures": len(id_fail_gross),
           "net_identity_failures": len(id_fail_net),
           "yes_price_cent_histogram": {str(k): v for k, v in sorted(px_hist.items())}}
    (OUT / "trade_accounting_audit.json").write_text(json.dumps(res, indent=1))
    print(f"\n  wrote {OUT/'trade_accounting_audit.json'}")
    if id_fail_gross or id_fail_net:
        sys.exit("ACCOUNTING IDENTITY FAILED - stopping per protocol")
    print("  identities hold exactly on every trade under both outcomes")
