"""LSM status report - the numbers that decide whether this is real.

Reads only the runner's own jsonl log. No dataframes, no network, negligible
RAM. Safe to run repeatedly.

THE NUMBER THAT MATTERS is net P&L per SUBMITTED order, not per fill. An
unfilled order is a zero, not an absence - counting only fills is exactly the
selection error that makes maker strategies look better than they are.
"""
import glob
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for fp in sorted(glob.glob(os.path.join(HERE, "lsm_2*.jsonl"))):
    for line in open(fp):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass

posts = [r for r in rows if r.get("ev") == "post" and r.get("status") in (200, 201)]
terms = [r for r in rows if r.get("ev") == "terminal"]
opps = [r for r in rows if r.get("ev") == "opportunity"]
halts = [r for r in rows if r.get("ev") in ("halted", "stop_loss", "cap_orders")]

print(f"opportunities seen     {len(opps)}")
print(f"orders submitted       {len(posts)}")
print(f"orders resolved        {len(terms)}")
if halts:
    print(f"!! HALT EVENTS: {[h.get('ev') for h in halts]}")

if not terms:
    print("\nno resolved orders yet - nothing to measure")
    raise SystemExit

tgt = sum(t.get("target") or 0 for t in terms)
fil = sum(t.get("FILLED") or 0 for t in terms)
print(f"\ncontracts submitted    {tgt}")
print(f"contracts FILLED       {fil}")
print(f"F_Q (quantity fill)    {fil/tgt:.4f}" if tgt else "")
nfull = sum(1 for t in terms if (t.get("FILLED") or 0) >= (t.get("target") or 1))
nnone = sum(1 for t in terms if (t.get("FILLED") or 0) == 0)
print(f"orders fully filled    {nfull}/{len(terms)}")
print(f"orders never filled    {nnone}/{len(terms)}")

print("\n=== FILL RATE BY QUEUE AHEAD (the question the research could not answer) ===")
buckets = [(0, 100), (100, 400), (400, 1000), (1000, 3000), (3000, 10**9)]
lab = ["<100", "100-400", "400-1k", "1k-3k", ">3k"]
by = defaultdict(lambda: [0, 0, 0])       # orders, submitted_ct, filled_ct
for t in terms:
    q = t.get("queue_ahead")
    if q is None:
        continue
    for i, (lo, hi) in enumerate(buckets):
        if lo <= q < hi:
            b = by[i]
            b[0] += 1
            b[1] += t.get("target") or 0
            b[2] += t.get("FILLED") or 0
            break
print(f"{'queue ahead':>12} {'orders':>7} {'submitted':>10} {'filled':>8} {'fill rate':>10}")
for i in sorted(by):
    o, s, f = by[i]
    print(f"{lab[i]:>12} {o:>7} {s:>10} {f:>8} {f/s if s else 0:>10.4f}")

print("\n=== ECONOMICS (settled markets only) ===")
setl = {}
for r in rows:
    if r.get("ev") == "settled":
        setl[r.get("ticker")] = r
if not setl:
    print("  no settlement records logged by the runner yet;")
    print("  check the exchange directly for realised P&L")
print("\nNOTE: net P&L per SUBMITTED order is the decision metric.")
print("      Unfilled orders count as zero, never as missing.")
