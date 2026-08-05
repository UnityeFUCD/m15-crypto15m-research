import glob, json
import numpy as np
ORD={}
for fp in sorted(glob.glob("lsm_*.jsonl"))+sorted(glob.glob("lsm_*.log")):
    try:
        for ln in open(fp,errors="ignore"):
            ln=ln.strip()
            if not ln.startswith("{"): continue
            try: d=json.loads(ln)
            except Exception: continue
            t=d.get("ticker")
            if not t: continue
            if d.get("ev")=="opportunity":
                ORD.setdefault(t,{})["left"]=d.get("min_left"); ORD[t]["px"]=d.get("post_at")
                ORD[t]["qa"]=d.get("queue_ahead")
            if d.get("ev")=="terminal":
                ORD.setdefault(t,{})["target"]=d.get("target"); ORD[t]["filled"]=d.get("FILLED")
    except Exception: pass
R=[(v["left"],float(v["filled"])/max(float(v["target"]),1),v.get("qa") or 0,v.get("px") or 0)
   for v in ORD.values() if v.get("left") is not None and v.get("target") and v.get("filled") is not None]
print(f"orders with entry time AND fill outcome: {len(R)}")
if len(R)<20: raise SystemExit("too few")
left=np.array([r[0] for r in R]); fq=np.array([r[1] for r in R])
print()
print("=== FILL RATE BY MINUTES REMAINING AT ENTRY ===")
print(f"{'min left':>12} {'orders':>7} {'F_Q':>8} {'fully filled':>13} {'never filled':>13}")
for lo,hi,lab in ((8,10,'8-10'),(10,12,'10-12'),(12,13.5,'12-13.5'),(13.5,15,'13.5-14')):
    m=(left>=lo)&(left<hi)
    if m.sum()<4: continue
    print(f"{lab:>12} {m.sum():>7} {fq[m].mean():>8.4f} {(fq[m]>=1).mean():>13.3f} {(fq[m]==0).mean():>13.3f}")
print()
early=left>=12; late=left<12
if early.sum()>=5 and late.sum()>=5:
    rng=np.random.default_rng(7)
    bs=np.sort(np.array([fq[late][rng.integers(0,late.sum(),late.sum())].mean()
                        -fq[early][rng.integers(0,early.sum(),early.sum())].mean() for _ in range(8000)]))
    print(f"  entered LATE (<12 min): F_Q {fq[late].mean():.4f}  n {late.sum()}")
    print(f"  entered EARLY (>=12)  : F_Q {fq[early].mean():.4f}  n {early.sum()}")
    print(f"  late minus early: {(fq[late].mean()-fq[early].mean()):+.4f}  "
          f"95% CI [{bs[200]:+.4f},{bs[7799]:+.4f}]  p {2*min((bs<=0).mean(),(bs>=0).mean()):.4f}")
    print()
    print("  If entering later fills MORE, the price has had less time to walk")
    print("  away - and my earlier timing test, which modelled no fill selection")
    print("  at all, understated the case for later entry.")
