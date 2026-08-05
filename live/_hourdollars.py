import numpy as np, pandas as pd
D=pd.read_parquet("regime_data.parquet"); days=sorted(D.day.unique()); QTY=30
rows=[]
for i in range(4,len(days)):
    past=D[D.day.isin(days[:i])]; now=D[D.day==days[i]]
    if len(now)<10: continue
    hm=past.groupby("hour").edge.mean()
    good=set(hm[hm>hm.median()].index)
    sel=now[now.hour.isin(good)]
    if len(sel)<5: continue
    rows.append((days[i],len(now),now.edge.sum()*QTY,len(sel),sel.edge.sum()*QTY))
print(f"{'day':>12} {'n all':>6} {'$ all':>10} {'n good':>7} {'$ good':>10} {'diff':>9}")
for d,na,da,ns,ds in rows:
    print(f"  {d:>10} {na:>6} {da:>+10.2f} {ns:>7} {ds:>+10.2f} {ds-da:>+9.2f}")
A=np.array([r[2] for r in rows]); B=np.array([r[4] for r in rows])
na=np.array([r[1] for r in rows]); ns=np.array([r[3] for r in rows])
print(f"\n  ALL hours  : {na.mean():.0f} entries/day  ${A.mean():+.2f}/day")
print(f"  GOOD hours : {ns.mean():.0f} entries/day  ${B.mean():+.2f}/day")
print(f"  difference : {ns.mean()-na.mean():+.0f} entries/day  ${B.mean()-A.mean():+.2f}/day")
print(f"  good-hours better in DOLLARS on {(B>A).sum()}/{len(rows)} days")
print()
print("  NOTE: this counts every qualifying market in the replay, not the ~3")
print("  per window the live runner actually takes. The RATIO is what matters.")
print(f"  dollar ratio good/all = {B.mean()/A.mean():.3f}")
