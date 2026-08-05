import numpy as np, pandas as pd
RNG=np.random.default_rng(20260904)
D=pd.read_parquet("regime_data.parquet")
days=sorted(D.day.unique())
agg=D.groupby("day").agg(edge=("edge","mean"),n=("edge","size"))
print("=== IS THE n-vs-PREMIUM CORRELATION AN ARTIFACT OF PARTIAL DAYS? ===")
print(f"{'day':>12} {'n':>5} {'premium':>9}")
for d,r in agg.iterrows(): print(f"  {d:>10} {int(r['n']):>5} {r['edge']*100:>+8.2f}c")
full=agg[(agg.n>=200)]
print(f"\n  all days      : corr(n, premium) = {np.corrcoef(agg.n,agg.edge)[0,1]:+.4f}  (n={len(agg)})")
if len(full)>=6:
    print(f"  FULL days only: corr(n, premium) = {np.corrcoef(full.n,full.edge)[0,1]:+.4f}  (n={len(full)})")
a2=agg.iloc[:-1]
print(f"  excluding last (partial) day: corr = {np.corrcoef(a2.n,a2.edge)[0,1]:+.4f}  (n={len(a2)})")
print("\n=== IS THE HOUR PATTERN REAL? proper split-half with random halves ===")
H=D.groupby(["day","hour"]).edge.mean().reset_index()
cs=[]
for _ in range(2000):
    perm=RNG.permutation(days); h=len(days)//2
    A=D[D.day.isin(perm[:h])].groupby("hour").edge.mean()
    B=D[D.day.isin(perm[h:])].groupby("hour").edge.mean()
    J=pd.concat([A,B],axis=1).dropna()
    if len(J)>=12: cs.append(J.iloc[:,0].corr(J.iloc[:,1]))
cs=np.array(cs)
print(f"  mean split-half corr over 2000 random splits: {cs.mean():+.4f}")
print(f"  95% range [{np.percentile(cs,2.5):+.4f}, {np.percentile(cs,97.5):+.4f}]")
null=[]
for _ in range(2000):
    S=D.copy(); S["hour"]=RNG.permutation(S.hour.values)
    perm=RNG.permutation(days); h=len(days)//2
    A=S[S.day.isin(perm[:h])].groupby("hour").edge.mean()
    B=S[S.day.isin(perm[h:])].groupby("hour").edge.mean()
    J=pd.concat([A,B],axis=1).dropna()
    if len(J)>=12: null.append(J.iloc[:,0].corr(J.iloc[:,1]))
null=np.array(null)
print(f"  NULL (hours shuffled): mean {null.mean():+.4f}  95% [{np.percentile(null,2.5):+.4f},{np.percentile(null,97.5):+.4f}]")
print(f"  p = {(null>=cs.mean()).mean():.4f}")
print("\n=== WOULD TRADING ONLY THE GOOD HOURS HAVE PAID? (walk-forward) ===")
res=[]
for i in range(4,len(days)):
    past=D[D.day.isin(days[:i])]; now=D[D.day==days[i]]
    if len(now)<10: continue
    hm=past.groupby("hour").edge.mean()
    good=set(hm[hm>hm.median()].index)
    sel=now[now.hour.isin(good)]
    if len(sel)<5: continue
    res.append((days[i],now.edge.mean()*100,sel.edge.mean()*100,len(now),len(sel)))
if res:
    print(f"{'day':>12} {'all':>8} {'good hrs':>9} {'n all':>6} {'n sel':>6}")
    for d,a,b,na,ns in res: print(f"  {d:>10} {a:>+8.2f} {b:>+9.2f} {na:>6} {ns:>6}")
    A=np.array([r[1] for r in res]); B=np.array([r[2] for r in res])
    print(f"\n  mean all {A.mean():+.2f}c   good-hours {B.mean():+.2f}c   diff {B.mean()-A.mean():+.2f}c")
    print(f"  good-hours better on {(B>A).sum()}/{len(res)} out-of-sample days")
