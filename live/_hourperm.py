import numpy as np, pandas as pd
RNG=np.random.default_rng(20260905)
D=pd.read_parquet("regime_data.parquet"); days=sorted(D.day.unique())
def walkfwd(S):
    out=[]
    for i in range(4,len(days)):
        past=S[S.day.isin(days[:i])]; now=S[S.day==days[i]]
        if len(now)<10: continue
        hm=past.groupby("hour").edge.mean()
        good=set(hm[hm>hm.median()].index)
        sel=now[now.hour.isin(good)]
        if len(sel)<5: continue
        out.append((now.edge.mean()*100, sel.edge.mean()*100))
    if not out: return None
    A=np.array([o[0] for o in out]); B=np.array([o[1] for o in out])
    return B.mean()-A.mean(), (B>A).sum(), len(out)
real=walkfwd(D)
print(f"REAL walk-forward gain: {real[0]:+.2f}c   better on {real[1]}/{real[2]} days")
print("\nPERMUTATION: shuffle the HOUR label within each day, redo the whole")
print("walk-forward. This destroys any real hour structure but keeps the")
print("day-to-day premium variation and the reuse-of-past-days dependence.\n")
gains=[];wins=[]
for _ in range(1500):
    S=D.copy()
    S["hour"]=S.groupby("day").hour.transform(lambda s: RNG.permutation(s.values))
    r=walkfwd(S)
    if r: gains.append(r[0]); wins.append(r[1])
gains=np.array(gains); wins=np.array(wins)
print(f"  null gain: mean {gains.mean():+.2f}c  sd {gains.std():.2f}  "
      f"95% [{np.percentile(gains,2.5):+.2f},{np.percentile(gains,97.5):+.2f}]")
print(f"  p(null gain >= real) = {(gains>=real[0]).mean():.4f}")
print(f"  null 'better on k/10': mean {wins.mean():.2f}   "
      f"p(null wins >= {real[1]}) = {(wins>=real[1]).mean():.4f}")
print()
if (gains>=real[0]).mean()<0.05:
    print("  -> the hour effect survives. Gating on hour is worth considering.")
else:
    print("  -> the hour effect does NOT survive. The 9/10 was an artifact of")
    print("     overlapping training windows, exactly as suspected.")
