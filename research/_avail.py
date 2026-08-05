import numpy as np, pandas as pd, itertools
RNG=np.random.default_rng(20260925); QTY=15
Y=pd.read_parquet("yesno.parquet")
q=Y[(Y.px>=.65)&(Y.px<.85)&(Y.minute>=8)&(Y.minute<=14)&(Y.vol>=2000)]
C=(q.sort_values("minute",ascending=False).groupby(["wkey","coin"],as_index=False).first()).copy()
g=C.groupby("wkey").agg(n=("won","size"),nyes=("maker_yes","sum")).reset_index()
g2=g[g.n>=2]
print("=== IS DIRECTION-MIXING EVEN AVAILABLE? ===")
print(f"  windows with 2+ candidates: {len(g2)}")
both=((g2.nyes>0)&(g2.nyes<g2.n)).sum()
print(f"  windows offering BOTH directions: {both} ({both/len(g2):.1%})")
print(f"  windows where ALL candidates point the SAME way: {len(g2)-both} ({1-both/len(g2):.1%})")
print()
print("  -> when the crypto complex moves together, every coin's FAVOURITE")
print("     points the same way. There is usually no opposite side to pick.")
print()
print("=== SO: DOES THE CHOICE OF COIN PAIR MATTER? (correlation structure) ===")
print("  If some pairs co-fail less than others, prefer those - also free.\n")
piv=C.pivot_table(index="wkey",columns="coin",values="won")
coins=[c for c in piv.columns if piv[c].notna().sum()>=80]
print(f"{'pair':>12} {'both traded':>12} {'P(both lose)':>13} {'indep':>8} {'ratio':>7}")
rows=[]
for a,b in itertools.combinations(coins,2):
    s=piv[[a,b]].dropna()
    if len(s)<40: continue
    pl=((s[a]==0)&(s[b]==0)).mean()
    ind=(1-s[a].mean())*(1-s[b].mean())
    rows.append((f"{a}-{b}",len(s),pl,ind,pl/ind if ind>0 else np.nan))
for r in sorted(rows,key=lambda x:-x[4]):
    print(f"{r[0]:>12} {r[1]:>12} {r[2]:>13.4f} {r[3]:>8.4f} {r[4]:>7.2f}")
print()
rs=[r[4] for r in rows]
print(f"  ratio spread: {min(rs):.2f} to {max(rs):.2f}")
print(f"  If all pairs cluster near the same ratio, pair choice buys nothing.")
print()
print("=== WHY: how correlated are the coins' OUTCOMES? ===")
cm=piv[coins].corr()
print(cm.round(3).to_string())
print()
iu=np.triu_indices(len(coins),1)
print(f"  mean pairwise outcome correlation: {cm.values[iu].mean():.3f}")
print(f"  range: {cm.values[iu].min():.3f} to {cm.values[iu].max():.3f}")
