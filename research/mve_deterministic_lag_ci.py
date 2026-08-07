from __future__ import annotations
import json, math, time
from pathlib import Path
from datetime import datetime, timezone
import requests

BASE='https://external-api.kalshi.com/trade-api/v2'
s=requests.Session()

def get(path, params=None, tries=8):
    for k in range(tries):
        r=s.get(BASE+path, params=params, timeout=30)
        if r.status_code==200: return r.json()
        if r.status_code==429:
            time.sleep(min(5, .5*(2**k))); continue
        print('ERR',r.status_code,r.url,r.text[:300],flush=True); return None
    return None

def ts(x):
    if not x: return None
    try: return datetime.fromisoformat(x.replace('Z','+00:00')).timestamp()
    except Exception: return None

def fnum(x):
    try: return float(x)
    except Exception: return float('nan')

def batches(xs,n=60):
    xs=list(xs)
    for i in range(0,len(xs),n): yield xs[i:i+n]

def fetch_markets(tickers):
    out={}
    for i,b in enumerate(batches(sorted(tickers),50),1):
        body=get('/markets', {'tickers':','.join(b),'limit':1000})
        if body:
            for m in body.get('markets') or []: out[m['ticker']]=m
        if i%20==0: print('COMPONENT batches',i,'got',len(out),'of',len(tickers),flush=True)
        time.sleep(.08)
    return out

def fetch_trades(ticker, lo, hi):
    rows=[]; cur=None
    for _ in range(10):
        p={'ticker':ticker,'limit':1000,'min_ts':max(0,int(lo)-2),'max_ts':int(hi)+2}
        if cur: p['cursor']=cur
        b=get('/markets/trades',p)
        if not b: break
        a=b.get('trades') or []; rows.extend(a); cur=b.get('cursor')
        if not cur or not a: break
        time.sleep(.05)
    return rows

def evaluate_combo(m, comp):
    legs=m.get('mve_selected_legs') or []
    if not legs: return None
    states=[]
    for leg in legs:
        c=comp.get(leg.get('market_ticker'))
        if not c: return None
        res=(c.get('result') or '').lower(); side=(leg.get('side') or '').lower()
        st=ts(c.get('settlement_ts'))
        if res not in ('yes','no') or st is None:
            states.append(('unknown',None,c,side,res)); continue
        ok=(res==side)
        states.append(('win' if ok else 'lose',st,c,side,res))
    losses=[z for z in states if z[0]=='lose']
    if losses:
        ct=min(z[1] for z in losses)
        return {'certain':'no','certainty_ts':ct,'states':states}
    if all(z[0]=='win' for z in states):
        ct=max(z[1] for z in states)
        return {'certain':'yes','certainty_ts':ct,'states':states}
    return {'certain':None,'certainty_ts':None,'states':states}

def main():
    D=json.loads(Path('mve_public_discovery.json').read_text())
    om=D['open']; sm=D['settled']
    # prioritize combos with actual book/volume, plus all recently traded settled combos
    oq=[m for m in om if max(fnum(m.get('yes_bid_dollars')), fnum(m.get('yes_ask_dollars')), fnum(m.get('volume_fp'))) > 0]
    sq=[m for m in sm if fnum(m.get('volume_fp'))>0]
    tick=set()
    for m in oq+sq:
        for l in m.get('mve_selected_legs') or []: tick.add(l.get('market_ticker'))
    tick.discard(None)
    print('TARGET combos open',len(oq),'settled_traded',len(sq),'unique components',len(tick),flush=True)
    comp=fetch_markets(tick)
    print('COMPONENT COVERAGE',len(comp),'/',len(tick),flush=True)

    live=[]
    for m in oq:
        e=evaluate_combo(m,comp)
        if not e or not e['certain']: continue
        if e['certain']=='no':
            bid=fnum(m.get('yes_bid_dollars')); size=fnum(m.get('yes_bid_size_fp'))
            edge=bid; cost=1-bid
        else:
            ask=fnum(m.get('yes_ask_dollars')); size=fnum(m.get('yes_ask_size_fp'))
            edge=(1-ask) if ask>0 else 0; cost=ask
        if edge>0 and size>0:
            live.append(dict(ticker=m['ticker'],certain=e['certain'],edge=edge,size=size,cost=cost,
                             volume=fnum(m.get('volume_fp')),created_time=m.get('created_time'),close_time=m.get('close_time'),
                             certainty_ts=e['certainty_ts'],collection=m.get('mve_collection_ticker'),legs=m.get('mve_selected_legs')))
    live.sort(key=lambda x:(x['edge']*x['size'],x['edge']),reverse=True)
    print('\nLIVE DETERMINISTIC EXECUTABLE GROSS',len(live))
    print('LIVE_TOTAL_GROSS_TOPLEVEL',sum(x['edge']*x['size'] for x in live))
    for x in live[:30]: print('LIVE',json.dumps(x,default=str)[:5000])

    candidates=[]
    for m in sq:
        e=evaluate_combo(m,comp)
        if not e or not e['certain']: continue
        ct=e['certainty_ts']; close=ts(m.get('close_time')); created=ts(m.get('created_time')); settle=ts(m.get('settlement_ts'))
        if ct is None: continue
        lag_close=(close-ct) if close is not None else None
        lag_settle=(settle-ct) if settle is not None else None
        if close is not None and ct < close:
            candidates.append((m,e,lag_close,lag_settle,created))
    print('\nHIST CERTAIN BEFORE CLOSE',len(candidates),'of traded',len(sq))
    print('LAG_CLOSE seconds quantiles',end=' ')
    ls=sorted(x[2] for x in candidates if x[2] is not None)
    if ls:
        print({q:ls[min(len(ls)-1,int(q*(len(ls)-1)))] for q in [0,.1,.25,.5,.75,.9,.99,1]})
    else: print({})

    # Fetch trades only for the strongest time-lag candidates; cap protects rate limits.
    candidates.sort(key=lambda x:x[2],reverse=True)
    post=[]; checked=0
    for m,e,lag_close,lag_settle,created in candidates[:1500]:
        tr=fetch_trades(m['ticker'],e['certainty_ts'],ts(m.get('close_time')) or e['certainty_ts']+max(1,lag_close))
        checked+=1
        for z in tr:
            tt=ts(z.get('created_time'))
            if tt is None or tt < e['certainty_ts']-1e-6: continue
            # If taker chose the already-certain outcome, this is incontrovertible executable taker profit ex fee.
            side=(z.get('taker_outcome_side') or '').lower()
            if side != e['certain']: continue
            px=fnum(z.get('yes_price_dollars')) if side=='yes' else fnum(z.get('no_price_dollars'))
            qty=fnum(z.get('count_fp') or z.get('count'))
            if not (0<px<1 and qty>0): continue
            edge=1-px
            post.append(dict(ticker=m['ticker'],certain=e['certain'],certainty_ts=e['certainty_ts'],trade_ts=tt,
                             delay_s=tt-e['certainty_ts'],px=px,qty=qty,gross_edge=edge,gross_dollars=edge*qty,
                             trade_id=z.get('trade_id'),taker_side=side,collection=m.get('mve_collection_ticker'),
                             legs=m.get('mve_selected_legs')))
        if checked%100==0: print('HIST trade checks',checked,'post certain taker trades',len(post),flush=True)
        time.sleep(.04)
    post.sort(key=lambda x:x['gross_dollars'],reverse=True)
    print('\nPOST-CERTAINTY SAME-SIDE TAKER TRADES',len(post),'markets',len(set(x['ticker'] for x in post)), 'checked',checked)
    print('GROSS_DOLLARS',sum(x['gross_dollars'] for x in post),'CONTRACTS',sum(x['qty'] for x in post))
    for x in post[:50]: print('POST',json.dumps(x,default=str)[:5000])

    Path('mve_deterministic_lag.json').write_text(json.dumps({'live':live,'post':post,'n_checked':checked,'n_candidates':len(candidates)},default=str))
if __name__=='__main__': main()
