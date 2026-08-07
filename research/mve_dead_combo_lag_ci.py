from __future__ import annotations
import json, time, math
from datetime import datetime, timezone
from pathlib import Path
import requests

BASE='https://external-api.kalshi.com/trade-api/v2'
s=requests.Session()

def get(path, params=None, tries=12):
    for k in range(tries):
        r=s.get(BASE+path,params=params,timeout=30)
        if r.status_code==200: return r.json()
        if r.status_code==429:
            wait=min(10.0,0.75*(k+1))
            print('429',path,'wait',wait,flush=True); time.sleep(wait); continue
        print('ERR',r.status_code,r.url,r.text[:500],flush=True); return None
    return None

def ts(x):
    if not x:return None
    try:return datetime.fromisoformat(x.replace('Z','+00:00')).timestamp()
    except:return None

def f(x):
    try:return float(x)
    except:return float('nan')

def fetch_settled(max_pages=20):
    out=[];cur=None
    # no min_settled_ts: API order itself gives latest first; stop after max_pages.
    for page in range(max_pages):
        p={'limit':1000,'mve_filter':'only','status':'settled'}
        if cur:p['cursor']=cur
        b=get('/markets',p)
        if not b:break
        a=b.get('markets') or [];out.extend(a);cur=b.get('cursor')
        print('settled page',page+1,'rows',len(out),flush=True)
        if not cur or not a:break
        time.sleep(.8)
    return out

def chunks(xs,n=50):
    xs=list(xs)
    for i in range(0,len(xs),n):yield xs[i:i+n]

def fetch_components(tickers):
    out={}
    for i,ch in enumerate(chunks(sorted(tickers),50),1):
        b=get('/markets',{'limit':1000,'tickers':','.join(ch)})
        if b:
            for m in b.get('markets') or []:out[m['ticker']]=m
        if i%10==0:print('component batch',i,'coverage',len(out),'/',len(tickers),flush=True)
        time.sleep(.35)
    return out

def fetch_trades(ticker,lo,hi):
    out=[];cur=None
    for _ in range(10):
        p={'ticker':ticker,'limit':1000,'min_ts':max(0,int(lo)-1),'max_ts':int(hi)+1}
        if cur:p['cursor']=cur
        b=get('/markets/trades',p)
        if not b:break
        a=b.get('trades') or [];out.extend(a);cur=b.get('cursor')
        if not cur or not a:break
        time.sleep(.2)
    # dedupe public trade ids
    seen=set();z=[]
    for x in out:
        tid=x.get('trade_id')
        if tid in seen:continue
        seen.add(tid);z.append(x)
    return z

def main():
    markets=fetch_settled(20)
    traded=[m for m in markets if f(m.get('volume_fp'))>0 and (m.get('mve_selected_legs') or [])]
    traded.sort(key=lambda m:f(m.get('volume_fp')),reverse=True)
    top=traded[:500]
    print('SETTLED',len(markets),'TRADED',len(traded),'TOP',len(top),'VOL_CUTOFF',f(top[-1].get('volume_fp')) if top else None,flush=True)
    tick={l.get('market_ticker') for m in top for l in m.get('mve_selected_legs') or [] if l.get('market_ticker')}
    comp=fetch_components(tick)
    print('COMP COVERAGE',len(comp),'/',len(tick),flush=True)

    dead=[]
    for m in top:
        losing=[];unknown=[]
        for leg in m.get('mve_selected_legs') or []:
            c=comp.get(leg.get('market_ticker'))
            if not c:
                unknown.append(('missing',leg));continue
            result=(c.get('result') or '').lower(); side=(leg.get('side') or '').lower(); st=ts(c.get('settlement_ts'))
            if result not in ('yes','no') or st is None:
                unknown.append(('unsettled',leg,result,st));continue
            if result!=side:
                losing.append((st,leg,c))
        if not losing:continue
        certainty=min(x[0] for x in losing)
        close=ts(m.get('close_time')); settle=ts(m.get('settlement_ts'))
        if close is None:continue
        # We only use normal resolved combo NO cases, no scalar fallback.
        if (m.get('result') or '').lower()!='no':
            print('ASSERT_FAIL losing component but combo result',m.get('result'),m['ticker']);continue
        dead.append((m,certainty,close,settle,losing,unknown))
    print('DEAD_FROM_COMPONENT',len(dead),'of top',len(top),flush=True)
    lags=sorted(close-cert for _,cert,close,_,_,_ in dead)
    if lags:
        print('DEAD_TO_CLOSE_QUANTILES',{str(q):lags[min(len(lags)-1,int(q*(len(lags)-1)))] for q in [0,.1,.25,.5,.75,.9,.99,1]},flush=True)

    # prioritize biggest-volume markets with a positive certainty->close interval
    dead=[x for x in dead if x[2]>x[1]]
    dead.sort(key=lambda x:f(x[0].get('volume_fp')),reverse=True)
    events=[]
    for i,(m,certainty,close,settle,losing,unknown) in enumerate(dead[:300],1):
        tr=fetch_trades(m['ticker'],certainty,close)
        for z in tr:
            t=ts(z.get('created_time'))
            if t is None or t+1e-6<certainty:continue
            side=(z.get('taker_outcome_side') or '').lower()
            yp=f(z.get('yes_price_dollars')); np=f(z.get('no_price_dollars')); qty=f(z.get('count_fp') or z.get('count'))
            if not (qty>0):continue
            # Once combo YES is dead, NO payout is exactly $1. A taker NO trade is directly executable evidence.
            if side=='no' and 0<np<1:
                events.append(dict(ticker=m['ticker'],trade_id=z.get('trade_id'),trade_ts=t,certainty_ts=certainty,delay_s=t-certainty,
                                   no_price=np,yes_price=yp,qty=qty,gross_edge=1-np,gross_dollars=(1-np)*qty,
                                   market_volume=f(m.get('volume_fp')),collection=m.get('mve_collection_ticker'),close_time=m.get('close_time'),
                                   first_losing_leg=losing[0][1],unknown_components=len(unknown)))
        if i%25==0:print('trade scan',i,'/',min(300,len(dead)),'direct events',len(events),flush=True)
        time.sleep(.25)
    events.sort(key=lambda x:x['gross_dollars'],reverse=True)
    by_market={}
    for e in events:
        x=by_market.setdefault(e['ticker'],{'gross':0.,'qty':0.,'trades':0,'min_px':1.,'max_px':0.,'first_delay':1e99})
        x['gross']+=e['gross_dollars'];x['qty']+=e['qty'];x['trades']+=1;x['min_px']=min(x['min_px'],e['no_price']);x['max_px']=max(x['max_px'],e['no_price']);x['first_delay']=min(x['first_delay'],e['delay_s'])
    print('DIRECT_TAKER_NO_EVENTS',len(events),'MARKETS',len(by_market),'GROSS',sum(e['gross_dollars'] for e in events),'QTY',sum(e['qty'] for e in events),flush=True)
    for k,v in sorted(by_market.items(),key=lambda kv:kv[1]['gross'],reverse=True)[:50]:print('MKT',k,json.dumps(v,sort_keys=True),flush=True)
    for e in events[:100]:print('TRADE',json.dumps(e,default=str,sort_keys=True),flush=True)
    Path('mve_dead_combo_lag.json').write_text(json.dumps({'events':events,'markets':by_market,'dead_n':len(dead),'top_n':len(top)},default=str))

if __name__=='__main__':main()
