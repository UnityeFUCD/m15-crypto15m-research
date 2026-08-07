from __future__ import annotations
import json,time,math
from pathlib import Path
from datetime import datetime
import requests
BASE='https://external-api.kalshi.com/trade-api/v2';s=requests.Session()
COINS={'KXBTC15M','KXETH15M','KXSOL15M','KXXRP15M','KXDOGE15M','KXHYPE15M','KXBNB15M'}
TICK=0.001

def get(path,params=None,tries=12):
  for k in range(tries):
    r=s.get(BASE+path,params=params,timeout=35)
    if r.status_code==200:return r.json()
    if r.status_code==429:time.sleep(min(8,.5*(k+1)));continue
    if r.status_code in (404,410):return None
    print('ERR',r.status_code,r.url,r.text[:400],flush=True);return None
  return None

def ts(x):
  if not x:return None
  try:return datetime.fromisoformat(x.replace('Z','+00:00')).timestamp()
  except:return None

def f(x):
  try:return float(x)
  except:return float('nan')

def settled(pages=30):
  out=[];cur=None
  for i in range(pages):
    p={'limit':1000,'mve_filter':'only','status':'settled'}
    if cur:p['cursor']=cur
    b=get('/markets',p)
    if not b:break
    a=b.get('markets') or [];out+=a;cur=b.get('cursor')
    print('mve page',i+1,'n',len(out),flush=True)
    if not cur or not a:break
    time.sleep(.12)
  return out

def first_trade(ticker):
  b=get('/markets/trades',{'ticker':ticker,'limit':1000})
  a=(b or {}).get('trades') or []
  return min(a,key=lambda z:z.get('created_time') or '') if a else None

def candle(ticker,decision):
  ser=ticker.split('-',1)[0]; lo=max(0,int(decision)-1000); hi=int(decision)
  p={'start_ts':lo,'end_ts':hi,'period_interval':1,'include_latest_before_start':'true'}
  b=get(f'/series/{ser}/markets/{ticker}/candlesticks',p)
  if not b:
    b=get(f'/historical/markets/{ticker}/candlesticks',{'start_ts':lo,'end_ts':hi,'period_interval':1})
  a=(b or {}).get('candlesticks') or []
  a=[x for x in a if (x.get('end_period_ts') or 0)<=decision]
  if not a:return None
  return max(a,key=lambda x:x.get('end_period_ts') or 0)

def pxclose(obj):
  if not obj:return float('nan')
  for k in ('close_dollars','close'):
    v=f(obj.get(k))
    if math.isfinite(v):return v
  return float('nan')

def main():
  ms=settled(30)
  crypto=[]
  for m in ms:
    legs=m.get('mve_selected_legs') or []; ss=[(l.get('event_ticker') or '').split('-',1)[0] for l in legs]
    if legs and all(x in COINS for x in ss) and f(m.get('volume_fp'))>0:
      z=first_trade(m['ticker'])
      if z and (z.get('taker_outcome_side') or '').lower()=='yes':crypto.append((m,z))
  print('CRYPTO YES-FIRST',len(crypto),flush=True)
  uniq={l['market_ticker'] for m,_ in crypto for l in m.get('mve_selected_legs') or []}
  # cache candles by (ticker, minute of decision) because RFQs refresh through the window
  cache={};rows=[]
  for i,(m,z) in enumerate(crypto,1):
    decision=ts(m.get('created_time')); trade_ts=ts(z.get('created_time'))
    if decision is None or trade_ts is None:continue
    py=f(z.get('yes_price_dollars')); qhist=f(z.get('count_fp') or z.get('count'))
    if not (0<py<1 and qhist>0):continue
    quote=max(0.001,py-TICK) # strict 0.1c improvement over accepted YES price
    for leg in m.get('mve_selected_legs') or []:
      tk=leg['market_ticker']; side=(leg.get('side') or '').lower(); key=(tk,int(decision//60))
      if key not in cache:cache[key]=candle(tk,decision)
      c=cache[key]
      if not c:continue
      yb=pxclose(c.get('yes_bid')); ya=pxclose(c.get('yes_ask'))
      if side=='yes': acq=yb
      else: acq=(1-ya) if math.isfinite(ya) else float('nan')
      if not (0<acq<1):continue
      # Deterministic identity: selected leg side L implies combo YES C. Long L + long combo NO has min payout 1.
      gross_lock=quote-acq
      # Conservative normal maker fee stress on both acquisitions: 1.75%*p*(1-p), no rounding benefit.
      fee_component=.0175*acq*(1-acq)
      no_cost=1-quote
      fee_combo=.0175*no_cost*(1-no_cost)
      net_lock=gross_lock-fee_component-fee_combo
      rows.append(dict(combo=m['ticker'],component=tk,side=side,decision=decision,trade_ts=trade_ts,delay_s=trade_ts-decision,
        accepted_yes=py,our_yes=quote,hist_qty=qhist,component_bid=acq,gross_lock=gross_lock,net_lock_fee_stress=net_lock,
        nlegs=len(m.get('mve_selected_legs') or []),close_key=tk.split('-',1)[1],candle_end=c.get('end_period_ts')))
    if i%50==0:print('processed',i,'rows',len(rows),'cache',len(cache),flush=True)
    time.sleep(.02)
  good=[r for r in rows if .65<=r['component_bid']<=.80 and r['net_lock_fee_stress']>0]
  # choose best one component hedge per combo, then measure exact historical accepted size capacity
  best={}
  for r in good:
    if r['combo'] not in best or r['net_lock_fee_stress']>best[r['combo']]['net_lock_fee_stress']:best[r['combo']]=r
  B=list(best.values())
  print('ROWS',len(rows),'GOOD 65-80',len(good),'COMBOS',len(B),flush=True)
  for q in [1,5,10,20,30,40,50,60,75,100]:
    pnl=sum(min(q,r['hist_qty'])*r['net_lock_fee_stress'] for r in B)
    ct=sum(min(q,r['hist_qty']) for r in B)
    print('Q',q,'contracts',round(ct,2),'guaranteed_net_fee_stress',round(pnl,4),'per_combo',round(pnl/max(1,len(B)),4),flush=True)
  # close-cluster opportunity view; same component inventory cannot be reused beyond its quantity, so do NOT sum all combos as capacity claim.
  byclose={}
  for r in B:
    byclose.setdefault(r['close_key'],[]).append(r)
  print('CLOSE_KEYS',len(byclose),flush=True)
  for ck,rr in sorted(byclose.items()):
    print('CLOSE',ck,'combos',len(rr),'best_net',max(x['net_lock_fee_stress'] for x in rr),'median_qty',sorted(x['hist_qty'] for x in rr)[len(rr)//2],flush=True)
  for r in sorted(B,key=lambda x:x['net_lock_fee_stress']*min(60,x['hist_qty']),reverse=True)[:50]:print('LOCK',json.dumps(r,sort_keys=True),flush=True)
  Path('ibdr_lock.json').write_text(json.dumps({'rows':rows,'good':B},indent=2))
if __name__=='__main__':main()
