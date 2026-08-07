from __future__ import annotations
import json,time,math,statistics
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import requests
BASE='https://external-api.kalshi.com/trade-api/v2';s=requests.Session();COINS={'KXBTC15M','KXETH15M','KXSOL15M','KXXRP15M','KXDOGE15M','KXHYPE15M','KXBNB15M'}
def get(path,params=None):
  for k in range(10):
    r=s.get(BASE+path,params=params,timeout=30)
    if r.status_code==200:return r.json()
    if r.status_code==429:time.sleep(.7*(k+1));continue
    print('ERR',r.status_code,r.url,r.text[:300]);return None
  return None
def f(x):
  try:return float(x)
  except:return float('nan')
def fetch_settled(pages=30):
  out=[];cur=None
  for i in range(pages):
    p={'limit':1000,'mve_filter':'only','status':'settled'}
    if cur:p['cursor']=cur
    b=get('/markets',p)
    if not b:break
    a=b.get('markets') or [];out+=a;cur=b.get('cursor');print('page',i+1,'n',len(out),flush=True)
    if not cur or not a:break
    time.sleep(.65)
  return out
def trades(t):
  b=get('/markets/trades',{'ticker':t,'limit':1000})
  return (b or {}).get('trades') or []
def main():
  ms=fetch_settled(30)
  crypto=[]
  for m in ms:
    legs=m.get('mve_selected_legs') or []
    ss=[(l.get('event_ticker') or '').split('-',1)[0] for l in legs]
    if legs and all(x in COINS for x in ss) and f(m.get('volume_fp'))>0 and (m.get('result') or '').lower() in ('yes','no'):
      crypto.append(m)
  print('SETTLED',len(ms),'CRYPTO_TRADED',len(crypto),flush=True)
  rows=[]
  for i,m in enumerate(crypto,1):
    tr=trades(m['ticker'])
    if not tr:continue
    # oldest public trade = first realized execution after combo existed
    tr=sorted(tr,key=lambda x:x.get('created_time') or '')
    z=tr[0]; side=(z.get('taker_outcome_side') or '').lower(); yp=f(z.get('yes_price_dollars')); np=f(z.get('no_price_dollars')); q=f(z.get('count_fp') or z.get('count'))
    y=1 if (m.get('result') or '').lower()=='yes' else 0
    legs=m.get('mve_selected_legs') or []; sides=[(l.get('side') or '').lower() for l in legs]
    allsame=len(set(sides))==1
    chosen_yes_price=yp
    pnl_yes=y-yp
    pnl_taker=(y-yp) if side=='yes' else ((1-y)-np)
    rows.append(dict(ticker=m['ticker'],created=m.get('created_time'),first_trade=z.get('created_time'),result=m.get('result'),nlegs=len(legs),pattern=''.join('Y' if a=='yes' else 'N' for a in sorted(sides)),allsame=allsame,allside=sides[0] if allsame else 'mixed',taker=side,yes_price=yp,no_price=np,qty=q,pnl_yes=pnl_yes,pnl_taker=pnl_taker,volume=f(m.get('volume_fp'))))
    if i%25==0:print('trade',i,'/',len(crypto),flush=True)
    time.sleep(.12)
  def stat(rr,key='pnl_yes'):
    if not rr:return None
    v=[x[key] for x in rr]
    return dict(n=len(rr),mean=sum(v)/len(v),wins=sum(1 for x in rr if x['result']=='yes')/len(rr),avg_px=sum(x['yes_price'] for x in rr)/len(rr),total=sum(v))
  groups={}
  for name,fn in [
   ('all',lambda x:True),('allsame',lambda x:x['allsame']),('mixed',lambda x:not x['allsame']),
   ('all_yes',lambda x:x['allsame'] and x['allside']=='yes'),('all_no',lambda x:x['allsame'] and x['allside']=='no'),
  ]:
    rr=[x for x in rows if fn(x)];groups[name]=stat(rr)
  for n in range(2,6):
    for typ in ['all_yes','all_no','mixed']:
      rr=[x for x in rows if x['nlegs']==n and ((typ=='mixed' and not x['allsame']) or (typ=='all_yes' and x['allsame'] and x['allside']=='yes') or (typ=='all_no' and x['allsame'] and x['allside']=='no'))]
      groups[f'{typ}_{n}']=stat(rr)
  # actual taker direction return is a market-efficiency control
  groups['actual_taker']=stat(rows,'pnl_taker')
  print('GROUPS',json.dumps(groups,sort_keys=True),flush=True)
  for r in rows:print('ROW',json.dumps(r,sort_keys=True),flush=True)
  Path('mve_crypto_combo_firsttrade.json').write_text(json.dumps({'groups':groups,'rows':rows},indent=2))
if __name__=='__main__':main()
