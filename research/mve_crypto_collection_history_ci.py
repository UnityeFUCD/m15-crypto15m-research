from __future__ import annotations
import json,time
from pathlib import Path
import requests
BASE='https://external-api.kalshi.com/trade-api/v2'; s=requests.Session()
COLLS=['KXMVECROSSCATEGORY-R','KXMVESPORTSMULTIGAMEEXTENDED-R']
COINS={'KXBTC15M','KXETH15M','KXSOL15M','KXXRP15M','KXDOGE15M','KXHYPE15M','KXBNB15M'}
def get(path,p=None):
  for k in range(15):
    r=s.get(BASE+path,params=p,timeout=40)
    if r.status_code==200:return r.json()
    if r.status_code==429:time.sleep(min(10,.8*(k+1)));continue
    print('ERR',r.status_code,r.url,r.text[:500],flush=True);return None
  return None
def market_day(m): return (m.get('created_time') or m.get('open_time') or '')[:10]
def main():
  out={};crypto=[]
  for coll in COLLS:
    cur=None;ev=[]
    for pg in range(100):
      p={'limit':200,'collection_ticker':coll,'with_nested_markets':'true'}
      if cur:p['cursor']=cur
      b=get('/events/multivariate',p)
      if not b:break
      a=b.get('events') or [];ev+=a;cur=b.get('cursor')
      md=[market_day(m) for e in a for m in (e.get('markets') or []) if market_day(m)]
      cpg=0
      for e in a:
        for m in e.get('markets') or []:
          legs=m.get('mve_selected_legs') or []
          ser=[(l.get('event_ticker') or '').split('-',1)[0] for l in legs]
          if legs and all(x in COINS for x in ser):cpg+=1
      print(coll,'page',pg+1,'events',len(ev),'market_dates',min(md) if md else None,max(md) if md else None,'crypto_pg',cpg,flush=True)
      if not cur or not a:break
      time.sleep(.18)
    out[coll]=ev
    for e in ev:
      for m in e.get('markets') or []:
        legs=m.get('mve_selected_legs') or []
        ser=[(l.get('event_ticker') or '').split('-',1)[0] for l in legs]
        if legs and all(x in COINS for x in ser):crypto.append(m)
  days=sorted(set(market_day(m) for m in crypto if market_day(m)))
  print('TOTAL EVENTS',{k:len(v) for k,v in out.items()},'CRYPTO MARKETS',len(crypto),'DAYS',days,flush=True)
  Path('mve_crypto_collection_history.json').write_text(json.dumps({'events':out,'crypto':crypto},default=str))
if __name__=='__main__':main()
