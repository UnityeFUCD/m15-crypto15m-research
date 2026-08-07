from __future__ import annotations
import json,time
from pathlib import Path
from datetime import datetime,timezone
import requests
BASE='https://external-api.kalshi.com/trade-api/v2'; s=requests.Session()
COLLS=['KXMVECROSSCATEGORY-R','KXMVESPORTSMULTIGAMEEXTENDED-R']
COINS={'KXBTC15M','KXETH15M','KXSOL15M','KXXRP15M','KXDOGE15M','KXHYPE15M','KXBNB15M'}
def get(path,p=None):
  for k in range(12):
    r=s.get(BASE+path,params=p,timeout=40)
    if r.status_code==200:return r.json()
    if r.status_code==429:time.sleep(min(10,.8*(k+1)));continue
    print('ERR',r.status_code,r.url,r.text[:500],flush=True);return None
  return None
def f(x):
  try:return float(x)
  except:return 0.0
def main():
  out={}; crypto=[]
  for coll in COLLS:
    cur=None;ev=[]
    for pg in range(150):
      p={'limit':200,'collection_ticker':coll,'with_nested_markets':'true'}
      if cur:p['cursor']=cur
      b=get('/events/multivariate',p)
      if not b:break
      a=b.get('events') or []; ev+=a; cur=b.get('cursor')
      dates=[(e.get('created_time') or e.get('last_updated_ts') or '')[:10] for e in a]
      print(coll,'page',pg+1,'events',len(ev),'dates',min(dates) if dates else None,max(dates) if dates else None,flush=True)
      if not cur or not a:break
      # stop once clearly older than Aug 1; seven days is enough first pass
      dd=[d for d in dates if d]
      if dd and min(dd)<'2026-08-01': break
      time.sleep(.2)
    out[coll]=ev
    for e in ev:
      ms=e.get('markets') or []
      for m in ms:
        legs=m.get('mve_selected_legs') or e.get('mve_selected_legs') or []
        series=[(l.get('event_ticker') or '').split('-',1)[0] for l in legs]
        if legs and all(x in COINS for x in series):
          crypto.append(m)
  print('TOTAL EVENTS', {k:len(v) for k,v in out.items()},'CRYPTO MARKETS',len(crypto),flush=True)
  for m in crypto[:10]: print('CRYPTO',json.dumps(m,default=str)[:10000],flush=True)
  Path('mve_crypto_collection_history.json').write_text(json.dumps({'events':out,'crypto':crypto},default=str))
if __name__=='__main__':main()
