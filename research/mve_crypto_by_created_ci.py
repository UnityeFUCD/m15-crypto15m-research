from __future__ import annotations
import json,time
from pathlib import Path
from datetime import datetime,timezone
import requests
BASE='https://external-api.kalshi.com/trade-api/v2';s=requests.Session()
COINS={'KXBTC15M','KXETH15M','KXSOL15M','KXXRP15M','KXDOGE15M','KXHYPE15M','KXBNB15M'}
def get(p):
  for k in range(12):
    r=s.get(BASE+'/markets',params=p,timeout=40)
    if r.status_code==200:return r.json()
    if r.status_code==429:time.sleep(min(8,.5*(k+1)));continue
    print('ERR',r.status_code,r.url,r.text[:500],flush=True);return None
  return None
def ep(day):return int(datetime.fromisoformat(day+'T00:00:00+00:00').timestamp())
def main():
  allc=[]
  for day in ['2026-07-31','2026-08-01','2026-08-02','2026-08-03','2026-08-04','2026-08-05','2026-08-06','2026-08-07']:
    lo=ep(day);hi=lo+86400;cur=None;arr=[]
    for pg in range(100):
      p={'limit':1000,'mve_filter':'only','min_created_ts':lo,'max_created_ts':hi}
      if cur:p['cursor']=cur
      b=get(p)
      if not b:break
      a=b.get('markets') or [];arr+=a;cur=b.get('cursor')
      if not cur or not a:break
      if pg>=49:break
      time.sleep(.1)
    crypto=[]
    for m in arr:
      legs=m.get('mve_selected_legs') or []
      ss=[(l.get('event_ticker') or '').split('-',1)[0] for l in legs]
      if legs and all(x in COINS for x in ss):crypto.append(m)
    print(day,'allmve',len(arr),'crypto',len(crypto),'mincreated',min((m.get('created_time') or '') for m in arr) if arr else None,'maxcreated',max((m.get('created_time') or '') for m in arr) if arr else None,flush=True)
    allc+=crypto
  print('TOTAL CRYPTO',len(allc),flush=True)
  Path('mve_crypto_by_created.json').write_text(json.dumps({'crypto':allc},default=str))
if __name__=='__main__':main()
