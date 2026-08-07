from __future__ import annotations
import json, time
from pathlib import Path
import requests

BASE='https://external-api.kalshi.com/trade-api/v2'
s=requests.Session()

def pag(path, params, key, max_pages=50):
    p=dict(params); cur=None; out=[]
    for _ in range(max_pages):
        if cur: p['cursor']=cur
        r=s.get(BASE+path, params=p, timeout=30)
        print('GET',r.url,'->',r.status_code, flush=True)
        if r.status_code!=200:
            print(r.text[:1000]); break
        b=r.json(); batch=b.get(key) or []; out.extend(batch)
        cur=b.get('cursor')
        if not cur or not batch: break
        time.sleep(.05)
    return out

def main():
    open_m=pag('/markets', {'limit':1000,'mve_filter':'only','status':'open'}, 'markets', 20)
    recent_settled=pag('/markets', {'limit':1000,'mve_filter':'only','status':'settled','min_settled_ts': int(time.time())-45*86400}, 'markets', 20)
    ev=pag('/events/multivariate', {'limit':200}, 'events', 10)
    print('COUNTS open',len(open_m),'settled45d',len(recent_settled),'events',len(ev))
    for label,arr in [('OPEN',open_m),('SETTLED',recent_settled),('EVENT',ev)]:
        print('\n###',label,'SAMPLES')
        for x in arr[:5]:
            print(json.dumps(x, sort_keys=True, default=str)[:12000])
    # Key inventory across market records
    ks={}
    for x in open_m+recent_settled:
        for k,v in x.items():
            ks.setdefault(k,set()).add(type(v).__name__)
    print('\nMARKET_KEYS',json.dumps({k:sorted(v) for k,v in sorted(ks.items())},indent=2))
    Path('mve_public_discovery.json').write_text(json.dumps({'open':open_m,'settled':recent_settled,'events':ev},default=str))
if __name__=='__main__': main()
