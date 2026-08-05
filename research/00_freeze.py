"""STAGE 0 - freeze the discovery state.

Read-only. Writes ONLY into this directory. Produces DATA_MANIFEST.json and
ENVIRONMENT.txt. Hashes every input file that the discovery depended on, counts
trades/contracts/windows, and records the exact scripts involved.

Nothing here interprets the discovery. It only pins down what was in front of
it, so that any later claim can be checked against a fixed corpus.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path("C:/Users/fycin/m15-claude-independent")
OUT = ROOT / "research/final_minute_favorite_maker_validation"
TRADES = ROOT / "data/sare/trades"
CF = ROOT / "data/sare/cf"


def sha256(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sh(*args):
    try:
        return subprocess.run(args, cwd=ROOT, capture_output=True,
                              text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"<error {e}>"


print("stage 0: freezing discovery state", flush=True)
man = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "git": {"commit": sh("git", "rev-parse", "HEAD"),
               "branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
               "status_porcelain": sh("git", "status", "--porcelain"),
               "remotes": sh("git", "remote", "-v")},
       "inputs": {}}
man["git"]["working_tree_clean"] = (man["git"]["status_porcelain"] == "")

# ---- trade tape -------------------------------------------------------------
print("hashing + counting trade tape...", flush=True)
files = sorted(TRADES.glob("*.json"))
per_coin = Counter()
n_trades = 0
contracts = Decimal(0)          # exact: contract counts are decimal strings
tmin, tmax = None, None
bad = []
hashes = {}
t0 = time.time()
for i, f in enumerate(files):
    coin = f.name.split("15M")[0].replace("KX", "")
    per_coin[coin] += 1
    hashes[f.name] = sha256(f)
    try:
        rec = json.loads(f.read_text())
    except Exception as e:
        bad.append({"file": f.name, "reason": f"unparseable: {e}"})
        continue
    if not isinstance(rec, list):
        bad.append({"file": f.name, "reason": f"not a list: {type(rec).__name__}"})
        continue
    n_trades += len(rec)
    for r in rec:
        try:
            contracts += Decimal(str(r["count_fp"]))
        except Exception:
            bad.append({"file": f.name, "reason": "count_fp missing/unparseable"})
        ct = r.get("created_time")
        if ct:
            tmin = ct if tmin is None or ct < tmin else tmin
            tmax = ct if tmax is None or ct > tmax else tmax
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(files)}  {n_trades:,} trades  "
              f"{time.time()-t0:.0f}s", flush=True)

man["inputs"]["trade_tape"] = {
    "dir": str(TRADES), "n_files": len(files),
    "windows_per_coin": dict(sorted(per_coin.items())),
    "n_trades": n_trades, "n_contracts": str(contracts),
    "created_time_min": tmin, "created_time_max": tmax,
    "malformed_records": bad, "n_malformed": len(bad),
    "sha256_by_file": hashes,
    "sha256_of_all_hashes": hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode()).hexdigest()}

# ---- CF index ---------------------------------------------------------------
print("hashing CF index...", flush=True)
cfs = sorted(CF.glob("*.npz"))
cf_h = {f.name: sha256(f) for f in cfs}
cf_coin = Counter(f.name.split("_")[0] for f in cfs)
hours = sorted(int(f.name.split("_")[1].split(".")[0]) for f in cfs)
man["inputs"]["cf_index"] = {
    "dir": str(CF), "n_files": len(cfs), "files_per_coin": dict(sorted(cf_coin.items())),
    "hour_min_unix": hours[0] if hours else None,
    "hour_max_unix": hours[-1] if hours else None,
    "hour_min_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(hours[0])) if hours else None,
    "hour_max_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(hours[-1] + 3600)) if hours else None,
    "sha256_of_all_hashes": hashlib.sha256(
        json.dumps(cf_h, sort_keys=True).encode()).hexdigest()}

# ---- the scripts that produced the discovery --------------------------------
prod = ["cleanroom/snipe.py", "cleanroom/snipe2.py", "cleanroom/pathshape.py",
        "cleanroom/calib_curve.py", "cleanroom/band70.py", "claude_frame_ext.py"]
man["discovery_scripts"] = {}
for p in prod:
    fp = ROOT / p
    man["discovery_scripts"][p] = ({"sha256": sha256(fp), "bytes": fp.stat().st_size}
                                   if fp.exists() else None)

(OUT / "DATA_MANIFEST.json").write_text(json.dumps(man, indent=1))

env = [f"created_utc: {man['created_utc']}",
       f"python: {sys.version}", f"platform: {platform.platform()}",
       f"machine: {platform.machine()}", f"cwd: {ROOT}"]
for mod in ("numpy", "pandas", "scipy", "pyarrow", "requests"):
    try:
        m = __import__(mod)
        env.append(f"{mod}: {getattr(m, '__version__', '?')}")
    except Exception as e:
        env.append(f"{mod}: NOT INSTALLED ({e.__class__.__name__})")
env.append(f"git commit: {man['git']['commit']}")
env.append(f"git clean: {man['git']['working_tree_clean']}")
(OUT / "ENVIRONMENT.txt").write_text("\n".join(env) + "\n")

print("\n=== STAGE 0 SUMMARY ===")
print(f"  git commit        {man['git']['commit'][:12]}  clean={man['git']['working_tree_clean']}")
t = man["inputs"]["trade_tape"]
print(f"  trade files       {t['n_files']:,}")
print(f"  windows per coin  {t['windows_per_coin']}")
print(f"  trades            {t['n_trades']:,}")
print(f"  contracts         {t['n_contracts']}")
print(f"  trade time range  {t['created_time_min']} .. {t['created_time_max']}")
print(f"  malformed         {t['n_malformed']}")
c = man["inputs"]["cf_index"]
print(f"  cf index files    {c['n_files']:,}  {c['files_per_coin']}")
print(f"  cf time range     {c['hour_min_utc']} .. {c['hour_max_utc']}")
print(f"\n  wrote {OUT/'DATA_MANIFEST.json'}")
print(f"  wrote {OUT/'ENVIRONMENT.txt'}")
