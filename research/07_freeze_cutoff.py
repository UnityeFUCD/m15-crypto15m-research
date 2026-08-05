"""STAGE 0 FREEZE - seal the discovery period.

Everything up to the cutoff recorded here is DISCOVERY data. It caused us to
focus on BTC/ETH and on the 65-85c x 8-14min box, so it can never serve as
prospective confirmation of those choices. Any confirmation sample must begin
strictly after this timestamp with the decision function unchanged.

Records: UTC cutoff, git commit, working-tree state, SHA-256 of the runner, of
the decision-relevant config, and of the decision function itself (extracted as
source text so that a later edit anywhere else in the file does not silently
invalidate the comparison).
"""
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

ROOT = Path("C:/Users/fycin/m15-claude-independent")
LIVE = ROOT / "live"
OUT = ROOT / "research/final_minute_favorite_maker_validation"
RUNNER = LIVE / "lsm_runner.py"


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()


def sh(*a):
    try:
        return subprocess.run(a, cwd=ROOT, capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except Exception as e:
        return f"<error {e}>"


src = RUNNER.read_text()

# the decision-relevant constants, pinned individually
cfg = {}
for k in ("BAND", "MIN_LEFT", "MAX_PER_WINDOW", "MAX_QTY", "SERIES",
          "MAX_GROSS", "MAX_LOSS"):
    m = re.search(rf"^{k}\s*=\s*(.+)$", src, re.M)
    cfg[k] = m.group(1).split("#")[0].strip() if m else None

# the decision function = the scan block that decides what to submit
# The decision function is the whole main loop: `while True:` to the end.
# The first attempt anchored on a comment string that does not exist in the
# file and silently hashed the EMPTY STRING (e3b0c442... is sha256 of ""),
# which would have made every future comparison trivially "match".
i = src.find("\nwhile True:")
decision_src = src[i:] if i >= 0 else ""
assert len(decision_src) > 2000, f"decision block too short: {len(decision_src)}"

freeze = {
    "cutoff_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "git_commit": sh("git", "rev-parse", "HEAD"),
    "git_branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
    "git_dirty": sh("git", "status", "--porcelain") != "",
    "runner_path": str(RUNNER),
    "runner_sha256": sha(RUNNER.read_bytes()),
    "runner_bytes": RUNNER.stat().st_size,
    "config": cfg,
    "config_sha256": sha(json.dumps(cfg, sort_keys=True)),
    "decision_function_sha256": sha(decision_src),
    "decision_function_bytes": len(decision_src),
    # RUNTIME values, not file defaults. The file defaults (MAX_GROSS 80,
    # MAX_LOSS 50) are overridden by environment at launch, so capturing the
    # source text alone would have frozen the wrong numbers.
    "runtime": {"LSM_QTY": 30, "LSM_MAX_GROSS": 110, "LSM_MAX_LOSS": 100,
                "LSM_PER_WINDOW": 2, "LSM_MIN_LEFT_LO": 8, "LSM_MIN_LEFT_HI": 14,
                "WATCH_MAX_DD": 75},
    "note": ("All live results up to cutoff_utc are DISCOVERY data. They "
             "selected BTC/ETH and the 65-85c x 8-14min box and therefore "
             "cannot confirm them. Confirmation requires a sample beginning "
             "strictly after this cutoff with these hashes unchanged."),
}
(OUT / "FREEZE_CUTOFF.json").write_text(json.dumps(freeze, indent=1))

print("=== STRATEGY FROZEN ===")
for k in ("cutoff_utc", "git_commit", "git_dirty", "runner_sha256",
          "config_sha256", "decision_function_sha256"):
    v = freeze[k]
    print(f"  {k:26} {str(v)[:64]}")
print("\n  config pinned:")
for k, v in cfg.items():
    print(f"    {k:16} {v}")
print(f"\n  wrote {OUT/'FREEZE_CUTOFF.json'}")
