"""One authoritative high-water mark that SURVIVES RESTARTS.

THE DEFECT THIS FIXES
  The system had three drawdown controls - the runner's -$150 self-stop, the
  watchdog's threshold, and a human check - and all three measured from a
  baseline captured at PROCESS START. On 2026-08-05 the runner was started 17
  times and the watchdog re-baselined 13 times, so none of them ever
  accumulated a meaningful measurement window. The watchdog's last restart
  re-baselined at $447.31, discarding the day's $524.71 high: a subsequent
  bleed to $400 would have been reported as a $66 drawdown when the true
  peak-to-trough was $124. The stops would fire late or never.

  Defence in depth is worthless when every layer shares the same blind spot.

THE FIX
  A single JSON file holding the all-time peak equity. Every process reads it
  at start and updates it when equity makes a new high. It is NEVER lowered, so
  concurrent writers cannot corrupt it - the worst case is a lost update that
  the next cycle re-applies. Writes are atomic (temp file + replace).

  This is deliberately a tiny, boring module. Risk controls should be the least
  clever code in a system.
"""
import json
import os
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
HWM_FILE = HERE / "lsm_hwm.json"


def read_hwm():
    """Returns (peak_equity, meta). peak 0.0 if unset or unreadable."""
    try:
        d = json.loads(HWM_FILE.read_text())
        return float(d.get("peak_equity") or 0.0), d
    except Exception:
        return 0.0, {}


def update_hwm(equity, who="?"):
    """Raise the mark if equity is a new high. Returns the authoritative peak.

    Only ever RAISES. A concurrent writer can at worst lose one update, which
    the next cycle restores - so no locking is needed and a crash mid-write
    cannot corrupt the value."""
    peak, d = read_hwm()
    if equity <= peak:
        return peak
    d = dict(d or {})
    d["peak_equity"] = round(float(equity), 4)
    d["updated_by"] = who
    try:
        fd, tmp = tempfile.mkstemp(dir=str(HERE), suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(d, f, indent=1)
        os.replace(tmp, HWM_FILE)          # atomic on Windows and POSIX
    except Exception:
        return peak
    return float(d["peak_equity"])


def drawdown(equity):
    """Dollars below the persistent all-time peak. 0.0 if at or above it."""
    peak, _ = read_hwm()
    if peak <= 0:
        return 0.0
    return max(0.0, peak - float(equity))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "set":
        print(update_hwm(float(sys.argv[2]), who="manual"))
    else:
        p, d = read_hwm()
        print(json.dumps(d, indent=1) if d else "(unset)")
