# RETRACTED — Probe-Then-Commit adversarial v1

**Do not use any number from the previous revision of this file.**

The v1 audit converted pandas microsecond timestamps by dividing by `1e9`, as
though they were nanoseconds. The nominal 60-second decision point became about
1.8 billion seconds after order creation, so eventual no-fills were incorrectly
classified as 60-second no-fills.

The corrected audit is:

- `research/ptc_adversarial_v2.py`
- `research/results/ptc_v2_report.md`
- `research/results/ptc_v2_summary.json`

The corrected v2 result is far weaker and also exposes non-random path
missingness: 220 path-matched orders made +$153.78, while the 83 unmatched
orders lost -$144.40. A targeted refetch is required before the direct path
study can be generalized.

This retraction is preserved in place so stale links fail safely rather than
continuing to display an invalid result.
