# SUPERSEDED — Probe-Then-Commit model audit

The previous revision reported a model-based PTC estimate of roughly
`+$5.72/day`. **Do not treat that as a historical or live result.**

That model drew fill/no-fill from aggregate two-day outcome rates and paired the
draw with delayed asks from the 73-day path population. It therefore assumed
conditional independence between fill state and the later quote. The direct
joint-path audit was built specifically to attack that assumption.

Authoritative current files:

- `research/ptc_adversarial_v2.py`
- `research/results/ptc_v2_report.md`
- `research/results/ptc_v2_summary.json`

The timestamp-corrected direct audit found:

- no-fill remains informative;
- only a very small number of no-fill markets remain below the frozen ask
  ceiling;
- PTC beat diagnostic-only by `$11.84` on the 220 matched orders, but the
  moving-block interval included zero;
- PTC underperformed the standardized maker control on that matched subset;
- 83 unmatched orders lost `$144.40`, proving path availability is severely
  non-random.

The next required step is to run `research/fetch_missing_lsm_paths.py` locally,
push the completed paths, and rerun v2. Until then PTC is a research
architecture, not a deployment PASS.
