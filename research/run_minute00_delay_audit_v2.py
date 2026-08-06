"""Run the minute-00 audit with explicit audit corrections.

Corrections applied without changing the frozen primary policy:
1. Sparse paths whose next quote is more than 90 seconds after the nominal
   120-second decision are censored instead of halting or pretending they are
   120-second observations.
2. Bootstrap repetitions are reduced to 4,000 (3,000 for the best-of-four
   permutation) so the exact same estimands finish reliably in CI.
3. Wait sensitivity starts from every initial :00 candidate rather than from
   the subset that qualified at the primary 120-second rule.
"""
from pathlib import Path

source_path = Path(__file__).with_name("minute00_delay_audit.py")
source = source_path.read_text(encoding="utf-8")
patches = {
    '            raise RuntimeError(f"invalid effective wait {effective_wait}")':
        '            continue  # censored: no complete quote near frozen wait',
    'reps: int = 12000': 'reps: int = 4000',
    'reps: int = 6000': 'reps: int = 3000',
    'wait_grid = wait_sensitivity(top)':
        'wait_grid = wait_sensitivity(all_rows[all_rows.minute == 0])',
}
for old, new in patches.items():
    if old not in source:
        raise RuntimeError(f"expected audit patch not found: {old}")
    source = source.replace(old, new)
namespace = {"__name__": "minute00_delay_audit_v2", "__file__": str(source_path)}
exec(compile(source, str(source_path), "exec"), namespace)
namespace["main"]()
