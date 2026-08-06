"""Run the minute-00 audit with one explicit censoring correction.

Some full paths skip from the entry observation to a quote five minutes later.
That row cannot represent a frozen 120-second decision. The first version
raised and stopped the entire audit. This runner changes that one branch to
`continue`, so sparse rows are excluded rather than silently treated as 120s.
"""
from pathlib import Path

source_path = Path(__file__).with_name("minute00_delay_audit.py")
source = source_path.read_text(encoding="utf-8")
old = '            raise RuntimeError(f"invalid effective wait {effective_wait}")'
new = '            continue  # censored: no complete quote near the frozen wait'
if source.count(old) != 1:
    raise RuntimeError("expected exactly one effective-wait guard to patch")
namespace = {"__name__": "minute00_delay_audit_v2", "__file__": str(source_path)}
exec(compile(source.replace(old, new), str(source_path), "exec"), namespace)
namespace["main"]()
