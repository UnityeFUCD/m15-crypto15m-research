from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import hbcd_causal_audit as audit

SERIES = [
    x.strip()
    for x in os.getenv(
        "HBCD_DIAGNOSTIC_SERIES",
        "KXMVECROSSCATEGORY,KXMVESPORTSMULTIGAMEEXTENDED",
    ).split(",")
    if x.strip()
]


def fetch_series_markets(api: audit.PublicAPI):
    out: dict[str, dict] = {}
    truncated: list[str] = []
    for day in audit.daterange(audit.START_DAY, audit.END_DAY):
        lo = int(datetime.combine(day, datetime.min.time(), timezone.utc).timestamp())
        hi = lo + 86400
        for series in SERIES:
            cursor = None
            exhausted = False
            for _ in range(audit.MAX_PAGES_PER_DAY):
                params = {
                    "limit": 1000,
                    "mve_filter": "only",
                    "series_ticker": series,
                    "min_created_ts": lo,
                    "max_created_ts": hi,
                }
                if cursor:
                    params["cursor"] = cursor
                body = api.get("/markets", params)
                if not body:
                    exhausted = True
                    break
                batch = body.get("markets") or []
                for market in batch:
                    ticker = market.get("ticker")
                    if ticker:
                        out[str(ticker)] = market
                cursor = body.get("cursor")
                if not cursor or not batch:
                    exhausted = True
                    break
            if not exhausted and cursor:
                truncated.append(f"{day.isoformat()}:{series}")
        crypto_n = sum(
            1
            for market in out.values()
            if str(market.get("created_time") or "")[:10] == day.isoformat()
            and audit.is_crypto_combo(market)
        )
        print(
            "DIAGNOSTIC_DAY",
            day.isoformat(),
            "crypto_unique",
            crypto_n,
            "series",
            SERIES,
            flush=True,
        )
    return list(out.values()), truncated


if __name__ == "__main__":
    audit.OUT = audit.ROOT / "research" / "results" / "hbcd_series_diagnostic"
    Path(audit.OUT).mkdir(parents=True, exist_ok=True)
    audit.fetch_mve_markets = fetch_series_markets
    audit.evaluate()
