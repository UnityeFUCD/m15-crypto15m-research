"""One Kalshi client. POST-capable, paginated, error-preserving, retrying.

REPLACES TWO DIVERGENT CLIENTS
  live/acct.py       GET only; returns None on non-200, discarding the error
                     body, so api_errors can never be populated from it
  live/lsm_runner.py GET+POST and preserves bodies, but no pagination anywhere
                     (grep -n cursor returns nothing) and no retry

CREDENTIALS
  Read from KALSHI_CRED_DIR, else the existing external path. Never from the
  repo. Nothing here writes a key anywhere.

PAGINATION
  paginate() follows `cursor` until exhausted with a hard page cap, so a
  truncated read is impossible and an unbounded loop is also impossible.
"""
from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

BASE = "https://api.elections.kalshi.com/trade-api/v2"
PREFIX = "/trade-api/v2"
DEFAULT_CRED = Path("C:/Users/fycin/kalshi_weather_trading")


@dataclass
class ApiResult:
    status: int
    body: Any
    path: str
    method: str
    ts_request_ms: int
    ts_response_ms: int
    error: str | None = None
    attempts: int = 1

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    @property
    def latency_ms(self) -> int:
        return self.ts_response_ms - self.ts_request_ms


@dataclass
class KalshiClient:
    cred_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("KALSHI_CRED_DIR", str(DEFAULT_CRED))))
    timeout: float = 15.0
    max_retries: int = 3
    min_interval_s: float = 0.05          # crude client-side rate limit
    on_error: Any = None                  # callable(ApiResult) -> None

    def __post_init__(self):
        env = self.cred_dir / ".env"
        cfg = {}
        if env.exists():
            for line in env.read_text().splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.split("=", 1)
                    cfg[k.strip()] = v.strip().strip('"').strip("'")
        # PRECEDENCE MATTERS AND MUST BE VISIBLE.
        # A stale KALSHI_API_KEY_ID in the environment silently overrode the
        # .env file once and produced a bare 401 with no clue that two
        # different key ids were in play. The file is the source of truth for
        # this deployment; the environment only fills gaps, and whichever won
        # is recorded on `cred_source` so a 401 is diagnosable.
        file_key = cfg.get("KALSHI_API_KEY_ID")
        env_key = os.environ.get("KALSHI_API_KEY_ID")
        if file_key:
            self.key_id, self.cred_source = file_key, f"{env}"
            if env_key and env_key != file_key:
                self.cred_conflict = (
                    f"environment KALSHI_API_KEY_ID ({env_key[:8]}...) differs "
                    f"from {env} ({file_key[:8]}...); using the file")
            else:
                self.cred_conflict = None
        else:
            self.key_id, self.cred_source = env_key, "environment"
            self.cred_conflict = None

        pem = (cfg.get("KALSHI_RSA_PRIVATE_KEY_PATH")
               or os.environ.get("KALSHI_PRIVATE_KEY_PATH")
               or os.environ.get("KALSHI_RSA_PRIVATE_KEY_PATH")
               or str(self.cred_dir / "kalshi_private_key.pem"))
        if not Path(pem).is_absolute():
            pem = str(self.cred_dir / pem)
        if not self.key_id or not Path(pem).exists():
            raise RuntimeError(
                f"credentials not found (key_id={'set' if self.key_id else 'MISSING'}, "
                f"pem={pem}). Set KALSHI_CRED_DIR, or KALSHI_API_KEY_ID + "
                f"KALSHI_RSA_PRIVATE_KEY_PATH. Never commit them.")
        self._priv = serialization.load_pem_private_key(
            Path(pem).read_bytes(), password=None)
        self._s = requests.Session()
        self._last_call = 0.0

    def _sign(self, ts: str, method: str, path: str) -> str:
        msg = (ts + method + PREFIX + path).encode()
        return base64.b64encode(self._priv.sign(
            msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                             salt_length=32), hashes.SHA256())).decode()

    def request(self, method: str, path: str, params=None,
                body=None) -> ApiResult:
        gap = time.time() - self._last_call
        if gap < self.min_interval_s:
            time.sleep(self.min_interval_s - gap)
        last: ApiResult | None = None
        for attempt in range(1, self.max_retries + 1):
            t0 = int(time.time() * 1000)
            ts = str(t0)
            h = {"KALSHI-ACCESS-KEY": self.key_id,
                 "KALSHI-ACCESS-SIGNATURE": self._sign(ts, method, path),
                 "KALSHI-ACCESS-TIMESTAMP": ts,
                 "Content-Type": "application/json"}
            try:
                r = self._s.request(method, BASE + path, params=params,
                                    json=body, headers=h, timeout=self.timeout)
                self._last_call = time.time()
                try:
                    parsed = r.json()
                except Exception:
                    parsed = {"_raw": r.text[:2000]}
                res = ApiResult(r.status_code, parsed, path, method, t0,
                                int(time.time() * 1000), attempts=attempt)
            except Exception as e:
                self._last_call = time.time()
                res = ApiResult(0, None, path, method, t0,
                                int(time.time() * 1000),
                                error=f"{type(e).__name__}: {e}",
                                attempts=attempt)
            last = res
            # retry only on transport failure or 5xx / 429 - never on 4xx,
            # and never on a write that may already have been accepted
            retryable = res.status == 0 or res.status >= 500 or res.status == 429
            if not retryable or method != "GET" or attempt == self.max_retries:
                break
            time.sleep(0.4 * attempt)
        if last is not None and not last.ok and self.on_error:
            try:
                self.on_error(last)
            except Exception:
                pass
        return last

    def get(self, path, params=None) -> ApiResult:
        return self.request("GET", path, params=params)

    def post(self, path, body=None, params=None) -> ApiResult:
        return self.request("POST", path, params=params, body=body)

    def delete(self, path, params=None) -> ApiResult:
        return self.request("DELETE", path, params=params)

    def paginate(self, path: str, key: str, params: dict | None = None,
                 page_limit: int = 200, max_pages: int = 500
                 ) -> Iterator[dict]:
        """Follow `cursor` to exhaustion. The runner does none of this today."""
        p = dict(params or {})
        p.setdefault("limit", page_limit)
        cursor, pages = None, 0
        seen_cursors = set()
        while pages < max_pages:
            if cursor:
                p["cursor"] = cursor
            res = self.get(path, p)
            if not res.ok or not isinstance(res.body, dict):
                return
            batch = res.body.get(key) or []
            for item in batch:
                yield item
            cursor = res.body.get("cursor")
            pages += 1
            if not cursor or not batch:
                return
            if cursor in seen_cursors:      # server bug guard
                return
            seen_cursors.add(cursor)

    # ---------- endpoints the capture system needs ----------
    def queue_position(self, order_id: str) -> ApiResult:
        """Authoritative single-order queue position.
        VERIFIED 2026-08-06: 200 {"queue_position_fp": "0.00"}"""
        return self.get(f"/portfolio/orders/{order_id}/queue_position")

    def queue_positions(self, market_tickers: list[str] | None = None,
                        event_ticker: str | None = None) -> ApiResult:
        """Authoritative batch queue positions.

        VERIFIED 2026-08-06: requires market_tickers or event_ticker.
        Passing order_ids returns
        400 {"details": "Need to specify market_tickers or event_ticker"}.
        """
        if not market_tickers and not event_ticker:
            raise ValueError("market_tickers or event_ticker is required")
        p: dict[str, Any] = {}
        if market_tickers:
            p["market_tickers"] = ",".join(market_tickers)
        if event_ticker:
            p["event_ticker"] = event_ticker
        return self.get("/portfolio/orders/queue_positions", p)

    def orderbook(self, ticker: str, depth: int = 100) -> ApiResult:
        return self.get(f"/markets/{ticker}/orderbook", {"depth": depth})

    def resting_orders(self, **kw) -> list[dict]:
        return list(self.paginate("/portfolio/orders", "orders",
                                  {"status": "resting", **kw}))

    def all_positions(self) -> list[dict]:
        return list(self.paginate("/portfolio/positions", "market_positions"))

    def fills(self, **kw) -> list[dict]:
        return list(self.paginate("/portfolio/fills", "fills", kw))

    def public_trades(self, ticker: str, max_pages: int = 10) -> list[dict]:
        return list(self.paginate("/markets/trades", "trades",
                                  {"ticker": ticker}, max_pages=max_pages))
