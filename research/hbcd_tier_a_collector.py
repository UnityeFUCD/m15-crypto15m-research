#!/usr/bin/env python3
"""Read-only Tier-A shadow collector for HBCD dominance-lock research.

HBCD (Hedge-Before-Confirm Dominance Lock) pairs one selected component leg
with combo NO.  This process never creates an RFQ, quote, order, confirmation,
or cancellation.  It records the causal data needed to decide whether such an
action *would* have been admissible:

* authenticated ``communications`` RFQ/quote lifecycle messages;
* sequence-numbered component order books;
* authenticated ``user_orders``, ``fill``, and ``market_positions`` messages;
* multivariate market lifecycle messages;
* signed, read-only GETs for RFQ, combo-market, and component-orderbook data;
* local wall and monotonic clocks, raw payloads, hashes, connection IDs, and
  frozen shadow decisions.

Credentials are read from environment variables or CLI arguments and are never
written to the log.  A later executor must be a separate program with an
explicit real-money enable flag and its own request/response journal.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import gzip
import hashlib
import json
import math
import os
import platform
import signal
import socket
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import requests
import websockets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

DEFAULT_REST_BASE = "https://external-api.kalshi.com/trade-api/v2"
DEFAULT_WS_URL = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
WS_SIGN_PATH = "/trade-api/ws/v2"

CRYPTO15_SERIES = {
    "KXBTC15M",
    "KXETH15M",
    "KXSOL15M",
    "KXXRP15M",
    "KXDOGE15M",
    "KXHYPE15M",
    "KXBNB15M",
}

FEE_RATE_STRESS = 0.07
MAX_ROUNDING_CHARGE_PER_ORDER = 0.0099
DEFAULT_MARGIN = 0.01
CONFIG_VERSION = "HBCD-shadow-v1"


def now_pair() -> tuple[int, int]:
    return time.time_ns(), time.monotonic_ns()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def fnum(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def load_private_key(path: str | None, pem: str | None):
    if pem:
        raw = pem.replace("\\n", "\n").encode()
    elif path:
        raw = Path(path).expanduser().read_bytes()
    else:
        raise SystemExit("Set KALSHI_PRIVATE_KEY_PATH or KALSHI_PRIVATE_KEY_PEM")
    return serialization.load_pem_private_key(raw, password=None)


def sign_text(private_key, text: str) -> str:
    signature = private_key.sign(
        text.encode(),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode()


def signed_headers(key_id: str, private_key, method: str, full_path: str) -> dict[str, str]:
    ts = str(int(time.time() * 1000))
    path_without_query = full_path.split("?", 1)[0]
    signature = sign_text(private_key, f"{ts}{method.upper()}{path_without_query}")
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": signature,
    }


def series_of_leg(leg: dict[str, Any]) -> str:
    source = str(leg.get("event_ticker") or leg.get("market_ticker") or "")
    return source.split("-", 1)[0]


def is_crypto15_combo(legs: Iterable[dict[str, Any]]) -> bool:
    legs = list(legs)
    return bool(legs) and all(series_of_leg(leg) in CRYPTO15_SERIES for leg in legs)


def parse_price_ranges(market: dict[str, Any]) -> list[dict[str, float]]:
    parsed: list[dict[str, float]] = []
    for row in market.get("price_ranges") or []:
        if not isinstance(row, dict):
            continue

        def take(*names: str) -> float:
            for name in names:
                value = fnum(row.get(name))
                if math.isfinite(value):
                    if value > 1 and name in {"start", "end", "step"}:
                        value /= 100.0
                    return value
            return float("nan")

        start = take("start_dollars", "start", "start_price_dollars", "start_price")
        end = take("end_dollars", "end", "end_price_dollars", "end_price")
        step = take("step_dollars", "step", "step_size_dollars", "step_size")
        if not math.isfinite(start):
            start = 0.0
        if not math.isfinite(end):
            end = 1.0
        if 0 < step <= 1 and start <= end:
            parsed.append({"start": start, "end": end, "step": step})
    return parsed


def tick_for_price(market: dict[str, Any], price: float) -> float:
    for row in parse_price_ranges(market):
        if row["start"] - 1e-12 <= price <= row["end"] + 1e-12:
            return row["step"]
    if str(market.get("price_level_structure") or "") == "deci_cent":
        return 0.001
    return 0.01


def floor_to_grid(market: dict[str, Any], value: float) -> tuple[float, float]:
    p = min(max(value, 0.0), 1.0)
    for _ in range(12):
        tick = tick_for_price(market, p)
        rounded = math.floor((p + 1e-12) / tick) * tick
        rounded = round(rounded, 4)
        if abs(rounded - p) <= 1e-10:
            return rounded, tick
        p = rounded
    return p, tick_for_price(market, p)


def stressed_fee(quantity: int, price: float) -> float:
    if quantity <= 0 or not (0 <= price <= 1):
        return float("inf")
    raw = FEE_RATE_STRESS * quantity * price * (1.0 - price)
    rounded = math.ceil(raw * 10000.0 - 1e-12) / 10000.0
    return rounded + MAX_ROUNDING_CHARGE_PER_ORDER


def solve_no_bid(
    market: dict[str, Any],
    component_ask: float,
    quantity: int,
    target_margin: float,
) -> dict[str, float] | None:
    """Highest valid combo-NO bid retaining the frozen guaranteed margin."""
    if quantity <= 0 or not (0 < component_ask < 1) or target_margin < 0:
        return None
    raw_max = 1.0 - component_ask - target_margin
    no_bid, tick = floor_to_grid(market, raw_max)
    for _ in range(2000):
        if not (0 < no_bid < 1):
            return None
        fees = stressed_fee(quantity, component_ask) + stressed_fee(quantity, no_bid)
        guaranteed = quantity * (1.0 - component_ask - no_bid) - fees
        if guaranteed + 1e-12 >= quantity * target_margin:
            return {
                "yes_bid": 0.0,
                "no_bid": no_bid,
                "requester_yes_price": 1.0 - no_bid,
                "tick": tick,
                "fee_stress": fees,
                "guaranteed_net": guaranteed,
                "guaranteed_net_per_contract": guaranteed / quantity,
            }
        no_bid, tick = floor_to_grid(market, no_bid - tick)
    return None


@dataclass
class BookView:
    ticker: str
    valid: bool
    sid: int | None
    seq: int | None
    yes_bids: dict[float, float]
    no_bids: dict[float, float]
    local_wall_ns: int | None = None
    local_mono_ns: int | None = None
    exchange_ts_ms: int | None = None

    @classmethod
    def empty(cls, ticker: str) -> "BookView":
        return cls(ticker, False, None, None, {}, {})

    def apply_snapshot(self, sid: int | None, seq: int | None, msg: dict[str, Any], wall: int, mono: int) -> None:
        self.sid = sid
        self.seq = seq
        self.yes_bids = self._levels(msg.get("yes_dollars_fp") or msg.get("yes_dollars"))
        self.no_bids = self._levels(msg.get("no_dollars_fp") or msg.get("no_dollars"))
        self.local_wall_ns = wall
        self.local_mono_ns = mono
        self.exchange_ts_ms = _int_or_none(msg.get("ts_ms"))
        self.valid = True

    def apply_delta(self, sid: int | None, seq: int | None, msg: dict[str, Any], wall: int, mono: int) -> bool:
        contiguous = self.valid and self.sid == sid and self.seq is not None and seq == self.seq + 1
        if not contiguous:
            self.valid = False
            self.sid = sid
            self.seq = seq
            self.local_wall_ns = wall
            self.local_mono_ns = mono
            return False
        side = str(msg.get("side") or "").lower()
        price = fnum(msg.get("price_dollars") or msg.get("price"))
        delta = fnum(msg.get("delta_fp") or msg.get("delta"))
        ladder = self.yes_bids if side == "yes" else self.no_bids if side == "no" else None
        if ladder is None or not (0 < price < 1) or not math.isfinite(delta):
            self.valid = False
            return False
        new_qty = ladder.get(price, 0.0) + delta
        if new_qty <= 1e-12:
            ladder.pop(price, None)
        else:
            ladder[price] = new_qty
        self.seq = seq
        self.local_wall_ns = wall
        self.local_mono_ns = mono
        self.exchange_ts_ms = _int_or_none(msg.get("ts_ms"))
        return True

    @staticmethod
    def _levels(rows: Any) -> dict[float, float]:
        out: dict[float, float] = {}
        for row in rows or []:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            price, qty = fnum(row[0]), fnum(row[1])
            if 0 < price < 1 and qty > 0:
                out[price] = qty
        return out

    def acquisition(self, selected_side: str) -> dict[str, Any] | None:
        if not self.valid:
            return None
        if selected_side == "yes" and self.no_bids:
            opposite_bid = max(self.no_bids)
            return {
                "ask": 1.0 - opposite_bid,
                "depth": self.no_bids[opposite_bid],
                "opposite_book_side": "no",
                "opposite_bid": opposite_bid,
            }
        if selected_side == "no" and self.yes_bids:
            opposite_bid = max(self.yes_bids)
            return {
                "ask": 1.0 - opposite_bid,
                "depth": self.yes_bids[opposite_bid],
                "opposite_book_side": "yes",
                "opposite_bid": opposite_bid,
            }
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class JsonlAuditLog:
    def __init__(self, path: Path, static: dict[str, Any]) -> None:
        self.path = path
        self.static = static
        self.fh = gzip.open(path, "at", encoding="utf-8", compresslevel=6)
        self.count = 0
        self.lock = threading.Lock()

    def write(self, kind: str, payload: Any, *, wall_ns: int | None = None, mono_ns: int | None = None) -> None:
        if wall_ns is None or mono_ns is None:
            wall_ns, mono_ns = now_pair()
        raw = json_bytes(payload)
        record = {
            **self.static,
            "kind": kind,
            "local_wall_ns": wall_ns,
            "local_monotonic_ns": mono_ns,
            "payload_sha256": sha256_bytes(raw),
            "payload": payload,
        }
        with self.lock:
            self.fh.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False, default=str) + "\n")
            self.fh.flush()
            self.count += 1

    def close(self) -> None:
        with self.lock:
            self.fh.flush()
            self.fh.close()


class SignedRestClient:
    def __init__(self, base_url: str, key_id: str, private_key, audit: JsonlAuditLog) -> None:
        self.base_url = base_url.rstrip("/")
        self.key_id = key_id
        self.private_key = private_key
        self.audit = audit

    async def get(self, path: str, params: dict[str, Any] | None = None, *, retry_404: bool = False) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get_sync, path, params, retry_404)

    def _get_sync(self, path: str, params: dict[str, Any] | None, retry_404: bool) -> dict[str, Any] | None:
        full_url = self.base_url + path
        full_path = urlparse(full_url).path
        for attempt in range(8):
            send_wall, send_mono = now_pair()
            headers = signed_headers(self.key_id, self.private_key, "GET", full_path)
            try:
                response = requests.get(full_url, params=params, headers=headers, timeout=15)
                recv_wall, recv_mono = now_pair()
                body = response.content
                event = {
                    "method": "GET",
                    "path": path,
                    "params": params or {},
                    "attempt": attempt + 1,
                    "status": response.status_code,
                    "request_send_wall_ns": send_wall,
                    "request_send_monotonic_ns": send_mono,
                    "response_wall_ns": recv_wall,
                    "response_monotonic_ns": recv_mono,
                    "elapsed_ns": recv_mono - send_mono,
                    "response_sha256": sha256_bytes(body),
                    "response_text": body.decode("utf-8", errors="replace"),
                    "auth_headers_redacted": True,
                }
                self.audit.write("rest_response", event, wall_ns=recv_wall, mono_ns=recv_mono)
            except requests.RequestException as exc:
                recv_wall, recv_mono = now_pair()
                self.audit.write(
                    "rest_exception",
                    {
                        "method": "GET",
                        "path": path,
                        "params": params or {},
                        "attempt": attempt + 1,
                        "request_send_wall_ns": send_wall,
                        "request_send_monotonic_ns": send_mono,
                        "response_wall_ns": recv_wall,
                        "response_monotonic_ns": recv_mono,
                        "elapsed_ns": recv_mono - send_mono,
                        "error": repr(exc),
                    },
                    wall_ns=recv_wall,
                    mono_ns=recv_mono,
                )
                if attempt == 7:
                    return None
                time.sleep(min(4.0, 0.25 * (2**attempt)))
                continue

            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError:
                    return None
            if response.status_code == 429 or (retry_404 and response.status_code in {404, 410}):
                time.sleep(min(4.0, 0.25 * (2**attempt)))
                continue
            return None
        return None


class HBCDCollector:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.key_id = args.key_id or os.getenv("KALSHI_API_KEY_ID")
        if not self.key_id:
            raise SystemExit("Set KALSHI_API_KEY_ID or --key-id")
        self.private_key = load_private_key(
            args.private_key_path or os.getenv("KALSHI_PRIVATE_KEY_PATH"),
            args.private_key_pem or os.getenv("KALSHI_PRIVATE_KEY_PEM"),
        )
        self.run_id = str(uuid.uuid4())
        self.out_dir = Path(args.out).expanduser()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.raw_path = self.out_dir / f"hbcd_shadow_{self.run_id}.jsonl.gz"
        self.manifest_path = self.out_dir / f"hbcd_shadow_{self.run_id}.manifest.json"
        code_hash = sha256_bytes(Path(__file__).read_bytes())
        config = {
            "version": CONFIG_VERSION,
            "rest_base": args.rest_base,
            "ws_url": args.ws_url,
            "target_margin": args.target_margin,
            "decision_quantity": args.decision_quantity,
            "snapshot_timeout_s": args.snapshot_timeout_s,
            "max_book_age_ms": args.max_book_age_ms,
            "crypto15_series": sorted(CRYPTO15_SERIES),
            "fee_rate_stress": FEE_RATE_STRESS,
            "rounding_stress_per_order": MAX_ROUNDING_CHARGE_PER_ORDER,
            "read_only": True,
        }
        self.config_hash = sha256_bytes(json_bytes(config))
        static = {
            "run_id": self.run_id,
            "host": socket.gethostname(),
            "region": args.region or os.getenv("COLLECTOR_REGION"),
            "pid": os.getpid(),
            "platform": platform.platform(),
            "config_version": CONFIG_VERSION,
            "config_sha256": self.config_hash,
            "code_sha256": code_hash,
        }
        self.audit = JsonlAuditLog(self.raw_path, static)
        self.rest = SignedRestClient(args.rest_base, self.key_id, self.private_key, self.audit)
        self.books: dict[str, BookView] = {}
        self.snapshot_events: dict[str, asyncio.Event] = {}
        self.desired_tickers: set[str] = set(
            x.strip() for x in args.initial_tickers.split(",") if x.strip()
        )
        self.orderbook_sid: int | None = None
        self.orderbook_sid_event = asyncio.Event()
        self.orderbook_subscribe_pending = False
        self.ws = None
        self.ws_send_lock = asyncio.Lock()
        self.subscription_lock = asyncio.Lock()
        self.message_id = 100
        self.stop_event = asyncio.Event()
        self.received_messages = 0
        self.connection_count = 0
        self.rfq_tasks: set[asyncio.Task] = set()
        self.manifest = {
            **static,
            "started_wall_ns": time.time_ns(),
            "started_monotonic_ns": time.monotonic_ns(),
            "raw_file": self.raw_path.name,
            "config": config,
            "notes": [
                "Read-only: this process contains no quote, order, accept, confirm, cancel, POST, PUT, or DELETE action.",
                "Credentials and signatures are never persisted.",
                "Shadow decisions are not fills, PnL, or execution proof.",
            ],
        }
        self._write_manifest()

    def _write_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2, default=str), encoding="utf-8")

    def request_stop(self) -> None:
        self.stop_event.set()

    async def send_ws(self, payload: dict[str, Any]) -> None:
        if self.ws is None:
            raise RuntimeError("WebSocket unavailable")
        async with self.ws_send_lock:
            wall, mono = now_pair()
            raw = json.dumps(payload, separators=(",", ":"))
            await self.ws.send(raw)
            self.audit.write("ws_outbound", {"raw_text": raw, "parsed": payload}, wall_ns=wall, mono_ns=mono)

    def next_message_id(self) -> int:
        self.message_id += 1
        return self.message_id

    async def initial_subscribe(self) -> None:
        commands = [
            {"id": 1, "cmd": "subscribe", "params": {"channels": ["communications"]}},
            {"id": 2, "cmd": "subscribe", "params": {"channels": ["user_orders"]}},
            {"id": 3, "cmd": "subscribe", "params": {"channels": ["fill"]}},
            {"id": 4, "cmd": "subscribe", "params": {"channels": ["market_positions"]}},
            {"id": 5, "cmd": "subscribe", "params": {"channels": ["multivariate_market_lifecycle"]}},
        ]
        if self.desired_tickers:
            commands.append(
                {
                    "id": 6,
                    "cmd": "subscribe",
                    "params": {
                        "channels": ["orderbook_delta"],
                        "market_tickers": sorted(self.desired_tickers),
                    },
                }
            )
        for command in commands:
            await self.send_ws(command)

    async def ensure_orderbooks(self, tickers: list[str]) -> None:
        tickers = sorted(set(tickers))
        if not tickers:
            return
        async with self.subscription_lock:
            new = [t for t in tickers if t not in self.desired_tickers]
            self.desired_tickers.update(tickers)
            for ticker in tickers:
                self.books.setdefault(ticker, BookView.empty(ticker))
                self.snapshot_events.setdefault(ticker, asyncio.Event())

            if self.orderbook_sid is None:
                if not self.orderbook_sid_event.is_set() and not self.orderbook_subscribe_pending:
                    self.orderbook_subscribe_pending = True
                    await self.send_ws(
                        {
                            "id": self.next_message_id(),
                            "cmd": "subscribe",
                            "params": {
                                "channels": ["orderbook_delta"],
                                "market_tickers": sorted(self.desired_tickers),
                            },
                        }
                    )
                try:
                    await asyncio.wait_for(self.orderbook_sid_event.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    return
            elif new:
                await self.send_ws(
                    {
                        "id": self.next_message_id(),
                        "cmd": "update_subscription",
                        "params": {
                            "sids": [self.orderbook_sid],
                            "market_tickers": new,
                            "action": "add_markets",
                        },
                    }
                )

            if self.orderbook_sid is not None:
                await self.send_ws(
                    {
                        "id": self.next_message_id(),
                        "cmd": "update_subscription",
                        "params": {
                            "sids": [self.orderbook_sid],
                            "market_tickers": tickers,
                            "action": "get_snapshot",
                        },
                    }
                )

        waits = [self.snapshot_events[ticker].wait() for ticker in tickers]
        try:
            await asyncio.wait_for(asyncio.gather(*waits), timeout=self.args.snapshot_timeout_s)
        except asyncio.TimeoutError:
            pass

    async def process_rfq(self, inbound: dict[str, Any], recv_wall: int, recv_mono: int) -> None:
        msg = inbound.get("msg") or {}
        rfq_id = str(msg.get("rfq_id") or msg.get("id") or "")
        combo_ticker = str(msg.get("market_ticker") or "")
        if not rfq_id:
            self.audit.write("hbcd_skip", {"reason": "rfq_id_missing", "message": msg})
            return

        body = await self.rest.get(f"/communications/rfqs/{rfq_id}", retry_404=True)
        rfq = (body or {}).get("rfq") or msg
        combo_ticker = str(rfq.get("market_ticker") or combo_ticker)
        legs = rfq.get("mve_selected_legs") or []

        market_body = await self.rest.get(f"/markets/{combo_ticker}", retry_404=True) if combo_ticker else None
        market = (market_body or {}).get("market") or {}
        if not legs:
            legs = market.get("mve_selected_legs") or []

        base = {
            "rfq_id": rfq_id,
            "combo_ticker": combo_ticker,
            "rfq_received_wall_ns": recv_wall,
            "rfq_received_monotonic_ns": recv_mono,
            "rfq": rfq,
            "market": market,
        }
        if not combo_ticker or not legs:
            self.audit.write("hbcd_skip", {**base, "reason": "combo_metadata_missing"})
            return
        if not is_crypto15_combo(legs):
            self.audit.write("hbcd_skip", {**base, "reason": "not_crypto15_combo"})
            return

        contracts = fnum(rfq.get("contracts_fp"))
        if not math.isfinite(contracts) or contracts <= 0 or abs(contracts - round(contracts)) > 1e-9:
            self.audit.write(
                "hbcd_skip",
                {**base, "reason": "non_contract_or_target_cost_rfq", "contracts_fp": rfq.get("contracts_fp")},
            )
            return
        requested_q = int(round(contracts))
        component_tickers = [str(leg.get("market_ticker") or "") for leg in legs]
        component_tickers = [ticker for ticker in component_tickers if ticker]
        await self.ensure_orderbooks(component_tickers)

        candidates: list[dict[str, Any]] = []
        decision_wall, decision_mono = now_pair()
        for leg in legs:
            ticker = str(leg.get("market_ticker") or "")
            side = str(leg.get("side") or "").lower()
            book = self.books.get(ticker)
            acquired = book.acquisition(side) if book else None
            if not acquired:
                candidates.append(
                    {
                        "component_ticker": ticker,
                        "selected_side": side,
                        "eligible": False,
                        "reason": "invalid_or_empty_sequence_book",
                        "book": asdict(book) if book else None,
                    }
                )
                continue
            age_ms = (
                (decision_mono - book.local_mono_ns) / 1_000_000
                if book and book.local_mono_ns is not None
                else float("inf")
            )
            candidates.append(
                {
                    "component_ticker": ticker,
                    "selected_side": side,
                    "eligible": age_ms <= self.args.max_book_age_ms,
                    "reason": None if age_ms <= self.args.max_book_age_ms else "book_too_old",
                    "ask": acquired["ask"],
                    "displayed_depth": acquired["depth"],
                    "opposite_book_side": acquired["opposite_book_side"],
                    "opposite_bid": acquired["opposite_bid"],
                    "book_age_ms": age_ms,
                    "book_sid": book.sid,
                    "book_seq": book.seq,
                    "book_exchange_ts_ms": book.exchange_ts_ms,
                    "book_local_wall_ns": book.local_wall_ns,
                    "book_local_monotonic_ns": book.local_mono_ns,
                }
            )

        eligible = [x for x in candidates if x.get("eligible")]
        if not eligible:
            self.audit.write(
                "hbcd_shadow_decision",
                {
                    **base,
                    "decision": "skip",
                    "reason": "no_valid_component_book",
                    "requested_q": requested_q,
                    "candidates": candidates,
                    "decision_wall_ns": decision_wall,
                    "decision_monotonic_ns": decision_mono,
                    "decision_latency_ms": (decision_mono - recv_mono) / 1_000_000,
                },
                wall_ns=decision_wall,
                mono_ns=decision_mono,
            )
            return

        eligible.sort(key=lambda x: (x["ask"], x["component_ticker"], x["selected_side"]))
        chosen = eligible[0]
        quantity = min(requested_q, self.args.decision_quantity)
        qspec = solve_no_bid(market, chosen["ask"], quantity, self.args.target_margin)
        reasons: list[str] = []
        if chosen["displayed_depth"] + 1e-12 < quantity:
            reasons.append("insufficient_displayed_component_depth")
        if qspec is None:
            reasons.append("no_price_clears_margin_and_fee_stress")

        decision = "would_quote" if not reasons else "skip"
        record = {
            **base,
            "decision": decision,
            "skip_reasons": reasons,
            "read_only": True,
            "requested_q": requested_q,
            "shadow_q": quantity,
            "target_margin": self.args.target_margin,
            "chosen_component": chosen,
            "all_component_candidates": candidates,
            "shadow_quote": qspec,
            "economic_side": "buy selected component + buy combo NO",
            "decision_wall_ns": decision_wall,
            "decision_monotonic_ns": decision_mono,
            "decision_latency_ms": (decision_mono - recv_mono) / 1_000_000,
            "config_sha256": self.config_hash,
        }
        self.audit.write("hbcd_shadow_decision", record, wall_ns=decision_wall, mono_ns=decision_mono)

    def launch_rfq_task(self, parsed: dict[str, Any], wall: int, mono: int) -> None:
        task = asyncio.create_task(self.process_rfq(parsed, wall, mono))
        self.rfq_tasks.add(task)
        task.add_done_callback(self.rfq_tasks.discard)

    def handle_book_message(self, parsed: dict[str, Any], wall: int, mono: int) -> None:
        msg_type = parsed.get("type")
        msg = parsed.get("msg") or {}
        ticker = str(msg.get("market_ticker") or "")
        if not ticker:
            return
        book = self.books.setdefault(ticker, BookView.empty(ticker))
        sid = _int_or_none(parsed.get("sid"))
        seq = _int_or_none(parsed.get("seq"))
        if msg_type == "orderbook_snapshot":
            book.apply_snapshot(sid, seq, msg, wall, mono)
            self.snapshot_events.setdefault(ticker, asyncio.Event()).set()
        elif msg_type == "orderbook_delta":
            if not book.apply_delta(sid, seq, msg, wall, mono):
                self.audit.write(
                    "orderbook_sequence_gap",
                    {"ticker": ticker, "sid": sid, "seq": seq, "current_seq": book.seq},
                    wall_ns=wall,
                    mono_ns=mono,
                )
                self.snapshot_events.setdefault(ticker, asyncio.Event()).clear()

    async def handle_inbound(self, raw: str | bytes) -> None:
        wall, mono = now_pair()
        raw_text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        try:
            parsed = json.loads(raw_text)
            parse_error = None
        except Exception as exc:
            parsed = None
            parse_error = repr(exc)
        self.audit.write(
            "ws_inbound",
            {"raw_text": raw_text, "parsed": parsed, "parse_error": parse_error},
            wall_ns=wall,
            mono_ns=mono,
        )
        self.received_messages += 1
        if not isinstance(parsed, dict):
            return

        msg_type = str(parsed.get("type") or "")
        if msg_type == "subscribed":
            msg = parsed.get("msg") or {}
            if msg.get("channel") == "orderbook_delta":
                self.orderbook_sid = _int_or_none(msg.get("sid") or parsed.get("sid"))
                self.orderbook_subscribe_pending = False
                if self.orderbook_sid is not None:
                    self.orderbook_sid_event.set()
            return
        if msg_type in {"orderbook_snapshot", "orderbook_delta"}:
            self.handle_book_message(parsed, wall, mono)
            return
        if msg_type == "rfq_created":
            self.launch_rfq_task(parsed, wall, mono)
            return
        if msg_type == "quote_accepted":
            self.audit.write(
                "hbcd_quote_acceptance_observed",
                {
                    "message": parsed.get("msg") or {},
                    "note": "Lifecycle evidence only; this collector never created a quote.",
                },
                wall_ns=wall,
                mono_ns=mono,
            )

    async def connect_once(self) -> None:
        headers = signed_headers(self.key_id, self.private_key, "GET", WS_SIGN_PATH)
        self.connection_count += 1
        connection_id = str(uuid.uuid4())
        self.audit.write(
            "ws_connect_attempt",
            {
                "connection_id": connection_id,
                "url": self.args.ws_url,
                "attempt": self.connection_count,
                "auth_headers_redacted": True,
            },
        )
        kwargs = dict(ping_interval=20, ping_timeout=20, max_size=None, close_timeout=5)
        try:
            ws = await websockets.connect(self.args.ws_url, additional_headers=headers, **kwargs)
        except TypeError:
            ws = await websockets.connect(self.args.ws_url, extra_headers=headers, **kwargs)
        self.ws = ws
        self.orderbook_sid = None
        self.orderbook_subscribe_pending = False
        self.orderbook_sid_event.clear()
        for event in self.snapshot_events.values():
            event.clear()
        for book in self.books.values():
            book.valid = False
        self.audit.write("ws_connected", {"connection_id": connection_id, "url": self.args.ws_url})
        try:
            await self.initial_subscribe()
            async for raw in ws:
                await self.handle_inbound(raw)
                if self.args.max_messages and self.received_messages >= self.args.max_messages:
                    self.stop_event.set()
                    break
                if self.stop_event.is_set():
                    break
        finally:
            await ws.close()
            self.ws = None
            self.audit.write("ws_disconnected", {"connection_id": connection_id})

    async def run(self) -> None:
        backoff = 1.0
        reconnects = 0
        try:
            while not self.stop_event.is_set():
                try:
                    await self.connect_once()
                    backoff = 1.0
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.audit.write("collector_connection_error", {"error": repr(exc), "reconnect": reconnects})
                if self.stop_event.is_set():
                    break
                reconnects += 1
                if self.args.max_reconnects >= 0 and reconnects > self.args.max_reconnects:
                    break
                try:
                    await asyncio.wait_for(self.stop_event.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(30.0, backoff * 2)
        finally:
            if self.rfq_tasks:
                await asyncio.gather(*list(self.rfq_tasks), return_exceptions=True)
            self.manifest.update(
                {
                    "finished_wall_ns": time.time_ns(),
                    "finished_monotonic_ns": time.monotonic_ns(),
                    "messages": self.received_messages,
                    "connections": self.connection_count,
                    "records": self.audit.count,
                }
            )
            self.audit.close()
            self.manifest["raw_sha256"] = sha256_bytes(self.raw_path.read_bytes())
            self._write_manifest()
            print(
                json.dumps(
                    {
                        "manifest": str(self.manifest_path),
                        "raw": str(self.raw_path),
                        "messages": self.received_messages,
                        "records": self.manifest["records"],
                        "sha256": self.manifest["raw_sha256"],
                    },
                    indent=2,
                )
            )


def self_test() -> None:
    binary_market = {"price_level_structure": "linear_cent", "price_ranges": []}
    assert is_crypto15_combo(
        [
            {"event_ticker": "KXBTC15M-26AUG130000", "market_ticker": "KXBTC15M-X", "side": "yes"},
            {"event_ticker": "KXETH15M-26AUG130000", "market_ticker": "KXETH15M-X", "side": "no"},
        ]
    )
    for values in ([0.0, 0.0], [0.2, 0.9], [1.0, 1.0], [0.4, 0.5, 0.7]):
        combo = math.prod(values)
        for selected in values:
            assert selected + 1.0 - combo >= 1.0 - 1e-12

    book = BookView.empty("X")
    book.apply_snapshot(
        4,
        10,
        {"market_ticker": "X", "yes_dollars_fp": [["0.70", "12"]], "no_dollars_fp": [["0.25", "8"]]},
        100,
        200,
    )
    yes = book.acquisition("yes")
    no = book.acquisition("no")
    assert yes and abs(yes["ask"] - 0.75) < 1e-12 and yes["depth"] == 8
    assert no and abs(no["ask"] - 0.30) < 1e-12 and no["depth"] == 12
    assert book.apply_delta(4, 11, {"side": "no", "price_dollars": "0.25", "delta_fp": "-3"}, 101, 201)
    assert book.no_bids[0.25] == 5
    assert not book.apply_delta(4, 13, {"side": "no", "price_dollars": "0.25", "delta_fp": "1"}, 102, 202)

    quote = solve_no_bid(binary_market, component_ask=0.70, quantity=1, target_margin=0.01)
    assert quote is not None
    assert quote["guaranteed_net"] >= 0.01 - 1e-12
    assert quote["no_bid"] < 0.30
    print(json.dumps({"self_test": "PASS", "quote": quote}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="hbcd_tier_a_shadow")
    ap.add_argument("--rest-base", default=DEFAULT_REST_BASE)
    ap.add_argument("--ws-url", default=DEFAULT_WS_URL)
    ap.add_argument("--key-id")
    ap.add_argument("--private-key-path")
    ap.add_argument("--private-key-pem")
    ap.add_argument("--region")
    ap.add_argument("--initial-tickers", default="")
    ap.add_argument("--target-margin", type=float, default=DEFAULT_MARGIN)
    ap.add_argument("--decision-quantity", type=int, default=1)
    ap.add_argument("--snapshot-timeout-s", type=float, default=1.5)
    ap.add_argument("--max-book-age-ms", type=float, default=500.0)
    ap.add_argument("--max-messages", type=int, default=0)
    ap.add_argument("--max-reconnects", type=int, default=-1)
    ap.add_argument("--self-test", action="store_true")
    return ap


async def async_main(args: argparse.Namespace) -> None:
    collector = HBCDCollector(args)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, collector.request_stop)
        except (NotImplementedError, RuntimeError):
            pass
    await collector.run()


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        self_test()
        return
    if args.decision_quantity <= 0:
        raise SystemExit("--decision-quantity must be positive")
    if args.target_margin < 0:
        raise SystemExit("--target-margin must be nonnegative")
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
