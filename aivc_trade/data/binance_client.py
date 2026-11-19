"""Binance REST client – candles, account, orders (incl. exchange-side stops).

Design notes (M2):

* The API host is injected via ``base_url``; callers resolve it from config with
  :func:`aivc_trade.config.loader.resolve_base_url`.  Nothing here hardcodes a
  venue, so prod / testnet switch is a config concern.
* Signed requests sign the exact query string that is sent (signature appended
  as a raw query parameter) instead of relying on the HTTP library to re-encode
  the parameters — that is the classic source of ``-1022`` signature errors.
* Server-time offset is tracked (``sync_time``) so ``timestamp`` does not drift
  against the exchange clock; a ``-1021`` (timestamp outside recvWindow) response
  triggers one resync + retry.
* Quantity / price formatting is derived from ``exchangeInfo`` filters so that
  values never carry more precision than LOT_SIZE.stepSize / PRICE_FILTER.tickSize
  and never fall into scientific notation.
"""

from __future__ import annotations

import hashlib
import hmac
import time
import urllib.parse
from datetime import datetime, timezone
from decimal import ROUND_DOWN, ROUND_HALF_UP, ROUND_UP, Decimal
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from aivc_trade.core.logger import get_logger

log = get_logger("binance_client")

BASE_URL = "https://api.binance.com"
DEFAULT_RECV_WINDOW = 5000
DEFAULT_TIMEOUT = 10.0
# Re-sync the server clock at most this often (seconds).
TIME_SYNC_INTERVAL = 30 * 60

# Order types accepted by place_stop_order / cancel_replace_order.
_STOP_ORDER_TYPES = ("STOP_LOSS", "STOP_LOSS_LIMIT", "TAKE_PROFIT", "TAKE_PROFIT_LIMIT")
_CANCEL_REPLACE_MODES = ("STOP_ON_FAILURE", "ALLOW_FAILURE")
_TIMESTAMP_OUT_OF_RANGE = -1021
_UNKNOWN_ORDER = -2011
_ORDER_NOT_FOUND = -2013

# Interval string → milliseconds
_INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


class BinanceAPIError(RuntimeError):
    """Raised when Binance returns an error payload (``{"code": ..., "msg": ...}``)."""

    def __init__(
        self,
        code: Optional[int],
        msg: Optional[str],
        status: Optional[int] = None,
        method: Optional[str] = None,
        path: Optional[str] = None,
    ) -> None:
        self.code = code
        self.msg = msg or ""
        self.status = status
        self.method = method
        self.path = path
        location = f"{method} {path}" if method and path else "request"
        super().__init__(f"Binance API error {code}: {self.msg} ({location}, HTTP {status})")


# ----------------------------------------------------------------------
# Step / precision helpers (pure functions – unit-testable without HTTP)
# ----------------------------------------------------------------------

def step_decimals(step: float) -> int:
    """Number of decimal places implied by *step* (0.001 → 3, 1 → 0)."""
    if step is None or step <= 0:
        return 0
    exponent = Decimal(str(step)).normalize().as_tuple().exponent
    return max(0, -int(exponent))


def round_to_step(value: float, step: float, mode: str = "down") -> float:
    """Round *value* to a multiple of *step*.

    ``mode``: ``down`` (never exceed the value – quantities), ``up`` (never
    below – conservative sell stops) or ``nearest`` (prices).
    """
    if step is None or step <= 0:
        return float(value)
    rounding = {
        "down": ROUND_DOWN,
        "up": ROUND_UP,
        "nearest": ROUND_HALF_UP,
    }.get(mode)
    if rounding is None:
        raise ValueError(f"Unsupported rounding mode: {mode!r}")

    units = Decimal(str(value)) / Decimal(str(step))
    return float(units.to_integral_value(rounding=rounding) * Decimal(str(step)))


def format_step(value: float, step: float, mode: str = "down") -> str:
    """Format *value* rounded to *step* with exactly ``step_decimals`` places."""
    rounded = round_to_step(value, step, mode=mode)
    return f"{rounded:.{step_decimals(step)}f}"


def format_float(value: float) -> str:
    """Format a float without scientific notation (fallback when filters are unknown)."""
    text = f"{float(value):.16f}".rstrip("0").rstrip(".")
    return text or "0"


class BinanceClient:
    """Thin wrapper around Binance public + authenticated REST endpoints."""

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        base_url: str = BASE_URL,
        recv_window: int = DEFAULT_RECV_WINDOW,
        timeout: float = DEFAULT_TIMEOUT,
        session: Optional[Any] = None,
        time_fn: Optional[Any] = None,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = (base_url or BASE_URL).rstrip("/")
        self.recv_window = int(recv_window)
        self.timeout = float(timeout)
        self._session = session if session is not None else requests.Session()
        self._time_fn = time_fn or time.time
        self._time_offset_ms = 0
        self._time_synced_at: Optional[float] = None
        self._symbol_info_cache: Dict[str, Dict[str, Any]] = {}
        self._filters_cache: Dict[str, Dict[str, Dict[str, float]]] = {}

        headers = getattr(self._session, "headers", None)
        if api_key and headers is not None:
            headers.update({"X-MBX-APIKEY": api_key})

    # ------------------------------------------------------------------
    # Time synchronisation
    # ------------------------------------------------------------------
    def sync_time(self) -> int:
        """Fetch server time and store the local↔server offset in ms."""
        data = self._public_get("/api/v3/time")
        server_ms = int(data["serverTime"])
        self._time_offset_ms = server_ms - int(self._time_fn() * 1000)
        self._time_synced_at = self._time_fn()
        log.debug(f"Time synced: offset={self._time_offset_ms}ms")
        return self._time_offset_ms

    def server_timestamp(self) -> int:
        """Current exchange-side timestamp in ms (local time + measured offset)."""
        return int(self._time_fn() * 1000) + self._time_offset_ms

    def _ensure_time_synced(self) -> None:
        stale = (
            self._time_synced_at is None
            or (self._time_fn() - self._time_synced_at) > TIME_SYNC_INTERVAL
        )
        if not stale:
            return
        try:
            self.sync_time()
        except Exception as e:  # never block trading on a clock probe
            log.warning(f"Time sync failed, using local clock: {e}")
            self._time_synced_at = self._time_fn()

    # ------------------------------------------------------------------
    # Public: Klines
    # ------------------------------------------------------------------
    def fetch_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 500,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch klines and return a DataFrame with OHLCV columns."""
        params: Dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms

        raw: List[list] = self._public_get("/api/v3/klines", params)
        return self._parse_klines(raw)

    def fetch_klines_full(
        self,
        symbol: str,
        interval: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        """Paginated fetch from *start_dt* to *end_dt* (inclusive)."""
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)
        limit = 1000
        all_frames: List[pd.DataFrame] = []
        cursor = start_ms

        while cursor < end_ms:
            df = self.fetch_klines(
                symbol, interval, limit=limit, start_ms=cursor, end_ms=end_ms
            )
            if df.empty:
                break
            all_frames.append(df)
            last_ts = int(df["ts"].iloc[-1].timestamp() * 1000)
            cursor = last_ts + _INTERVAL_MS.get(interval, 3_600_000)
            time.sleep(0.15)  # rate-limit courtesy

        if not all_frames:
            return pd.DataFrame()
        result = pd.concat(all_frames, ignore_index=True).drop_duplicates(
            subset=["ts"]
        )
        return result.sort_values("ts").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Public: Exchange info (lot size / tick size filters)
    # ------------------------------------------------------------------
    def get_symbol_info(self, symbol: str, refresh: bool = False) -> Dict[str, Any]:
        """Return the raw exchangeInfo entry for *symbol* (cached per process)."""
        if not refresh and symbol in self._symbol_info_cache:
            return self._symbol_info_cache[symbol]

        data = self._public_get("/api/v3/exchangeInfo", {"symbol": symbol})
        for s in data.get("symbols", []):
            if s["symbol"] == symbol:
                filters: Dict[str, Dict[str, float]] = {}
                for f in s.get("filters", []):
                    parsed: Dict[str, float] = {}
                    for key, value in f.items():
                        if key == "filterType":
                            continue
                        try:
                            parsed[key] = float(value)
                        except (TypeError, ValueError):
                            continue
                    filters[f["filterType"]] = parsed
                self._symbol_info_cache[symbol] = s
                self._filters_cache[symbol] = filters
                return s
        raise ValueError(f"Symbol {symbol} not found in exchangeInfo")

    def get_lot_size(self, symbol: str) -> Dict[str, float]:
        """Return {minQty, maxQty, stepSize, minNotional} for *symbol*.

        ``minNotional`` comes from the NOTIONAL filter (older exchanges expose
        MIN_NOTIONAL); it is 0.0 when the venue does not publish one.
        """
        filters = self._get_filters(symbol)
        lot = filters.get("LOT_SIZE")
        if lot is None:
            raise ValueError(f"LOT_SIZE filter not found for {symbol}")
        notional = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL") or {}
        return {
            "minQty": float(lot.get("minQty", 0.0)),
            "maxQty": float(lot.get("maxQty", 0.0)),
            "stepSize": float(lot.get("stepSize", 0.0)),
            "minNotional": float(notional.get("minNotional", 0.0)),
        }

    def get_price_tick(self, symbol: str) -> float:
        """Return PRICE_FILTER.tickSize for *symbol* (0.0 when unconstrained)."""
        price_filter = self._get_filters(symbol).get("PRICE_FILTER")
        if price_filter is None:
            return 0.0
        return float(price_filter.get("tickSize", 0.0))

    def _get_filters(self, symbol: str) -> Dict[str, Dict[str, float]]:
        if symbol not in self._filters_cache:
            self.get_symbol_info(symbol)
        return self._filters_cache[symbol]

    # ------------------------------------------------------------------
    # Public: Ticker (latest price)
    # ------------------------------------------------------------------
    def get_ticker_price(self, symbol: str) -> float:
        data = self._public_get("/api/v3/ticker/price", {"symbol": symbol})
        return float(data["price"])

    # ------------------------------------------------------------------
    # Quantity / price formatting
    # ------------------------------------------------------------------
    def format_quantity(self, symbol: str, qty: float) -> str:
        """Format *qty* down to the symbol's LOT_SIZE stepSize."""
        try:
            step = self.get_lot_size(symbol)["stepSize"]
        except Exception as e:
            log.warning(f"LOT_SIZE unavailable for {symbol}, falling back: {e}")
            return format_float(qty)
        return format_step(qty, step, mode="down")

    def format_price(self, symbol: str, price: float, mode: str = "nearest") -> str:
        """Format *price* to the symbol's PRICE_FILTER tickSize."""
        try:
            tick = self.get_price_tick(symbol)
        except Exception as e:
            log.warning(f"PRICE_FILTER unavailable for {symbol}, falling back: {e}")
            return format_float(price)
        return format_step(price, tick, mode=mode)

    # ------------------------------------------------------------------
    # Authenticated: Account
    # ------------------------------------------------------------------
    def get_account_balance(self, asset: str = "JPY") -> float:
        """Return free balance for *asset*."""
        resp = self._signed_request("GET", "/api/v3/account")
        for b in resp.get("balances", []):
            if b["asset"] == asset:
                return float(b["free"])
        return 0.0

    def get_spot_balance(self, asset: str) -> float:
        """Return total (free + locked) for *asset*."""
        resp = self._signed_request("GET", "/api/v3/account")
        for b in resp.get("balances", []):
            if b["asset"] == asset:
                return float(b["free"]) + float(b["locked"])
        return 0.0

    # ------------------------------------------------------------------
    # Authenticated: Orders
    # ------------------------------------------------------------------
    def place_market_order(
        self, symbol: str, side: str, qty: float
    ) -> Dict[str, Any]:
        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": self.format_quantity(symbol, qty),
        }
        return self._signed_request("POST", "/api/v3/order", params)

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        time_in_force: str = "GTC",
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        params = {
            "symbol": symbol,
            "side": side,
            "type": "LIMIT",
            "timeInForce": time_in_force,
            "quantity": self.format_quantity(symbol, qty),
            "price": self.format_price(symbol, price),
        }
        if client_order_id:
            params["newClientOrderId"] = client_order_id
        return self._signed_request("POST", "/api/v3/order", params)

    def place_stop_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        order_type: str = "STOP_LOSS",
        price: Optional[float] = None,
        time_in_force: str = "GTC",
        client_order_id: Optional[str] = None,
        stop_limit_offset_pct: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Place an exchange-side stop (STOP_LOSS or STOP_LOSS_LIMIT).

        ``STOP_LOSS`` is a stop-market order and takes no price / timeInForce.
        For ``STOP_LOSS_LIMIT`` either *price* or *stop_limit_offset_pct* (limit
        placed that far below the stop for a sell) must be supplied.
        """
        order_type = order_type.upper()
        if order_type not in _STOP_ORDER_TYPES:
            raise ValueError(f"Unsupported stop order type: {order_type!r}")

        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": self.format_quantity(symbol, qty),
            "stopPrice": self.format_price(symbol, stop_price, mode="up"),
        }
        if order_type.endswith("_LIMIT"):
            if price is None:
                if stop_limit_offset_pct is None:
                    raise ValueError(
                        f"{order_type} requires a price or stop_limit_offset_pct"
                    )
                price = stop_price * (1 - float(stop_limit_offset_pct) / 100.0)
            params["price"] = self.format_price(symbol, price, mode="down")
            params["timeInForce"] = time_in_force
        if client_order_id:
            params["newClientOrderId"] = client_order_id
        return self._signed_request("POST", "/api/v3/order", params)

    def get_order(
        self,
        symbol: str,
        order_id: Optional[int | str] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query one order (by exchange id or client id)."""
        if order_id is None and client_order_id is None:
            raise ValueError("get_order requires order_id or client_order_id")
        params: Dict[str, Any] = {"symbol": symbol}
        if order_id is not None:
            params["orderId"] = order_id
        else:
            params["origClientOrderId"] = client_order_id
        return self._signed_request("GET", "/api/v3/order", params)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        return self._signed_request("GET", "/api/v3/openOrders", params)

    def cancel_order(
        self,
        symbol: str,
        order_id: Optional[int | str] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Cancel one order (by exchange id or client id)."""
        if order_id is None and client_order_id is None:
            raise ValueError("cancel_order requires order_id or client_order_id")
        params: Dict[str, Any] = {"symbol": symbol}
        if order_id is not None:
            params["orderId"] = order_id
        else:
            params["origClientOrderId"] = client_order_id
        return self._signed_request("DELETE", "/api/v3/order", params)

    def cancel_replace_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        order_type: str = "STOP_LOSS",
        price: Optional[float] = None,
        time_in_force: str = "GTC",
        cancel_order_id: Optional[int | str] = None,
        cancel_client_order_id: Optional[str] = None,
        new_client_order_id: Optional[str] = None,
        mode: str = "STOP_ON_FAILURE",
        stop_limit_offset_pct: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Atomically cancel the working stop and place a new one.

        ``mode=STOP_ON_FAILURE`` keeps the existing order when the replacement
        fails to validate, which is what a stop ratchet wants.  Returns the
        ``data`` payload (the new order) on success.
        """
        if cancel_order_id is None and cancel_client_order_id is None:
            raise ValueError(
                "cancel_replace_order requires cancel_order_id or cancel_client_order_id"
            )
        mode = mode.upper()
        if mode not in _CANCEL_REPLACE_MODES:
            raise ValueError(f"Unsupported cancelReplaceMode: {mode!r}")

        order_type = order_type.upper()
        if order_type not in _STOP_ORDER_TYPES:
            raise ValueError(f"Unsupported stop order type: {order_type!r}")

        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": self.format_quantity(symbol, qty),
            "stopPrice": self.format_price(symbol, stop_price, mode="up"),
            "cancelReplaceMode": mode,
        }
        if order_type.endswith("_LIMIT"):
            if price is None:
                if stop_limit_offset_pct is None:
                    raise ValueError(
                        f"{order_type} requires a price or stop_limit_offset_pct"
                    )
                price = stop_price * (1 - float(stop_limit_offset_pct) / 100.0)
            params["price"] = self.format_price(symbol, price, mode="down")
            params["timeInForce"] = time_in_force
        if cancel_order_id is not None:
            params["cancelOrderId"] = cancel_order_id
        else:
            params["cancelOrigClientOrderId"] = cancel_client_order_id
        if new_client_order_id:
            params["newClientOrderId"] = new_client_order_id

        resp = self._signed_request("POST", "/api/v3/order/cancelReplace", params)
        # Success envelope: {"code": 200, "msg": "...", "data": {...}}.
        # A 200 response may still carry an error code when the cancel succeeded
        # but the replacement was rejected (ALLOW_FAILURE).
        if isinstance(resp, dict) and resp.get("code") not in (None, 200, "200"):
            raise BinanceAPIError(
                resp.get("code"), resp.get("msg"), 200, "POST", "/api/v3/order/cancelReplace"
            )
        if isinstance(resp, dict) and "data" in resp:
            return resp["data"]
        return resp

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _public_get(
        self, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        resp = self._session.get(
            f"{self.base_url}{path}", params=params, timeout=self.timeout
        )
        return self._handle_response(resp, "GET", path)

    def _signed_request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        _retry: bool = True,
    ) -> Any:
        """Sign and send an authenticated request.

        The signature covers exactly the query string that is transmitted
        (parameters keep their insertion order, ``timestamp``/``recvWindow``
        appended, ``signature`` last).
        """
        self._ensure_time_synced()

        payload: Dict[str, Any] = dict(params or {})
        payload["timestamp"] = self.server_timestamp()
        payload["recvWindow"] = self.recv_window
        query = urllib.parse.urlencode(payload, doseq=True)
        signature = hmac.new(
            self.api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        url = f"{self.base_url}{path}?{query}&signature={signature}"

        resp = self._session.request(method, url, timeout=self.timeout)
        try:
            return self._handle_response(resp, method, path)
        except BinanceAPIError as e:
            if e.code == _TIMESTAMP_OUT_OF_RANGE and _retry:
                log.warning("Timestamp outside recvWindow – resyncing clock and retrying")
                self.sync_time()
                return self._signed_request(method, path, params, _retry=False)
            raise

    @staticmethod
    def _handle_response(resp: Any, method: str, path: str) -> Any:
        status = int(getattr(resp, "status_code", 0) or 0)
        try:
            data = resp.json()
        except Exception:
            data = None

        if 200 <= status < 300:
            return data

        code = data.get("code") if isinstance(data, dict) else None
        msg = data.get("msg") if isinstance(data, dict) else None
        if msg is None:
            msg = getattr(resp, "text", "") or "request failed"
        raise BinanceAPIError(code, msg, status, method, path)

    @staticmethod
    def _parse_klines(raw: List[list]) -> pd.DataFrame:
        rows = []
        for k in raw:
            rows.append(
                {
                    "ts": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                }
            )
        return pd.DataFrame(rows)
