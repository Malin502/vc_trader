"""Binance REST / WebSocket client for candle (kline) retrieval."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
try:
    from binance.client import Client as BinanceSDKClient
except Exception:  # pragma: no cover - fallback when package is unavailable
    BinanceSDKClient = None

from aivc_trade.core.logger import get_logger

log = get_logger("binance_client")

BASE_URL = "https://api.binance.com"

# Interval string → milliseconds
_INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def _binance_dt_str(dt: datetime) -> str:
    """Convert datetime to Binance-compatible UTC datetime string."""
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.strftime("%d %b, %Y %H:%M:%S UTC")


class BinanceClient:
    """Thin wrapper around Binance public + authenticated REST endpoints."""

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        base_url: str = BASE_URL,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self._session = requests.Session()
        self._session.timeout = 20  # type: ignore[attr-defined]
        if api_key:
            self._session.headers.update({"X-MBX-APIKEY": api_key})
        self._public_client = None
        if BinanceSDKClient is not None:
            self._public_client = BinanceSDKClient(
                api_key=api_key or None,
                api_secret=api_secret or None,
                requests_params={"timeout": 20},
            )

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

        if self._public_client is not None:
            raw = self._public_client.get_klines(**params)
            return self._parse_klines(raw)

        resp = self._session.get(f"{self.base_url}/api/v3/klines", params=params)
        resp.raise_for_status()
        raw: List[list] = resp.json()
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

        if self._public_client is not None:
            raw = self._public_client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=_binance_dt_str(start_dt),
                end_str=_binance_dt_str(end_dt),
            )
            sdk_df = self._parse_klines(raw)
            interval_ms = _INTERVAL_MS.get(interval, 3_600_000)
            expected_bars = max(1, ((end_ms - start_ms) // interval_ms) + 1)
            if not (expected_bars > limit and len(sdk_df) <= limit):
                sdk_df = sdk_df.drop_duplicates(subset=["ts"]).sort_values("ts")
                sdk_df = sdk_df[(sdk_df["ts"] >= start_dt) & (sdk_df["ts"] <= end_dt)]
                return sdk_df.reset_index(drop=True)
            log.warning(
                f"get_historical_klines returned only {len(sdk_df)} rows for {symbol} {interval}; "
                "falling back to paginated /api/v3/klines fetch."
            )

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
        result = pd.concat(all_frames, ignore_index=True)

        result = result.drop_duplicates(subset=["ts"]).sort_values("ts")
        result = result[(result["ts"] >= start_dt) & (result["ts"] <= end_dt)]
        return result.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Public: Exchange info (lot size filters)
    # ------------------------------------------------------------------
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Return filters (LOT_SIZE, MIN_NOTIONAL, etc.) for *symbol*."""
        resp = self._session.get(
            f"{self.base_url}/api/v3/exchangeInfo", params={"symbol": symbol}
        )
        resp.raise_for_status()
        data = resp.json()
        for s in data.get("symbols", []):
            if s["symbol"] == symbol:
                return s
        raise ValueError(f"Symbol {symbol} not found in exchangeInfo")

    def get_lot_size(self, symbol: str) -> Dict[str, float]:
        """Return {minQty, maxQty, stepSize} for *symbol*."""
        info = self.get_symbol_info(symbol)
        for f in info.get("filters", []):
            if f["filterType"] == "LOT_SIZE":
                return {
                    "minQty": float(f["minQty"]),
                    "maxQty": float(f["maxQty"]),
                    "stepSize": float(f["stepSize"]),
                }
        raise ValueError(f"LOT_SIZE filter not found for {symbol}")

    # ------------------------------------------------------------------
    # Public: Ticker (latest price)
    # ------------------------------------------------------------------
    def get_ticker_price(self, symbol: str) -> float:
        resp = self._session.get(
            f"{self.base_url}/api/v3/ticker/price", params={"symbol": symbol}
        )
        resp.raise_for_status()
        return float(resp.json()["price"])

    # ------------------------------------------------------------------
    # Authenticated: Account
    # ------------------------------------------------------------------
    def get_account_balance(self, asset: str = "USDC") -> float:
        """Return free balance for *asset*."""
        resp = self._signed_get("/api/v3/account")
        for b in resp.get("balances", []):
            if b["asset"] == asset:
                return float(b["free"])
        return 0.0

    def get_spot_balance(self, asset: str) -> float:
        """Return total (free + locked) for *asset*."""
        resp = self._signed_get("/api/v3/account")
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
            "quantity": f"{qty}",
        }
        return self._signed_post("/api/v3/order", params)

    def place_limit_order(
        self, symbol: str, side: str, qty: float, price: float
    ) -> Dict[str, Any]:
        params = {
            "symbol": symbol,
            "side": side,
            "type": "LIMIT",
            "timeInForce": "GTC",
            "quantity": f"{qty}",
            "price": f"{price}",
        }
        return self._signed_post("/api/v3/order", params)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _signed_get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._signed_request("GET", path, params or {})

    def _signed_post(self, path: str, params: Dict[str, Any]) -> Any:
        return self._signed_request("POST", path, params)

    def _signed_request(
        self, method: str, path: str, params: Dict[str, Any]
    ) -> Any:
        import hashlib
        import hmac
        import urllib.parse

        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000
        query = urllib.parse.urlencode(params)
        signature = hmac.new(
            self.api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        params["signature"] = signature

        if method == "GET":
            resp = self._session.get(f"{self.base_url}{path}", params=params)
        else:
            resp = self._session.post(f"{self.base_url}{path}", params=params)

        resp.raise_for_status()
        return resp.json()

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
