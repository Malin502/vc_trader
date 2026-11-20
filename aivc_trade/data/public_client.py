"""認証不要の公開エンドポイントだけを叩くクライアント（実板ペーパー用）.

``exchange.env: paper`` のとき ``main_live`` はこのクラスだけを使う。
署名・API キー・アカウント系のコードを **一切持たない** ことが、そのまま
「署名付きエンドポイントへの経路が存在しない」保証になる
(:mod:`aivc_trade.execution.broker` の ``LiveBroker`` と対になる)。

提供するのは次の公開 API のみ:

* ``GET /api/v3/klines``      – 足（BTCJPY / ETHJPY / XRPJPY / SOLJPY）
* ``GET /api/v3/ticker/price`` – 現在値
* ``GET /api/v3/depth``        – 板（模擬約定の入力）
* ``GET /api/v3/exchangeInfo`` – lot size / tick size
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from aivc_trade.core.logger import get_logger
from aivc_trade.data.binance_client import BinanceAPIError, parse_klines

log = get_logger("public_client")

DEFAULT_TIMEOUT = 10.0


class PublicMarketClient:
    """Public-only market data client (no credentials, no signing)."""

    def __init__(
        self,
        base_url: str,
        timeout: float = DEFAULT_TIMEOUT,
        session: Optional[Any] = None,
    ) -> None:
        if not str(base_url or "").startswith("https://"):
            raise ValueError(f"public client requires an https base_url, got {base_url!r}")
        self.base_url = str(base_url).rstrip("/")
        self.timeout = float(timeout)
        self._session = session if session is not None else requests.Session()
        self._symbol_info_cache: Dict[str, Dict[str, Any]] = {}
        self._filters_cache: Dict[str, Dict[str, Dict[str, float]]] = {}

    # ------------------------------------------------------------------
    # Klines / ticker / depth
    # ------------------------------------------------------------------
    def fetch_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 500,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
    ) -> pd.DataFrame:
        params: Dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms
        return parse_klines(self._public_get("/api/v3/klines", params))

    def get_ticker_price(self, symbol: str) -> float:
        data = self._public_get("/api/v3/ticker/price", {"symbol": symbol})
        return float(data["price"])

    def fetch_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        return self._public_get(
            "/api/v3/depth", {"symbol": symbol, "limit": int(limit)}
        )

    # ------------------------------------------------------------------
    # Exchange info (lot / tick size)
    # ------------------------------------------------------------------
    def get_symbol_info(self, symbol: str, refresh: bool = False) -> Dict[str, Any]:
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
        price_filter = self._get_filters(symbol).get("PRICE_FILTER")
        if price_filter is None:
            return 0.0
        return float(price_filter.get("tickSize", 0.0))

    def _get_filters(self, symbol: str) -> Dict[str, Dict[str, float]]:
        if symbol not in self._filters_cache:
            self.get_symbol_info(symbol)
        return self._filters_cache[symbol]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _public_get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        resp = self._session.get(
            f"{self.base_url}{path}", params=params, timeout=self.timeout
        )
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
        raise BinanceAPIError(code, msg, status, "GET", path)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"PublicMarketClient(base_url={self.base_url!r})"
