"""模擬ブローカー – 実際の板で約定させる（実板ペーパー）.

``exchange.env: paper`` のとき ``LiveBroker`` の代わりに使う。実注文は出さず、
**本番の公開板**（``/api/v3/depth``）の厚みを消費して約定価格・スリッページを
決める（:func:`aivc_trade.data.orderbook.simulate_market_fill`）。板が取れない
ときは現在値（``/api/v3/ticker/price``）に ``paper.fallback_slippage_bps`` を
不利方向へ足した価格で約定とみなし、``source: ticker_fallback`` として残す。

仮想口座（quote 残高 / base 残高）は ``paper.account_file`` に保存し、再起動を
跨いで維持する。``main_live`` は ``LiveBroker`` と同じ ``execute(order)`` と、
``BinanceClient`` と同じ ``get_account_balance`` / ``get_spot_balance`` を呼ぶ。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from aivc_trade.core.logger import get_logger
from aivc_trade.core.types import Order
from aivc_trade.data.orderbook import simulate_market_fill

log = get_logger("paper_broker")

DEFAULT_ACCOUNT_FILE = "data/paper/paper_account.json"


def _base_asset(symbol: str, quote_asset: str) -> str:
    if quote_asset and symbol.upper().endswith(quote_asset.upper()):
        return symbol[: -len(quote_asset)]
    return symbol


class PaperBroker:
    """Simulated exchange that fills against the real order book."""

    def __init__(
        self,
        client: Any,
        cfg: Dict[str, Any],
        *,
        state_path: Optional[str | Path] = None,
    ) -> None:
        self.client = client
        self.cfg = cfg
        exchange = cfg.get("exchange", {}) or {}
        self.quote_asset = str(exchange.get("quote_asset") or "JPY").upper()

        paper_cfg = cfg.get("paper", {}) or {}
        costs = cfg.get("costs", {}) or {}
        self.initial_equity = float(
            paper_cfg.get("initial_equity")
            or (cfg.get("backtest", {}) or {}).get("initial_equity")
            or 100_000.0
        )
        self.commission_bps = float(costs.get("commission_bps", 10.0))
        self.fallback_slippage_bps = float(
            paper_cfg.get("fallback_slippage_bps", costs.get("slippage_bps", 5.0))
        )
        self.depth_limit = int((cfg.get("orderbook", {}) or {}).get("depth_limit", 100))

        configured = state_path or paper_cfg.get("account_file") or DEFAULT_ACCOUNT_FILE
        self.state_path = Path(configured)

        self.quote_balance = self.initial_equity
        self.base_balances: Dict[str, float] = {}
        self.realized_pnl = 0.0
        self.fills = 0
        self.last_fill: Optional[Dict[str, Any]] = None
        self._load()
        if not self.state_path.exists():
            # 起動直後に口座ファイルを作る（運用者が残高を確認できるように）。
            self.save()

    # ------------------------------------------------------------------
    # Account view (BinanceClient-compatible)
    # ------------------------------------------------------------------
    def get_account_balance(self, asset: str = "") -> float:
        """Free balance for *asset* (quote → 仮想現金)."""
        asset = str(asset or self.quote_asset).upper()
        if asset == self.quote_asset:
            return float(self.quote_balance)
        return float(self.base_balances.get(asset, 0.0))

    def get_spot_balance(self, asset: str) -> float:
        return self.get_account_balance(asset)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def execute(self, order: Order) -> Order:
        """Fill *order* against the public book and update the virtual account."""
        symbol = order.symbol
        side = str(getattr(order.side, "value", order.side)).upper()
        qty = float(order.qty)
        if qty <= 0:
            raise ValueError(f"order qty must be > 0, got {qty}")

        depth, book_error = self._depth(symbol)
        ticker = self._ticker(symbol, fallback=order.price)

        fill = simulate_market_fill(
            depth or {},
            side,
            qty,
            fallback_price=ticker,
            fallback_slippage_bps=self.fallback_slippage_bps,
        )
        if not fill.get("ok"):
            order.status = "ERROR"
            raise RuntimeError(
                f"paper fill failed for {symbol}: {fill.get('reason')}"
                f" (book_error={book_error})"
            )

        price = float(fill["avg_price"])
        filled = float(fill["filled_qty"])
        notional = float(fill["notional"])
        fee = notional * self.commission_bps / 10_000.0
        base = _base_asset(symbol, self.quote_asset)

        if side == "BUY":
            self.quote_balance -= notional + fee
            self.base_balances[base] = self.base_balances.get(base, 0.0) + filled
        else:
            self.quote_balance += notional - fee
            remaining = self.base_balances.get(base, 0.0) - filled
            # 浮動小数の残骸を残さない（0 近傍は 0 とみなす）。
            self.base_balances[base] = remaining if remaining > 1e-12 else 0.0

        order.filled_price = price
        order.filled_qty = filled
        order.quote_qty = notional
        order.fee = fee
        order.fee_asset = self.quote_asset
        order.order_id = f"PAPER-{uuid.uuid4().hex[:12]}"
        order.status = "FILLED"

        self.fills += 1
        self.last_fill = {
            "symbol": symbol,
            "side": side,
            "qty": filled,
            "avg_price": price,
            "notional": notional,
            "fee": fee,
            "fee_asset": self.quote_asset,
            "levels_consumed": int(fill.get("levels_consumed", 0)),
            "shortfall_qty": float(fill.get("shortfall_qty", 0.0) or 0.0),
            "source": fill.get("source"),
            "best_price": fill.get("best_price"),
            "mid": fill.get("mid"),
            "book_available": depth is not None,
            "book_error": book_error,
            "commission_bps": self.commission_bps,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        self.save()

        log.info(
            f"PAPER fill: {symbol} {side} qty={filled} px={price} "
            f"levels={fill.get('levels_consumed')} source={fill.get('source')} "
            f"fee={fee:.4f} {self.quote_asset} id={order.order_id}"
        )
        return order

    # ------------------------------------------------------------------
    # Book / ticker
    # ------------------------------------------------------------------
    def _depth(self, symbol: str):
        try:
            return self.client.fetch_order_book(symbol, limit=self.depth_limit), ""
        except Exception as e:  # 板が取れなくても取引は止めない（fallback へ）
            log.warning(f"PAPER depth fetch failed for {symbol}: {e}")
            return None, str(e)

    def _ticker(self, symbol: str, fallback: Optional[float] = None) -> Optional[float]:
        try:
            return float(self.client.get_ticker_price(symbol))
        except Exception as e:
            log.warning(f"PAPER ticker fetch failed for {symbol}: {e}")
            return float(fallback) if fallback else None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load(self) -> None:
        try:
            if not self.state_path.exists():
                return
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning(f"PAPER account load failed ({self.state_path}): {e}")
            return
        self.quote_balance = float(data.get("quote_balance", self.initial_equity))
        self.base_balances = {
            str(k).upper(): float(v) for k, v in (data.get("base_balances") or {}).items()
        }
        self.realized_pnl = float(data.get("realized_pnl", 0.0))
        self.fills = int(data.get("fills", 0))
        log.info(
            f"PAPER account restored: {self.quote_balance:.2f} {self.quote_asset}, "
            f"base={self.base_balances or '{}'}, fills={self.fills}"
        )

    def save(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(
                json.dumps(
                    {
                        "quote_asset": self.quote_asset,
                        "quote_balance": self.quote_balance,
                        "base_balances": self.base_balances,
                        "realized_pnl": self.realized_pnl,
                        "fills": self.fills,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as e:  # 記録失敗で取引を止めない
            log.warning(f"PAPER account save failed ({self.state_path}): {e}")
