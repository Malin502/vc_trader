"""Signal-time order book snapshot (MS6 Step 3).

目的: シグナルが発生した瞬間の板を残し、後日「そのときの期待約定価格」と
実際の fill を突き合わせて **実効スリッページ** を裏取りできるようにする。

構成:

* :func:`summarize_depth` – ``/api/v3/depth`` の生ペイロードを、ベスト bid/ask・
  スプレッド・±X% の板の厚み（base / quote）・需給インバランスに要約する純関数。
  HTTP なしでテストできる。
* :class:`OrderBookRecorder` – クライアントから板を取り、要約を返しつつ JSONL に
  追記する。取得失敗はシグナル処理を止めない（``ok: False`` を返すだけ）。

実運用での蓄積は連続運転（M6 本体）で始まる。本モジュールは仕組みとテストまで。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from aivc_trade.core.logger import get_logger

log = get_logger("orderbook")

DEFAULT_BANDS_PCT: Tuple[float, ...] = (0.5, 1.0)


def _iso(ts: Optional[datetime]) -> Optional[str]:
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.isoformat()


def _levels(raw: Iterable[Sequence[Any]]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for lvl in raw or []:
        try:
            out.append((float(lvl[0]), float(lvl[1])))
        except (TypeError, ValueError, IndexError):
            continue
    return out


def summarize_depth(
    depth: Dict[str, Any],
    bands_pct: Sequence[float] = DEFAULT_BANDS_PCT,
    ts: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Summarise a raw depth payload into the fields we keep per signal.

    Returns ``{"ok": False, "reason": ...}`` when the book is unusable (empty
    side, non-positive prices) so a broken snapshot never looks like a real one.
    """
    bids = _levels((depth or {}).get("bids") or [])
    asks = _levels((depth or {}).get("asks") or [])
    if not bids or not asks:
        return {
            "ok": False,
            "reason": "empty book",
            "ts": _iso(ts),
            "last_update_id": (depth or {}).get("lastUpdateId"),
        }

    best_bid = bids[0][0]
    best_ask = asks[0][0]
    if best_bid <= 0 or best_ask <= 0 or best_ask < best_bid:
        return {
            "ok": False,
            "reason": "invalid best bid/ask",
            "ts": _iso(ts),
            "last_update_id": (depth or {}).get("lastUpdateId"),
        }

    mid = (best_bid + best_ask) / 2.0
    spread = best_ask - best_bid

    out: Dict[str, Any] = {
        "ok": True,
        "ts": _iso(ts),
        "last_update_id": (depth or {}).get("lastUpdateId"),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread_abs": spread,
        "spread_bps": spread / mid * 10_000.0,
        "levels": {"bids": len(bids), "asks": len(asks)},
        "depth": {},
    }

    for band in bands_pct:
        band = float(band)
        # 境界ちょうどの板も含める（mid が 2 つの価格の平均なので float 誤差で
        # ±0.5% ちょうどのレベルが落ちるのを防ぐ）。
        lower = mid * (1.0 - band / 100.0) * (1.0 - 1e-12)
        upper = mid * (1.0 + band / 100.0) * (1.0 + 1e-12)
        bid_legs = [(p, q) for p, q in bids if p >= lower]
        ask_legs = [(p, q) for p, q in asks if p <= upper]
        bid_base = sum(q for _, q in bid_legs)
        ask_base = sum(q for _, q in ask_legs)
        bid_quote = sum(p * q for p, q in bid_legs)
        ask_quote = sum(p * q for p, q in ask_legs)
        total = bid_base + ask_base
        out["depth"][f"{band:g}pct"] = {
            "bid_base": bid_base,
            "ask_base": ask_base,
            "bid_quote": bid_quote,
            "ask_quote": ask_quote,
            "levels": {"bids": len(bid_legs), "asks": len(ask_legs)},
            # (bid - ask) / (bid + ask): +1 は買い板だけ、-1 は売り板だけ
            "imbalance": ((bid_base - ask_base) / total) if total > 0 else 0.0,
        }

    return out


def simulate_market_fill(
    depth: Dict[str, Any],
    side: str,
    qty: float,
    *,
    fallback_price: Optional[float] = None,
    fallback_slippage_bps: float = 0.0,
) -> Dict[str, Any]:
    """成行注文を板の厚みで約定させる（実板ペーパーの模擬約定）.

    ``depth`` は ``/api/v3/depth`` の生ペイロード（:meth:`OrderBookRecorder.snapshot`
    と同じ取得経路）。BUY は asks を、SELL は bids を安い/高い方から消費し、
    消費した数量の加重平均を約定価格にする。板が薄く ``qty`` を満たせない場合は
    最良レベル（最後に消費した価格）で残りを埋め、``shortfall_qty`` に残す
    （＝実運用では成行が板を食い切る場面。レポートで検証できるよう事実を残す）。

    板が取れていない / 片側が空のときは ``fallback_price``（通常は現在値）に
    ``fallback_slippage_bps`` を不利方向へ足した価格で全量約定とみなす。
    この場合 ``source`` は ``ticker_fallback`` になる。
    """
    side = str(side).upper()
    if side not in ("BUY", "SELL"):
        raise ValueError(f"Unsupported side: {side!r}")
    qty = float(qty)
    if qty <= 0:
        raise ValueError(f"qty must be > 0, got {qty}")

    raw = (depth or {}).get("asks" if side == "BUY" else "bids") or []
    levels = _levels(raw)
    best = levels[0][0] if levels else None
    other = _levels((depth or {}).get("bids" if side == "BUY" else "asks") or [])
    mid = ((best + other[0][0]) / 2.0) if (best and other) else None

    remaining = qty
    notional = 0.0
    filled = 0.0
    consumed = 0
    shortfall = 0.0
    for price, available in levels:
        if price <= 0 or available <= 0:
            continue
        take = min(remaining, available)
        notional += take * price
        filled += take
        remaining -= take
        consumed += 1
        if remaining <= 1e-12:
            break

    if filled <= 0:
        if fallback_price is None or float(fallback_price) <= 0:
            return {
                "ok": False,
                "reason": "empty book and no fallback price",
                "side": side,
                "filled_qty": 0.0,
                "notional": 0.0,
                "avg_price": None,
                "levels_consumed": 0,
                "shortfall_qty": qty,
                "source": "none",
            }
        slip = float(fallback_price) * float(fallback_slippage_bps) / 10_000.0
        price = float(fallback_price) + (slip if side == "BUY" else -slip)
        return {
            "ok": True,
            "side": side,
            "filled_qty": qty,
            "notional": price * qty,
            "avg_price": price,
            "levels_consumed": 0,
            "shortfall_qty": 0.0,
            "source": "ticker_fallback",
        }

    if remaining > 1e-12:
        # 板を食い切った: 最後のレベルで残りを埋める（実運用ではさらに不利に滑る）。
        last_price = levels[consumed - 1][0]
        notional += remaining * last_price
        filled += remaining
        shortfall = remaining
        remaining = 0.0

    return {
        "ok": True,
        "side": side,
        "filled_qty": filled,
        "notional": notional,
        "avg_price": notional / filled,
        "levels_consumed": consumed,
        "shortfall_qty": shortfall,
        "best_price": best,
        "mid": mid,
        "source": "book",
    }


class OrderBookRecorder:
    """Fetch and persist the order book at signal time.

    ``enabled=False`` (or ``orderbook.enabled: false``) makes :meth:`snapshot`
    a no-op returning ``None``, so callers do not need to branch.
    """

    def __init__(
        self,
        client: Any,
        cfg: Dict[str, Any],
        path: Optional[str | Path] = None,
        time_fn: Optional[Any] = None,
    ) -> None:
        ob_cfg = cfg.get("orderbook", {}) or {}
        self.client = client
        self.cfg = cfg
        self.enabled = bool(ob_cfg.get("enabled", True))
        self.depth_limit = int(ob_cfg.get("depth_limit", 100))
        self.bands_pct = tuple(float(b) for b in (ob_cfg.get("bands_pct") or DEFAULT_BANDS_PCT))
        configured = (
            path
            or ob_cfg.get("journal_path")
            or (cfg.get("persistence", {}) or {}).get("orderbook_journal_path")
        )
        self.path = Path(configured) if configured else None
        self._time_fn = time_fn or (lambda: datetime.now(timezone.utc))

    def snapshot(self, symbol: str, ts: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """Return a summary for *symbol*; never raises.

        Failures are logged and returned as ``{"ok": False, "error": ...}`` so
        the signal record keeps the fact that a snapshot was attempted.
        """
        if not self.enabled:
            return None

        if ts is None:
            try:
                ts = self._time_fn()
            except Exception:  # pragma: no cover - clock is injectable
                ts = datetime.now(timezone.utc)

        try:
            depth = self.client.fetch_order_book(symbol, limit=self.depth_limit)
            summary = summarize_depth(depth, self.bands_pct, ts=ts)
        except Exception as e:
            log.warning(f"Order book snapshot failed for {symbol}: {e}")
            summary = {
                "ok": False,
                "reason": "fetch failed",
                "error": str(e),
                "ts": _iso(ts),
            }

        summary["symbol"] = symbol
        self._append(summary)
        return summary

    def _append(self, record: Dict[str, Any]) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as e:  # 記録できなくても取引は止めない
            log.warning(f"Order book journal write failed ({self.path}): {e}")
