"""Signal journal – シグナル時点の記録を板スナップショットと紐づけて残す (MS6 Step 3).

ペーパー運転（Step 1'）の評価指標を 1 つの JSONL から集計できるようにする。
記録は ``event`` で種類を分ける（古い行は ``event`` を持たないので SIGNAL 扱い）。

* ``SIGNAL`` – シグナル時点の想定（``entry_price`` / ``stop_price``）と板
* ``FILL``   – 実際の約定（``expected_price`` との乖離＝実効スリッページ、手数料）
* ``STOP``   – 取引所側 STOP の設置 / ラチェット / 取消の成否

行の形::

    {"event": "SIGNAL", "signal_id": "20261119T120000-BTCJPY-1a2b3c4d",
     "ts": "...", "bar_ts": "...", "mode": "paper", "action": "ENTRY_CANDIDATE",
     "symbol": "BTCJPY", "side": "BUY", "entry_type": "BREAKOUT", "score": 0.123,
     "entry_price": 12345678.0, "stop_price": 12000000.0, "orderbook": {...}}

    {"event": "FILL", "ts": "...", "signal_id": "...", "kind": "ENTRY",
     "symbol": "BTCJPY", "side": "BUY", "qty": 0.001, "price": 12346000.0,
     "expected_price": 12345678.0, "slippage_bps": 0.26,
     "fee": 12.3, "fee_asset": "JPY", "order_id": "123"}

    {"event": "STOP", "ts": "...", "symbol": "BTCJPY", "kind": "PLACED",
     "ok": true, "step": "initial", "stop_price": 12000000.0, "order_id": "456"}

``bar_ts`` はシグナル判定に使った 1h 足の開始時刻。バックテストのシグナルと
(symbol, bar_ts) で突き合わせる＝シグナル一致率の元データになる。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from aivc_trade.core.logger import get_logger
from aivc_trade.core.types import Signal

log = get_logger("signal_journal")

#: レコード種別（``event`` フィールド）
EVENT_SIGNAL = "SIGNAL"
EVENT_FILL = "FILL"
EVENT_STOP = "STOP"
EVENT_GATE = "GATE"
EVENT_CONFIG = "CONFIG"


def _iso(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.isoformat()


def _side_value(side: Any) -> str:
    return getattr(side, "value", str(side))


def slippage_bps(price: float, expected: float, side: str = "BUY") -> Optional[float]:
    """約定価格が想定からどれだけ離れたか（bps、不利方向を正）。

    BUY は高く約定するほど、SELL は安く約定するほどコストなので、符号を
    「不利＝正」に揃えておくとレポートでそのまま合算できる。
    """
    if not expected or expected <= 0 or price is None:
        return None
    raw = (float(price) - float(expected)) / float(expected) * 10_000.0
    return raw if str(side).upper() == "BUY" else -raw


class SignalJournal:
    """Append-only JSONL writer for signal / fill / stop records."""

    def __init__(
        self,
        path: str | Path,
        mode: str = "live",
        *,
        enabled: bool = True,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.path = Path(path)
        self.mode = str(mode)
        self.enabled = bool(enabled)
        #: 全レコードに付ける既定フィールド（実板ペーパーは ``venue=paper_mock``）。
        #: 約定が模擬であることを後から機械的に見分けるために使う。
        self.defaults: Dict[str, Any] = dict(defaults or {})

    def _stamp(self, record: Dict[str, Any], extra: Optional[Dict[str, Any]]) -> None:
        for key, value in self.defaults.items():
            record.setdefault(key, value)
        if extra:
            record.update(extra)

    # ------------------------------------------------------------------
    # SIGNAL
    # ------------------------------------------------------------------
    def record(
        self,
        signal: Signal,
        *,
        ts: Optional[datetime] = None,
        orderbook: Optional[Dict[str, Any]] = None,
        action: str = "ENTRY_CANDIDATE",
        extra: Optional[Dict[str, Any]] = None,
        bar_ts: Optional[datetime] = None,
        signal_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build, persist and return the record for *signal*.

        The write is best effort: a journal failure must never stop trading, so
        the record is returned regardless and the error is logged.
        """
        ts = ts or signal.ts or datetime.now(timezone.utc)
        signal_id = signal_id or (
            f"{ts.strftime('%Y%m%dT%H%M%S')}-{signal.symbol}-{uuid.uuid4().hex[:8]}"
        )

        record: Dict[str, Any] = {
            "event": EVENT_SIGNAL,
            "signal_id": signal_id,
            "ts": _iso(ts),
            "bar_ts": _iso(bar_ts) if bar_ts is not None else None,
            "mode": self.mode,
            "action": action,
            "symbol": signal.symbol,
            "side": _side_value(signal.side),
            "entry_type": signal.entry_type,
            "score": float(signal.score),
            "entry_price": float(signal.entry_price),
            "stop_price": float(signal.stop_price),
            "orderbook": orderbook,
        }
        self._stamp(record, extra)

        self.append(record)
        return record

    # ------------------------------------------------------------------
    # FILL
    # ------------------------------------------------------------------
    def record_fill(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        ts: Optional[datetime] = None,
        kind: str = "ENTRY",
        signal_id: Optional[str] = None,
        expected_price: Optional[float] = None,
        fee: float = 0.0,
        fee_asset: str = "",
        order_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record one real fill (entry / exit / partial).

        ``expected_price`` はシグナル時点の想定価格（``signal.entry_price``）。
        ここから ``slippage_bps`` を計算して残すので、レポート側で再計算不要。
        """
        record: Dict[str, Any] = {
            "event": EVENT_FILL,
            "ts": _iso(ts or datetime.now(timezone.utc)),
            "mode": self.mode,
            "kind": str(kind).upper(),
            "signal_id": signal_id,
            "symbol": symbol,
            "side": str(side).upper(),
            "qty": float(qty),
            "price": float(price),
            "expected_price": float(expected_price) if expected_price else None,
            "slippage_bps": slippage_bps(price, expected_price or 0.0, side),
            "fee": float(fee or 0.0),
            "fee_asset": str(fee_asset or ""),
            "order_id": str(order_id) if order_id else None,
        }
        self._stamp(record, extra)
        self.append(record)
        return record

    # ------------------------------------------------------------------
    # STOP (取引所側 STOP_LOSS の成否)
    # ------------------------------------------------------------------
    def record_stop(
        self,
        *,
        symbol: str,
        kind: str,
        ok: bool,
        ts: Optional[datetime] = None,
        step: str = "",
        stop_price: Optional[float] = None,
        qty: Optional[float] = None,
        order_id: Optional[str] = None,
        detail: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record the outcome of one exchange-stop operation.

        ``kind`` は ``PLACED`` / ``RATCHET`` / ``RESIZE`` / ``CANCELLED`` /
        ``RECONCILED``。失敗も同じ形で残すので「STOP 失敗 1 件＝停止条件」を
        レポートで数えられる。
        """
        record: Dict[str, Any] = {
            "event": EVENT_STOP,
            "ts": _iso(ts or datetime.now(timezone.utc)),
            "mode": self.mode,
            "kind": str(kind).upper(),
            "ok": bool(ok),
            "step": str(step),
            "symbol": symbol,
            "stop_price": float(stop_price) if stop_price is not None else None,
            "qty": float(qty) if qty is not None else None,
            "order_id": str(order_id) if order_id else None,
            "detail": detail,
        }
        self._stamp(record, extra)
        self.append(record)
        return record

    # ------------------------------------------------------------------
    # GATE (高時間軸ゲート G3 の状態変化 / 判定不能の警告)
    # ------------------------------------------------------------------
    def record_gate(
        self,
        status: Dict[str, Any],
        *,
        ts: Optional[datetime] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a G3 state change. ``ok=False`` は「判定不能」。

        live の fail-closed では ``ok=False`` かつ ``blocked=True``
        （``reason`` が ``data_unavailable: ...``）になり、新規エントリーは止まる。
        """
        record: Dict[str, Any] = {
            "event": EVENT_GATE,
            "ts": _iso(ts or datetime.now(timezone.utc)),
            "mode": self.mode,
            "level": "INFO" if status.get("ok") else "WARNING",
        }
        record.update(status)
        self._stamp(record, extra)
        self.append(record)
        return record

    # ------------------------------------------------------------------
    # CONFIG (起動時の構成スナップショット = 期間ごとの構成の再構成用)
    # ------------------------------------------------------------------
    def record_config(
        self,
        snapshot: Dict[str, Any],
        *,
        ts: Optional[datetime] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record the configuration a run started with (``event: CONFIG``).

        有効なゲート（G3 のみ / G3+W2 等）と版を起動ごとに 1 行残す。ジャーナル
        上の隣接する CONFIG レコードの区間が、その構成で走っていた期間になる。
        """
        record: Dict[str, Any] = {
            "event": EVENT_CONFIG,
            "ts": _iso(ts or datetime.now(timezone.utc)),
            "mode": self.mode,
            "level": "INFO",
        }
        record.update(snapshot)
        self._stamp(record, extra)
        self.append(record)
        return record

    # ------------------------------------------------------------------
    # Low level
    # ------------------------------------------------------------------
    def append(self, record: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as e:  # 記録失敗で発注を止めない
            log.warning(f"Signal journal write failed ({self.path}): {e}")

    @staticmethod
    def load(path: str | Path) -> list:
        """Read back every record (skips malformed lines)."""
        p = Path(path)
        if not p.exists():
            return []
        out = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out
