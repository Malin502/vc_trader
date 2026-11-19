"""Signal journal – シグナル時点の記録を板スナップショットと紐づけて残す (MS6 Step 3).

1 シグナル = 1 行の JSON（JSONL）。後日「シグナル時点の板」と「想定約定価格
（``signal.entry_price``）」と「実際の fill」を突き合わせれば、実効スリッページを
シグナル単位で裏取りできる。

行の形::

    {
      "signal_id": "20261119T120000-BTCJPY-1a2b3c4d",
      "ts": "2026-11-19T12:00:00+00:00",
      "mode": "paper",                  # sizing.risk_mode
      "action": "ENTRY_CANDIDATE",      # 呼び出し側の任意ラベル
      "symbol": "BTCJPY",
      "side": "BUY",
      "entry_type": "BREAKOUT",
      "score": 0.123,
      "entry_price": 12345678.0,        # シグナル時点の想定約定価格
      "stop_price": 12000000.0,
      "orderbook": { ... }              # data.orderbook.summarize_depth の出力
    }
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


def _iso(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.isoformat()


def _side_value(side: Any) -> str:
    return getattr(side, "value", str(side))


class SignalJournal:
    """Append-only JSONL writer for signal-time records."""

    def __init__(
        self,
        path: str | Path,
        mode: str = "live",
        *,
        enabled: bool = True,
    ) -> None:
        self.path = Path(path)
        self.mode = str(mode)
        self.enabled = bool(enabled)

    def record(
        self,
        signal: Signal,
        *,
        ts: Optional[datetime] = None,
        orderbook: Optional[Dict[str, Any]] = None,
        action: str = "ENTRY_CANDIDATE",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build, persist and return the record for *signal*.

        The write is best effort: a journal failure must never stop trading, so
        the record is returned regardless and the error is logged.
        """
        ts = ts or signal.ts or datetime.now(timezone.utc)
        signal_id = f"{ts.strftime('%Y%m%dT%H%M%S')}-{signal.symbol}-{uuid.uuid4().hex[:8]}"

        record: Dict[str, Any] = {
            "signal_id": signal_id,
            "ts": _iso(ts),
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
        if extra:
            record.update(extra)

        self.append(record)
        return record

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
