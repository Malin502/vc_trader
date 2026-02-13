"""Risk management – circuit breakers, drawdown monitoring."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from aivc_trade.core.logger import get_logger

log = get_logger("risk")


class CircuitBreaker:
    """Monitor equity snapshots and halt trading if loss limits are breached."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cb = cfg["circuit_breaker"]
        self.halt_until: Optional[datetime] = None
        self.consecutive_api_errors: int = 0

    def is_halted(self, now: datetime) -> bool:
        if self.halt_until and now < self.halt_until:
            return True
        return False

    def check_equity(
        self,
        equity_snapshots: List[Dict[str, Any]],
        now: datetime,
    ) -> bool:
        """Check 24h and 7d loss thresholds. Return True if halt triggered."""
        if len(equity_snapshots) < 2:
            return False

        current_equity = equity_snapshots[-1]["equity"]

        # 24h check
        threshold_24h = now - timedelta(hours=24)
        eq_24h = self._find_equity_at(equity_snapshots, threshold_24h)
        if eq_24h and eq_24h > 0:
            loss_24h = (current_equity - eq_24h) / eq_24h
            if loss_24h <= -self.cb["daily_loss_pct"]:
                self.halt_until = now + timedelta(hours=self.cb["daily_halt_hours"])
                log.warning(
                    f"CIRCUIT BREAKER: 24h loss {loss_24h:.2%} → halt until {self.halt_until}"
                )
                return True

        # 7d check
        threshold_7d = now - timedelta(days=7)
        eq_7d = self._find_equity_at(equity_snapshots, threshold_7d)
        if eq_7d and eq_7d > 0:
            loss_7d = (current_equity - eq_7d) / eq_7d
            if loss_7d <= -self.cb["weekly_loss_pct"]:
                self.halt_until = now + timedelta(hours=self.cb["weekly_halt_hours"])
                log.warning(
                    f"CIRCUIT BREAKER: 7d loss {loss_7d:.2%} → halt until {self.halt_until}"
                )
                return True

        return False

    def record_api_error(self) -> bool:
        """Increment error count, return True if threshold breached."""
        self.consecutive_api_errors += 1
        if self.consecutive_api_errors >= self.cb["max_api_errors"]:
            log.error(f"API error streak = {self.consecutive_api_errors} → HALT")
            return True
        return False

    def reset_api_errors(self) -> None:
        self.consecutive_api_errors = 0

    @staticmethod
    def _find_equity_at(
        snapshots: List[Dict[str, Any]], target_ts: datetime
    ) -> Optional[float]:
        """Find the equity snapshot closest to *target_ts* (looking backward)."""
        best = None
        for snap in snapshots:
            ts = snap["ts"]
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            if ts <= target_ts:
                best = snap["equity"]
        return best
