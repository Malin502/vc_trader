"""Discord webhook notifier."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

import requests

from aivc_trade.core.logger import get_logger

log = get_logger("notifier")


class DiscordNotifier:
    """Send notifications via Discord webhook."""

    def __init__(self, webhook_url: str = "") -> None:
        self.webhook_url = webhook_url
        self.enabled = bool(webhook_url)

    def send(self, title: str, message: str, color: int = 0x00FF00) -> bool:
        """Send an embed message to Discord. Returns True on success."""
        if not self.enabled:
            log.debug(f"Notification (disabled): {title} | {message}")
            return False

        payload = {
            "embeds": [
                {
                    "title": title,
                    "description": message,
                    "color": color,
                    "timestamp": datetime.utcnow().isoformat(),
                    "footer": {"text": "AIVC Trade Bot"},
                }
            ]
        }
        try:
            resp = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            if resp.status_code in (200, 204):
                return True
            log.warning(f"Discord returned {resp.status_code}: {resp.text}")
            return False
        except Exception as e:
            log.error(f"Discord notification failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------
    def notify_entry(self, symbol: str, qty: float, price: float, stop: float) -> None:
        self.send(
            "📈 Entry",
            f"**{symbol}** BUY\n"
            f"Qty: {qty}\nPrice: {price:.2f}\nStop: {stop:.2f}",
            color=0x00FF00,
        )

    def notify_exit(
        self, symbol: str, qty: float, price: float, reason: str, pnl: float
    ) -> None:
        color = 0x00FF00 if pnl >= 0 else 0xFF0000
        self.send(
            "📉 Exit",
            f"**{symbol}** SELL ({reason})\n"
            f"Qty: {qty}\nPrice: {price:.2f}\nPnL: {pnl:+.2f}",
            color=color,
        )

    def notify_halt(self, reason: str, until: datetime) -> None:
        self.send(
            "⚠️ Trading Halted",
            f"Reason: {reason}\nResumes: {until.isoformat()}",
            color=0xFFAA00,
        )

    def notify_partial_tp(
        self, symbol: str, qty: float, price: float, pnl: float,
        remaining_qty: float,
    ) -> None:
        self.send(
            "Partial Take-Profit",
            f"**{symbol}** Partial SELL\n"
            f"Sold Qty: {qty}\nPrice: {price:.2f}\nPnL: {pnl:+.2f}\n"
            f"Remaining (runner): {remaining_qty}",
            color=0x00FF00,
        )

    def notify_error(self, error: str) -> None:
        self.send("🚨 Error", error, color=0xFF0000)
