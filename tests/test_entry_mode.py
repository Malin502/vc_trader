"""Entry mode switch (entry.mode) – full vs regime_only.

簡素版（TREND_UP 判定のみ）の切替が generate_signals に効くことだけを確認する。
決済・サイジング・クールダウンは両モードで共通なので、ここでは
「追加条件を満たさない行が regime_only でだけ通る」ことを見る。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from aivc_trade.config.loader import load_config
from aivc_trade.core.types import Regime
from aivc_trade.strategy.signal import generate_signals

NOW = datetime(2026, 11, 20, tzinfo=timezone.utc)
SYMBOL = "XRPJPY"


def _cfg():
    cfg = load_config()
    cfg["exchange"]["symbols"] = [SYMBOL]
    cfg["exchange"]["shadow_symbols"] = []
    cfg["position"]["max_concurrent"] = 1
    cfg["position"]["max_per_symbol"] = 1
    return cfg


def _row(**overrides):
    """TREND_UP だが現行の追加条件（ADX/ブレイク/出来高/ret24h）を満たさない行。"""
    row = {
        "ts": NOW,
        "regime": Regime.TREND_UP,
        "ret_24h": 0.005,  # < ret24h_min_pct 2.0%
        "atrp": 0.012,
        "close": 300.0,
        "ema_fast": 299.0,
        "ema_slow": 290.0,
        "trend_ma": 290.0,
        "donchian_high_prev": 320.0,  # ブレイクしていない
        "breakout_high_prev": 320.0,
        "volume": 50.0,               # volume_sma 割れ
        "volume_sma": 100.0,
        "atrp_z": 0.2,
        "slope": 0.003,
        "atr": 3.0,
        "recent_swing_low": 296.0,
        "adx": 21.0,                  # entry_filter 22 / entry_quality 25 未満
        "high": 301.0,
        "low": 298.0,
    }
    row.update(overrides)
    return row


def _features(**overrides):
    return {SYMBOL: pd.DataFrame([_row(**overrides)])}


def _signals(cfg):
    return generate_signals(_features(), {}, cfg, NOW)


class TestEntryMode:
    def test_default_mode_is_full(self):
        cfg = _cfg()
        assert cfg["entry"]["mode"] == "full"
        assert _signals(cfg) == []

    def test_regime_only_bypasses_extra_gates(self):
        cfg = _cfg()
        cfg["entry"]["mode"] = "regime_only"
        sigs = _signals(cfg)
        assert len(sigs) == 1
        assert sigs[0].symbol == SYMBOL
        assert sigs[0].entry_type == "REGIME_ONLY"

    def test_regime_only_still_requires_trend_up(self):
        cfg = _cfg()
        cfg["entry"]["mode"] = "regime_only"
        sigs = generate_signals(_features(regime=Regime.RANGE), {}, cfg, NOW)
        assert sigs == []

    def test_regime_only_keeps_cooldown(self):
        cfg = _cfg()
        cfg["entry"]["mode"] = "regime_only"
        sigs = generate_signals(
            _features(), {}, cfg, NOW, cooldowns={SYMBOL: NOW + timedelta(hours=1)}
        )
        assert sigs == []

    def test_full_mode_still_takes_a_qualified_bar(self):
        """full モードの既存挙動は変えない（条件を満たせば通る）。"""
        cfg = _cfg()
        sigs = generate_signals(
            _features(
                ret_24h=0.05,
                adx=40.0,
                volume=150.0,
                close=330.0,
            ),
            {},
            cfg,
            NOW,
        )
        assert len(sigs) == 1
        assert sigs[0].entry_type == "BREAKOUT"
