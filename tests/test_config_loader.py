"""Unit tests for config resolution (exchange env / base_url / risk mode)."""

from __future__ import annotations

import pytest
import yaml

from aivc_trade.config.loader import (
    load_config,
    resolve_base_url,
    resolve_risk_per_trade,
)


def _minimal_cfg(**overrides):
    cfg = {
        "exchange": {
            "name": "binance",
            "env": "prod",
            "base_url": "https://api.binance.com",
            "testnet_base_url": "https://testnet.binance.vision",
            "quote_asset": "JPY",
            "symbols": ["BTCJPY", "ETHJPY"],
            "market_type": "spot",
        },
        "position": {"max_concurrent": 2, "max_per_symbol": 1},
        "sizing": {
            "risk_mode": "paper",
            "risk_per_trade_paper": 0.02,
            "risk_per_trade_live": 0.01,
            "max_notional_pct": 0.5,
        },
        "stop_order": {
            "enabled": True,
            "order_type": "STOP_LOSS",
            "client_order_prefix": "aivc-sl",
            "limit_offset_pct": 0.1,
        },
    }
    for key, value in overrides.items():
        cfg[key].update(value)
    return cfg


def _write_cfg(tmp_path, cfg):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


class TestResolveBaseUrl:
    def test_prod_uses_base_url(self):
        cfg = _minimal_cfg()
        assert resolve_base_url(cfg) == "https://api.binance.com"

    def test_testnet_uses_testnet_base_url(self):
        cfg = _minimal_cfg(exchange={"env": "testnet"})
        assert resolve_base_url(cfg) == "https://testnet.binance.vision"

    def test_trailing_slash_is_stripped(self):
        cfg = _minimal_cfg(exchange={"base_url": "https://api.example.test/"})
        assert resolve_base_url(cfg) == "https://api.example.test"

    def test_missing_host_fails_fast(self):
        cfg = _minimal_cfg(exchange={"base_url": ""})
        with pytest.raises(ValueError, match="base_url"):
            resolve_base_url(cfg)

    def test_missing_testnet_host_fails_fast(self):
        cfg = _minimal_cfg(exchange={"env": "testnet", "testnet_base_url": ""})
        with pytest.raises(ValueError, match="testnet_base_url"):
            resolve_base_url(cfg)

    def test_unknown_env_rejected(self):
        cfg = _minimal_cfg(exchange={"env": "staging"})
        with pytest.raises(ValueError, match="exchange.env"):
            resolve_base_url(cfg)

    def test_non_https_rejected(self):
        cfg = _minimal_cfg(exchange={"base_url": "http://api.example.test"})
        with pytest.raises(ValueError, match="https"):
            resolve_base_url(cfg)


class TestRiskMode:
    def test_paper_mode_default(self):
        cfg = _minimal_cfg()
        assert resolve_risk_per_trade(cfg["sizing"]) == 0.02

    def test_live_mode(self):
        cfg = _minimal_cfg(sizing={"risk_mode": "live"})
        assert resolve_risk_per_trade(cfg["sizing"]) == 0.01

    def test_explicit_value_wins(self):
        cfg = _minimal_cfg(sizing={"risk_per_trade": 0.005})
        assert resolve_risk_per_trade(cfg["sizing"]) == 0.005

    def test_missing_mode_value_raises(self):
        sizing = {"risk_mode": "live", "risk_per_trade_paper": 0.02}
        with pytest.raises(ValueError, match="risk_per_trade_live"):
            resolve_risk_per_trade(sizing)


class TestLoadConfig:
    def test_default_config_injects_effective_risk(self):
        cfg = load_config()
        assert cfg["sizing"]["risk_per_trade"] == 0.02
        assert cfg["exchange"]["env"] == "prod"
        # Binance Japan runs on the global platform: api.binance.com is the host.
        assert cfg["exchange"]["base_url"] == "https://api.binance.com"
        assert resolve_base_url(cfg) == "https://api.binance.com"

    def test_risk_mode_env_override(self, tmp_path, monkeypatch):
        path = _write_cfg(tmp_path, _minimal_cfg())
        monkeypatch.setenv("AIVC_RISK_MODE", "live")
        cfg = load_config(path)
        assert cfg["sizing"]["risk_mode"] == "live"
        assert cfg["sizing"]["risk_per_trade"] == 0.01

    def test_exchange_env_override(self, tmp_path, monkeypatch):
        path = _write_cfg(tmp_path, _minimal_cfg())
        monkeypatch.setenv("AIVC_EXCHANGE_ENV", "testnet")
        cfg = load_config(path)
        assert cfg["exchange"]["env"] == "testnet"
        assert resolve_base_url(cfg) == "https://testnet.binance.vision"

    def test_base_url_env_override(self, tmp_path, monkeypatch):
        path = _write_cfg(tmp_path, _minimal_cfg(exchange={"base_url": ""}))
        monkeypatch.setenv("BINANCE_BASE_URL", "https://api.example.test")
        cfg = load_config(path)
        assert resolve_base_url(cfg) == "https://api.example.test"

    def test_invalid_risk_mode_rejected(self, tmp_path):
        path = _write_cfg(tmp_path, _minimal_cfg(sizing={"risk_mode": "yolo"}))
        with pytest.raises(ValueError, match="risk_mode"):
            load_config(path)

    def test_risk_above_one_rejected(self, tmp_path):
        path = _write_cfg(tmp_path, _minimal_cfg(sizing={"risk_per_trade_live": 1.5}))
        with pytest.raises(ValueError, match="risk_per_trade_live"):
            load_config(path)

    def test_unused_mode_value_is_still_validated(self, tmp_path):
        # risk_mode=paper, but the live value must still be sane for later switching.
        path = _write_cfg(tmp_path, _minimal_cfg(sizing={"risk_per_trade_live": 2.0}))
        with pytest.raises(ValueError, match="risk_per_trade_live"):
            load_config(path)

    def test_explicit_risk_per_trade_overrides_modes(self, tmp_path):
        cfg = _minimal_cfg()
        cfg["sizing"] = {"risk_per_trade": 0.015, "max_notional_pct": 0.5}
        loaded = load_config(_write_cfg(tmp_path, cfg))
        assert loaded["sizing"]["risk_per_trade"] == 0.015

    def test_max_notional_guard_rejected_when_out_of_range(self, tmp_path):
        path = _write_cfg(tmp_path, _minimal_cfg(sizing={"max_notional_pct": 1.5}))
        with pytest.raises(ValueError, match="max_notional_pct"):
            load_config(path)

    def test_symbol_quote_mismatch_rejected(self, tmp_path):
        cfg = _minimal_cfg(exchange={"symbols": ["BTCUSDC", "ETHJPY"]})
        with pytest.raises(ValueError, match="quoted in JPY"):
            load_config(_write_cfg(tmp_path, cfg))

    def test_duplicate_symbols_rejected(self, tmp_path):
        cfg = _minimal_cfg(exchange={"symbols": ["BTCJPY", "BTCJPY"]})
        with pytest.raises(ValueError, match="duplicates"):
            load_config(_write_cfg(tmp_path, cfg))

    def test_max_per_symbol_cannot_exceed_concurrent(self, tmp_path):
        cfg = _minimal_cfg(position={"max_concurrent": 1, "max_per_symbol": 2})
        with pytest.raises(ValueError, match="max_per_symbol"):
            load_config(_write_cfg(tmp_path, cfg))

    def test_stop_order_type_validated(self, tmp_path):
        cfg = _minimal_cfg(stop_order={"order_type": "MARKET_IF_TOUCHED"})
        with pytest.raises(ValueError, match="stop_order.order_type"):
            load_config(_write_cfg(tmp_path, cfg))

    def test_stop_order_prefix_validated(self, tmp_path):
        cfg = _minimal_cfg(stop_order={"client_order_prefix": "bad prefix!"})
        with pytest.raises(ValueError, match="client_order_prefix"):
            load_config(_write_cfg(tmp_path, cfg))
