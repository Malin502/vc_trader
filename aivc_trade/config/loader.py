"""Configuration loader – reads YAML, merges env overrides, resolves derived values.

Derived values injected by :func:`load_config`:

* ``exchange.env``         – normalised to ``prod`` / ``testnet``
* ``sizing.risk_per_trade`` – effective per-trade risk for the active ``risk_mode``

The API host is intentionally **not** injected: call :func:`resolve_base_url`
where the connection is actually built (main_live wiring is M4).  This keeps
``main_live`` untouched while prod / testnet can be switched from config.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"

EXCHANGE_ENVS = ("prod", "testnet")
RISK_MODES = ("paper", "live")
STOP_ORDER_TYPES = ("STOP_LOSS", "STOP_LOSS_LIMIT")

_CLIENT_ORDER_PREFIX_RE = re.compile(r"^[A-Za-z0-9_.-]{1,20}$")


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load config YAML and overlay environment variable overrides."""
    if path is not None:
        cfg_path = Path(path)
    else:
        env_cfg_path = os.environ.get("AIVC_CONFIG_PATH")
        cfg_path = Path(env_cfg_path) if env_cfg_path else _DEFAULT_CONFIG_PATH
    with open(cfg_path, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    # --- Environment variable overrides ---
    _env = os.environ.get
    if _env("BINANCE_API_KEY"):
        cfg.setdefault("secrets", {})["api_key"] = _env("BINANCE_API_KEY", "")
    if _env("BINANCE_API_SECRET"):
        cfg.setdefault("secrets", {})["api_secret"] = _env("BINANCE_API_SECRET", "")
    if _env("DISCORD_WEBHOOK_URL"):
        cfg["notification"]["discord_webhook_url"] = _env("DISCORD_WEBHOOK_URL", "")
    if _env("AIVC_INITIAL_EQUITY"):
        cfg["backtest"]["initial_equity"] = float(_env("AIVC_INITIAL_EQUITY", "10000"))
    if _env("AIVC_EXCHANGE_ENV"):
        cfg.setdefault("exchange", {})["env"] = _env("AIVC_EXCHANGE_ENV", "")
    if _env("AIVC_RISK_MODE"):
        cfg.setdefault("sizing", {})["risk_mode"] = _env("AIVC_RISK_MODE", "")
    if _env("BINANCE_BASE_URL"):
        cfg.setdefault("exchange", {})["base_url"] = _env("BINANCE_BASE_URL", "")

    _validate_config(cfg)
    _apply_derived(cfg)
    return cfg


# ----------------------------------------------------------------------
# Derived / resolved values
# ----------------------------------------------------------------------

def _apply_derived(cfg: Dict[str, Any]) -> None:
    """Normalise mode selectors and inject the effective risk per trade."""
    exchange = cfg.setdefault("exchange", {})
    exchange["env"] = str(exchange.get("env") or "prod").strip().lower()

    sizing = cfg.setdefault("sizing", {})
    sizing["risk_mode"] = str(sizing.get("risk_mode") or "paper").strip().lower()
    sizing["risk_per_trade"] = resolve_risk_per_trade(sizing)


def resolve_risk_per_trade(sizing_cfg: Dict[str, Any]) -> float:
    """Return the effective per-trade risk fraction.

    An explicit ``risk_per_trade`` wins (manual override); otherwise the value
    is selected from ``risk_per_trade_paper`` / ``risk_per_trade_live`` by
    ``risk_mode``.
    """
    explicit = sizing_cfg.get("risk_per_trade")
    if explicit is not None:
        return float(explicit)

    mode = str(sizing_cfg.get("risk_mode") or "paper").strip().lower()
    key = f"risk_per_trade_{mode}"
    if key not in sizing_cfg or sizing_cfg[key] is None:
        raise ValueError(
            f"Invalid config: sizing.{key} is required when risk_mode={mode!r}"
        )
    return float(sizing_cfg[key])


def resolve_base_url(cfg: Dict[str, Any]) -> str:
    """Return the API base URL for the configured ``exchange.env``.

    ``prod`` reads ``exchange.base_url`` (deliberately left unset until the
    Binance Japan host is confirmed), ``testnet`` reads
    ``exchange.testnet_base_url``.  Raises ``ValueError`` when the selected
    host is missing so a misconfigured deployment fails fast instead of
    silently trading against the wrong venue.
    """
    exchange = cfg.get("exchange", {})
    env = str(exchange.get("env") or "prod").strip().lower()
    if env not in EXCHANGE_ENVS:
        raise ValueError(
            f"Invalid config: exchange.env must be one of {EXCHANGE_ENVS}, got {env!r}"
        )

    key = "base_url" if env == "prod" else "testnet_base_url"
    url = str(exchange.get(key) or "").strip().rstrip("/")
    if not url:
        raise ValueError(
            f"Invalid config: exchange.{key} is empty (env={env}). "
            "Set the API host in config.yaml or BINANCE_BASE_URL."
        )
    if not url.startswith("https://"):
        raise ValueError(
            f"Invalid config: exchange.{key} must be an https URL, got {url!r}"
        )
    return url


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

def _validate_config(cfg: Dict[str, Any]) -> None:
    _validate_exchange(cfg)
    _validate_sizing(cfg)
    _validate_stop_order(cfg)

    risk_cfg = cfg.get("risk", {})
    initial_sl = risk_cfg.get("initial_sl", {})
    min_sl_pct = float(initial_sl.get("min_sl_pct", 0.0))
    max_sl_pct = float(initial_sl.get("max_sl_pct", 100.0))
    if min_sl_pct > max_sl_pct:
        raise ValueError("Invalid config: risk.initial_sl.min_sl_pct must be <= max_sl_pct")

    early_fail = risk_cfg.get("early_fail_exit", {})
    if early_fail:
        window_bars = int(early_fail.get("window_bars", 1))
        if window_bars < 1:
            raise ValueError("Invalid config: risk.early_fail_exit.window_bars must be >= 1")

    be_cfg = risk_cfg.get("breakeven", {})
    be_buffer = be_cfg.get("buffer_pct")
    if be_buffer is not None:
        be_buffer = float(be_buffer)
        if not (0.0 < be_buffer < 1.0):
            raise ValueError("Invalid config: risk.breakeven.buffer_pct must satisfy 0 < x < 1.0")

    breakout_cfg = cfg.get("entry_signal", {}).get("breakout", {})
    if breakout_cfg:
        lookback_bars = int(breakout_cfg.get("lookback_bars", 2))
        if lookback_bars < 2:
            raise ValueError("Invalid config: entry_signal.breakout.lookback_bars must be >= 2")
        breakout_buffer = float(breakout_cfg.get("buffer_pct", 0.0))
        if not (0.0 < breakout_buffer < 1.0):
            raise ValueError("Invalid config: entry_signal.breakout.buffer_pct must satisfy 0 < x < 1.0")

    runner_cfg = cfg.get("runner", {})
    trail_atr_k = float(runner_cfg.get("trail_atr_k", 1.0))
    if trail_atr_k < 1.0:
        raise ValueError("Invalid config: runner.trail_atr_k must be >= 1.0")

    exit_tuning = cfg.get("exit_tuning", {})
    strong_trend_hold = exit_tuning.get("strong_trend_hold", {})
    if strong_trend_hold:
        adx_min = float(strong_trend_hold.get("adx_min", 0.0))
        if adx_min < 0:
            raise ValueError("Invalid config: exit_tuning.strong_trend_hold.adx_min must be >= 0")
        trail_mult = float(strong_trend_hold.get("trail_atr_k_multiplier", 1.0))
        if trail_mult <= 0:
            raise ValueError(
                "Invalid config: exit_tuning.strong_trend_hold.trail_atr_k_multiplier must be > 0"
            )

    trail_two_stage = exit_tuning.get("trail_two_stage", {})
    if trail_two_stage:
        mfe_switch_pct = float(trail_two_stage.get("mfe_switch_pct", 0.0))
        if mfe_switch_pct < 0:
            raise ValueError(
                "Invalid config: exit_tuning.trail_two_stage.mfe_switch_pct must be >= 0"
            )
        atr_k_before = float(trail_two_stage.get("atr_k_before", 1.0))
        atr_k_after = float(trail_two_stage.get("atr_k_after", 1.0))
        if atr_k_before <= 0 or atr_k_after <= 0:
            raise ValueError(
                "Invalid config: exit_tuning.trail_two_stage.atr_k_before/atr_k_after must be > 0"
            )


def _validate_exchange(cfg: Dict[str, Any]) -> None:
    exchange = cfg.get("exchange", {})

    env = str(exchange.get("env") or "prod").strip().lower()
    if env not in EXCHANGE_ENVS:
        raise ValueError(
            f"Invalid config: exchange.env must be one of {EXCHANGE_ENVS}, got {env!r}"
        )

    symbols = exchange.get("symbols") or []
    if not symbols:
        raise ValueError("Invalid config: exchange.symbols must not be empty")
    if len(set(symbols)) != len(symbols):
        raise ValueError("Invalid config: exchange.symbols contains duplicates")

    quote_asset = str(exchange.get("quote_asset") or "").strip().upper()
    if quote_asset:
        for symbol in symbols:
            if not str(symbol).upper().endswith(quote_asset):
                raise ValueError(
                    f"Invalid config: symbol {symbol!r} is not quoted in {quote_asset}"
                )

    for key in ("base_url", "testnet_base_url"):
        url = str(exchange.get(key) or "").strip()
        if url and not url.startswith("https://"):
            raise ValueError(
                f"Invalid config: exchange.{key} must be an https URL, got {url!r}"
            )

    position = cfg.get("position", {})
    max_concurrent = int(position.get("max_concurrent", 1))
    max_per_symbol = int(position.get("max_per_symbol", 1))
    if max_concurrent < 1:
        raise ValueError("Invalid config: position.max_concurrent must be >= 1")
    if max_per_symbol < 1:
        raise ValueError("Invalid config: position.max_per_symbol must be >= 1")
    if max_per_symbol > max_concurrent:
        raise ValueError(
            "Invalid config: position.max_per_symbol must be <= max_concurrent"
        )


def _validate_sizing(cfg: Dict[str, Any]) -> None:
    sizing = cfg.get("sizing", {})

    mode = str(sizing.get("risk_mode") or "paper").strip().lower()
    if mode not in RISK_MODES:
        raise ValueError(
            f"Invalid config: sizing.risk_mode must be one of {RISK_MODES}, got {mode!r}"
        )

    # Mode-specific values are required unless risk_per_trade is set explicitly.
    if sizing.get("risk_per_trade") is None:
        for key in ("risk_per_trade_paper", "risk_per_trade_live"):
            value = sizing.get(key)
            if value is None:
                raise ValueError(f"Invalid config: sizing.{key} is required")
            value = float(value)
            if not (0.0 < value <= 1.0):
                raise ValueError(
                    f"Invalid config: sizing.{key} must satisfy 0 < x <= 1.0, got {value}"
                )

    effective = resolve_risk_per_trade(sizing)
    if not (0.0 < effective <= 1.0):
        raise ValueError(
            f"Invalid config: sizing.risk_per_trade must satisfy 0 < x <= 1.0, got {effective}"
        )

    max_notional = float(sizing.get("max_notional_pct", 1.0))
    if not (0.0 < max_notional <= 1.0):
        raise ValueError(
            "Invalid config: sizing.max_notional_pct must satisfy 0 < x <= 1.0"
        )


def _validate_stop_order(cfg: Dict[str, Any]) -> None:
    stop_cfg = cfg.get("stop_order", {})
    if not stop_cfg:
        return

    order_type = str(stop_cfg.get("order_type", "STOP_LOSS")).upper()
    if order_type not in STOP_ORDER_TYPES:
        raise ValueError(
            f"Invalid config: stop_order.order_type must be one of {STOP_ORDER_TYPES}, "
            f"got {order_type!r}"
        )

    prefix = str(stop_cfg.get("client_order_prefix", "aivc-sl"))
    if not _CLIENT_ORDER_PREFIX_RE.match(prefix):
        raise ValueError(
            "Invalid config: stop_order.client_order_prefix must match "
            "^[A-Za-z0-9_.-]{1,20}$"
        )

    offset = float(stop_cfg.get("limit_offset_pct", 0.0))
    if not (0.0 <= offset < 100.0):
        raise ValueError(
            "Invalid config: stop_order.limit_offset_pct must satisfy 0 <= x < 100"
        )
