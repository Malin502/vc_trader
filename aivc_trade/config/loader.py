"""Configuration loader – reads YAML and merges environment overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


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

    return cfg
