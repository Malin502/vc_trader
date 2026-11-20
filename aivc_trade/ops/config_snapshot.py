"""起動時の構成スナップショット（``event: CONFIG`` 用）.

ペーパー運転では「いつからどの構成で走っていたか」が後から必要になる
（G3 のみ / G3＋W2 など）。起動時に 1 行残しておけば、ジャーナル上の CONFIG
レコードの区間＝その構成で走っていた期間として再構成できる。

ここで扱うのは**有効・無効と版**だけ。秘密情報（API キー等）は含めない
（``load_config`` が paper で落としている前提だが、念のため明示的に拾わない）。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

#: スナップショットの形式版。フィールドを増やすときは上げる。
CONFIG_SNAPSHOT_VERSION = 1


def _enabled(flag: Any) -> bool:
    return bool((flag or {}).get("enabled", False))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def code_revision(repo_root: Optional[Path] = None) -> Optional[str]:
    """実行中コードの版（環境変数優先、無ければ git の短縮 HEAD）。

    配布物に ``.git`` が無い場合は ``None``（記録は落とさない）。
    """
    env = os.environ.get("AIVC_CODE_REVISION")
    if env:
        return env.strip()[:64]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root or _repo_root()),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:  # git が無い・遅い等。記録のための付加情報なので諦める
        return None
    value = (out.stdout or "").strip()
    return value or None


def build_config_snapshot(
    cfg: Dict[str, Any],
    *,
    config_path: Optional[str] = None,
    revision: Optional[str] = None,
) -> Dict[str, Any]:
    """構成の要約（有効なゲート・版・識別子）を返す。

    ``config_id`` は同じ意味の構成なら同じ値になる（関係する設定だけを
    正規化した JSON の SHA-256 先頭 12 桁）。``ts`` は呼び出し側が付ける。
    """
    htf = (cfg.get("htf_gate") or {}).get("g3") or {}
    backtest_gates = cfg.get("gates") or {}
    whipsaw = cfg.get("whipsaw") or {}
    entry = cfg.get("entry") or {}
    exchange = cfg.get("exchange") or {}
    position = cfg.get("position") or {}

    snapshot: Dict[str, Any] = {
        "config_snapshot_version": CONFIG_SNAPSHOT_VERSION,
        "exchange_env": str(exchange.get("env") or "").strip().lower(),
        "symbols": [str(s) for s in (exchange.get("symbols") or [])],
        "shadow_symbols": [str(s) for s in (exchange.get("shadow_symbols") or [])],
        "entry_mode": str(entry.get("mode") or "full"),
        "risk_mode": str((cfg.get("sizing") or {}).get("risk_mode") or ""),
        "max_concurrent": int(position.get("max_concurrent", 1) or 1),
        "htf_gate": {
            "g3": {
                "enabled": bool(htf.get("enabled", False)),
                "symbol": str(htf.get("symbol") or "BTCJPY"),
                "sma_period": int(htf.get("sma_period", 200) or 200),
                "fail_closed": bool(htf.get("fail_closed", True)),
            }
        },
        # バックテスト専用（live では列を作らない）。記録だけしておく。
        "backtest_gates": {
            key: _enabled(backtest_gates.get(key)) for key in ("g1", "g2", "g3")
        },
        "whipsaw": {
            key: _enabled(whipsaw.get(key)) for key in ("w1", "w2", "w3")
        },
        "code_revision": revision if revision is not None else code_revision(),
        "config_path": config_path or os.environ.get("AIVC_CONFIG_PATH") or None,
    }
    snapshot["enabled_components"] = enabled_components(snapshot)
    snapshot["config_id"] = _config_id(snapshot)
    return snapshot


def enabled_components(snapshot: Dict[str, Any]) -> List[str]:
    """人が読む用の有効構成リスト（例 ``["G3", "W2"]``）。"""
    out: List[str] = []
    if (snapshot.get("htf_gate") or {}).get("g3", {}).get("enabled"):
        out.append("G3")
    for key, on in sorted((snapshot.get("whipsaw") or {}).items()):
        if on:
            out.append(key.upper())
    for key, on in sorted((snapshot.get("backtest_gates") or {}).items()):
        if on:
            out.append(f"BT-{key.upper()}")
    return out


def _config_id(snapshot: Dict[str, Any]) -> str:
    """構成の同一性だけを拾った短いハッシュ（版・パス・時刻は含めない）。"""
    keys = (
        "exchange_env",
        "symbols",
        "shadow_symbols",
        "entry_mode",
        "risk_mode",
        "max_concurrent",
        "htf_gate",
        "backtest_gates",
        "whipsaw",
    )
    material = {k: snapshot.get(k) for k in keys}
    blob = json.dumps(material, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
