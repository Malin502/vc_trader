"""Testnet での main_live ループ通し確認（有限回で必ず止まるハーネス）.

M4b では ``main_live`` 本体を通していなかった。ここでは **本番と同じコード** を
そのまま呼び、起動 → 照合 → ループ → (境界なら) シグナル生成 → 状態保存 までを
N 回転だけ回して抜ける。

signal を送れない環境（コンテナ / CI / 一部のサンドボックス）でも安全に終わる
ように、``_sleep_interruptible`` を差し替えて「N 回目の終わりに停止フラグ」を
立てる。停止経路は systemd の SIGTERM とまったく同じ（ループ先頭でフラグを見て
状態を保存してから break）なので、停止処理そのものの確認にもなる。

使い方::

    # .env.testnet を読み、設定は scratch 用のものを使う
    .venv/bin/python paper_loop_check.py --config /path/to/config.yaml --iterations 4

終了コード: 0 = N 回転して正常終了 / 1 = 途中で致命的エラー
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ENV_FILE = Path(__file__).resolve().parent / ".env.testnet"


def load_env(path: Path) -> None:
    if not path.exists():
        sys.exit(f"{path} がありません（Testnet 資格情報を配置してください）")
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")
    os.environ["BINANCE_API_KEY"] = os.environ.get("BINANCE_TESTNET_API_KEY", "")
    os.environ["BINANCE_API_SECRET"] = os.environ.get("BINANCE_TESTNET_API_SECRET", "")


def main() -> int:
    ap = argparse.ArgumentParser(description="main_live の Testnet 通し確認")
    ap.add_argument("--config", required=True, help="Testnet 用 config.yaml")
    ap.add_argument("--iterations", type=int, default=3, help="回すループ回数")
    ap.add_argument("--interval", type=float, default=5.0, help="ループ間隔（秒）")
    args = ap.parse_args()

    load_env(ENV_FILE)
    os.environ["AIVC_CONFIG_PATH"] = str(Path(args.config).resolve())
    os.environ["AIVC_EXCHANGE_ENV"] = "testnet"
    os.environ["AIVC_LOOP_INTERVAL"] = str(args.interval)
    if "testnet" not in os.environ.get(
        "BINANCE_BASE_URL", "https://testnet.binance.vision"
    ):
        sys.exit("refusing to run: base URL is not a testnet host")

    from aivc_trade import main_live

    # 本番 API を触らないための最後の砦（設定解決後にもう一度確認する）
    from aivc_trade.config.loader import load_config, resolve_base_url

    resolved = resolve_base_url(load_config())
    if "testnet" not in resolved:
        sys.exit(f"refusing to run: resolved endpoint {resolved} is not testnet")

    iterations = {"n": 0}

    def _bounded_sleep(seconds: float) -> None:
        iterations["n"] += 1
        print(
            f"[loop-check] iteration {iterations['n']}/{args.iterations} done",
            flush=True,
        )
        if iterations["n"] >= args.iterations:
            # SIGTERM と同じ経路（ループ先頭で検知 → 状態保存 → break）
            main_live._SHUTDOWN["requested"] = True

    main_live._sleep_interruptible = _bounded_sleep
    main_live.main()

    if iterations["n"] < args.iterations:
        print(f"[loop-check] fewer iterations than requested: {iterations['n']}")
        return 1
    print(f"[loop-check] OK: {iterations['n']} iterations, endpoint={resolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
