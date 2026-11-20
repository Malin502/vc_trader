"""main_live ループの通し確認（有限回で必ず止まるハーネス）.

``--env`` で対象を選ぶ:

* ``paper``（既定）: 実板ペーパー。**資格情報を読まない**。接続先は本番の公開
  ホスト（``https://api.binance.com``）で、発注は模擬ブローカーが行う。
* ``testnet``: 旧 Testnet 経路（``.env.testnet`` を読む）。M4b の再現用に残す。

signal を送れない環境（コンテナ / CI / 一部のサンドボックス）でも安全に終わる
ように、``_sleep_interruptible`` を差し替えて「N 回目の終わりに停止フラグ」を
立てる。停止経路は systemd の SIGTERM とまったく同じ（ループ先頭でフラグを見て
状態を保存してから break）なので、停止処理そのものの確認にもなる。

使い方::

    .venv/bin/python paper_loop_check.py --config config.paper.yaml --iterations 4

終了コード: 0 = N 回転して正常終了 / 1 = 途中で致命的エラー
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ENV_FILE = Path(__file__).resolve().parent / ".env.testnet"
PUBLIC_HOST = "https://api.binance.com"


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
    ap = argparse.ArgumentParser(description="main_live の通し確認（paper / testnet）")
    ap.add_argument("--config", required=True, help="対象 env 用 config.yaml")
    ap.add_argument("--iterations", type=int, default=3, help="回すループ回数")
    ap.add_argument("--interval", type=float, default=5.0, help="ループ間隔（秒）")
    ap.add_argument(
        "--env",
        choices=("paper", "testnet"),
        default="paper",
        help="paper = 実板ペーパー（資格情報不要） / testnet = 旧 Testnet 経路",
    )
    args = ap.parse_args()

    if args.env == "testnet":
        load_env(ENV_FILE)
    os.environ["AIVC_CONFIG_PATH"] = str(Path(args.config).resolve())
    os.environ["AIVC_EXCHANGE_ENV"] = args.env
    os.environ["AIVC_LOOP_INTERVAL"] = str(args.interval)

    from aivc_trade import main_live

    # 誤って別の env を触らないための最後の砦（設定解決後にもう一度確認する）
    from aivc_trade.config.loader import load_config, resolve_base_url

    resolved = resolve_base_url(load_config())
    if args.env == "paper":
        if resolved != PUBLIC_HOST:
            sys.exit(f"refusing to run: resolved endpoint {resolved} is not {PUBLIC_HOST}")
    elif "testnet" not in resolved:
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
    print(f"[loop-check] OK: {iterations['n']} iterations, env={args.env}, endpoint={resolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

