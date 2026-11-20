"""起動前チェック（preflight）– Testnet 用.

systemd の ``ExecStartPre`` から呼ぶ前提。ここが落ちると ``main_live`` は起動しない
（＝「取引所の状態が分からないまま動き出す」ことを防ぐ）。

確認する内容:

1. 接続先が testnet であること（本番 URL なら即エラー）
2. 署名付き GET が通ること（時刻同期・残高）
3. 未約定注文の一覧（人の目で見るため表示）
4. ``StopOrderManager.reconcile`` による「建玉 ⇔ 取引所 STOP」の照合
   - 建玉ありで stop なし → その場で設置（＝起動前 reconcile）
   - stop のみで建玉なし → 孤児として取消
   - 数量不一致 → 数量を合わせる

終了コード: 0 = 起動可 / 2 = 要確認（エラーまたは注意） / 1 = 実行不能

使い方::

    .venv/bin/python preflight_reconcile.py \
        --config /opt/aivc_trade/config.testnet.yaml \
        --env-file /opt/aivc_trade/.env.testnet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aivc_trade.config.loader import load_config, operational_symbols, resolve_base_url
from aivc_trade.data.binance_client import BinanceClient
from aivc_trade.execution.stop_order_manager import StopOrderManager


def load_env(path: Path) -> None:
    """Load KEY=VALUE lines without printing any value."""
    if not path.exists():
        sys.exit(f"[preflight] env file not found: {path}")
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")
    # main_live は BINANCE_API_KEY / BINANCE_API_SECRET を読む。.env.testnet が
    # TESTNET_* しか持たない場合はここで別名を付ける（値は出力しない）。
    if not os.environ.get("BINANCE_API_KEY") and os.environ.get("BINANCE_TESTNET_API_KEY"):
        os.environ["BINANCE_API_KEY"] = os.environ["BINANCE_TESTNET_API_KEY"]
    if not os.environ.get("BINANCE_API_SECRET") and os.environ.get(
        "BINANCE_TESTNET_API_SECRET"
    ):
        os.environ["BINANCE_API_SECRET"] = os.environ["BINANCE_TESTNET_API_SECRET"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Testnet 起動前チェック")
    ap.add_argument("--config", default=None, help="config.yaml（既定: AIVC_CONFIG_PATH）")
    ap.add_argument("--env-file", default=".env.testnet")
    ap.add_argument("--state-file", default=None, help="既定: config.persistence.state_file")
    args = ap.parse_args()

    load_env(Path(args.env_file))
    cfg = load_config(args.config)
    base_url = resolve_base_url(cfg)
    if "testnet" not in base_url:
        print(f"[preflight] FAIL: endpoint is not testnet: {base_url}")
        return 1
    print(f"[preflight] endpoint: {base_url} (env={cfg['exchange']['env']})")

    api_key = cfg.get("secrets", {}).get("api_key", "") or os.environ.get("BINANCE_API_KEY", "")
    api_secret = cfg.get("secrets", {}).get("api_secret", "") or os.environ.get(
        "BINANCE_API_SECRET", ""
    )
    if not api_key or not api_secret:
        print("[preflight] FAIL: BINANCE_API_KEY / BINANCE_API_SECRET が未設定")
        return 1

    client = BinanceClient(api_key=api_key, api_secret=api_secret, base_url=base_url)
    quote = str(cfg.get("exchange", {}).get("quote_asset", "JPY"))
    symbols = operational_symbols(cfg)

    try:
        offset = client.sync_time()
        quote_balance = client.get_account_balance(quote)
    except Exception as e:
        print(f"[preflight] FAIL: signed GET failed: {e}")
        return 1
    print(f"[preflight] signed GET ok (clock offset {offset}ms), {quote} free={quote_balance:.2f}")

    for sym in symbols:
        try:
            price = client.get_ticker_price(sym)
            print(f"[preflight] {sym} last={price}")
        except Exception as e:
            print(f"[preflight] WARN: price unavailable for {sym}: {e}")

    open_orders = []
    try:
        open_orders = client.get_open_orders() or []
    except Exception as e:
        print(f"[preflight] WARN: open orders unavailable: {e}")
    print(f"[preflight] open orders: {len(open_orders)}")
    for order in open_orders:
        print(
            "  - {symbol} {side} {type} qty={origQty} stop={stopPrice} id={orderId}".format(
                symbol=order.get("symbol"),
                side=order.get("side"),
                type=order.get("type"),
                origQty=order.get("origQty"),
                stopPrice=order.get("stopPrice"),
                orderId=order.get("orderId"),
            )
        )

    # --- 建玉 ⇔ 取引所 STOP の照合（起動前 reconcile） ------------------
    state_path = Path(args.state_file or cfg["persistence"]["state_file"])
    positions = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            for pos in state.get("positions") or []:
                positions[pos["symbol"]] = pos
        except Exception as e:
            print(f"[preflight] FAIL: state file unreadable ({state_path}): {e}")
            return 1
    print(f"[preflight] state: {state_path} ({len(positions)} position(s))")

    steps = {}
    for sym in symbols:
        try:
            steps[sym] = float(client.get_lot_size(sym)["stepSize"])
        except Exception as e:
            print(f"[preflight] WARN: lot size unavailable for {sym}: {e}")

    stop_om = StopOrderManager(client, cfg)
    if not stop_om.enabled:
        print("[preflight] stop_order disabled – exchange-side safety net is OFF")
    try:
        report = stop_om.reconcile(positions, steps=steps)
    except Exception as e:
        print(f"[preflight] FAIL: reconcile crashed: {e}")
        return 2

    print(f"[preflight] reconcile: {report.summary()}")
    for line in report.details():
        print(f"  - {line}")
    if report.needs_attention or report.has_errors:
        print("[preflight] NEEDS ATTENTION – 内容を確認してから起動してください")
        return 2

    print("[preflight] OK: ready to start")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
