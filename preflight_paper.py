"""起動前チェック（preflight）– 実板ペーパー用.

systemd の ``ExecStartPre`` から呼ぶ。Testnet 版（``preflight_reconcile.py``）の
置き換えで、**署名付き GET を行わない**（資格情報を読まない）点が最大の違い。

確認する内容:

1. ``exchange.env`` が ``paper`` であること（prod / testnet なら即エラー）
2. 解決した接続先が本番の公開ホスト（``https://api.binance.com``）であること
3. 監視銘柄すべてで公開エンドポイントが応答すること
   - ``/api/v3/ticker/price``（現在値）
   - ``/api/v3/depth``（板: 模擬約定の入力）
   - ``/api/v3/klines``（1h 足）
4. 模擬口座ファイルがあれば残高を表示する（無ければ初期資金から開始）

終了コード: 0 = 起動可 / 2 = 要確認 / 1 = 実行不能

使い方::

    .venv/bin/python preflight_paper.py --config /opt/aivc_trade/config.paper.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aivc_trade.config.loader import (  # noqa: E402
    is_paper_env,
    load_config,
    resolve_base_url,
    watched_symbols,
)
from aivc_trade.data.public_client import PublicMarketClient  # noqa: E402

PUBLIC_HOST = "https://api.binance.com"


def main() -> int:
    ap = argparse.ArgumentParser(description="実板ペーパーの起動前チェック")
    ap.add_argument("--config", required=True, help="paper 用 config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    env = str((cfg.get("exchange", {}) or {}).get("env") or "").strip().lower()

    print(f"[preflight-paper] config={args.config}")
    print(f"[preflight-paper] exchange.env={env}")
    if env != "paper" or not is_paper_env(cfg):
        print("[preflight-paper] FAIL: exchange.env must be 'paper'", file=sys.stderr)
        return 1

    try:
        base_url = resolve_base_url(cfg)
    except Exception as e:
        print(f"[preflight-paper] FAIL: {e}", file=sys.stderr)
        return 1
    print(f"[preflight-paper] endpoint={base_url}")
    if base_url != PUBLIC_HOST:
        print(
            f"[preflight-paper] FAIL: paper must use the public host {PUBLIC_HOST}",
            file=sys.stderr,
        )
        return 1

    # 資格情報は読まない（PublicMarketClient は署名コードを持たない）。
    client = PublicMarketClient(base_url=base_url)
    symbols = watched_symbols(cfg)
    if not symbols:
        print("[preflight-paper] FAIL: no symbols configured", file=sys.stderr)
        return 1

    problems = []
    for sym in symbols:
        try:
            price = client.get_ticker_price(sym)
            book = client.fetch_order_book(sym, limit=5)
            bars = client.fetch_klines(sym, "1h", limit=3)
            bids = len((book or {}).get("bids") or [])
            asks = len((book or {}).get("asks") or [])
            print(
                f"  {sym}: price={price} book=bids{bids}/asks{asks} "
                f"1h_bars={len(bars)}"
            )
            if bids == 0 or asks == 0:
                problems.append(f"{sym}: empty book")
            if len(bars) == 0:
                problems.append(f"{sym}: no klines")
        except Exception as e:
            problems.append(f"{sym}: {e}")

    account_file = (cfg.get("paper", {}) or {}).get("account_file")
    if account_file and Path(account_file).exists():
        try:
            data = json.loads(Path(account_file).read_text(encoding="utf-8"))
            print(
                f"[preflight-paper] paper account: "
                f"{data.get('quote_balance')} {data.get('quote_asset', '')} "
                f"base={data.get('base_balances')} fills={data.get('fills')}"
            )
        except Exception as e:
            problems.append(f"paper account file unreadable: {e}")
    else:
        print("[preflight-paper] paper account: fresh (initial_equity から開始)")

    if problems:
        for p in problems:
            print(f"[preflight-paper] PROBLEM: {p}", file=sys.stderr)
        print("[preflight-paper] 要確認: 解消するまで起動しない", file=sys.stderr)
        return 2

    print("[preflight-paper] OK: public data reachable, no credentials required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
