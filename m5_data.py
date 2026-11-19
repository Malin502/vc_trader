"""M5 – 公開データ取得（バックテスト用 1h 足 + exchangeInfo のロット情報）.

公開エンドポイントのみ使用。API キーは不要で、実発注も行わない。

* 1h 足: 2024-03-01 (ウォームアップ込み) 〜 現在
* 5m 足: simulator の trend 版は 5m 特徴量を使わないため取得しない
  （``Simulator.run`` は feat_5m を構築するがシグナル/決済では参照しない）
* exchangeInfo: LOT_SIZE / NOTIONAL を JSON に保存し、simulator の
  ``exchange.lot_info`` に渡して数量丸めを実口座と同じ条件にする

使い方::

    .venv/bin/python m5_data.py            # 取得してキャッシュ
    .venv/bin/python m5_data.py --force    # 再取得
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aivc_trade.config.loader import load_config
from aivc_trade.core.logger import get_logger, setup_logger
from aivc_trade.data.binance_client import BinanceClient
from aivc_trade.data.candles_store import CandlesStore

log = get_logger("m5_data")

DATA_START = datetime(2024, 3, 1, tzinfo=timezone.utc)
LOT_INFO_PATH = Path("data/lot_info.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="キャッシュを無視して再取得")
    args = ap.parse_args()

    setup_logger()
    cfg = load_config()
    store = CandlesStore(cfg["persistence"]["candles_dir"])
    client = BinanceClient()  # public endpoints only

    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    fresh_limit = end - pd.Timedelta(hours=2)

    lot_info = {}
    for sym in cfg["exchange"]["symbols"]:
        lot = client.get_lot_size(sym)
        lot_info[sym] = {
            "step_size": lot["stepSize"],
            "min_qty": lot["minQty"],
            "min_notional": lot["minNotional"],
        }
        log.info(f"lot_info {sym}: {lot_info[sym]}")

        cached = store.load(sym, "1h")
        if not cached.empty and not args.force:
            cached["ts"] = pd.to_datetime(cached["ts"], utc=True)
            if cached["ts"].min() <= DATA_START and cached["ts"].max() >= fresh_limit:
                log.info(f"{sym}: cache hit ({len(cached)} rows)")
                continue

        log.info(f"{sym}: downloading 1h {DATA_START.date()} → {end.date()}")
        df = client.fetch_klines_full(sym, "1h", DATA_START, end)
        if df.empty:
            log.error(f"{sym}: no data returned")
            continue
        store.save(sym, "1h", df)
        log.info(f"{sym}: saved {len(df)} rows ({df['ts'].min()} → {df['ts'].max()})")

    LOT_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOT_INFO_PATH.write_text(json.dumps(lot_info, indent=2))


if __name__ == "__main__":
    main()
