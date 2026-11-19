"""M4b – Testnet 実走スモーク（STOP_LOSS 結線の実地確認）.

`env=testnet` の取引所に対して、次の 1〜6 を最小数量で実地確認する使い捨てハーネス。
自然な値動きは待たない（手動トリガ）。

  1. 署名付き GET（残高）
  2. エントリ約定 → 取引所側 STOP_LOSS 設置
  3. ラチェット（cancelReplace で stop 切り上げ）
  3b. 置換拒否時に無防備にしない（直前の水準で復旧）
  4. 部分利確相当の数量減 → stop 数量の縮小
  5. 決済 → stop 取り消し（孤児を残さない）
  6. 再起動相当の照合（採用 / 孤児 stop の検出と掃除）

資格情報は作業コピー直下の ``.env.testnet`` から読む。**値は出力しない**。
本番 API には絶対に接続しない（testnet 以外の URL を検出したら即終了）。

使い方::

    .venv/bin/python testnet_smoke_m4b.py [--symbol BTCJPY] [--orphan-symbol ETHJPY]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aivc_trade.config.loader import resolve_base_url
from aivc_trade.core.types import Position
from aivc_trade.data.binance_client import BinanceClient
from aivc_trade.execution.stop_order_manager import StopOrderManager

ENV_FILE = Path(__file__).resolve().parent / ".env.testnet"
RESULTS: dict[str, bool] = {}


def load_env() -> None:
    if not ENV_FILE.exists():
        sys.exit(f"{ENV_FILE} がありません（Testnet 資格情報を配置してください）")
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")
    os.environ["BINANCE_API_KEY"] = os.environ["BINANCE_TESTNET_API_KEY"]
    os.environ["BINANCE_API_SECRET"] = os.environ["BINANCE_TESTNET_API_SECRET"]


def make_config(symbols: list[str]) -> dict:
    cfg = {
        "exchange": {
            "env": "testnet",
            "base_url": "https://api.binance.com",
            "testnet_base_url": "https://testnet.binance.vision",
            "quote_asset": "JPY",
            "symbols": symbols,
            "market_type": "spot",
        },
        "position": {"max_concurrent": 2, "max_per_symbol": 1, "cooldown_hours": 24},
        "sizing": {
            "risk_mode": "paper",
            "risk_per_trade_paper": 0.02,
            "risk_per_trade_live": 0.01,
            "max_notional_pct": 0.5,
        },
        "risk": {
            "initial_sl": {"min_sl_pct": 0.6, "max_sl_pct": 2.5},
            "early_fail_exit": {"window_bars": 6},
            "breakeven": {"buffer_pct": 0.05},
        },
        "entry_signal": {"breakout": {"buffer_pct": 0.10}},
        "runner": {"trail_atr_k": 3.8},
        "stop_order": {
            "enabled": True,
            "order_type": "STOP_LOSS",
            "client_order_prefix": "aivc-sl",
            "limit_offset_pct": 0.10,
            "safety_buffer_pct": 0.0,
        },
        "notification": {"enabled": False, "discord_webhook_url": ""},
        "persistence": {
            "state_file": "data/position_state_testnet.json",
            "candles_dir": "data/candles",
            "log_dir": "logs",
        },
    }
    base_url = resolve_base_url(cfg)
    if "testnet" not in base_url:
        sys.exit(f"refusing to run: base_url={base_url} is not a testnet host")
    cfg["_base_url"] = base_url
    return cfg


def min_trade_qty(client: BinanceClient, symbol: str, price: float) -> float:
    """半分売っても minNotional を満たす最小数量（LOT_SIZE step に切り上げ）。"""
    lot = client.get_lot_size(symbol)
    step = lot["stepSize"] or 1e-8
    need = max(lot["minQty"], (lot["minNotional"] * 2.0) / price)
    return round(math.ceil(need / step) * step, 10)


def open_stops(client: BinanceClient, symbol: str) -> list[tuple]:
    return [
        (o["orderId"], o["clientOrderId"], o["stopPrice"], o["origQty"], o["type"],
         o["side"], o["status"])
        for o in client.get_open_orders(symbol)
    ]


def mark(step: str, ok: bool, note: str) -> None:
    RESULTS[step] = bool(ok)
    print(f"[{step}] {'PASS' if ok else 'FAIL'} {note}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCJPY")
    ap.add_argument("--orphan-symbol", default="ETHJPY")
    args = ap.parse_args()
    sym, orphan_sym = args.symbol, args.orphan_symbol

    load_env()
    cfg = make_config([sym, orphan_sym])
    client = BinanceClient(
        api_key=os.environ["BINANCE_TESTNET_API_KEY"],
        api_secret=os.environ["BINANCE_TESTNET_API_SECRET"],
        base_url=cfg["_base_url"],
    )
    steps = {s: client.get_lot_size(s)["stepSize"] for s in (sym, orphan_sym)}
    print(f"endpoint: {cfg['_base_url']} (env=testnet)")

    # -- 1. 署名付き GET ------------------------------------------------
    offset = client.sync_time()
    quote = client.get_account_balance("JPY")
    base_asset = sym[: -len("JPY")] if sym.endswith("JPY") else sym
    base_free = client.get_account_balance(base_asset)
    mark("step1", quote > 0, f"signed GET ok (offset={offset}ms), "
                             f"JPY free={quote}, {base_asset} free={base_free}")

    price = client.get_ticker_price(sym)
    qty = min_trade_qty(client, sym, price)
    print(f"[info] {sym} price={price} qty={qty} lot={client.get_lot_size(sym)}")
    if client.get_open_orders(sym):
        sys.exit(f"{sym} に未処理の注文があります。片付けてから実行してください: "
                 f"{open_stops(client, sym)}")

    # -- 2. エントリ → 取引所側 STOP_LOSS -------------------------------
    entry = client.place_market_order(sym, "BUY", qty)
    filled = float(entry["executedQty"])
    avg = float(entry["cummulativeQuoteQty"]) / filled
    pos = Position(symbol=sym, qty=filled, entry_price=avg,
                   stop_price=round(avg * 0.97, 1), entry_ts=None)
    om = StopOrderManager(client, cfg)
    st = om.place_initial_stop(pos, om.safety_net_price(pos.stop_price))
    stops = open_stops(client, sym)
    mark("step2",
         st is not None and len(stops) == 1 and stops[0][4] == "STOP_LOSS"
         and stops[0][5] == "SELL" and abs(float(stops[0][3]) - filled) < 1e-9,
         f"entry {entry['orderId']} filled {filled} @ {avg:.2f}; "
         f"exchange stop id={stops[0][0] if stops else None} "
         f"stop={stops[0][2] if stops else None} qty={stops[0][3] if stops else None}")

    # -- 3. ラチェット ---------------------------------------------------
    first_id = stops[0][0] if stops else None
    target = round(client.get_ticker_price(sym) * 0.99, 1)
    st3 = om.update_stop(pos, target)
    stops = open_stops(client, sym)
    mark("step3",
         st3 is not None and st3.order_id is not None and len(stops) == 1
         and abs(float(stops[0][2]) - target) <= max(steps[sym], 1.0)
         and str(stops[0][0]) != str(first_id)
         and abs(st3.stop_price - float(stops[0][2])) < 1e-9,
         f"stop {first_id} -> {stops[0][0] if stops else None} "
         f"@ {stops[0][2] if stops else None} (requested {target})")

    # -- 3b. 置換拒否でも無防備にしない ----------------------------------
    resting = float(stops[0][2]) if stops else 0.0
    bad = round(client.get_ticker_price(sym) * 1.05, 1)   # 即トリガー=拒否
    st3b = om.update_stop(pos, bad)
    stops = open_stops(client, sym)
    mark("step3b",
         st3b is not None and len(stops) == 1
         and abs(float(stops[0][2]) - resting) <= max(steps[sym], 1.0),
         f"rejected {bad} -> re-armed at {stops[0][2] if stops else None}")

    # -- 4. 数量追随 -----------------------------------------------------
    partial = round(filled / 2.0, 10)
    client.place_market_order(sym, "SELL", partial)
    pos.qty = round(filled - partial, 10)
    st4 = om.resize_stop(pos, pos.qty, step=steps[sym])
    stops = open_stops(client, sym)
    mark("step4",
         st4 is not None and len(stops) == 1
         and abs(float(stops[0][3]) - pos.qty) < 1e-9,
         f"stop qty {filled} -> {stops[0][3] if stops else None} "
         f"(position {pos.qty})")

    # -- 5. 決済 → stop 取消 ---------------------------------------------
    client.place_market_order(sym, "SELL", pos.qty)
    cancelled = om.cancel(sym)
    mark("step5", cancelled and open_stops(client, sym) == [],
         f"cancel={cancelled}, open orders={open_stops(client, sym)}")

    # -- 6. 再起動照合 ---------------------------------------------------
    entry6 = client.place_market_order(sym, "BUY", qty)
    filled6 = float(entry6["executedQty"])
    avg6 = float(entry6["cummulativeQuoteQty"]) / filled6
    pos6 = Position(symbol=sym, qty=filled6, entry_price=avg6,
                    stop_price=round(avg6 * 0.98, 1), entry_ts=None)
    om6 = StopOrderManager(client, cfg)
    om6.place_initial_stop(pos6, om6.safety_net_price(pos6.stop_price))
    before = open_stops(client, sym)

    restarted = StopOrderManager(client, cfg)          # 新しいプロセス相当
    order = client.get_open_orders(sym)[0]
    pos_r = Position(symbol=sym, qty=float(order["origQty"]), entry_price=avg6,
                     stop_price=float(order["stopPrice"]), entry_ts=None)
    report = restarted.reconcile({sym: pos_r}, steps=steps)
    adopted = restarted.get_state(sym)
    mark("step6a",
         report.adopted == [sym] and not report.has_errors
         and open_stops(client, sym) == before
         and adopted is not None and adopted.order_id == str(order["orderId"]),
         f"adopted {sym} id={getattr(adopted, 'order_id', None)} "
         f"stop={getattr(adopted, 'stop_price', None)} (no writes)")

    oprice = client.get_ticker_price(orphan_sym)
    client.place_stop_order(
        symbol=orphan_sym, side="SELL",
        qty=min_trade_qty(client, orphan_sym, oprice),
        stop_price=round(oprice * 0.95, 1), order_type="STOP_LOSS",
        client_order_id=f"aivc-sl-{orphan_sym}-999",
    )
    report2 = restarted.reconcile({sym: pos_r}, steps=steps)
    mark("step6b",
         orphan_sym in report2.orphans_cancelled
         and open_stops(client, orphan_sym) == []
         and len(open_stops(client, sym)) == 1,
         f"orphan {orphan_sym} cancelled, held stop kept, errors={report2.errors}")

    # -- 後片付け --------------------------------------------------------
    client.place_market_order(sym, "SELL", filled6)
    restarted.cancel(sym)
    print(f"[cleanup] open orders: {sym}={open_stops(client, sym)} "
          f"{orphan_sym}={open_stops(client, orphan_sym)}")
    print(f"[cleanup] JPY free={client.get_account_balance('JPY')}")

    failed = [k for k, ok in RESULTS.items() if not ok]
    print("SUMMARY:", ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in RESULTS.items()))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
