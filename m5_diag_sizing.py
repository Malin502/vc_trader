"""M5 – 診断: 数量が「リスク上限」と「枠上限」のどちらで決まっているか.

``sizing.risk_per_trade`` (paper 2% / live 1%) と ``sizing.max_notional_pct`` (0.5)
のどちらが binding かを、実際のエントリーで数える。あわせてストップ幅の分布を出す。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from aivc_trade.config.loader import load_config
from aivc_trade.core.logger import setup_logger
from aivc_trade.data.candles_store import CandlesStore
from aivc_trade.backtest.simulator import Simulator
from aivc_trade.core.types import Position

PERIOD_START = datetime(2024, 5, 1, tzinfo=timezone.utc)
SYMBOLS = ["BTCJPY", "ETHJPY", "XRPJPY", "SOLJPY"]

rows: list[dict] = []
_orig_open = Simulator._open_position


def _wrapped(self, signal, feat_at, feat_1h, ts, equity):
    pos: Position | None = _orig_open(self, signal, feat_at, feat_1h, ts, equity)
    if pos is not None:
        notional = pos.initial_qty * pos.entry_price
        rows.append(
            {
                "symbol": pos.symbol,
                "ts": ts,
                "entry_price": pos.entry_price,
                "stop_price": pos.stop_price,
                "stop_pct": (pos.entry_price - pos.stop_price) / pos.entry_price * 100,
                "qty": pos.initial_qty,
                "equity": equity,
                "notional_pct": notional / equity,
                "risk_pct": pos.initial_qty * (pos.entry_price - pos.stop_price) / equity * 100,
            }
        )
    return pos


def main() -> None:
    setup_logger()
    base = load_config()
    store = CandlesStore(base["persistence"]["candles_dir"])
    candles = {}
    for sym in SYMBOLS:
        df = store.load(sym, "1h")
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        candles[sym] = df.sort_values("ts").reset_index(drop=True)

    lot_info = json.loads(open("data/lot_info.json").read())
    Simulator._open_position = _wrapped  # type: ignore[method-assign]

    for mode in ("paper", "live"):
        rows.clear()
        cfg = json.loads(json.dumps(base, default=str))
        # deep copy しつつ、YAML 由来の値の型を保つため load_config を再取得する
        from aivc_trade.config.loader import load_config as _lc, resolve_risk_per_trade

        cfg = _lc()
        cfg["exchange"]["symbols"] = list(SYMBOLS)
        cfg["exchange"]["lot_info"] = lot_info
        cfg["sizing"]["risk_mode"] = mode
        cfg["sizing"].pop("risk_per_trade", None)
        cfg["sizing"]["risk_per_trade"] = resolve_risk_per_trade(cfg["sizing"])

        sim = Simulator(cfg)
        sim.run(candles, {}, trade_start=PERIOD_START)

        df = pd.DataFrame(rows)
        if df.empty:
            print(f"[{mode}] no entries")
            continue
        capped = (df["notional_pct"] >= 0.49).mean()
        print(
            f"[{mode}] entries={len(df)} "
            f"risk_target={cfg['sizing']['risk_per_trade']:.2%} "
            f"notional_cap_binding={capped:.1%} "
            f"median_notional_pct={df['notional_pct'].median():.3f} "
            f"median_risk_pct={df['risk_pct'].median():.2f}%"
        )
        print(
            "  stop_pct: "
            f"min={df['stop_pct'].min():.2f} p25={df['stop_pct'].quantile(.25):.2f} "
            f"median={df['stop_pct'].median():.2f} p75={df['stop_pct'].quantile(.75):.2f} "
            f"max={df['stop_pct'].max():.2f}"
        )
        by = df.groupby("symbol").agg(
            n=("qty", "size"),
            median_stop_pct=("stop_pct", "median"),
            median_risk_pct=("risk_pct", "median"),
            capped=("notional_pct", lambda s: float((s >= 0.49).mean())),
        )
        print(by.to_string())


if __name__ == "__main__":
    main()
