"""AIVC Trade – Backtest runner (Phase A).

Downloads (or loads cached) historic data, runs the simulator,
and prints a performance report.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# Allow `python aivc_trade/main_backtest.py` from any cwd.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aivc_trade.config.loader import load_config
from aivc_trade.core.logger import get_logger, setup_logger
from aivc_trade.data.binance_client import BinanceClient
from aivc_trade.data.candles_store import CandlesStore
from aivc_trade.backtest.simulator import Simulator
from aivc_trade.backtest.metrics import compute_metrics, print_report
from aivc_trade.backtest.reporting import save_backtest_reports, trades_to_dataframe
from aivc_trade.backtest.walkforward import walk_forward

log = get_logger("main_backtest")


def _ensure_data(
    cfg: Dict[str, Any],
    store: CandlesStore,
    client: Any,
    offline: bool = False,
) -> tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Load or download candle data for the backtest period.

    Parameters
    ----------
    client : BinanceClient or None (when offline=True)
    offline : if True, only use cached data, never hit the network.
    """
    start = datetime.strptime(cfg["backtest"]["start_date"], "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )
    end = datetime.strptime(cfg["backtest"]["end_date"], "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )

    candles_1h: Dict[str, pd.DataFrame] = {}
    candles_5m: Dict[str, pd.DataFrame] = {}

    for sym in cfg["exchange"]["symbols"]:
        # 1h
        cached = store.load(sym, "1h")
        if not cached.empty:
            cached["ts"] = pd.to_datetime(cached["ts"], utc=True)
            min_ts, max_ts = cached["ts"].min(), cached["ts"].max()
            if min_ts <= start and max_ts >= end:
                log.info(f"Using cached 1h data for {sym} ({len(cached)} rows)")
                candles_1h[sym] = cached[
                    (cached["ts"] >= start) & (cached["ts"] <= end)
                ].reset_index(drop=True)
            elif not offline:
                log.info(f"Downloading 1h data for {sym}...")
                df = client.fetch_klines_full(sym, "1h", start, end)
                if not df.empty:
                    store.save(sym, "1h", df)
                    candles_1h[sym] = df
            else:
                log.warning(
                    f"Offline mode: cached 1h data for {sym} does not cover "
                    f"requested period ({min_ts.date()}..{max_ts.date()}), skipping"
                )
        elif not offline:
            log.info(f"Downloading 1h data for {sym}...")
            df = client.fetch_klines_full(sym, "1h", start, end)
            if not df.empty:
                store.save(sym, "1h", df)
                candles_1h[sym] = df
        else:
            log.warning(f"Offline mode: no cached 1h data for {sym}, skipping")

        # 5m
        cached_5m = store.load(sym, "5m")
        if not cached_5m.empty:
            cached_5m["ts"] = pd.to_datetime(cached_5m["ts"], utc=True)
            min_ts5, max_ts5 = cached_5m["ts"].min(), cached_5m["ts"].max()
            if min_ts5 <= start and max_ts5 >= end:
                log.info(f"Using cached 5m data for {sym} ({len(cached_5m)} rows)")
                candles_5m[sym] = cached_5m[
                    (cached_5m["ts"] >= start) & (cached_5m["ts"] <= end)
                ].reset_index(drop=True)
            elif not offline:
                log.info(f"Downloading 5m data for {sym}...")
                df = client.fetch_klines_full(sym, "5m", start, end)
                if not df.empty:
                    store.save(sym, "5m", df)
                    candles_5m[sym] = df
            else:
                log.warning(
                    f"Offline mode: cached 5m data for {sym} does not cover "
                    f"requested period, skipping"
                )
        elif not offline:
            log.info(f"Downloading 5m data for {sym}...")
            df = client.fetch_klines_full(sym, "5m", start, end)
            if not df.empty:
                store.save(sym, "5m", df)
                candles_5m[sym] = df
        else:
            log.warning(f"Offline mode: no cached 5m data for {sym}, skipping")

    return candles_1h, candles_5m


def _data_quality_checks(
    candles_1h: Dict[str, pd.DataFrame],
    candles_5m: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for tf, blob in [("1h", candles_1h), ("5m", candles_5m)]:
        for sym, df in blob.items():
            if df.empty:
                rows.append(
                    {
                        "symbol": sym,
                        "timeframe": tf,
                        "rows": 0,
                        "duplicates": 0,
                        "null_rows": 0,
                        "naive_ts": 0,
                    }
                )
                continue
            ts = pd.to_datetime(df["ts"], errors="coerce", utc=False)
            naive_ts = int(sum(getattr(x, "tzinfo", None) is None for x in ts.dropna()))
            rows.append(
                {
                    "symbol": sym,
                    "timeframe": tf,
                    "rows": len(df),
                    "duplicates": int(ts.duplicated().sum()),
                    "null_rows": int(df[["open", "high", "low", "close", "volume"]].isna().any(axis=1).sum()),
                    "naive_ts": naive_ts,
                }
            )
    return pd.DataFrame(rows)


def _risk_and_fee_checks(cfg: Dict[str, Any], trades_df: pd.DataFrame) -> pd.DataFrame:
    fees_cfg = cfg.get("fees", {})
    costs_cfg = cfg.get("costs", {})
    slippage = float(costs_cfg.get("slippage_bps", 0.0))
    if fees_cfg.get("enabled", False):
        model = str(fees_cfg.get("model", "flat")).lower()
        if model == "binance_spot":
            spot_cfg = fees_cfg.get("binance_spot", {})
            commission = float(spot_cfg.get("taker", 0.0010)) * 10_000.0
            discount = float(spot_cfg.get("bnb_discount", 0.0))
            commission = commission * max(0.0, 1.0 - discount)
        else:
            commission = float(fees_cfg.get("flat_rate", 0.001)) * 10_000.0
    else:
        commission = float(costs_cfg.get("commission_bps", 0.0))
    expected_roundtrip = 2.0 * (commission + slippage)
    configured_roundtrip = float(costs_cfg.get("roundtrip_cost_bps", expected_roundtrip))
    rows = [
        {
            "check": "roundtrip_cost_bps_consistency",
            "value": configured_roundtrip,
            "expected": expected_roundtrip,
            "ok": abs(configured_roundtrip - expected_roundtrip) < 1e-9,
        }
    ]
    if not trades_df.empty and "planned_risk" in trades_df.columns:
        breach = trades_df[
            (trades_df["net_pnl"] < 0)
            & (trades_df["planned_risk"] > 0)
            & (trades_df["net_pnl"].abs() > trades_df["planned_risk"] * 1.05)
        ]
        rows.append(
            {
                "check": "max_loss_vs_planned_risk",
                "value": int(len(breach)),
                "expected": 0,
                "ok": len(breach) == 0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    setup_logger()
    cfg = load_config()

    log.info("=" * 50)
    log.info("AIVC Trade – Phase A Backtest")
    log.info(f"Period: {cfg['backtest']['start_date']} → {cfg['backtest']['end_date']}")
    log.info(f"Initial equity: {cfg['backtest']['initial_equity']}")
    log.info("=" * 50)

    offline = bool(cfg["backtest"].get("offline", False))
    client = None
    if not offline:
        client = BinanceClient()  # public endpoints only
    else:
        log.info("Offline mode: skipping Binance client initialization")
    store = CandlesStore("data/candles/test")
    log.info("Using candles dir for backtest: data/candles/test")

    candles_1h, candles_5m = _ensure_data(cfg, store, client, offline=offline)

    if not candles_1h:
        log.error("No 1h candle data available – aborting")
        sys.exit(1)

    if cfg["backtest"].get("walkforward", False):
        log.info("Running walk-forward analysis...")
        results = walk_forward(candles_1h, candles_5m, cfg)
        print("\n\nWalk-Forward Results:")
        for r in results:
            m = r["metrics"]
            print(
                f"  Window {r['window']}: "
                f"{r['test_start'].date()}→{r['test_end'].date()} | "
                f"trades={r['n_trades']} PF={m['profit_factor']:.2f} "
                f"DD={m['max_dd']:.2%} "
                f"{'PASS' if m['passed'] else 'FAIL'}"
            )
    else:
        sim = Simulator(cfg)
        trades, equity_curve = sim.run(candles_1h, candles_5m)
        metrics = compute_metrics(
            trades, equity_curve, cfg["backtest"]["initial_equity"]
        )
        if hasattr(sim, "last_run_stats"):
            metrics["regime_halts"] = int(sim.last_run_stats.get("regime_halts", 0))
        print_report(metrics)

        # Save results
        results_dir = Path("data/backtest_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        save_backtest_reports(metrics, trades, equity_curve, results_dir, candles_1h=candles_1h)
        log.info(f"Backtest reports saved → {results_dir}")
        skips = []
        if hasattr(sim, "last_run_stats"):
            skips = sim.last_run_stats.get("phase_b_skips", []) or []
        if skips:
            pb_cfg = cfg.get("phaseb", cfg.get("phase_b", {}))
            log_skips = bool(pb_cfg.get("log_skips", pb_cfg.get("gate", {}).get("log_skips", True)))
            if log_skips:
                log_path = Path("logs/phaseb_skips.csv")
                log_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(skips).to_csv(log_path, index=False)
                # Keep a copy in backtest results for experiment tracking.
                pd.DataFrame(skips).to_csv(results_dir / "phaseb_skips.csv", index=False)
                log.info(f"PhaseB skips saved → {log_path} ({len(skips)} rows)")

        quality_df = _data_quality_checks(candles_1h, candles_5m)
        quality_df.to_csv(results_dir / "data_quality_checks.csv", index=False)
        trades_df = trades_to_dataframe(trades)
        checks_df = _risk_and_fee_checks(cfg, trades_df)
        checks_df.to_csv(results_dir / "safety_checks.csv", index=False)
        log.info(f"Data/safety checks saved → {results_dir / 'data_quality_checks.csv'} , {results_dir / 'safety_checks.csv'}")


if __name__ == "__main__":
    main()
