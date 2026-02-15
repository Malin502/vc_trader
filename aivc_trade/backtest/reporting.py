"""Backtest decomposition report outputs (tables + charts)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aivc_trade.core.types import TradeRecord


def trades_to_dataframe(trades: List[TradeRecord]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "symbol": t.symbol,
                "side": t.side.value if hasattr(t.side, "value") else str(t.side),
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "qty": t.qty,
                "entry_ts": t.entry_ts,
                "exit_ts": t.exit_ts,
                "exit_reason": t.exit_reason.value,
                "pnl": t.pnl,
                "net_pnl": t.net_pnl if t.net_pnl is not None else t.pnl,
                "gross_pnl": t.gross_pnl,
                "pnl_pct": t.pnl_pct,
                "holding_hours": t.holding_hours,
                "hold_hours": t.holding_hours,
                "entry_cost": t.entry_cost,
                "exit_cost": t.exit_cost,
                "total_cost": t.total_cost if t.total_cost != 0 else t.cost,
                "mae": t.mae,
                "mfe": t.mfe,
                "mae_pct": t.mae_pct,
                "mfe_pct": t.mfe_pct,
                "planned_risk": t.planned_risk,
                "event_type": t.event_type,
                "runner_mode": t.runner_mode,
                "regime_at_exit": t.regime_at_exit,
                "regime_at_entry": t.regime_at_entry,
                "entry_type": t.entry_type,
                "entry_filters_passed": t.entry_filters_passed,
                "unrealized_pct_at_event": t.unrealized_pct_at_event,
            }
            for t in trades
        ]
    )


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def _plot_equity_curve(equity_curve: pd.DataFrame, out_path: Path) -> None:
    if equity_curve.empty:
        return
    eq = equity_curve.copy().sort_values("ts")
    eq["ts"] = pd.to_datetime(eq["ts"], utc=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(eq["ts"], eq["equity"], lw=1.5)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_trade_pnl_timeline(trades_df: pd.DataFrame, out_path: Path) -> None:
    if trades_df.empty:
        return
    df = trades_df.copy().sort_values("exit_ts")
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True)
    colors = ["#2e7d32" if p > 0 else "#c62828" for p in df["net_pnl"]]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(df["exit_ts"], df["net_pnl"], color=colors, width=0.03)
    ax.set_title("Trade PnL Timeline")
    ax.set_xlabel("Exit Time")
    ax.set_ylabel("Net PnL")
    ax.axhline(0, color="black", lw=1)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_losing_streak_distribution(metrics: Dict[str, Any], out_path: Path) -> None:
    dist = metrics.get("losing_streak", {}).get("distribution", {})
    if not dist:
        return
    x = sorted(int(k) for k in dist.keys())
    y = [dist[k] if k in dist else dist[str(k)] for k in x]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, y, color="#455a64")
    ax.set_title("Losing Streak Distribution")
    ax.set_xlabel("Consecutive Losses")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_mae_mfe_by_outcome(trades_df: pd.DataFrame, out_path: Path) -> None:
    if trades_df.empty:
        return
    df = trades_df.copy()
    df["outcome"] = df["net_pnl"].apply(lambda x: "win" if x > 0 else "loss")
    groups = [df.loc[df["outcome"] == "win", "mae"], df.loc[df["outcome"] == "loss", "mae"]]
    labels = ["MAE win", "MAE loss"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.boxplot(groups, labels=labels, showfliers=False)
    ax.set_title("MAE by Outcome")
    ax.set_ylabel("MAE")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_1h_trade_charts(
    candles_1h: Dict[str, pd.DataFrame],
    trades_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    if not candles_1h:
        return
    charts_dir = out_dir / "trade_charts_1h"
    charts_dir.mkdir(parents=True, exist_ok=True)

    for sym, raw_df in candles_1h.items():
        if raw_df is None or raw_df.empty:
            continue

        df = raw_df.copy().sort_values("ts")
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts", "open", "high", "low", "close"]).reset_index(drop=True)
        if df.empty:
            continue

        x = np.arange(len(df))
        candle_up = df["close"] >= df["open"]

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.vlines(x, df["low"], df["high"], color="#546e7a", linewidth=1.0, alpha=0.75)
        ax.bar(
            x[candle_up],
            (df.loc[candle_up, "close"] - df.loc[candle_up, "open"]).to_numpy(),
            bottom=df.loc[candle_up, "open"].to_numpy(),
            width=0.6,
            color="#2e7d32",
            alpha=0.9,
            linewidth=0,
        )
        ax.bar(
            x[~candle_up],
            (df.loc[~candle_up, "open"] - df.loc[~candle_up, "close"]).to_numpy(),
            bottom=df.loc[~candle_up, "close"].to_numpy(),
            width=0.6,
            color="#c62828",
            alpha=0.9,
            linewidth=0,
        )

        sym_trades = pd.DataFrame()
        if not trades_df.empty:
            sym_trades = trades_df[trades_df["symbol"] == sym].copy()
            if not sym_trades.empty:
                sym_trades["entry_ts"] = pd.to_datetime(sym_trades["entry_ts"], utc=True, errors="coerce")
                sym_trades["exit_ts"] = pd.to_datetime(sym_trades["exit_ts"], utc=True, errors="coerce")
                sym_trades = sym_trades.dropna(subset=["entry_ts", "exit_ts"]).reset_index(drop=True)

        if not sym_trades.empty:
            ts_values = df["ts"].to_numpy(dtype="datetime64[ns]")
            entry_idx = np.searchsorted(ts_values, sym_trades["entry_ts"].to_numpy(dtype="datetime64[ns]"), side="left")
            exit_idx = np.searchsorted(ts_values, sym_trades["exit_ts"].to_numpy(dtype="datetime64[ns]"), side="left")
            entry_idx = np.clip(entry_idx, 0, len(df) - 1)
            exit_idx = np.clip(exit_idx, 0, len(df) - 1)

            side_series = sym_trades.get("side", pd.Series(index=sym_trades.index, dtype=str)).astype(str).str.upper()
            long_mask = side_series == "BUY"
            short_mask = side_series == "SELL"
            unknown_mask = ~(long_mask | short_mask)

            if long_mask.any():
                ax.scatter(
                    entry_idx[long_mask.to_numpy()],
                    sym_trades.loc[long_mask, "entry_price"],
                    marker="^",
                    s=60,
                    color="#1565c0",
                    edgecolors="white",
                    linewidths=0.6,
                    label="LONG IN",
                    zorder=4,
                )
                ax.scatter(
                    exit_idx[long_mask.to_numpy()],
                    sym_trades.loc[long_mask, "exit_price"],
                    marker="v",
                    s=60,
                    color="#42a5f5",
                    edgecolors="white",
                    linewidths=0.6,
                    label="LONG OUT",
                    zorder=4,
                )

            if short_mask.any():
                ax.scatter(
                    entry_idx[short_mask.to_numpy()],
                    sym_trades.loc[short_mask, "entry_price"],
                    marker="v",
                    s=60,
                    color="#8e24aa",
                    edgecolors="white",
                    linewidths=0.6,
                    label="SHORT IN",
                    zorder=4,
                )
                ax.scatter(
                    exit_idx[short_mask.to_numpy()],
                    sym_trades.loc[short_mask, "exit_price"],
                    marker="^",
                    s=60,
                    color="#ef5350",
                    edgecolors="white",
                    linewidths=0.6,
                    label="SHORT OUT",
                    zorder=4,
                )

            if unknown_mask.any():
                ax.scatter(
                    entry_idx[unknown_mask.to_numpy()],
                    sym_trades.loc[unknown_mask, "entry_price"],
                    marker="o",
                    s=50,
                    color="#546e7a",
                    edgecolors="white",
                    linewidths=0.6,
                    label="IN (UNKNOWN)",
                    zorder=4,
                )
                ax.scatter(
                    exit_idx[unknown_mask.to_numpy()],
                    sym_trades.loc[unknown_mask, "exit_price"],
                    marker="x",
                    s=55,
                    color="#ef6c00",
                    linewidths=1.0,
                    label="OUT (UNKNOWN)",
                    zorder=4,
                )

            for row_id, trade in sym_trades.iterrows():
                color = "#2e7d32" if float(trade.get("net_pnl", 0.0)) >= 0 else "#c62828"
                ax.plot(
                    [entry_idx[row_id], exit_idx[row_id]],
                    [trade["entry_price"], trade["exit_price"]],
                    color=color,
                    alpha=0.35,
                    linewidth=1.0,
                    zorder=3,
                )

        tick_count = min(10, len(df))
        if tick_count > 0:
            tick_idx = np.unique(np.linspace(0, len(df) - 1, tick_count, dtype=int))
            ax.set_xticks(tick_idx)
            ax.set_xticklabels(df.loc[tick_idx, "ts"].dt.strftime("%Y-%m-%d\n%H:%M"), fontsize=8)

        ax.set_title(f"{sym} 1h Candles with Entry/Exit")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Price")
        ax.grid(alpha=0.2)
        if not sym_trades.empty:
            ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(charts_dir / f"{sym.lower()}_1h_trades.png", dpi=150)
        plt.close(fig)


def save_backtest_reports(
    metrics: Dict[str, Any],
    trades: List[TradeRecord],
    equity_curve: pd.DataFrame,
    results_dir: Path,
    candles_1h: Dict[str, pd.DataFrame] | None = None,
) -> None:
    """Persist decomposition tables and plots for post-analysis."""
    results_dir.mkdir(parents=True, exist_ok=True)
    trades_df = trades_to_dataframe(trades)

    _save_json(results_dir / "metrics.json", metrics)
    if not trades_df.empty:
        trades_df.to_parquet(results_dir / "trades.parquet", index=False)
        trades_df.to_csv(results_dir / "trades.csv", index=False)
    if not equity_curve.empty:
        equity_curve.to_parquet(results_dir / "equity_curve.parquet", index=False)
        equity_curve.to_csv(results_dir / "equity_curve.csv", index=False)

    pd.DataFrame.from_dict(
        metrics.get("exit_reason_performance", {}), orient="index"
    ).reset_index(names=["exit_reason"]).to_csv(
        results_dir / "exit_reason_performance.csv", index=False
    )
    pd.DataFrame.from_dict(
        metrics.get("mae_mfe_by_outcome", {}), orient="index"
    ).reset_index(names=["bucket"]).to_csv(
        results_dir / "mae_mfe_by_outcome.csv", index=False
    )
    pd.DataFrame.from_dict(
        metrics.get("mae_mfe_by_exit_reason", {}), orient="index"
    ).reset_index(names=["exit_reason"]).to_csv(
        results_dir / "mae_mfe_by_exit_reason.csv", index=False
    )
    streak_df = pd.DataFrame(
        [
            {"streak_len": int(k), "count": int(v)}
            for k, v in metrics.get("losing_streak", {}).get("distribution", {}).items()
        ]
    )
    if not streak_df.empty:
        streak_df = streak_df.sort_values("streak_len")
    streak_df.to_csv(results_dir / "losing_streak_distribution.csv", index=False)
    pd.DataFrame([metrics.get("fees", {})]).to_csv(results_dir / "fee_impact.csv", index=False)
    pd.DataFrame([metrics.get("dd_diagnostics", {})]).to_csv(results_dir / "dd_diagnostics.csv", index=False)

    _plot_equity_curve(equity_curve, results_dir / "equity_curve.png")
    _plot_trade_pnl_timeline(trades_df, results_dir / "trade_pnl_timeline.png")
    _plot_losing_streak_distribution(metrics, results_dir / "losing_streak_distribution.png")
    _plot_mae_mfe_by_outcome(trades_df, results_dir / "mae_by_outcome.png")
    _plot_1h_trade_charts(candles_1h or {}, trades_df, results_dir)
