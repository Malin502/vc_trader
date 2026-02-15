"""Threshold search for directional quantile PhaseB models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict

import pandas as pd


def _proxy_objective(
    selected: pd.DataFrame,
    dd_lambda: float,
    cost_cfg: Dict[str, Any] | None = None,
) -> tuple[float, float, float, float]:
    if selected.empty:
        return -1e9, 0.0, 0.0, 0.0

    gross_edge = (selected["y_up_true"] - selected["y_down_true"]).astype(float)
    net_edge = gross_edge - _estimate_cost_fraction(selected, cost_cfg)
    if len(net_edge) < 2:
        sharpe_like = 0.0
    else:
        std = float(net_edge.std(ddof=0))
        sharpe_like = float(net_edge.mean() / std) if std > 1e-12 else 0.0
    curve = net_edge.cumsum()
    dd = float((curve - curve.cummax()).min()) if len(curve) else 0.0
    objective = sharpe_like - dd_lambda * abs(dd)
    avg_net_edge = float(net_edge.mean()) if len(net_edge) else 0.0
    return objective, sharpe_like, dd, avg_net_edge


def _estimate_cost_fraction(
    selected: pd.DataFrame,
    cost_cfg: Dict[str, Any] | None = None,
) -> pd.Series:
    """Estimate per-trade cost in return fraction (not bps)."""
    cfg = cost_cfg or {}
    roundtrip_bps = float(cfg.get("roundtrip_bps", 0.0))
    regime_extra_bps = cfg.get("regime_extra_bps", {}) or {}

    base = pd.Series(roundtrip_bps / 10_000.0, index=selected.index, dtype=float)
    if not regime_extra_bps:
        return base

    regime_name = selected.get("regime_name")
    if regime_name is None and "regime_id" in selected.columns:
        regime_name = selected["regime_id"].map(_regime_name)
    if regime_name is None:
        return base

    extra = regime_name.map(
        lambda rn: float(regime_extra_bps.get(str(rn), 0.0)) / 10_000.0
    ).astype(float)
    return base + extra


def search_thresholds(
    val_scores: pd.DataFrame,
    backtester: Callable[[float, float], Dict[str, float]] | None,
    metric_cfg: Dict[str, Any],
) -> tuple[float, float]:
    """Search long/short thresholds on validation scores."""
    thr_grid = metric_cfg.get("thr_grid", {})
    long_grid = list(thr_grid.get("long", [0.0015]))
    short_grid = list(thr_grid.get("short", [0.0020]))
    constraints = metric_cfg.get("constraints", {})
    min_trades = int(constraints.get("min_trades", 0))
    max_dd = float(constraints.get("max_dd", -1.0))
    dd_lambda = float(metric_cfg.get("metric", {}).get("dd_lambda", 0.7))
    cost_cfg = metric_cfg.get("metric", {}).get("cost", {})

    best_long = float(long_grid[0])
    best_short = float(short_grid[0])
    best_score = -1e18
    sweep_rows = []

    for thr_long in long_grid:
        for thr_short in short_grid:
            if backtester is not None:
                bt = backtester(float(thr_long), float(thr_short))
                n_trades = int(bt.get("n_trades", 0))
                max_drawdown = float(bt.get("max_drawdown", 0.0))
                objective = float(bt.get("objective", -1e9))
                sharpe_like = float(bt.get("sharpe_like", 0.0))
                avg_net_edge = float(bt.get("avg_net_edge", 0.0))
            else:
                allowed = (
                    ((val_scores["direction"] == "long") & (val_scores["score"] >= float(thr_long)))
                    | ((val_scores["direction"] == "short") & (val_scores["score"] >= float(thr_short)))
                )
                selected = val_scores.loc[allowed]
                n_trades = int(len(selected))
                objective, sharpe_like, max_drawdown, avg_net_edge = _proxy_objective(
                    selected, dd_lambda, cost_cfg=cost_cfg
                )

            valid = True
            if n_trades < min_trades:
                valid = False
            if max_drawdown < max_dd:
                valid = False

            sweep_rows.append(
                {
                    "thr_long": float(thr_long),
                    "thr_short": float(thr_short),
                    "objective": float(objective),
                    "sharpe_like": float(sharpe_like),
                    "max_drawdown": float(max_drawdown),
                    "avg_net_edge": float(avg_net_edge),
                    "n_trades": n_trades,
                    "valid": valid,
                }
            )

            if valid and objective > best_score:
                best_score = objective
                best_long = float(thr_long)
                best_short = float(thr_short)

    save_dir = metric_cfg.get("save_dir")
    if save_dir:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "threshold_search_results.json", "w") as f:
            json.dump(sweep_rows, f, indent=2)
        with open(out_dir / "thresholds.json", "w") as f:
            json.dump({"long": best_long, "short": best_short}, f, indent=2)

    return best_long, best_short


def _regime_name(regime_id: Any) -> str:
    try:
        rid = int(float(regime_id))
    except Exception:
        return "range"
    if rid == 0:
        return "trend"
    if rid == 2:
        return "chaos"
    return "range"


def _search_single_threshold(
    df: pd.DataFrame,
    threshold_grid: list[float],
    min_trades: int,
    max_dd: float,
    dd_lambda: float,
    cost_cfg: Dict[str, Any],
) -> tuple[float, list[Dict[str, Any]]]:
    if not threshold_grid:
        threshold_grid = [0.0]
    best_thr = float(threshold_grid[0])
    best_obj = -1e18
    rows: list[Dict[str, Any]] = []
    for thr in threshold_grid:
        selected = df[df["score"] >= float(thr)]
        n_trades = int(len(selected))
        objective, sharpe_like, max_drawdown, avg_net_edge = _proxy_objective(
            selected, dd_lambda, cost_cfg=cost_cfg
        )
        valid = (n_trades >= min_trades) and (max_drawdown >= max_dd)
        rows.append(
            {
                "threshold": float(thr),
                "objective": float(objective),
                "sharpe_like": float(sharpe_like),
                "max_drawdown": float(max_drawdown),
                "avg_net_edge": float(avg_net_edge),
                "n_trades": n_trades,
                "valid": bool(valid),
            }
        )
        if valid and objective > best_obj:
            best_obj = objective
            best_thr = float(thr)
    return best_thr, rows


def search_thresholds_by_regime(
    val_scores: pd.DataFrame,
    metric_cfg: Dict[str, Any],
    global_thresholds: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    """Search thresholds per direction and regime.

    Returns:
      {
        "long": {"trend": x, "range": y, "chaos": z},
        "short": {"trend": a, "range": b, "chaos": c},
      }
    """
    thr_grid = metric_cfg.get("thr_grid", {})
    long_grid = list(thr_grid.get("long", [global_thresholds.get("long", 0.0015)]))
    short_grid = list(thr_grid.get("short", [global_thresholds.get("short", 0.0020)]))
    constraints = metric_cfg.get("constraints", {})
    min_trades = int(constraints.get("min_trades", 0))
    min_trades_per_regime = int(constraints.get("min_trades_per_regime", max(10, min_trades // 3)))
    max_dd = float(constraints.get("max_dd", -1.0))
    dd_lambda = float(metric_cfg.get("metric", {}).get("dd_lambda", 0.7))
    cost_cfg = metric_cfg.get("metric", {}).get("cost", {})

    scores = val_scores.copy()
    if "regime_name" not in scores.columns:
        scores["regime_name"] = scores["regime_id"].map(_regime_name)

    result = {
        "long": {"trend": float(global_thresholds["long"]), "range": float(global_thresholds["long"]), "chaos": float(global_thresholds["long"])},
        "short": {"trend": float(global_thresholds["short"]), "range": float(global_thresholds["short"]), "chaos": float(global_thresholds["short"])},
    }
    sweep: Dict[str, Any] = {"long": {}, "short": {}}

    for direction, grid in [("long", long_grid), ("short", short_grid)]:
        df_dir = scores[scores["direction"] == direction]
        for regime in ["trend", "range", "chaos"]:
            df_rg = df_dir[df_dir["regime_name"] == regime]
            if df_rg.empty:
                sweep[direction][regime] = {"reason": "empty", "rows": []}
                continue
            best_thr, rows = _search_single_threshold(
                df_rg,
                grid,
                min_trades=min_trades_per_regime,
                max_dd=max_dd,
                dd_lambda=dd_lambda,
                cost_cfg=cost_cfg,
            )
            result[direction][regime] = float(best_thr)
            sweep[direction][regime] = {"reason": "ok", "rows": rows}

    save_dir = metric_cfg.get("save_dir")
    if save_dir:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "threshold_search_results_by_regime.json", "w") as f:
            json.dump(sweep, f, indent=2)
        with open(out_dir / "thresholds_by_regime.json", "w") as f:
            json.dump(result, f, indent=2)

    return result
