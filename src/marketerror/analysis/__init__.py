"""Statistical analysis of backtests, robustness sweeps and failure scenarios."""

from __future__ import annotations

from .statistics import (
    ar1_coefficient,
    drawdown_series,
    longest_run_below,
    max_drawdown,
    realized_statistics,
    sharpe_ratio,
)

__all__ = [
    "ar1_coefficient",
    "drawdown_series",
    "longest_run_below",
    "max_drawdown",
    "realized_statistics",
    "sharpe_ratio",
]
