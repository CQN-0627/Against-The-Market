"""Backtesting: orders, execution, portfolio accounting, metrics and the engine."""

from __future__ import annotations

from .engine import BacktestConfig, BacktestResult, Backtester
from .execution import ExecutionConfig, ExecutionModel
from .metrics import PerformanceMetrics, compute_metrics
from .orders import Fill, Order, Side
from .portfolio import Portfolio
from .universe_engine import UniverseBacktestResult, UniverseBacktester
from .universe_execution import UniverseExecutionModel
from .universe_orders import SymbolOrder
from .universe_portfolio import UniversePortfolio

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "Backtester",
    "ExecutionConfig",
    "ExecutionModel",
    "Fill",
    "Order",
    "PerformanceMetrics",
    "Portfolio",
    "Side",
    "SymbolOrder",
    "UniverseBacktestResult",
    "UniverseBacktester",
    "UniverseExecutionModel",
    "UniversePortfolio",
    "compute_metrics",
]
