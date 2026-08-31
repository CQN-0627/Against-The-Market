"""Shared fixtures.

The deterministic markets here are the backbone of the backtester tests: when
the price path is known exactly, the correct P&L can be computed by hand, and a
discrepancy is unambiguously a bug rather than a modelling disagreement.
"""

from __future__ import annotations

import numpy as np
import pytest

from marketerror.backtest.engine import BacktestConfig
from marketerror.backtest.execution import ExecutionConfig
from marketerror.data.schema import MarketData, make_timestamps
from marketerror.market.parameters import MarketParameters


def build_market(
    price: np.ndarray,
    spread_bps: float = 0.0,
    volume: float = 1e9,
    depth: float = 1e9,
    periods_per_year: int = 252,
    metadata: dict | None = None,
) -> MarketData:
    """A market with an exactly specified price path."""
    price = np.asarray(price, dtype=np.float64)
    n = len(price)
    returns = np.zeros(n)
    returns[1:] = np.diff(np.log(price))
    spread = price * spread_bps / 1e4
    return MarketData(
        timestamp=make_timestamps(n, periods_per_year=periods_per_year),
        price=price,
        returns=returns,
        volume=np.full(n, volume),
        bid=price - spread / 2.0,
        ask=price + spread / 2.0,
        spread=spread,
        bid_size=np.full(n, depth),
        ask_size=np.full(n, depth),
        periods_per_year=periods_per_year,
        metadata=metadata if metadata is not None else {},
    )


@pytest.fixture
def flat_market() -> MarketData:
    """Ten bars at a constant price of 100, with no spread."""
    return build_market(np.full(10, 100.0))


@pytest.fixture
def frictionless() -> BacktestConfig:
    """A backtest with every cost switched off, for exact arithmetic."""
    return BacktestConfig(
        initial_capital=100_000.0,
        execution=ExecutionConfig(
            commission_bps=0.0,
            use_spread=False,
            slippage_coefficient=0.0,
            latency_periods=0,
        ),
    )


@pytest.fixture
def baseline_parameters() -> MarketParameters:
    return MarketParameters()
