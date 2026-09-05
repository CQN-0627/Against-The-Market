"""Tests for multi-asset accounting and engine causality."""

from __future__ import annotations

import numpy as np

from marketerror.backtest import (
    BacktestConfig,
    ExecutionConfig,
    SymbolOrder,
    UniverseBacktester,
)
from marketerror.data.synthetic_universe import SyntheticUniverseGenerator
from marketerror.market.parameters import MarketParameters
from marketerror.market.universe import UniverseParameters
from marketerror.strategies import UniverseStrategy


class BuyAllOnce(UniverseStrategy):
    def on_data(self, view):
        if view.t == 0:
            return [self.order_to_weight(view, symbol, 0.1) for symbol in view.symbols]
        return []

    def requires(self):
        return ()


class LongTop(UniverseStrategy):
    def requires(self):
        return ("return_5",)

    def on_data(self, view):
        if not view.ready("return_5"):
            return []
        return [self.order_to_weight(view, view.top("return_5", 1)[0], 0.5)]


def make_data(periods=40, seed=3):
    params = UniverseParameters.homogeneous(
        3, MarketParameters(drift=0.0, annualized_volatility=0.1), market_beta=0.5
    )
    return SyntheticUniverseGenerator(params).generate(periods, seed=seed)


def test_multi_asset_engine_tracks_positions_and_equity():
    config = BacktestConfig(
        execution=ExecutionConfig(
            commission_bps=0.0,
            use_spread=False,
            slippage_coefficient=0.0,
            max_participation=1.0,
        )
    )
    result = UniverseBacktester(config).run(make_data(), BuyAllOnce())
    assert result.positions.shape == (40, 3)
    assert result.metrics.n_trades == 3
    assert np.all(result.exposure <= 1.0 + 1e-9)
    assert result.position("SYN00")[-1] > 0.0
    assert len(result.to_frame()) == 40 * 3


def test_cross_sectional_strategy_runs_with_features():
    result = UniverseBacktester().run(make_data(), LongTop())
    assert result.metrics.periods == 40
    assert result.symbols == ("SYN00", "SYN01", "SYN02")


def test_basket_leverage_scaling_is_order_independent():
    class OrdersInOrder(UniverseStrategy):
        def on_data(self, view):
            return [self.order_to_weight(view, symbol, 1.0) for symbol in view.symbols] if view.t == 0 else []

    class OrdersReversed(UniverseStrategy):
        def on_data(self, view):
            return [self.order_to_weight(view, symbol, 1.0) for symbol in reversed(view.symbols)] if view.t == 0 else []

    config = BacktestConfig(
        execution=ExecutionConfig(
            commission_bps=0.0,
            use_spread=False,
            slippage_coefficient=0.0,
            max_participation=1.0,
        )
    )
    first = UniverseBacktester(config).run(make_data(), OrdersInOrder())
    second = UniverseBacktester(config).run(make_data(), OrdersReversed())
    assert np.allclose(first.positions[0], second.positions[0])
    assert first.equity[0] == second.equity[0]
