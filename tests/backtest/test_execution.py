"""Execution model: quotes, impact, partial fills, limits -- exact arithmetic."""

from __future__ import annotations

import math

import numpy as np
import pytest

from marketerror.backtest.execution import ExecutionConfig, ExecutionModel
from marketerror.backtest.orders import Order, Side
from marketerror.backtest.portfolio import Portfolio

from ..conftest import build_market


def _model(config: ExecutionConfig, **market_kwargs) -> tuple[ExecutionModel, object]:
    market = build_market(np.full(20, 100.0), **market_kwargs)
    return ExecutionModel(config, market), market


class TestQuoteCrossing:
    def test_buy_lifts_the_ask(self):
        config = ExecutionConfig(commission_bps=0.0, slippage_coefficient=0.0, use_spread=True)
        model, market = _model(config, spread_bps=20.0)
        fill = model.execute(Order("BUY", 100), 0, market, Portfolio(100_000.0))
        assert fill.price == pytest.approx(market.ask[0])
        assert fill.spread_cost == pytest.approx(100.0 * (market.ask[0] - market.price[0]))

    def test_sell_hits_the_bid(self):
        config = ExecutionConfig(commission_bps=0.0, slippage_coefficient=0.0, use_spread=True)
        model, market = _model(config, spread_bps=20.0)
        # Sell from a long position so the short-selling switch is not involved.
        portfolio = Portfolio(100_000.0)
        portfolio.position = 500.0
        fill = model.execute(Order("SELL", 100), 0, market, portfolio)
        assert fill.price == pytest.approx(market.bid[0])

    def test_use_spread_false_trades_at_mid(self):
        config = ExecutionConfig(commission_bps=0.0, slippage_coefficient=0.0, use_spread=False)
        model, market = _model(config, spread_bps=20.0)
        fill = model.execute(Order("BUY", 100), 0, market, Portfolio(100_000.0))
        assert fill.price == pytest.approx(market.price[0])
        assert fill.spread_cost == 0.0


class TestMarketImpact:
    def test_square_root_law_is_exact(self):
        config = ExecutionConfig(commission_bps=0.0, slippage_coefficient=1.0, use_spread=False)
        market = build_market(
            np.full(20, 100.0),
            volume=1_000_000.0,
            metadata={"parameters": {"annualized_volatility": 0.20}},
        )
        model = ExecutionModel(config, market)
        fill = model.execute(Order("BUY", 1000), 0, market, Portfolio(100_000.0))
        sigma = 0.20 / math.sqrt(252)
        expected_impact = 1.0 * sigma * math.sqrt(1000.0 / 1_000_000.0)
        assert fill.price == pytest.approx(100.0 * (1.0 + expected_impact))
        assert fill.slippage_cost == pytest.approx(1000.0 * 100.0 * expected_impact)

    def test_impact_grows_with_size(self):
        config = ExecutionConfig(commission_bps=0.0, slippage_coefficient=1.0, use_spread=False)
        model, market = _model(config, volume=1e6)
        small = model._impact_fraction(100.0, 0, market)
        large = model._impact_fraction(10_000.0, 0, market)
        assert large > small > 0.0
        # sqrt law: 100x the size is 10x the impact.
        assert large / small == pytest.approx(10.0, rel=1e-6)

    def test_impact_falls_with_volume(self):
        config = ExecutionConfig(slippage_coefficient=1.0)
        thin, market_thin = _model(config, volume=1e5)
        thick, market_thick = _model(config, volume=1e7)
        assert thin._impact_fraction(1000.0, 0, market_thin) > thick._impact_fraction(
            1000.0, 0, market_thick
        )

    def test_impact_uses_declared_volatility(self):
        """A volatility shock must flow straight into impact."""
        calm = build_market(
            np.full(20, 100.0), volume=1e6, metadata={"parameters": {"annualized_volatility": 0.20}}
        )
        wild = build_market(
            np.full(20, 100.0), volume=1e6, metadata={"parameters": {"annualized_volatility": 0.80}}
        )
        config = ExecutionConfig(slippage_coefficient=1.0)
        calm_impact = ExecutionModel(config, calm)._impact_fraction(1000.0, 0, calm)
        wild_impact = ExecutionModel(config, wild)._impact_fraction(1000.0, 0, wild)
        assert wild_impact == pytest.approx(4.0 * calm_impact, rel=1e-9)


class TestPartialFills:
    def test_depth_cap_limits_size(self):
        config = ExecutionConfig(commission_bps=0.0, slippage_coefficient=0.0, max_participation=0.25)
        model, market = _model(config, depth=1000.0)
        fill = model.execute(Order("BUY", 1000), 0, market, Portfolio(100_000.0))
        assert fill.filled_quantity == pytest.approx(250.0)
        assert "depth" in fill.reasons
        assert fill.is_partial

    def test_full_fill_within_depth(self):
        config = ExecutionConfig(commission_bps=0.0, slippage_coefficient=0.0, max_participation=0.5)
        model, market = _model(config, depth=1000.0)
        fill = model.execute(Order("BUY", 100), 0, market, Portfolio(100_000.0))
        assert fill.filled_quantity == pytest.approx(100.0)
        assert not fill.is_partial


class TestPositionLimits:
    def test_leverage_cap_clips_the_order(self):
        config = ExecutionConfig(commission_bps=0.0, slippage_coefficient=0.0, max_leverage=1.0)
        model, market = _model(config)
        portfolio = Portfolio(100_000.0)  # equity 100k, price 100 -> 1000 shares max
        fill = model.execute(Order("BUY", 5000), 0, market, portfolio)
        assert fill.filled_quantity == pytest.approx(1000.0)
        assert "leverage" in fill.reasons

    def test_short_selling_can_be_blocked(self):
        config = ExecutionConfig(allow_short=False)
        model, market = _model(config)
        fill = model.execute(Order("SELL", 100), 0, market, Portfolio(100_000.0))
        assert fill.filled_quantity == 0.0
        assert "short_disabled" in fill.reasons

    def test_selling_an_existing_long_is_allowed_when_short_blocked(self):
        config = ExecutionConfig(commission_bps=0.0, slippage_coefficient=0.0, allow_short=False)
        model, market = _model(config)
        portfolio = Portfolio(100_000.0)
        portfolio.position = 500.0
        portfolio.cash = 50_000.0
        fill = model.execute(Order("SELL", 300), 0, market, portfolio)
        assert fill.filled_quantity == pytest.approx(300.0)

    def test_no_equity_means_no_fill(self):
        config = ExecutionConfig()
        model, market = _model(config)
        portfolio = Portfolio(100_000.0)
        portfolio.cash = -1_000.0
        portfolio.position = 0.0
        fill = model.execute(Order("BUY", 10), 0, market, portfolio)
        assert fill.filled_quantity == 0.0


class TestLimitOrders:
    def test_marketable_limit_fills(self):
        config = ExecutionConfig(commission_bps=0.0, slippage_coefficient=0.0, use_spread=False)
        model, market = _model(config)
        fill = model.execute(Order("BUY", 100, limit_price=101.0), 0, market, Portfolio(100_000.0))
        assert fill.filled_quantity == pytest.approx(100.0)

    def test_unmarketable_limit_is_rejected(self):
        config = ExecutionConfig(commission_bps=0.0, slippage_coefficient=0.0, use_spread=False)
        model, market = _model(config)
        fill = model.execute(Order("BUY", 100, limit_price=99.0), 0, market, Portfolio(100_000.0))
        assert fill.filled_quantity == 0.0
        assert "limit" in fill.reasons


class TestLatencyResolution:
    def test_explicit_config_wins(self):
        market = build_market(np.full(10, 100.0), metadata={"parameters": {"latency_periods": 5}})
        model = ExecutionModel(ExecutionConfig(latency_periods=2), market)
        assert model.latency == 2

    def test_inherits_from_market_parameters(self):
        market = build_market(np.full(10, 100.0), metadata={"parameters": {"latency_periods": 3}})
        model = ExecutionModel(ExecutionConfig(latency_periods=None), market)
        assert model.latency == 3

    def test_defaults_to_zero(self):
        market = build_market(np.full(10, 100.0))
        model = ExecutionModel(ExecutionConfig(latency_periods=None), market)
        assert model.latency == 0


class TestConfigValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"commission_bps": -1.0},
            {"max_participation": 0.0},
            {"max_participation": 1.5},
            {"max_leverage": 0.0},
            {"latency_periods": -1},
            {"slippage_coefficient": -0.1},
        ],
    )
    def test_invalid_config_rejected(self, kwargs):
        with pytest.raises(ValueError):
            ExecutionConfig(**kwargs)


class TestHoldOrders:
    def test_hold_produces_an_empty_fill(self):
        config = ExecutionConfig()
        model, market = _model(config)
        fill = model.execute(Order.hold(), 0, market, Portfolio(100_000.0))
        assert fill.filled_quantity == 0.0
        assert fill.side is Side.HOLD
