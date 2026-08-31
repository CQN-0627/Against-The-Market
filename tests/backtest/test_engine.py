"""Engine correctness: hand-calculable P&L, and no look-ahead bias."""

from __future__ import annotations

import numpy as np
import pytest

from marketerror.backtest.engine import BacktestConfig, Backtester
from marketerror.backtest.execution import ExecutionConfig
from marketerror.backtest.orders import Order, Side
from marketerror.data.schema import LookAheadError, MarketView
from marketerror.data.synthetic_market import SyntheticMarketGenerator
from marketerror.market.parameters import MarketParameters
from marketerror.strategies.base import Strategy
from marketerror.strategies.buy_and_hold import BuyAndHoldStrategy
from marketerror.strategies.momentum import MomentumStrategy

from ..conftest import build_market


class BuyOnceStrategy(Strategy):
    """Buys a fixed number of shares on the first bar, then holds."""

    def __init__(self, quantity: float = 1000.0, bar: int = 0) -> None:
        self.quantity = quantity
        self.bar = bar

    def on_data(self, market_data):
        if market_data.t == self.bar:
            return Order("BUY", self.quantity)
        return Order.hold()


class TestHandCalculatedCases:
    """Exact arithmetic on a known price path -- specification phase 4."""

    def test_flat_market_frictionless_returns_zero(self, flat_market, frictionless):
        result = Backtester(frictionless).run(flat_market, BuyAndHoldStrategy())
        assert result.metrics.total_return == pytest.approx(0.0, abs=1e-12)
        assert result.metrics.n_trades == 1
        assert result.position[-1] == pytest.approx(1000.0)

    def test_commission_is_exact(self, flat_market):
        """10 bps on 100,000 of notional is exactly 100."""
        config = BacktestConfig(
            execution=ExecutionConfig(
                commission_bps=10.0,
                use_spread=False,
                slippage_coefficient=0.0,
                latency_periods=0,
            )
        )
        result = Backtester(config).run(flat_market, BuyAndHoldStrategy())
        assert result.metrics.commission == pytest.approx(100.0)
        assert result.metrics.final_equity == pytest.approx(99_900.0)
        assert result.metrics.total_return == pytest.approx(-0.001)

    def test_half_spread_is_exact(self):
        """A 20 bps spread on a 100 price costs 0.10 per share, once."""
        market = build_market(np.full(10, 100.0), spread_bps=20.0)
        config = BacktestConfig(
            execution=ExecutionConfig(
                commission_bps=0.0,
                use_spread=True,
                slippage_coefficient=0.0,
                latency_periods=0,
            )
        )
        result = Backtester(config).run(market, BuyAndHoldStrategy())
        assert result.metrics.spread_cost == pytest.approx(100.0)
        assert result.metrics.total_return == pytest.approx(-0.001)

    def test_known_price_move_gives_known_profit(self, frictionless):
        """1,000 shares bought at 100 and marked at 110 is +10,000."""
        price = np.concatenate([np.full(5, 100.0), np.full(5, 110.0)])
        result = Backtester(frictionless).run(build_market(price), BuyOnceStrategy())
        assert result.metrics.final_equity == pytest.approx(110_000.0)
        assert result.metrics.total_return == pytest.approx(0.10)

    def test_square_root_impact_is_exact(self):
        """impact = Y * sigma_period * sqrt(Q/V), applied to the quote."""
        market = build_market(
            np.full(10, 100.0),
            volume=1_000_000.0,
            metadata={"parameters": {"annualized_volatility": 0.20}},
        )
        config = BacktestConfig(
            execution=ExecutionConfig(
                commission_bps=0.0,
                use_spread=False,
                slippage_coefficient=1.0,
                latency_periods=0,
            )
        )
        result = Backtester(config).run(market, BuyOnceStrategy(quantity=1000.0))
        sigma = 0.20 / np.sqrt(252)
        expected = 100.0 * (1.0 + sigma * np.sqrt(1000.0 / 1_000_000.0))
        assert result.fills[0].price == pytest.approx(expected)

    def test_partial_fill_at_the_depth_cap(self):
        """max_participation of top-of-book, and the rest is not filled."""
        market = build_market(np.full(10, 100.0), depth=1000.0)
        config = BacktestConfig(
            execution=ExecutionConfig(
                commission_bps=0.0,
                use_spread=False,
                slippage_coefficient=0.0,
                max_participation=0.25,
                latency_periods=0,
            )
        )
        result = Backtester(config).run(market, BuyOnceStrategy(quantity=1000.0))
        fill = result.fills[0]
        assert fill.filled_quantity == pytest.approx(250.0)
        assert fill.unfilled == pytest.approx(750.0)
        assert "depth" in fill.reasons
        assert fill.is_partial

    def test_leverage_cap_is_exact_without_frictions(self, frictionless):
        """Gross exposure must never exceed max_leverage when costs are zero."""
        price = 100.0 * np.exp(np.cumsum(np.full(60, 0.01)))
        market = build_market(price)
        result = Backtester(frictionless).run(market, MomentumStrategy(lookback=5))
        assert result.exposure.max() <= 1.0 + 1e-12

    def test_short_selling_can_be_disabled(self):
        price = 100.0 * np.exp(np.cumsum(np.full(60, -0.01)))
        config = BacktestConfig(
            execution=ExecutionConfig(allow_short=False, latency_periods=0)
        )
        result = Backtester(config).run(build_market(price), MomentumStrategy(lookback=5))
        assert result.position.min() >= -1e-9


class TestLatency:
    def test_zero_latency_executes_on_the_same_bar(self, flat_market, frictionless):
        result = Backtester(frictionless).run(flat_market, BuyOnceStrategy(bar=3))
        assert result.fills[0].index == 3
        assert result.position[3] == pytest.approx(1000.0)
        assert result.position[2] == 0.0

    def test_one_bar_latency_delays_execution(self, flat_market):
        config = BacktestConfig(
            execution=ExecutionConfig(
                commission_bps=0.0,
                use_spread=False,
                slippage_coefficient=0.0,
                latency_periods=1,
            )
        )
        result = Backtester(config).run(flat_market, BuyOnceStrategy(bar=3))
        assert result.fills[0].index == 4
        assert result.position[3] == 0.0
        assert result.position[4] == pytest.approx(1000.0)

    def test_latency_is_read_from_market_parameters(self, flat_market):
        """The latency perturbation dimension must actually reach the engine."""
        market = build_market(
            np.full(10, 100.0), metadata={"parameters": {"latency_periods": 2}}
        )
        config = BacktestConfig(
            execution=ExecutionConfig(
                commission_bps=0.0,
                use_spread=False,
                slippage_coefficient=0.0,
                latency_periods=None,  # inherit
            )
        )
        result = Backtester(config).run(market, BuyOnceStrategy(bar=1))
        assert result.fills[0].index == 3

    def test_explicit_config_overrides_the_market(self):
        market = build_market(
            np.full(10, 100.0), metadata={"parameters": {"latency_periods": 5}}
        )
        config = BacktestConfig(execution=ExecutionConfig(latency_periods=0))
        result = Backtester(config).run(market, BuyOnceStrategy(bar=1))
        assert result.fills[0].index == 1


class TestNoLookAhead:
    """Specification §26: strategies must not see future observations."""

    def test_corrupting_the_future_cannot_change_the_past(self):
        """The decisive empirical test.

        Run a strategy on a path, then replace everything after bar ``k`` with
        wildly different data and run again.  If any quantity before ``k``
        changes, information flowed backwards in time.
        """
        parameters = MarketParameters()
        data = SyntheticMarketGenerator(parameters).generate(200, seed=3)
        k = 120

        corrupted_price = data.price.copy()
        corrupted_price[k:] *= np.linspace(1.0, 5.0, len(corrupted_price) - k)
        corrupted = build_market(corrupted_price, spread_bps=5.0)
        original = build_market(data.price, spread_bps=5.0)

        config = BacktestConfig()
        strategy = MomentumStrategy(lookback=20)
        a = Backtester(config).run(original, strategy)
        b = Backtester(config).run(corrupted, strategy)

        assert np.allclose(a.equity[:k], b.equity[:k])
        assert np.allclose(a.position[:k], b.position[:k])
        assert np.allclose(a.cash[:k], b.cash[:k])

    def test_view_cannot_index_beyond_the_current_bar(self):
        data = SyntheticMarketGenerator(MarketParameters()).generate(50, seed=1)
        view = MarketView(data, {}, index=10)
        assert view["price"] == pytest.approx(data.price[10])
        assert view.history("price", 5)[-1] == pytest.approx(data.price[10])
        assert len(view.history("price")) == 11
        # Asking for more history than exists truncates rather than wrapping.
        assert len(view.history("price", 999)) == 11

    def test_view_exposes_no_route_to_the_raw_arrays(self):
        """A strategy must not be able to reach the full series off the view."""
        data = SyntheticMarketGenerator(MarketParameters()).generate(50, seed=1)
        view = MarketView(data, {}, index=5)
        for attribute in ("data", "_data_public", "prices", "all"):
            assert not hasattr(view, attribute) or attribute.startswith("_")
        # history() always ends at t, whatever is asked for.
        for key in ("price", "bid", "ask", "volume", "return"):
            assert len(view.history(key)) == 6

    def test_view_only_moves_forward(self):
        data = SyntheticMarketGenerator(MarketParameters()).generate(50, seed=1)
        view = MarketView(data, {}, index=10)
        with pytest.raises(ValueError):
            view._advance(5)

    def test_unknown_feature_is_an_error_not_a_nan(self):
        data = SyntheticMarketGenerator(MarketParameters()).generate(50, seed=1)
        view = MarketView(data, {}, index=5)
        with pytest.raises(KeyError):
            view["sma_20"]  # not declared in requires()


class TestStateIsolation:
    def test_strategy_state_does_not_leak_between_paths(self):
        """The engine resets the strategy, so path order cannot matter."""
        parameters = MarketParameters()
        generator = SyntheticMarketGenerator(parameters)
        paths = [generator.generate(120, seed=s) for s in range(3)]
        backtester = Backtester(BacktestConfig())

        from marketerror.strategies.mean_reversion import MeanReversionStrategy

        shared = MeanReversionStrategy(lookback=10)
        forward = [backtester.run(p, shared).metrics.total_return for p in paths]
        backward = [
            backtester.run(p, shared).metrics.total_return for p in reversed(paths)
        ]
        assert forward == pytest.approx(list(reversed(backward)))

    def test_fresh_instances_match_a_reused_one(self):
        from marketerror.strategies.mean_reversion import MeanReversionStrategy

        data = SyntheticMarketGenerator(MarketParameters()).generate(150, seed=5)
        backtester = Backtester(BacktestConfig())
        reused = MeanReversionStrategy(lookback=10)
        backtester.run(data, reused)  # dirty the state
        again = backtester.run(data, reused).metrics.total_return
        fresh = backtester.run(data, MeanReversionStrategy(lookback=10)).metrics.total_return
        assert again == pytest.approx(fresh)


class TestRuin:
    def test_wiped_out_path_is_flattened_and_frozen(self):
        """A short position with unbounded loss must not poison the metrics.

        The price here gaps ~17% per bar, so the once-per-bar liquidation check
        overshoots zero and equity ends negative. That is genuine gap risk and is
        reported as a total return worse than -100%; what must *not* happen is
        NaNs, continued trading, or an annualised return below -100%.
        """
        price = np.concatenate([np.full(5, 100.0), np.geomspace(100.0, 100_000.0, 45)])
        market = build_market(price)

        class AlwaysShort(Strategy):
            def on_data(self, market_data):
                if market_data.t == 0:
                    return Order("SELL", 1000.0)
                return Order.hold()

        config = BacktestConfig(
            execution=ExecutionConfig(latency_periods=0, max_leverage=1.0)
        )
        result = Backtester(config).run(market, AlwaysShort())
        assert result.metrics.ruined
        assert result.metrics.total_return < -1.0  # a gap can exceed the capital
        assert result.metrics.annualized_return == -1.0  # but this is floored
        assert np.all(np.isfinite(result.equity))
        assert np.all(np.isfinite(result.cumulative_return))
        assert result.cumulative_return.min() >= -1.0  # floored for comparability
        assert result.position[-1] == 0.0  # flattened, and stays flat
        assert result.metrics.n_trades == 1  # no trading after ruin

    def test_gentle_loss_does_not_trigger_ruin(self, frictionless):
        price = np.linspace(100.0, 60.0, 50)
        result = Backtester(frictionless).run(build_market(price), BuyOnceStrategy())
        assert not result.metrics.ruined
        assert result.metrics.total_return == pytest.approx(-0.40)


class TestReturnAccounting:
    def test_opening_cost_is_inside_the_reported_return(self, flat_market):
        """With same-bar execution, equity[0] is already post-trade.

        Measuring the return from equity[0] would erase the opening trade's cost,
        so it is measured from the starting capital instead.
        """
        config = BacktestConfig(
            initial_capital=100_000.0,
            execution=ExecutionConfig(
                commission_bps=10.0, use_spread=False, slippage_coefficient=0.0,
                latency_periods=0,
            ),
        )
        result = Backtester(config).run(flat_market, BuyAndHoldStrategy())
        assert result.equity[0] == pytest.approx(99_900.0)  # post-trade
        assert result.metrics.initial_equity == pytest.approx(100_000.0)
        assert result.metrics.total_return == pytest.approx(-0.001)
        assert result.cumulative_return[0] == pytest.approx(-0.001)


class TestOrders:
    def test_string_side_is_accepted(self):
        assert Order("BUY", 100).side is Side.BUY
        assert Order("sell", 100).side is Side.SELL

    def test_hold_has_no_quantity(self):
        assert Order("HOLD", 500).quantity == 0.0
        assert not Order.hold()

    def test_negative_quantity_rejected(self):
        with pytest.raises(ValueError):
            Order("BUY", -100)

    def test_signed_constructor(self):
        assert Order.signed(-50).side is Side.SELL
        assert Order.signed(-50).quantity == 50.0
        assert Order.signed(0).side is Side.HOLD

    def test_unknown_side_rejected(self):
        with pytest.raises(ValueError):
            Order("PURCHASE", 10)
