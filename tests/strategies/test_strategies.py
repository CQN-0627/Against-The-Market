"""Strategies: correct signals, clean warm-up, declared features, resettable state."""

from __future__ import annotations

import numpy as np
import pytest

from marketerror.backtest.engine import BacktestConfig, Backtester
from marketerror.backtest.execution import ExecutionConfig
from marketerror.backtest.orders import Side
from marketerror.strategies.base import Strategy
from marketerror.strategies.buy_and_hold import BuyAndHoldStrategy
from marketerror.strategies.loader import (
    StrategySpec,
    coerce_parameters,
    load_strategy,
    resolve_strategy_class,
)
from marketerror.strategies.mean_reversion import MeanReversionStrategy
from marketerror.strategies.momentum import MomentumStrategy
from marketerror.strategies.moving_average import MovingAverageCrossoverStrategy

from ..conftest import build_market

FRICTIONLESS = BacktestConfig(
    execution=ExecutionConfig(
        commission_bps=0.0, use_spread=False, slippage_coefficient=0.0, latency_periods=0
    )
)


def _trending_up(n: int = 120) -> object:
    return build_market(100.0 * np.exp(np.cumsum(np.full(n, 0.01))))


def _trending_down(n: int = 120) -> object:
    return build_market(100.0 * np.exp(np.cumsum(np.full(n, -0.01))))


class TestMomentum:
    def test_requires_the_return_feature(self):
        assert MomentumStrategy(lookback=20).requires() == ("return_20",)

    def test_goes_long_in_an_uptrend(self):
        result = Backtester(FRICTIONLESS).run(_trending_up(), MomentumStrategy(lookback=10))
        assert result.position[-1] > 0.0

    def test_goes_short_in_a_downtrend(self):
        result = Backtester(FRICTIONLESS).run(_trending_down(), MomentumStrategy(lookback=10))
        assert result.position[-1] < 0.0

    def test_no_trade_before_warmup(self):
        result = Backtester(FRICTIONLESS).run(_trending_up(), MomentumStrategy(lookback=10))
        # The feature return_10 is NaN until enough bars exist; no position before then.
        assert np.all(result.position[:10] == 0.0)

    def test_allow_short_false_stays_flat_or_long(self):
        result = Backtester(FRICTIONLESS).run(
            _trending_down(), MomentumStrategy(lookback=10, allow_short=False)
        )
        assert result.position.min() >= -1e-9

    def test_rejects_bad_lookback(self):
        with pytest.raises(ValueError):
            MomentumStrategy(lookback=0)


class TestMeanReversion:
    def test_requires_the_zscore_feature(self):
        assert MeanReversionStrategy(lookback=20).requires() == ("zscore_20",)

    def test_buys_a_dip_and_sells_a_spike(self):
        # A single trough below the moving average should trigger a long.
        price = np.concatenate([np.full(30, 100.0), [80.0], np.full(30, 100.0)])
        result = Backtester(FRICTIONLESS).run(
            build_market(price), MeanReversionStrategy(lookback=20, entry_z=1.0)
        )
        assert result.position.max() > 0.0

    def test_reset_clears_position_state(self):
        strategy = MeanReversionStrategy(lookback=10)
        price = np.concatenate([np.full(20, 100.0), [70.0], np.full(20, 100.0)])
        backtester = Backtester(FRICTIONLESS)
        first = backtester.run(build_market(price), strategy).metrics.total_return
        second = backtester.run(build_market(price), strategy).metrics.total_return
        assert first == pytest.approx(second)

    def test_rejects_bad_thresholds(self):
        with pytest.raises(ValueError):
            MeanReversionStrategy(entry_z=1.0, exit_z=1.5)  # exit must be < entry


class TestMovingAverage:
    def test_requires_both_averages(self):
        assert set(MovingAverageCrossoverStrategy(fast=10, slow=50).requires()) == {
            "sma_10",
            "sma_50",
        }

    def test_fast_must_be_shorter_than_slow(self):
        with pytest.raises(ValueError):
            MovingAverageCrossoverStrategy(fast=50, slow=10)

    def test_long_when_fast_above_slow(self):
        result = Backtester(FRICTIONLESS).run(
            _trending_up(), MovingAverageCrossoverStrategy(fast=5, slow=20)
        )
        assert result.position[-1] > 0.0


class TestBuyAndHold:
    def test_buys_once(self):
        result = Backtester(FRICTIONLESS).run(_trending_up(), BuyAndHoldStrategy())
        assert result.metrics.n_trades == 1

    def test_return_matches_the_market(self):
        """With no frictions, buy-and-hold return equals the price return."""
        price = _trending_up()
        result = Backtester(FRICTIONLESS).run(price, BuyAndHoldStrategy())
        market_return = price.price[-1] / price.price[0] - 1.0
        assert result.metrics.total_return == pytest.approx(market_return, rel=1e-6)

    def test_does_not_compound_into_leverage(self):
        result = Backtester(FRICTIONLESS).run(_trending_up(), BuyAndHoldStrategy())
        assert result.exposure.max() <= 1.0 + 1e-9


class TestNoLookAheadInFeatures:
    def test_features_are_causal(self):
        """A feature at bar t must not depend on data after t."""
        from marketerror.data.features import build_features
        from marketerror.data.synthetic_market import SyntheticMarketGenerator
        from marketerror.market.parameters import MarketParameters

        data = SyntheticMarketGenerator(MarketParameters()).generate(200, seed=1)
        full = build_features(data, ("sma_20", "return_10", "zscore_15"))
        k = 120
        truncated = build_features(data.slice(0, k), ("sma_20", "return_10", "zscore_15"))
        for name, series in truncated.items():
            assert np.allclose(series, full[name][:k], equal_nan=True)


class TestLoader:
    @pytest.mark.parametrize(
        "reference,expected",
        [
            ("momentum", MomentumStrategy),
            ("mom", MomentumStrategy),
            ("mean_reversion", MeanReversionStrategy),
            ("mr", MeanReversionStrategy),
            ("moving_average", MovingAverageCrossoverStrategy),
            ("ma", MovingAverageCrossoverStrategy),
            ("buy_and_hold", BuyAndHoldStrategy),
            ("hold", BuyAndHoldStrategy),
        ],
    )
    def test_resolves_builtin_aliases(self, reference, expected):
        assert resolve_strategy_class(reference) is expected

    def test_case_and_dash_insensitive(self):
        assert resolve_strategy_class("Mean-Reversion") is MeanReversionStrategy

    def test_unknown_reference_rejected(self):
        with pytest.raises(ValueError):
            resolve_strategy_class("nonsense")

    def test_coerces_string_parameters_to_declared_types(self):
        params = coerce_parameters(MomentumStrategy, {"lookback": "40", "allow_short": "false"})
        assert params == {"lookback": 40, "allow_short": False}

    def test_rejects_unknown_parameter(self):
        with pytest.raises(ValueError):
            coerce_parameters(MomentumStrategy, {"not_a_param": "1"})

    def test_load_strategy_builds_configured_instance(self):
        strategy = load_strategy("momentum", lookback=40)
        assert isinstance(strategy, MomentumStrategy)
        assert strategy.lookback == 40

    def test_loads_from_a_file(self, tmp_path):
        source = tmp_path / "my_strategy.py"
        source.write_text(
            "from marketerror import Order, Strategy\n"
            "class Contrarian(Strategy):\n"
            "    def on_data(self, view):\n"
            "        return Order('SELL', 1) if view.t == 0 else Order.hold()\n"
        )
        cls = resolve_strategy_class(str(source))
        assert cls.__name__ == "Contrarian"
        assert issubclass(cls, Strategy)

    def test_file_with_named_class(self, tmp_path):
        source = tmp_path / "two.py"
        source.write_text(
            "from marketerror import Order, Strategy\n"
            "class A(Strategy):\n"
            "    def on_data(self, view): return Order.hold()\n"
            "class B(Strategy):\n"
            "    def on_data(self, view): return Order.hold()\n"
        )
        with pytest.raises(ValueError):  # ambiguous without a class name
            resolve_strategy_class(str(source))
        assert resolve_strategy_class(f"{source}:B").__name__ == "B"


class TestStrategySpec:
    def test_round_trips_through_a_dict(self):
        spec = StrategySpec.parse("momentum", ["lookback=30"])
        assert spec.class_name == "MomentumStrategy"
        assert spec.build().lookback == 30

    def test_parse_rejects_malformed_assignment(self):
        with pytest.raises(ValueError):
            StrategySpec.parse("momentum", ["lookback"])

    def test_spec_is_serialisable(self):
        spec = StrategySpec.parse("mean_reversion", ["lookback=15"])
        payload = spec.to_dict()
        assert payload["class_name"] == "MeanReversionStrategy"
        assert payload["parameters"]["lookback"] == 15
