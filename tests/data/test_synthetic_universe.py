"""Tests for correlated multi-asset generation and causal views."""

from __future__ import annotations

import numpy as np
import pytest

from marketerror.data.synthetic_universe import SyntheticUniverseGenerator
from marketerror.data.universe_schema import UniverseView
from marketerror.market.parameters import MarketParameters
from marketerror.market.universe import UniverseParameters


def test_universe_shape_symbols_and_reproducibility():
    params = UniverseParameters.homogeneous(10, market_beta=0.6)
    generator = SyntheticUniverseGenerator(params)
    first = generator.generate(100, seed=42)
    second = generator.generate(100, seed=42)
    assert first.symbols == tuple(f"SYN{i:02d}" for i in range(10))
    assert first.price.shape == (100, 10)
    assert np.array_equal(first.price, second.price)
    assert first.price[0].tolist() == [100.0] * 10


def test_factor_correlation_and_volatility_are_calibrated():
    params = UniverseParameters.homogeneous(
        4, MarketParameters(annualized_volatility=0.20), market_beta=0.6
    )
    data = SyntheticUniverseGenerator(params).generate(50_000, seed=7)
    correlation = data.realized_correlation()
    off_diagonal = correlation[np.triu_indices(4, k=1)]
    assert float(off_diagonal.mean()) == pytest.approx(0.36, abs=0.03)
    volatility = data.returns[1:].std(axis=0, ddof=1) * np.sqrt(252)
    assert np.allclose(volatility, 0.20, rtol=0.03)


def test_symbol_streams_are_stable_when_universe_grows():
    base = MarketParameters()
    small = SyntheticUniverseGenerator(
        UniverseParameters.homogeneous(2, base, symbols=("A", "B"))
    ).generate(200, seed=12)
    large = SyntheticUniverseGenerator(
        UniverseParameters.homogeneous(3, base, symbols=("A", "B", "C"))
    ).generate(200, seed=12)
    assert np.array_equal(small.price[:, 0], large.price[:, 0])
    assert np.array_equal(small.price[:, 1], large.price[:, 1])


def test_view_is_causal_and_ranks_valid_symbols_only():
    params = UniverseParameters.homogeneous(3)
    data = SyntheticUniverseGenerator(params).generate(50, seed=2)
    features = {"signal_1": np.array([[np.nan, np.nan, np.nan], [3.0, 1.0, 2.0]] + [[3.0, 1.0, 2.0]] * 48)}
    view = UniverseView(data, features, index=1)
    assert view.top("signal_1", 2) == ("SYN00", "SYN02")
    assert view.bottom("signal_1", 2) == ("SYN01", "SYN02")
    assert len(view.history("price", "SYN00", 999)) == 2
    view._advance(10)
    with pytest.raises(ValueError):
        view._advance(9)
