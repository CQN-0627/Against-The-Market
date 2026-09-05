"""Focused tests for universe perturbation and optimizer orchestration."""

from __future__ import annotations

import numpy as np

from marketerror.backtest import BacktestConfig
from marketerror.market.parameters import MarketParameters
from marketerror.market.universe import UniverseParameters
from marketerror.optimization.grid_search import GridSpec
from marketerror.optimization.objective import FailureCriteria
from marketerror.optimization.universe import UniverseExperimentSpec, run_universe_experiment
from marketerror.perturbations.universe import UniversePerturbationSpace
from marketerror.strategies.universe_loader import UniverseStrategySpec


def test_common_shock_preserves_cross_sectional_shape():
    base = UniverseParameters.dispersed(3, MarketParameters())
    space = UniversePerturbationSpace(("volatility", "liquidity"))
    stressed, realised = space.realise(base, (1.0, -1.0))
    assert realised == (1.0, -1.0)
    ratios = [
        stressed.assets[i].market.annualized_volatility / base.assets[i].market.annualized_volatility
        for i in range(3)
    ]
    assert np.allclose(ratios, ratios[0])
    liquidity_ratios = [
        stressed.assets[i].market.liquidity / base.assets[i].market.liquidity
        for i in range(3)
    ]
    assert np.allclose(liquidity_ratios, liquidity_ratios[0])


def test_universe_optimizer_runs_and_caches_grid_points():
    spec = UniverseExperimentSpec(
        strategy=UniverseStrategySpec("examples/universe_backtest.py:CrossSectionalMomentum"),
        market=UniverseParameters.homogeneous(3, MarketParameters(drift=0.0)),
        dimensions=("volatility",),
        grid=GridSpec((0.0, 1.0)),
        criteria=FailureCriteria(minimum_paths=1),
        backtest=BacktestConfig(record_fills=False),
        periods=30,
        paths=2,
        seed=11,
        exhaustive=True,
        refine=False,
        axis_scan=False,
    )
    record = run_universe_experiment(spec)
    assert record.results.n_evaluated == 2
    assert record.n_backtests == 4
    assert record.baseline.summary.n_paths == 2
    assert record.spec.market.symbols == ("SYN00", "SYN01", "SYN02")
