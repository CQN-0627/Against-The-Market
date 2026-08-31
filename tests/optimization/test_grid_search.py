"""Grid search: generic over dimensions, ascending-severity order, early stop."""

from __future__ import annotations

import numpy as np
import pytest

from marketerror.backtest.engine import BacktestConfig
from marketerror.backtest.execution import ExecutionConfig
from marketerror.market.parameters import MarketParameters
from marketerror.optimization.constraints import PerturbationConstraints
from marketerror.optimization.grid_search import DEFAULT_LEVELS, GridSearch, GridSpec
from marketerror.optimization.objective import FailureCriteria, FailureObjective
from marketerror.perturbations.dimensions import build_space
from marketerror.strategies.loader import StrategySpec


class TestGridSpec:
    def test_default_levels_match_the_specification(self):
        assert GridSpec().levels == (-2.0, -1.0, 0.0, 1.0, 2.0)

    def test_symmetric_reproduces_the_default(self):
        assert GridSpec.symmetric(2.0, 2).levels == DEFAULT_LEVELS

    def test_size_is_the_product_of_axis_lengths(self):
        space = build_space(("volatility", "spread", "liquidity"))
        assert GridSpec().size(space) == 5**3

    def test_per_dimension_levels(self):
        space = build_space(("volatility", "spread"))
        spec = GridSpec(levels=(-1.0, 0.0, 1.0), per_dimension={"spread": (0.0,)})
        assert spec.size(space) == 3 * 1

    def test_points_are_filtered_by_constraints(self):
        space = build_space(("volatility", "spread"))
        spec = GridSpec(levels=(-2.0, 0.0, 2.0))
        constraints = PerturbationConstraints(max_severity=2.0)
        points = list(spec.points(space, constraints))
        # (2, 2) has severity sqrt(8) > 2 and must be excluded.
        assert (2.0, 2.0) not in points
        assert (2.0, 0.0) in points

    def test_levels_are_sorted(self):
        assert GridSpec(levels=(2.0, -1.0, 0.0)).levels == (-1.0, 0.0, 2.0)


def _objective(reference="momentum", phi=0.15, paths=8, periods=120, **strat):
    """A small but real objective, for integration-style search tests."""
    return FailureObjective(
        baseline=MarketParameters(trend_persistence=phi),
        space=build_space(("volatility", "spread", "liquidity", "trend", "jump")),
        strategy_spec=StrategySpec(reference, strat),
        seeds=list(range(paths)),
        periods=periods,
        criteria=FailureCriteria(min_loss_probability=0.5, minimum_paths=paths),
        config=BacktestConfig(record_fills=False),
    )


class TestGridSearch:
    def test_ordered_search_stops_at_the_first_failure(self):
        objective = _objective(reference="momentum", lookback=5)
        search = GridSearch(objective, GridSpec(), PerturbationConstraints(), exhaustive=False)
        results = search.run()
        assert results.n_evaluated <= results.n_candidates
        failures = results.failures()
        if failures:
            # In ascending-severity order, the only failure is the last point.
            assert results.evaluations[-1].failed
            assert results.early_stopped == (results.n_evaluated < results.n_candidates)

    def test_candidates_are_ordered_by_severity(self):
        objective = _objective()
        search = GridSearch(objective, GridSpec(), exhaustive=False)
        severities = [objective.severity_of(p) for p in search.candidates()]
        assert severities == sorted(severities)
        assert search.candidates()[0] == (0.0, 0.0, 0.0, 0.0, 0.0)  # baseline first

    def test_exhaustive_evaluates_every_point(self):
        objective = _objective(paths=6, periods=80)
        spec = GridSpec(levels=(-1.0, 0.0, 1.0))
        search = GridSearch(objective, spec, exhaustive=True)
        results = search.run()
        assert results.n_evaluated == spec.size(objective.space)
        assert not results.early_stopped

    def test_caching_avoids_recomputation(self):
        objective = _objective(paths=6, periods=80)
        spec = GridSpec(levels=(-1.0, 0.0, 1.0))
        GridSearch(objective, spec, exhaustive=True).run()
        evaluated_once = objective.n_evaluations
        # A second identical search must reuse the cache entirely.
        GridSearch(objective, spec, exhaustive=True).run()
        assert objective.n_evaluations == evaluated_once

    def test_coverage_note_reports_what_was_skipped(self):
        objective = _objective(reference="momentum", lookback=5)
        results = GridSearch(objective, GridSpec(), exhaustive=False).run()
        note = results.coverage_note()
        assert str(results.n_evaluated) in note or "all" in note


class TestGenericDimensionality:
    """The search must not care how many dimensions there are."""

    @pytest.mark.parametrize("dims", [("volatility",), ("volatility", "spread"),
                                      ("volatility", "spread", "liquidity", "trend")])
    def test_runs_for_any_dimension_count(self, dims):
        objective = FailureObjective(
            baseline=MarketParameters(trend_persistence=0.15),
            space=build_space(dims),
            strategy_spec=StrategySpec("momentum", {"lookback": 5}),
            seeds=list(range(6)),
            periods=80,
            criteria=FailureCriteria(min_loss_probability=0.5, minimum_paths=6),
            config=BacktestConfig(record_fills=False),
        )
        results = GridSearch(objective, GridSpec(levels=(-1.0, 0.0, 1.0)), exhaustive=True).run()
        assert results.n_evaluated == 3 ** len(dims)
        for evaluation in results.evaluations:
            assert len(evaluation.realised.z) == len(dims)
