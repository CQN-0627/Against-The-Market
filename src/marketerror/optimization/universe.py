"""Monte Carlo failure-boundary search for multi-asset universes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..backtest.engine import BacktestConfig
from ..data.distributions import path_seeds
from ..data.synthetic_universe import SyntheticUniverseGenerator
from ..market.universe import UniverseParameters
from ..perturbations.base import PerturbationSpace
from ..perturbations.universe import UniversePerturbationSpace
from ..strategies.universe_loader import UniverseStrategySpec
from .constraints import PerturbationConstraints
from .directional_search import RadialBisection
from .failure_boundary import FailureBoundary, minimum_failure
from .grid_search import GridSearch, GridSpec, SearchResults
from .objective import EuclideanSeverity, FailureCriteria, ScenarioEvaluation
from ..perturbations.vector import PerturbationVector
from ..simulation.monte_carlo import MonteCarloSummary, _outcome
from ..backtest.universe_engine import UniverseBacktester

__all__ = ["UniverseExperimentSpec", "UniverseExperimentRecord", "run_universe_experiment"]


@dataclass(frozen=True)
class UniverseExperimentSpec:
    strategy: UniverseStrategySpec
    market: UniverseParameters
    dimensions: tuple[str, ...] = ("volatility", "spread", "liquidity", "trend", "jump")
    constraints: PerturbationConstraints = PerturbationConstraints()
    grid: GridSpec = GridSpec()
    criteria: FailureCriteria = FailureCriteria()
    backtest: BacktestConfig = BacktestConfig()
    periods: int = 252
    paths: int = 32
    seed: int = 42
    exhaustive: bool = False
    refine: bool = True
    axis_scan: bool = True

    def __post_init__(self) -> None:
        if self.periods < 2 or self.paths < 1:
            raise ValueError("periods must be >= 2 and paths must be >= 1")

    def build_space(self) -> UniversePerturbationSpace:
        return UniversePerturbationSpace(self.dimensions)

    def seeds(self) -> list[Any]:
        return path_seeds(self.seed, self.paths)

    def validation_seeds(self) -> list[Any]:
        return path_seeds(self.seed + 1_000_000, self.paths)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.to_dict(),
            "universe": self.market.to_dict(),
            "dimensions": list(self.dimensions),
            "grid": self.grid.to_dict(),
            "constraints": self.constraints.to_dict(),
            "failure_criteria": self.criteria.to_dict(),
            "backtest": self.backtest.to_dict(),
            "periods": self.periods,
            "paths": self.paths,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class UniverseExperimentRecord:
    spec: UniverseExperimentSpec
    baseline: ScenarioEvaluation
    boundary: FailureBoundary
    results: SearchResults
    validation: MonteCarloSummary | None = None
    axis_results: tuple[Any, ...] = ()
    n_backtests: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "specification": self.spec.to_dict(),
            "baseline": self.baseline.to_row(),
            "failure_boundary": self.boundary.to_dict(),
            "search": {
                "n_candidates": self.results.n_candidates,
                "n_evaluated": self.results.n_evaluated,
                "coverage_note": self.results.coverage_note(),
                "n_backtests": self.n_backtests,
            },
            "validation": self.validation.to_dict() if self.validation else None,
        }

    def save(self, directory: str | Path) -> Path:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"{self.spec.strategy.class_name}_universe_seed{self.spec.seed}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path


class UniverseFailureObjective:
    def __init__(self, baseline: UniverseParameters, space: UniversePerturbationSpace, strategy: UniverseStrategySpec, seeds: Sequence[Any], periods: int, criteria: FailureCriteria, config: BacktestConfig):
        self.baseline = baseline
        self.space = space
        self.strategy_spec = strategy
        self.seeds = tuple(seeds)
        self.periods = periods
        self.criteria = criteria
        self.config = config
        self.severity_metric = EuclideanSeverity()
        self._cache: dict[tuple[float, ...], ScenarioEvaluation] = {}
        self._next_id = 0

    @property
    def n_evaluations(self) -> int:
        return len(self._cache)

    @property
    def n_backtests(self) -> int:
        return self.n_evaluations * len(self.seeds)

    def evaluate(self, z: Sequence[float] | Mapping[str, float], keep_results: int = 0) -> ScenarioEvaluation:
        if isinstance(z, Mapping):
            z = self.space.from_mapping(z)
        z = self.space._check(z)
        key = tuple(round(v, 6) for v in z)
        if key in self._cache and not keep_results:
            return self._cache[key]
        parameters, realised = self.space.realise(self.baseline, z)
        summary = run_universe_monte_carlo(parameters, self.strategy_spec, self.seeds, self.periods, self.config, self.criteria)
        evaluation = ScenarioEvaluation(
            scenario_id=self._next_id,
            vector=PerturbationVector(self.space.names, z),
            realised=PerturbationVector(self.space.names, realised),
            severity=self.severity_metric(realised),
            parameters=parameters,
            summary=summary,
            verdict=self.criteria.evaluate(summary),
        )
        self._next_id += 1
        if not keep_results:
            self._cache[key] = evaluation
        return evaluation

    def baseline_evaluation(self) -> ScenarioEvaluation:
        return self.evaluate(self.space.zeros())

    def severity_of(self, z: Sequence[float]) -> float:
        return self.severity_metric(self.space.realise(self.baseline, z)[1])

    def cached(self) -> list[ScenarioEvaluation]:
        return sorted(self._cache.values(), key=lambda e: e.scenario_id)


def run_universe_monte_carlo(parameters: UniverseParameters, strategy_spec: UniverseStrategySpec, seeds: Sequence[Any], periods: int, config: BacktestConfig, criteria: FailureCriteria) -> MonteCarloSummary:
    outcomes = []
    for seed in seeds:
        data = SyntheticUniverseGenerator(parameters).generate(periods=periods, seed=seed)
        result = UniverseBacktester(config).run(data, strategy_spec.build())
        outcomes.append(_outcome(result, criteria))
    from ..simulation.monte_carlo import _summarise
    return _summarise(outcomes, periods, ())


def run_universe_experiment(spec: UniverseExperimentSpec, progress: Callable[[int, int, ScenarioEvaluation], None] | None = None) -> UniverseExperimentRecord:
    space = spec.build_space()
    objective = UniverseFailureObjective(spec.market, space, spec.strategy, spec.seeds(), spec.periods, spec.criteria, spec.backtest)
    baseline = objective.baseline_evaluation()
    search = GridSearch(objective, spec.grid, spec.constraints, exhaustive=spec.exhaustive, progress=progress)
    results = search.run()
    best = minimum_failure(results.evaluations)
    refinement = None
    if best is not None and spec.refine and not best.vector.is_baseline:
        refinement = RadialBisection(objective, spec.constraints).refine(best.vector, known_failing_radius=best.severity, label=best.vector.label())
        if refinement.found and refinement.evaluation is not None and refinement.evaluation.severity < best.severity:
            best = refinement.evaluation
    validation = None
    if best is not None:
        validation = run_universe_monte_carlo(best.parameters, spec.strategy, spec.validation_seeds(), spec.periods, spec.backtest, spec.criteria)
    boundary = FailureBoundary(best, baseline, space, spec.market, results.n_evaluated, results.coverage_note(), max((e.severity for e in results.evaluations), default=0.0), refinement)
    return UniverseExperimentRecord(spec, baseline, boundary, results, validation, (), objective.n_backtests)
