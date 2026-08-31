"""The full experiment pipeline, from baseline to validated failure boundary.

This is specification §28's procedure, in order, with nothing skipped:

1. establish baseline performance on the unperturbed market;
2. search the perturbation grid in ascending severity;
3. identify the minimum-severity failure;
4. refine it below grid resolution by radial bisection;
5. measure single-axis sensitivities for context;
6. re-validate the boundary scenario on *independent* paths;
7. assemble a reproducible record.

Step 1 is not optional and step 6 is the one that keeps the whole thing honest.
The grid selects a scenario *because* it looked like a failure on the search
paths, which is a selection effect: with enough scenarios, some will fail by
luck.  Re-running the winner on seeds it has never seen is the out-of-sample
check, and when it disagrees with the search the record says so rather than
quietly reporting the flattering number.
"""

from __future__ import annotations

import time
from concurrent.futures import Executor
from typing import Any, Callable

from ..optimization.directional_search import BisectionResult, RadialBisection
from ..optimization.failure_boundary import FailureBoundary, minimum_failure
from ..optimization.grid_search import GridSearch
from ..optimization.objective import FailureObjective, ScenarioEvaluation
from ..perturbations.calibration import EmpiricalCalibration, estimate_dispersions
from .monte_carlo import MonteCarloSummary, run_monte_carlo
from .scenarios import ExperimentRecord, ExperimentSpec

__all__ = ["calibrate_spec", "run_experiment"]

#: Called with short status strings so the CLI can narrate a long run.
Reporter = Callable[[str], None]


def _noop(_message: str) -> None:
    pass


def calibrate_spec(
    spec: ExperimentSpec, reporter: Reporter = _noop
) -> tuple[ExperimentSpec, EmpiricalCalibration | None]:
    """Apply ``--sigma-source empirical`` if requested, else return ``spec`` as-is."""
    if spec.sigma_source != "empirical":
        return spec, None
    reporter("Estimating empirical dispersions from baseline paths...")
    calibration = estimate_dispersions(
        spec.baseline_parameters,
        spec.build_space(),
        periods=spec.periods,
        paths=max(64, spec.paths),
        seed=spec.seed,
    )
    merged = dict(spec.dispersion_overrides)
    merged.update(calibration.overrides)
    return spec.with_(dispersion_overrides=merged), calibration


def run_experiment(
    spec: ExperimentSpec,
    executor: Executor | None = None,
    reporter: Reporter = _noop,
    progress: Callable[[int, int, ScenarioEvaluation], None] | None = None,
) -> ExperimentRecord:
    """Run the whole pipeline for one strategy and return a reproducible record."""
    started = time.perf_counter()
    space = spec.build_space()
    baseline_parameters = spec.baseline_parameters

    objective = FailureObjective(
        baseline=baseline_parameters,
        space=space,
        strategy_spec=spec.strategy,
        seeds=spec.seeds(),
        periods=spec.periods,
        criteria=spec.criteria,
        config=spec.backtest,
        executor=executor,
    )

    # 1. Baseline ----------------------------------------------------------------
    reporter("Establishing baseline...")
    baseline = objective.baseline_evaluation()
    if baseline.failed:
        reporter(
            "NOTE: the strategy already meets the failure criterion on the "
            "unperturbed market, so the minimum failure severity is 0."
        )

    # 2-3. Grid search ----------------------------------------------------------
    search = GridSearch(
        objective=objective,
        spec=spec.grid,
        constraints=spec.constraints,
        exhaustive=spec.exhaustive,
        progress=progress,
    )
    n_candidates = spec.grid.size(space)
    reporter(f"Searching {n_candidates:,} perturbation scenarios...")
    results = search.run()
    best = minimum_failure(results.evaluations)

    # 4. Radial refinement ------------------------------------------------------
    refinement: BisectionResult | None = None
    if best is not None and spec.refine and not best.vector.is_baseline:
        reporter("Refining the boundary by radial bisection...")
        bisection = RadialBisection(objective, spec.constraints)
        refinement = bisection.refine(
            best.vector, known_failing_radius=best.severity, label=best.vector.label()
        )
        if refinement.found and refinement.evaluation is not None:
            if refinement.evaluation.severity < best.severity:
                best = refinement.evaluation

    # 5. Single-axis sensitivities ---------------------------------------------
    axis_results: tuple[BisectionResult, ...] = ()
    if spec.axis_scan:
        reporter("Scanning single-axis sensitivities...")
        bisection = RadialBisection(objective, spec.constraints)
        axis_results = tuple(bisection.axis_scan())
        # A single axis can beat the grid's best combination when the grid's
        # resolution straddles the boundary; take it if so.
        single = RadialBisection.best(axis_results)
        if single is not None and single.evaluation is not None:
            if best is None or single.evaluation.severity < best.severity:
                best = single.evaluation

    # 6. Out-of-sample validation ----------------------------------------------
    validation: MonteCarloSummary | None = None
    if best is not None:
        reporter("Validating the boundary scenario on independent paths...")
        validation = run_monte_carlo(
            parameters=best.parameters,
            strategy_spec=spec.strategy,
            seeds=spec.validation_seeds(),
            periods=spec.periods,
            config=spec.backtest,
            failure_test=spec.criteria,
            executor=executor,
        )
        verdict = spec.criteria.evaluate(validation)
        if not verdict.failed:
            reporter(
                "WARNING: the boundary scenario did NOT reproduce its failure on "
                "independent paths. Treat the reported severity as optimistic and "
                "re-run with more --paths."
            )

    boundary = FailureBoundary(
        minimum=best,
        baseline=baseline,
        space=space,
        parameters=baseline_parameters,
        n_evaluated=results.n_evaluated,
        coverage_note=results.coverage_note(),
        max_severity_searched=max(
            (e.severity for e in results.evaluations), default=0.0
        ),
        refinement=refinement,
    )

    return ExperimentRecord(
        spec=spec,
        baseline=baseline,
        boundary=boundary,
        results=results,
        axis_results=axis_results,
        refinement=refinement,
        validation=validation,
        n_backtests=objective.n_backtests,
        elapsed_seconds=time.perf_counter() - started,
    )
