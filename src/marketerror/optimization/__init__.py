"""Search for the minimum market disruption that makes a strategy fail.

Layered so that the definition of failure never depends on the search method:

``objective``
    The severity metric ``D(x) = ||x||_2``, the failure criteria (including
    ``--losstime``), and the evaluator that turns a z-vector into a verdict.
``constraints``
    The plausible region -- the ``+/-4 sigma`` box and economic validity.
``grid_search``
    Method 1: brute force, evaluated in ascending severity.
``directional_search``
    Method 2: radial bisection, refining a direction below grid resolution.
``failure_boundary``
    Pure analysis of evaluated scenarios: which failure was the smallest.

A future ``bayesian_search`` / ``evolutionary_search`` / ``ai_guided_search``
would sit alongside the two search modules and consume the same
``FailureObjective``; nothing in ``objective`` or ``failure_boundary`` would need
to change.
"""

from __future__ import annotations

from .constraints import PerturbationConstraints
from .directional_search import BisectionResult, RadialBisection
from .failure_boundary import (
    FailureBoundary,
    axis_sensitivity,
    minimum_failure,
    severity_profile,
    slice_2d,
)
from .grid_search import GridSearch, GridSpec, SearchResults
from .objective import (
    EuclideanSeverity,
    FailureCriteria,
    FailureObjective,
    FailureVerdict,
    ScenarioEvaluation,
    parse_loss_time,
)
from .universe import UniverseExperimentRecord, UniverseExperimentSpec, run_universe_experiment

__all__ = [
    "BisectionResult",
    "EuclideanSeverity",
    "FailureBoundary",
    "FailureCriteria",
    "FailureObjective",
    "FailureVerdict",
    "GridSearch",
    "GridSpec",
    "PerturbationConstraints",
    "RadialBisection",
    "ScenarioEvaluation",
    "SearchResults",
    "axis_sensitivity",
    "minimum_failure",
    "parse_loss_time",
    "severity_profile",
    "slice_2d",
    "UniverseExperimentRecord",
    "UniverseExperimentSpec",
    "run_universe_experiment",
]
