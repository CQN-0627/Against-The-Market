r"""Locating the failure boundary in the search output.

Given a set of evaluated scenarios, the boundary is defined by one question:
among the scenarios that failed, which had the smallest severity?

Specification §16 is emphatic that this is *not* "the first scenario with a
negative return".  Scenarios are ranked by :math:`D(x)`, and ties are broken
toward the more decisively failing one, so a boundary point is never an artefact
of evaluation order.

The functions here are pure: they consume evaluations and produce answers,
running no simulations of their own.  That keeps "how we searched" and "what we
concluded" separable -- any future optimiser can hand its results to the same
analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..market.parameters import MarketParameters
from ..perturbations.base import PerturbationSpace
from .objective import ScenarioEvaluation

__all__ = [
    "FailureBoundary",
    "axis_sensitivity",
    "minimum_failure",
    "severity_profile",
    "slice_2d",
]


def minimum_failure(
    evaluations: Iterable[ScenarioEvaluation],
) -> ScenarioEvaluation | None:
    """The lowest-severity failing scenario, or ``None`` if none failed.

    Ties on severity are broken by the lower mean return: if two equally distant
    perturbations both break the strategy, the more damaging one is the better
    representative of the boundary.
    """
    failures = [e for e in evaluations if e.failed]
    if not failures:
        return None
    return min(failures, key=lambda e: (e.severity, e.summary.mean_return))


def severity_profile(
    evaluations: Iterable[ScenarioEvaluation],
) -> list[tuple[float, float, bool]]:
    """``(severity, mean_return, failed)`` sorted by severity.

    This is the data behind the "severity vs return" plot, where the minimum
    failure point should be visually obvious.
    """
    rows = [(e.severity, e.summary.mean_return, e.failed) for e in evaluations]
    return sorted(rows, key=lambda row: row[0])


def axis_sensitivity(
    evaluations: Iterable[ScenarioEvaluation], space: PerturbationSpace
) -> dict[str, dict[str, float]]:
    """Per-dimension marginal effect on mean return.

    For each axis, the mean return averaged over the scenarios at each level of
    that axis, collapsed into a single slope via least squares.  A crude but
    interpretable answer to "which dimension is this strategy most exposed to?",
    and a useful cross-check that the search's chosen direction makes sense.

    Only meaningful on an exhaustive search; an early-stopped one has visited a
    biased, low-severity subset.
    """
    evaluations = list(evaluations)
    out: dict[str, dict[str, float]] = {}
    for index, name in enumerate(space.names):
        z = np.array([e.realised.z[index] for e in evaluations], dtype=np.float64)
        y = np.array([e.summary.mean_return for e in evaluations], dtype=np.float64)
        spread = z.max() - z.min() if len(z) else 0.0
        if len(z) < 3 or spread <= 0.0:
            out[name] = {"slope": float("nan"), "range": float(spread)}
            continue
        centred = z - z.mean()
        denominator = float(centred @ centred)
        slope = float(centred @ (y - y.mean()) / denominator) if denominator > 0 else float("nan")
        out[name] = {"slope": slope, "range": float(spread)}
    return out


def slice_2d(
    evaluations: Iterable[ScenarioEvaluation],
    space: PerturbationSpace,
    dim_x: str,
    dim_y: str,
    aggregate: str = "min",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project the response surface onto two dimensions.

    Returns ``(x_levels, y_levels, mean_return, failure_fraction)`` with the
    latter two shaped ``(len(y_levels), len(x_levels))`` for direct use with
    ``pcolormesh``/``contourf``.

    ``aggregate`` controls how the other dimensions are collapsed: ``"min"``
    takes the worst return at each ``(x, y)`` cell (the pessimistic envelope,
    which is what a failure-boundary plot should show), ``"mean"`` averages.
    Cells with no evaluated scenario are ``NaN`` -- an early-stopped search
    leaves most of the plane empty, which the plot then renders as a gap rather
    than fabricating a value.
    """
    evaluations = list(evaluations)
    ix, iy = space.index(dim_x), space.index(dim_y)
    xs = sorted({round(e.realised.z[ix], 6) for e in evaluations})
    ys = sorted({round(e.realised.z[iy], 6) for e in evaluations})
    buckets: dict[tuple[float, float], list[ScenarioEvaluation]] = {}
    for evaluation in evaluations:
        key = (round(evaluation.realised.z[ix], 6), round(evaluation.realised.z[iy], 6))
        buckets.setdefault(key, []).append(evaluation)

    returns = np.full((len(ys), len(xs)), np.nan)
    failures = np.full((len(ys), len(xs)), np.nan)
    for row, y in enumerate(ys):
        for column, x in enumerate(xs):
            cell = buckets.get((x, y))
            if not cell:
                continue
            values = [e.summary.mean_return for e in cell]
            returns[row, column] = min(values) if aggregate == "min" else float(np.mean(values))
            failures[row, column] = float(np.mean([e.failed for e in cell]))
    return np.array(xs), np.array(ys), returns, failures


@dataclass(frozen=True)
class FailureBoundary:
    """The conclusion of a search: the minimum failure, in context."""

    minimum: ScenarioEvaluation | None
    baseline: ScenarioEvaluation
    space: PerturbationSpace
    parameters: MarketParameters
    n_evaluated: int
    coverage_note: str
    #: Largest severity anywhere in the search, so a null result can be stated
    #: as a bound rather than as an absence.
    max_severity_searched: float = 0.0
    refinement: Any = None

    @property
    def found(self) -> bool:
        return self.minimum is not None

    @property
    def severity(self) -> float:
        return self.minimum.severity if self.minimum else float("inf")

    def report_lines(self) -> list[str]:
        """The headline block of a robustness report."""
        if self.minimum is None:
            return [
                "MINIMUM FAILURE",
                "---------------",
                "None found.",
                "",
                f"No perturbation within the searched region (severity up to "
                f"{self.max_severity_searched:.2f}s) met the failure criterion.",
                "This is a bound, not proof of robustness: a finer grid, more "
                "paths, or additional dimensions may still find one.",
            ]

        minimum = self.minimum
        lines = [
            "MINIMUM FAILURE",
            "---------------",
            f"Severity:      {minimum.severity:.3f}s",
            "",
        ]
        lines += self.space.describe(self.parameters, minimum.realised.z)
        lines += [
            "",
            f"Baseline Return: {self.baseline.summary.mean_return:>+8.2%}",
            f"Stressed Return: {minimum.summary.mean_return:>+8.2%}",
            f"Loss Probability:{minimum.summary.loss_probability:>8.0%}",
            f"Failure Prob.:   {minimum.summary.failure_probability:>8.0%}",
        ]
        if minimum.verdict.underpowered:
            lines.append(
                f"WARNING: judged on {minimum.verdict.n_paths} paths, fewer than the "
                f"recommended minimum; re-run with more --paths before relying on this."
            )
        return lines

    def to_dict(self) -> dict[str, Any]:
        return {
            "found": self.found,
            "severity": self.severity if self.found else None,
            "minimum": self.minimum.to_row() if self.minimum else None,
            "baseline": self.baseline.to_row(),
            "n_evaluated": self.n_evaluated,
            "coverage_note": self.coverage_note,
            "max_severity_searched": self.max_severity_searched,
        }
