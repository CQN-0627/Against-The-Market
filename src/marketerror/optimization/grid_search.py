r"""Grid search: the brute-force baseline optimiser.

Version 1's search is intentionally the simplest thing that is *correct*.  Grid
search has no tuning parameters, no convergence assumptions and no dependence on
the response surface being smooth or monotone -- which matters, because a
strategy's response to market conditions is frequently neither.

Severity ordering
-----------------
The one refinement worth making is the order of evaluation.  Specification §16
says to collect the failures and return the minimum-severity one.  Evaluating
candidates in *ascending severity* is equivalent and much cheaper: the first
failure encountered is necessarily the minimum-severity failure in the grid, so
the search can stop there.  A 5-dimensional, 5-level grid is 3,125 points, and a
strategy that fails near 1.7 sigma is usually found within the first few dozen.

That shortcut trades away the full response surface, which the visualisations
want.  ``exhaustive=True`` evaluates every point instead; the CLI turns it on
automatically when plots are requested.  Either way the *answer* is identical --
only the amount of surrounding data differs.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Sequence

from ..perturbations.base import PerturbationSpace
from .constraints import PerturbationConstraints
from .objective import FailureObjective, ScenarioEvaluation

__all__ = ["GridSearch", "GridSpec", "SearchResults"]

#: The levels specification §14 nominates.
DEFAULT_LEVELS: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)


@dataclass(frozen=True)
class GridSpec:
    """Which z values to try on each axis.

    ``levels`` applies to every dimension; ``per_dimension`` overrides
    individual axes, which is how you give volatility a finer grid than jump
    intensity without multiplying the whole search.
    """

    levels: tuple[float, ...] = DEFAULT_LEVELS
    per_dimension: Mapping[str, Sequence[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.levels:
            raise ValueError("grid needs at least one level")
        object.__setattr__(self, "levels", tuple(sorted(float(v) for v in self.levels)))
        object.__setattr__(
            self,
            "per_dimension",
            {k: tuple(sorted(float(v) for v in vals)) for k, vals in self.per_dimension.items()},
        )

    @classmethod
    def symmetric(cls, max_abs_z: float = 2.0, steps: int = 2) -> "GridSpec":
        """``steps`` levels either side of zero, evenly spaced up to ``max_abs_z``.

        ``symmetric(2.0, 2)`` reproduces the default ``(-2, -1, 0, 1, 2)``.
        """
        if steps < 1:
            raise ValueError("steps must be >= 1")
        step = max_abs_z / steps
        return cls(tuple(round(i * step, 10) for i in range(-steps, steps + 1)))

    def levels_for(self, name: str) -> tuple[float, ...]:
        return tuple(self.per_dimension.get(name, self.levels))

    def size(self, space: PerturbationSpace) -> int:
        total = 1
        for name in space.names:
            total *= len(self.levels_for(name))
        return total

    def points(
        self,
        space: PerturbationSpace,
        constraints: PerturbationConstraints | None = None,
    ) -> Iterator[tuple[float, ...]]:
        """Every grid point, filtered to the feasible region."""
        axes = [self.levels_for(name) for name in space.names]
        for point in itertools.product(*axes):
            if constraints is None or constraints.is_feasible(space, point):
                yield point

    def to_dict(self) -> dict[str, Any]:
        return {
            "levels": list(self.levels),
            "per_dimension": {k: list(v) for k, v in self.per_dimension.items()},
        }


@dataclass(frozen=True)
class SearchResults:
    """Everything a search evaluated, plus an account of what it skipped."""

    evaluations: tuple[ScenarioEvaluation, ...]
    n_candidates: int
    early_stopped: bool = False
    method: str = "grid"

    @property
    def n_evaluated(self) -> int:
        return len(self.evaluations)

    @property
    def n_skipped(self) -> int:
        """Candidates never simulated because the search stopped early."""
        return max(0, self.n_candidates - self.n_evaluated)

    def failures(self) -> tuple[ScenarioEvaluation, ...]:
        return tuple(e for e in self.evaluations if e.failed)

    def survivors(self) -> tuple[ScenarioEvaluation, ...]:
        return tuple(e for e in self.evaluations if not e.failed)

    def sorted_by_severity(self) -> tuple[ScenarioEvaluation, ...]:
        return tuple(sorted(self.evaluations, key=lambda e: e.severity))

    def to_rows(self) -> list[dict[str, Any]]:
        return [e.to_row() for e in self.evaluations]

    def to_frame(self) -> Any:
        import pandas as pd

        return pd.DataFrame(self.to_rows())

    def coverage_note(self) -> str:
        """A sentence stating explicitly what was and was not evaluated."""
        if not self.early_stopped:
            return (
                f"evaluated all {self.n_evaluated:,} feasible grid points "
                f"({self.method})"
            )
        return (
            f"evaluated {self.n_evaluated:,} of {self.n_candidates:,} feasible grid "
            f"points in ascending severity and stopped at the first failure; the "
            f"{self.n_skipped:,} unevaluated points all have severity >= the "
            f"reported minimum, so none could improve on it"
        )


class GridSearch:
    """Brute-force search over a :class:`GridSpec`.

    Generic in the number of dimensions: the grid is the Cartesian product of
    whatever axes the :class:`PerturbationSpace` contains, so adding a
    perturbation dimension requires no change here.
    """

    def __init__(
        self,
        objective: FailureObjective,
        spec: GridSpec | None = None,
        constraints: PerturbationConstraints | None = None,
        exhaustive: bool = False,
        progress: Callable[[int, int, ScenarioEvaluation], None] | None = None,
    ) -> None:
        self.objective = objective
        self.spec = spec or GridSpec()
        self.constraints = constraints or PerturbationConstraints()
        self.exhaustive = exhaustive
        self.progress = progress

    @property
    def space(self) -> PerturbationSpace:
        return self.objective.space

    def candidates(self) -> list[tuple[float, ...]]:
        """Feasible grid points, ordered as they will be evaluated.

        Sorted by the severity the objective will actually report (i.e. after
        bounds and rounding), then lexicographically so the order is
        deterministic across runs and platforms.
        """
        points = list(self.spec.points(self.space, self.constraints))
        if self.exhaustive:
            return points
        return sorted(points, key=lambda p: (self.objective.severity_of(p), p))

    def run(self) -> SearchResults:
        """Evaluate the grid, stopping at the first failure unless exhaustive."""
        points = self.candidates()
        total = len(points)
        evaluations: list[ScenarioEvaluation] = []
        early_stopped = False

        for index, point in enumerate(points):
            evaluation = self.objective.evaluate(point)
            evaluations.append(evaluation)
            if self.progress is not None:
                self.progress(index + 1, total, evaluation)
            if evaluation.failed and not self.exhaustive:
                early_stopped = index + 1 < total
                break

        return SearchResults(
            evaluations=tuple(evaluations),
            n_candidates=total,
            early_stopped=early_stopped,
            method="grid-exhaustive" if self.exhaustive else "grid-ordered",
        )
