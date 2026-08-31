r"""Radial bisection: sharpening a boundary the grid can only bracket.

Specification §17's second method.  A grid with levels at :math:`\pm1, \pm2` can
only ever report severities drawn from a discrete set -- if a strategy really
fails at 1.42 sigma, a grid whose nearest failing point is
:math:`(1, 1, -1)` will report 1.73 sigma.  That is an *over*-statement of how
much disruption is needed, so it must be refined rather than left alone.

The method fixes a direction :math:`u` (a unit z-vector) and bisects on the
radius :math:`r`:

.. code-block:: text

    r = 0.00  ->  profitable      (the baseline, known)
    r = 1.73  ->  unprofitable    (the grid's failing point)
    r = 0.87  ->  profitable
    r = 1.30  ->  unprofitable
    r = 1.08  ->  profitable
    ...                            converges on the boundary along u

Each halving costs one Monte Carlo evaluation, so 12 iterations resolve the
boundary to about 0.04 sigma for the price of 12 scenarios -- versus the
thousands a grid of equivalent resolution would need.

Honest limits
-------------
Bisection assumes failure is monotone in :math:`r` along the direction searched.
That is usually true and is helped considerably by common random numbers (all
radii are evaluated on identical market draws, so the comparison is paired), but
it is not guaranteed: a strategy can survive a moderate shock and fail at both
smaller and larger ones.  What bisection returns is therefore *a* boundary along
:math:`u`, bracketed by an explicit surviving radius below and failing radius
above.  The grid remains the global check; bisection only refines within a
direction the grid already identified.  ``BisectionResult`` reports the bracket
so the residual uncertainty is visible rather than implied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from ..perturbations.base import PerturbationSpace
from ..perturbations.vector import PerturbationVector
from .constraints import PerturbationConstraints
from .objective import FailureObjective, ScenarioEvaluation

__all__ = ["BisectionResult", "RadialBisection"]


@dataclass(frozen=True)
class BisectionResult:
    """The outcome of bisecting along one direction."""

    direction: PerturbationVector
    label: str
    surviving_radius: float
    failing_radius: float
    evaluation: ScenarioEvaluation | None
    iterations: int
    max_radius: float

    @property
    def found(self) -> bool:
        return self.evaluation is not None and math.isfinite(self.failing_radius)

    @property
    def severity(self) -> float:
        """Severity of the boundary scenario, or ``inf`` if none was found."""
        return self.evaluation.severity if self.evaluation else float("inf")

    @property
    def uncertainty(self) -> float:
        """Width of the surviving/failing bracket, in sigma."""
        if not self.found:
            return float("nan")
        return self.failing_radius - self.surviving_radius

    def summary(self) -> str:
        if not self.found:
            return (
                f"{self.label}: no failure up to {self.max_radius:.2f}s "
                f"(the constraint boundary)"
            )
        return (
            f"{self.label}: fails at {self.severity:.3f}s "
            f"(survives {self.surviving_radius:.3f}s, bracket {self.uncertainty:.3f}s)"
        )


class RadialBisection:
    """Bisects the failure boundary along fixed directions in z-space."""

    def __init__(
        self,
        objective: FailureObjective,
        constraints: PerturbationConstraints | None = None,
        tolerance: float = 0.05,
        max_iterations: int = 12,
    ) -> None:
        if tolerance <= 0.0:
            raise ValueError("tolerance must be > 0")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self.objective = objective
        self.constraints = constraints or PerturbationConstraints()
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    @property
    def space(self) -> PerturbationSpace:
        return self.objective.space

    # ------------------------------------------------------------------ single
    def refine(
        self,
        direction: Sequence[float] | PerturbationVector,
        known_failing_radius: float | None = None,
        label: str = "",
    ) -> BisectionResult:
        """Find the smallest failing radius along ``direction``.

        ``known_failing_radius`` seeds the upper bracket with a radius already
        known to fail (typically the grid's minimum failure), saving one
        evaluation and guaranteeing the bisection starts from a valid bracket.
        """
        vector = (
            direction
            if isinstance(direction, PerturbationVector)
            else PerturbationVector(self.space.names, tuple(direction))
        )
        norm = vector.severity
        if norm <= 0.0:
            raise ValueError("cannot bisect along a zero-length direction")
        unit = vector.unit()
        label = label or unit.label()

        max_radius = self.constraints.max_radius(self.space, unit.z)
        if max_radius <= 0.0:
            return BisectionResult(unit, label, 0.0, math.inf, None, 0, 0.0)

        # Establish a failing upper bracket.
        if known_failing_radius is not None and known_failing_radius <= max_radius:
            high = known_failing_radius
            high_evaluation = self.objective.evaluate(unit.scaled(high).z)
            if not high_evaluation.failed:  # hint was stale; fall back to the box
                high, high_evaluation = max_radius, self.objective.evaluate(
                    unit.scaled(max_radius).z
                )
        else:
            high = max_radius
            high_evaluation = self.objective.evaluate(unit.scaled(high).z)

        if not high_evaluation.failed:
            return BisectionResult(unit, label, high, math.inf, None, 0, max_radius)

        low = 0.0
        iterations = 0
        while high - low > self.tolerance and iterations < self.max_iterations:
            middle = 0.5 * (low + high)
            evaluation = self.objective.evaluate(unit.scaled(middle).z)
            iterations += 1
            if evaluation.failed:
                high, high_evaluation = middle, evaluation
            else:
                low = middle

        return BisectionResult(
            direction=unit,
            label=label,
            surviving_radius=low,
            failing_radius=high,
            evaluation=high_evaluation,
            iterations=iterations,
            max_radius=max_radius,
        )

    # ------------------------------------------------------------------- scans
    def axis_scan(
        self,
        dimensions: Iterable[str] | None = None,
        progress: Callable[[BisectionResult], None] | None = None,
    ) -> list[BisectionResult]:
        """Bisect each dimension alone, in both directions.

        Produces the single-factor sensitivity table: "volatility on its own
        breaks this strategy at +2.4 sigma; spread never does".  Because these
        are one-dimensional, their severities are directly comparable with the
        multi-dimensional minimum, and the gap between the best single-axis
        result and the joint minimum quantifies how much of the fragility comes
        from *combinations* of conditions.
        """
        names = tuple(dimensions) if dimensions is not None else self.space.names
        results = []
        for name in names:
            index = self.space.index(name)
            for sign in (1.0, -1.0):
                z = [0.0] * len(self.space)
                z[index] = sign
                lower, upper = self.constraints.bounds_for(self.space, name)
                if (sign > 0 and upper <= 0.0) or (sign < 0 and lower >= 0.0):
                    continue  # this side is excluded by the constraints
                result = self.refine(z, label=f"{name}{'+' if sign > 0 else '-'}")
                results.append(result)
                if progress is not None:
                    progress(result)
        return results

    @staticmethod
    def best(results: Iterable[BisectionResult]) -> BisectionResult | None:
        """The lowest-severity boundary among several bisections."""
        found = [r for r in results if r.found]
        if not found:
            return None
        return min(found, key=lambda r: r.severity)
