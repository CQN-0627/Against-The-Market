r"""Plausibility constraints: what the optimiser is *not* allowed to propose.

Without a constraint set, "find the perturbation that breaks this strategy" has a
trivial and useless answer -- 100 sigma of volatility breaks everything.  A
result is only interesting if the market it describes could actually occur, so
the search is confined to a box:

.. math::  -z_{max} \le z_i \le +z_{max}

with :math:`z_{max} = 4` by default.  Four sigma is already a severe, rare
market; anything found beyond it says more about the model's tails than about the
strategy.

Two kinds of constraint are enforced, and they are not the same thing:

*Statistical* plausibility
    The box above, plus an optional cap on total severity.  These are choices
    about what counts as a believable market, and they are configurable.

*Economic* validity
    Volatility, spread, liquidity and price must remain positive.  These are not
    negotiable, and they are structurally guaranteed rather than checked: the log
    standardisation in
    :mod:`marketerror.perturbations.standardization` maps every real ``z`` to a
    valid positive parameter.  :meth:`PerturbationConstraints.validate_market`
    re-checks anyway, because a silent violation here would invalidate results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ..market.parameters import MarketParameters, ParameterError
from ..perturbations.base import PerturbationSpace

__all__ = ["PerturbationConstraints"]


@dataclass(frozen=True)
class PerturbationConstraints:
    """The feasible region of the perturbation space.

    Attributes
    ----------
    max_abs_z
        Symmetric box half-width applied to every dimension.
    per_dimension
        ``{name: (lower, upper)}`` overrides for individual axes.  Use this to
        pin a dimension at zero (``(0.0, 0.0)``) or to make it one-sided.
    max_severity
        Optional cap on ``D(x)``.  Independent of the box: a 5-dimensional
        ``+4 sigma`` corner has severity 8.9, which may be further from normal
        than you intend to consider.
    adverse_only
        Restrict each dimension to the side of zero its ``adverse_sign``
        nominates.  Halves the search per dimension, at the cost of being unable
        to discover that a strategy dies from a *favourable*-looking shock --
        which mean-reversion strategies do under rising trend persistence, so
        this is off by default.
    """

    max_abs_z: float = 4.0
    per_dimension: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    max_severity: float | None = None
    adverse_only: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_abs_z) or self.max_abs_z <= 0.0:
            raise ValueError("max_abs_z must be finite and > 0")
        if self.max_severity is not None and self.max_severity <= 0.0:
            raise ValueError("max_severity must be > 0")
        for name, bounds in self.per_dimension.items():
            lower, upper = bounds
            if lower > upper:
                raise ValueError(
                    f"per-dimension bounds for {name!r} are inverted: {bounds}"
                )
        object.__setattr__(
            self,
            "per_dimension",
            {k: (float(v[0]), float(v[1])) for k, v in self.per_dimension.items()},
        )

    # ------------------------------------------------------------------ bounds
    def bounds_for(self, space: PerturbationSpace, name: str) -> tuple[float, float]:
        """Effective ``(lower, upper)`` z bounds for one dimension."""
        override = self.per_dimension.get(name)
        if override is not None:
            return override
        lower, upper = -self.max_abs_z, self.max_abs_z
        if self.adverse_only:
            sign = space[name].adverse_sign
            if sign > 0:
                lower = 0.0
            else:
                upper = 0.0
        return lower, upper

    def all_bounds(self, space: PerturbationSpace) -> tuple[tuple[float, float], ...]:
        return tuple(self.bounds_for(space, name) for name in space.names)

    def clip(self, space: PerturbationSpace, z: Sequence[float]) -> tuple[float, ...]:
        """Project ``z`` into the box (severity caps are not enforced here)."""
        return tuple(
            min(max(value, lower), upper)
            for value, (lower, upper) in zip(z, self.all_bounds(space))
        )

    # -------------------------------------------------------------- feasibility
    def violations(self, space: PerturbationSpace, z: Sequence[float]) -> list[str]:
        """Human-readable list of every constraint ``z`` breaks."""
        problems = []
        for name, value, (lower, upper) in zip(
            space.names, z, self.all_bounds(space)
        ):
            if value < lower - 1e-12 or value > upper + 1e-12:
                problems.append(
                    f"{name}={value:+.3f} outside [{lower:+.2f}, {upper:+.2f}]"
                )
        if self.max_severity is not None:
            total = math.sqrt(math.fsum(v * v for v in z))
            if total > self.max_severity + 1e-12:
                problems.append(
                    f"severity {total:.3f} exceeds max_severity {self.max_severity:.3f}"
                )
        return problems

    def is_feasible(self, space: PerturbationSpace, z: Sequence[float]) -> bool:
        return not self.violations(space, z)

    def max_radius(self, space: PerturbationSpace, direction: Sequence[float]) -> float:
        """Largest ``r`` with ``r * direction`` still inside the box.

        Used by the radial searches to bracket their bisection without ever
        stepping outside the plausible region.
        """
        radius = math.inf
        for value, (lower, upper) in zip(direction, self.all_bounds(space)):
            if value > 1e-15:
                radius = min(radius, upper / value)
            elif value < -1e-15:
                radius = min(radius, lower / value)
        if self.max_severity is not None:
            norm = math.sqrt(math.fsum(v * v for v in direction))
            if norm > 0.0:
                radius = min(radius, self.max_severity / norm)
        return 0.0 if radius is math.inf else max(0.0, radius)

    # ---------------------------------------------------------------- validity
    @staticmethod
    def validate_market(parameters: MarketParameters) -> None:
        """Re-assert economic validity of a stressed parameter set."""
        try:
            parameters.validate()
        except ParameterError as exc:  # pragma: no cover - structurally prevented
            raise ParameterError(
                f"perturbation produced an economically invalid market: {exc}"
            ) from exc

    def describe(self, space: PerturbationSpace) -> list[str]:
        lines = []
        for name in space.names:
            lower, upper = self.bounds_for(space, name)
            lines.append(f"{name:<14} z in [{lower:+.2f}, {upper:+.2f}]")
        if self.max_severity is not None:
            lines.append(f"{'severity':<14} D(x) <= {self.max_severity:.2f}")
        return lines

    def to_dict(self) -> dict[str, object]:
        return {
            "max_abs_z": self.max_abs_z,
            "per_dimension": {k: list(v) for k, v in self.per_dimension.items()},
            "max_severity": self.max_severity,
            "adverse_only": self.adverse_only,
        }
