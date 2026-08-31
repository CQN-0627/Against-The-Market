"""Applying perturbations to market parameters, and locating markets in z-space.

The market-facing view of the perturbation machinery: given a baseline and a
z-vector, produce the stressed parameters; and inversely, given any parameter
set, say where it sits in sigma units.

The inverse direction is what makes severities interpretable.  A search that
reports "this strategy fails at 1.7 sigma" is only meaningful if you can also be
told that a crisis is roughly 5 sigma away, and that is exactly what
:func:`locate_parameters` and :func:`regime_table` compute -- from the same
calibration the optimiser uses, so the comparison cannot drift.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from ..perturbations.base import PerturbationSpace
from ..perturbations.vector import PerturbationVector
from .parameters import MarketParameters
from .regimes import REGIME_OVERRIDES, Regime, apply_regime

__all__ = [
    "apply_perturbation",
    "locate_parameters",
    "locate_regime",
    "regime_table",
    "stressed_parameter_lines",
]


def apply_perturbation(
    parameters: MarketParameters,
    space: PerturbationSpace,
    z: Sequence[float] | Mapping[str, float] | PerturbationVector,
) -> MarketParameters:
    """Return ``parameters`` shocked by ``z``.

    Accepts a full sequence, a partial ``{name: z}`` mapping, or a
    :class:`PerturbationVector`.  Every dimension is standardised against the
    baseline, so the result does not depend on application order.
    """
    return space.apply(parameters, _as_sequence(space, z))


def locate_parameters(
    parameters: MarketParameters,
    baseline: MarketParameters,
    space: PerturbationSpace,
) -> PerturbationVector:
    """Express an arbitrary parameter set as a perturbation of ``baseline``."""
    return PerturbationVector(space.names, space.z_of(parameters, baseline))


def locate_regime(
    regime: "str | Regime",
    baseline: MarketParameters,
    space: PerturbationSpace,
) -> PerturbationVector:
    """Where a named regime sits in z-space, relative to ``baseline``.

    Only the dimensions in ``space`` are counted, so a regime that also changes
    ``average_volume`` or ``drift`` will read as less severe than it is.  The
    returned severity is a lower bound on the regime's true distance.
    """
    return locate_parameters(apply_regime(baseline, regime), baseline, space)


def regime_table(
    baseline: MarketParameters, space: PerturbationSpace
) -> list[tuple[str, PerturbationVector, tuple[str, ...]]]:
    """``(regime, z-vector, unmeasured-parameters)`` for every regime.

    The third element names the parameters a regime changes that the current
    perturbation space cannot represent -- the reason its severity is a lower
    bound.
    """
    representable = {d.parameter for d in space}
    rows = []
    for regime in Regime:
        vector = locate_regime(regime, baseline, space)
        missing = tuple(
            sorted(set(REGIME_OVERRIDES[regime]) - representable)
        )
        rows.append((regime.value, vector, missing))
    return rows


def stressed_parameter_lines(
    baseline: MarketParameters,
    space: PerturbationSpace,
    z: Sequence[float] | Mapping[str, float] | PerturbationVector,
) -> list[str]:
    """Report lines showing each dimension's baseline value, z and stressed value."""
    return space.describe(baseline, _as_sequence(space, z))


def _as_sequence(
    space: PerturbationSpace,
    z: Sequence[float] | Mapping[str, float] | PerturbationVector,
) -> tuple[float, ...]:
    if isinstance(z, PerturbationVector):
        if tuple(z.dimensions) == space.names:
            return z.z
        return space.from_mapping(z.as_mapping())
    if isinstance(z, Mapping):
        return space.from_mapping(z)
    return tuple(float(v) for v in z)
