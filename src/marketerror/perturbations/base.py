r"""What a perturbable market dimension *is*.

A :class:`PerturbationDimension` binds four things together:

* the market parameter it moves (``annualized_volatility``, ``spread_bps``, ...),
* the dispersion :math:`\sigma` that defines what "one standard deviation" means
  for that parameter,
* the :class:`~marketerror.perturbations.standardization.Standardizer` that maps
  between the parameter's units and z-scores,
* the validity bounds that keep the resulting market economically meaningful.

:class:`PerturbationSpace` is a chosen set of those dimensions -- the coordinate
system the optimiser searches in.

Realised vs requested z
-----------------------
A dimension can be bounded (trend persistence must stay inside ``(-1, 1)``) or
discrete (latency is a whole number of bars).  When either applies, the market
actually simulated may not sit at exactly the requested ``z``.  Every
:meth:`PerturbationSpace.realise` call therefore returns the z-vector
*corresponding to the parameters it built*, and severity is computed from that.
Reporting the requested severity instead would overstate how small a
disruption was needed -- the one direction of error this framework must not make.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ..market.parameters import MarketParameters
from .standardization import LinearStandardizer, Standardizer

__all__ = ["PerturbationDimension", "PerturbationSpace"]


@dataclass(frozen=True)
class PerturbationDimension:
    """One axis of the perturbation space.

    Attributes
    ----------
    name
        Short key used on the command line and in result tables.
    parameter
        Field of :class:`~marketerror.market.parameters.MarketParameters` moved.
    std
        Baseline dispersion.  Absolute units for a linear standardiser, log units
        for a log standardiser.
    standardizer
        Strategy object performing the ``x <-> z`` conversion.
    adverse_sign
        The sign of ``z`` that is *generally* stressful (``+1`` for volatility,
        ``-1`` for liquidity).  Used only for labelling and for optional
        one-sided searches; severity is sign-blind because it squares ``z``.
    lower, upper
        Hard validity bounds on the parameter value.
    integer
        Round the parameter to a whole number (latency in bars).
    """

    name: str
    parameter: str
    std: float
    standardizer: Standardizer = field(default_factory=LinearStandardizer)
    adverse_sign: int = 1
    lower: float | None = None
    upper: float | None = None
    integer: bool = False
    label: str = ""
    units: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if self.std <= 0.0 or not math.isfinite(self.std):
            raise ValueError(f"{self.name}: std must be finite and > 0")
        if self.adverse_sign not in (-1, 1):
            raise ValueError(f"{self.name}: adverse_sign must be -1 or +1")
        if self.lower is not None and self.upper is not None and self.lower >= self.upper:
            raise ValueError(f"{self.name}: lower bound must be below upper bound")
        if not self.label:
            object.__setattr__(self, "label", self.name.replace("_", " ").title())

    # ------------------------------------------------------------ conversions
    def baseline_value(self, parameters: MarketParameters) -> float:
        """The parameter's unperturbed value, i.e. the ``mu`` of the z-score."""
        return float(parameters.get(self.parameter))

    def value_at(self, parameters: MarketParameters, z: float) -> float:
        """The parameter value implied by ``z``, after bounds and rounding."""
        mean = self.baseline_value(parameters)
        value = self.standardizer.from_z_score(z, mean, self.std)
        if self.integer:
            value = float(round(value))
        if self.lower is not None:
            value = max(value, self.lower)
        if self.upper is not None:
            value = min(value, self.upper)
        return value

    def z_of(self, value: float, parameters: MarketParameters) -> float:
        """The z-score of an explicit parameter value."""
        return self.standardizer.to_z_score(
            value, self.baseline_value(parameters), self.std
        )

    def realised_z(self, parameters: MarketParameters, z: float) -> float:
        """The z actually achieved once bounds and rounding are applied.

        Equal to ``z`` for every continuous, unbounded-in-range dimension, which
        is all of the defaults within the usual ``+/-4 sigma`` box.
        """
        return self.z_of(self.value_at(parameters, z), parameters)

    def apply(self, parameters: MarketParameters, z: float) -> MarketParameters:
        """Return ``parameters`` with this single dimension shocked to ``z``."""
        return parameters.replace(**{self.parameter: self.value_at(parameters, z)})

    # ----------------------------------------------------------------- display
    def format_value(self, value: float) -> str:
        if self.units == "%":
            return f"{value:.2%}"
        if self.units:
            return f"{value:,.4g} {self.units}"
        return f"{value:,.4g}"

    def sigma_table(
        self, parameters: MarketParameters, levels: Sequence[float] = (-2, -1, 0, 1, 2)
    ) -> list[tuple[float, float]]:
        """``(z, value)`` pairs, for documenting what a sigma means here."""
        return [(float(z), self.value_at(parameters, z)) for z in levels]


@dataclass(frozen=True)
class PerturbationSpace:
    """An ordered set of dimensions: the coordinate system of a search.

    >>> from marketerror.perturbations.dimensions import build_space
    >>> space = build_space(("volatility", "spread"))
    >>> space.names
    ('volatility', 'spread')
    """

    dimensions: tuple[PerturbationDimension, ...]

    def __post_init__(self) -> None:
        if not self.dimensions:
            raise ValueError("a perturbation space needs at least one dimension")
        seen = [d.name for d in self.dimensions]
        duplicates = {n for n in seen if seen.count(n) > 1}
        if duplicates:
            raise ValueError(f"duplicate dimensions: {sorted(duplicates)}")

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return len(self.dimensions)

    def __iter__(self) -> Iterator[PerturbationDimension]:
        return iter(self.dimensions)

    def __getitem__(self, key: "int | str") -> PerturbationDimension:
        if isinstance(key, int):
            return self.dimensions[key]
        for dimension in self.dimensions:
            if dimension.name == key:
                return dimension
        raise KeyError(f"no dimension named {key!r} in this space")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(d.name for d in self.dimensions)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def zeros(self) -> tuple[float, ...]:
        return (0.0,) * len(self.dimensions)

    # ------------------------------------------------------------ application
    def apply(
        self, parameters: MarketParameters, z: Sequence[float]
    ) -> MarketParameters:
        """Apply a full z-vector, shocking every dimension simultaneously.

        Each dimension is standardised against the *baseline* parameters, not
        against the partially perturbed ones, so the shocks are independent of
        the order in which they are applied.
        """
        z = self._check(z)
        changes = {
            dimension.parameter: dimension.value_at(parameters, value)
            for dimension, value in zip(self.dimensions, z)
        }
        return parameters.replace(**changes)

    def realise(
        self, parameters: MarketParameters, z: Sequence[float]
    ) -> tuple[MarketParameters, tuple[float, ...]]:
        """Apply ``z`` and report the z-vector actually achieved."""
        z = self._check(z)
        stressed = self.apply(parameters, z)
        realised = tuple(
            dimension.z_of(stressed.get(dimension.parameter), parameters)
            for dimension in self.dimensions
        )
        return stressed, realised

    def z_of(self, parameters: MarketParameters, baseline: MarketParameters) -> tuple[float, ...]:
        """Locate an arbitrary parameter set in this space's z-coordinates.

        Used to answer "how many sigmas from normal is the CRISIS regime?".
        """
        return tuple(
            dimension.z_of(parameters.get(dimension.parameter), baseline)
            for dimension in self.dimensions
        )

    def as_mapping(self, z: Sequence[float]) -> dict[str, float]:
        return dict(zip(self.names, self._check(z)))

    def from_mapping(self, mapping: Mapping[str, float]) -> tuple[float, ...]:
        """Build a full z-vector from a partial ``{name: z}`` mapping."""
        unknown = set(mapping) - set(self.names)
        if unknown:
            raise ValueError(
                f"unknown perturbation dimensions: {sorted(unknown)}; "
                f"this space has {list(self.names)}"
            )
        return tuple(float(mapping.get(name, 0.0)) for name in self.names)

    def describe(self, parameters: MarketParameters, z: Sequence[float]) -> list[str]:
        """Per-dimension ``label  +1.00 sigma   value`` lines for reports."""
        z = self._check(z)
        lines = []
        for dimension, value in zip(self.dimensions, z):
            stressed = dimension.value_at(parameters, value)
            base = dimension.baseline_value(parameters)
            lines.append(
                f"{dimension.label:<18} {value:>+6.2f}s   "
                f"{dimension.format_value(base)} -> {dimension.format_value(stressed)}"
            )
        return lines

    def _check(self, z: Sequence[float]) -> tuple[float, ...]:
        values = tuple(float(v) for v in z)
        if len(values) != len(self.dimensions):
            raise ValueError(
                f"expected {len(self.dimensions)} z values for "
                f"{list(self.names)}, got {len(values)}"
            )
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"z values must be finite, got {values!r}")
        return values

    def to_dict(self) -> list[dict[str, Any]]:
        return [
            {
                "name": d.name,
                "parameter": d.parameter,
                "std": d.std,
                "scale": d.standardizer.name,
                "adverse_sign": d.adverse_sign,
            }
            for d in self.dimensions
        ]
