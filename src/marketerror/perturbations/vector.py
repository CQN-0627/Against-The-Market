r"""The perturbation vector and its severity.

A market disruption is a point in z-space:

.. math::

    x = [z_{volatility}, z_{spread}, z_{liquidity}, z_{trend}, z_{jump}]

and its severity is the Euclidean norm

.. math::

    D(x) = \sqrt{z_1^2 + z_2^2 + \cdots + z_n^2}

The consequence worth internalising is that **many small shocks are a smaller
disruption than one large one**.  Compare

===========================================  ==========================
``volatility = +2``, everything else 0       :math:`D = 2.00`
``volatility, spread, liquidity = +1,+1,-1`` :math:`D = \sqrt{3} = 1.73`
===========================================  ==========================

The second is the *smaller* perturbation under this metric, even though it
disturbs three properties instead of one.  That is the intended behaviour: it
reflects the fact that a market moving a little in several ways at once is a
more ordinary event than one moving a lot in a single dimension.  Under a
multivariate normal prior with independent dimensions, :math:`D(x)` is the
Mahalanobis distance from the baseline, and contours of constant :math:`D` are
contours of equal prior likelihood -- so "smallest disruption" means "most
probable disruption that breaks the strategy".

The independence assumption is not perfectly satisfied here, because liquidity
also widens the spread inside the market model.  ``docs/perturbations.md``
quantifies the resulting correlation and explains how to remove it
(``spread_liquidity_exponent=0``) if you want strictly orthogonal axes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

__all__ = ["PerturbationVector", "severity"]


def severity(z: Sequence[float], weights: Sequence[float] | None = None) -> float:
    r"""Euclidean norm of a z-vector, optionally weighted.

    ``weights`` rescales individual axes before the norm, for when you want to
    declare that one sigma of liquidity is a bigger deal than one sigma of
    spread.  Left at ``None``, every dimension counts equally, which is the
    definition the specification fixes.

    >>> round(severity([1.0, 1.0, -1.0]), 6)
    1.732051
    >>> severity([2.0, 0.0, 0.0])
    2.0
    """
    values = [float(v) for v in z]
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"z values must be finite, got {values!r}")
    if weights is None:
        return math.sqrt(math.fsum(v * v for v in values))
    weight_list = [float(w) for w in weights]
    if len(weight_list) != len(values):
        raise ValueError("weights must have the same length as the z vector")
    if any(w < 0.0 for w in weight_list):
        raise ValueError("weights must be non-negative")
    return math.sqrt(math.fsum((w * v) ** 2 for w, v in zip(weight_list, values)))


@dataclass(frozen=True)
class PerturbationVector:
    """A named z-vector: which dimensions were shocked, and by how much.

    >>> v = PerturbationVector(("volatility", "spread", "liquidity"), (1.0, 1.0, -1.0))
    >>> round(v.severity, 4)
    1.7321
    >>> v["liquidity"]
    -1.0
    >>> v.active
    (('volatility', 1.0), ('spread', 1.0), ('liquidity', -1.0))
    """

    dimensions: tuple[str, ...]
    z: tuple[float, ...]
    weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        set_(self, "dimensions", tuple(str(d) for d in self.dimensions))
        set_(self, "z", tuple(float(v) for v in self.z))
        if len(self.dimensions) != len(self.z):
            raise ValueError(
                f"got {len(self.dimensions)} dimension names but {len(self.z)} z values"
            )
        if not self.dimensions:
            raise ValueError("a perturbation vector needs at least one dimension")
        if not all(math.isfinite(v) for v in self.z):
            raise ValueError(f"z values must be finite, got {self.z!r}")
        if self.weights is not None:
            set_(self, "weights", tuple(float(w) for w in self.weights))
            if len(self.weights) != len(self.z):
                raise ValueError("weights must match the number of dimensions")

    # ----------------------------------------------------------------- factory
    @classmethod
    def zeros(cls, dimensions: Sequence[str]) -> "PerturbationVector":
        """The baseline: no shock at all."""
        return cls(tuple(dimensions), (0.0,) * len(dimensions))

    @classmethod
    def from_mapping(
        cls, dimensions: Sequence[str], mapping: Mapping[str, float]
    ) -> "PerturbationVector":
        """Build from a partial ``{name: z}`` mapping; omitted axes stay at 0."""
        names = tuple(dimensions)
        unknown = set(mapping) - set(names)
        if unknown:
            raise ValueError(
                f"unknown dimensions {sorted(unknown)}; expected some of {list(names)}"
            )
        return cls(names, tuple(float(mapping.get(n, 0.0)) for n in names))

    # ------------------------------------------------------------------ access
    def __len__(self) -> int:
        return len(self.z)

    def __iter__(self) -> Iterator[float]:
        return iter(self.z)

    def __getitem__(self, key: "int | str") -> float:
        if isinstance(key, int):
            return self.z[key]
        try:
            return self.z[self.dimensions.index(key)]
        except ValueError:
            raise KeyError(f"no dimension named {key!r}") from None

    def as_mapping(self) -> dict[str, float]:
        return dict(zip(self.dimensions, self.z))

    @property
    def active(self) -> tuple[tuple[str, float], ...]:
        """Only the non-zero components -- what a report should actually show."""
        return tuple(
            (name, value) for name, value in zip(self.dimensions, self.z) if value != 0.0
        )

    @property
    def is_baseline(self) -> bool:
        return not self.active

    # ---------------------------------------------------------------- geometry
    @property
    def severity(self) -> float:
        r"""The severity :math:`D(x) = \|x\|_2`."""
        return severity(self.z, self.weights)

    @property
    def max_abs_z(self) -> float:
        """The largest single-dimension shock -- the box constraint's binding side."""
        return max(abs(v) for v in self.z)

    @property
    def l1_norm(self) -> float:
        """Sum of absolute shocks. Reported for comparison, never optimised."""
        return math.fsum(abs(v) for v in self.z)

    def scaled(self, factor: float) -> "PerturbationVector":
        """Move along the same direction, scaling severity by ``factor``.

        This is the operation the radial bisection search iterates on: the
        direction is fixed and only the distance from baseline changes.
        """
        return PerturbationVector(
            self.dimensions, tuple(v * float(factor) for v in self.z), self.weights
        )

    def unit(self) -> "PerturbationVector":
        """The same direction at severity 1 (baseline vectors are returned as-is)."""
        norm = self.severity
        if norm == 0.0:
            return self
        return self.scaled(1.0 / norm)

    # ----------------------------------------------------------------- display
    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {f"{name}_z": value for name, value in zip(self.dimensions, self.z)}
        payload["severity"] = self.severity
        return payload

    def label(self) -> str:
        """Compact one-line rendering, e.g. ``volatility+1.0 liquidity-1.0``."""
        if self.is_baseline:
            return "baseline"
        return " ".join(f"{name}{value:+.2f}" for name, value in self.active)

    def __str__(self) -> str:
        return f"{self.label()} (D={self.severity:.3f})"
