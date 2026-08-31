"""Standard-deviation-based market perturbations.

The pieces:

``standardization``
    ``x <-> z`` conversion, linear and log scales.
``base``
    :class:`PerturbationDimension` (one axis) and :class:`PerturbationSpace`
    (the coordinate system).
``dimensions``
    The registry of shockable dimensions and their calibrated dispersions.
``vector``
    :class:`PerturbationVector` and the severity metric ``D(x) = ||x||_2``.
``calibration``
    Optional empirical estimation of the dispersions.

Note on layout: the specification sketches one module per shocked variable
(``volatility.py``, ``spread.py``, ...).  Those are collapsed into
``dimensions.py`` deliberately -- each would hold a single declarative record,
and choosing a dispersion is a comparative judgement that is only auditable when
all of them are visible in one table.  ``docs/perturbations.md`` explains the
reasoning.
"""

from __future__ import annotations

from .base import PerturbationDimension, PerturbationSpace
from .dimensions import (
    ALL_DIMENSION_NAMES,
    DEFAULT_DIMENSION_NAMES,
    DIMENSIONS,
    build_space,
    get_dimension,
)
from .standardization import (
    LinearStandardizer,
    LogStandardizer,
    Standardizer,
    from_z_score,
    to_z_score,
)
from .vector import PerturbationVector, severity

__all__ = [
    "ALL_DIMENSION_NAMES",
    "DEFAULT_DIMENSION_NAMES",
    "DIMENSIONS",
    "LinearStandardizer",
    "LogStandardizer",
    "PerturbationDimension",
    "PerturbationSpace",
    "PerturbationVector",
    "Standardizer",
    "build_space",
    "from_z_score",
    "get_dimension",
    "severity",
    "to_z_score",
]
