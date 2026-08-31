r"""Converting market parameters to and from standard deviations.

This module is the reason MarketError can add a volatility shock to a spread
shock and get a meaningful number.  Volatility is a percentage, spread is basis
points and liquidity is a dimensionless multiplier; they cannot be compared in
their native units.  Standardising each one against the dispersion of its own
baseline distribution puts them all on the same axis:

.. math::

    z = \frac{x - \mu}{\sigma}, \qquad x = \mu + z\sigma

Linear vs log scales
--------------------
The linear form above is the textbook definition and is used for parameters that
may legitimately be negative -- trend persistence, for instance, which is
negative in a mean-reverting market.

For strictly positive parameters (volatility, spread, liquidity, jump
intensity) a linear scale is actively wrong.  With :math:`\mu = 5` bps and
:math:`\sigma = 2` bps, :math:`z = -3` implies a spread of :math:`-1` bp: not a
market.  The usual fix is to clip, but clipping silently makes several distinct
``z`` values map to the same market, so the optimiser would report a severity
that does not correspond to what it actually simulated.  We instead standardise
in log space:

.. math::

    z = \frac{\ln(x/\mu)}{\sigma_{\log}}, \qquad x = \mu\,e^{z\sigma_{\log}}

which is a bijection onto :math:`(0, \infty)`.  Every :math:`z` maps to exactly
one valid market and vice versa, so specification §9's positivity constraints
hold for arbitrary ``z`` with no clipping at all, and the round trip
:math:`x \to z \to x` is exact.

Calibrating :math:`\sigma_{\log}` keeps the interpretation the specification
asks for.  Spread uses :math:`\sigma_{\log} = 0.30`, so a stressed spread of
9 bps against a 5 bps baseline gives
:math:`z = \ln(9/5)/0.30 = 1.96 \approx +2\sigma` -- the worked example in §6,
reproduced without ever admitting a negative spread.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

__all__ = [
    "LinearStandardizer",
    "LogStandardizer",
    "Standardizer",
    "from_z_score",
    "to_z_score",
]


class Standardizer(ABC):
    """Maps a parameter value to a z-score and back.

    The signatures take ``mean`` and ``std`` explicitly, rather than storing
    them, because the baseline mean is a property of the *experiment* (it moves
    when you change regime) while the scale is a property of the *parameter*.
    """

    #: Short label used in reports and docs.
    name: str = "standardizer"

    @abstractmethod
    def to_z_score(self, value: float, mean: float, std: float) -> float:
        """Standardise ``value`` against a baseline of ``mean`` and ``std``."""

    @abstractmethod
    def from_z_score(self, z: float, mean: float, std: float) -> float:
        """Invert :meth:`to_z_score`."""

    def describe(self, mean: float, std: float, z: float) -> str:
        return f"{self.from_z_score(z, mean, std):.6g}"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}()"


class LinearStandardizer(Standardizer):
    """The textbook z-score, for parameters that may be negative.

    >>> s = LinearStandardizer()
    >>> s.to_z_score(value=9, mean=5, std=2)
    2.0
    >>> s.from_z_score(z=2, mean=5, std=2)
    9.0
    """

    name = "linear"

    def to_z_score(self, value: float, mean: float, std: float) -> float:
        _check_std(std)
        return (float(value) - float(mean)) / float(std)

    def from_z_score(self, z: float, mean: float, std: float) -> float:
        _check_std(std)
        return float(mean) + float(z) * float(std)


class LogStandardizer(Standardizer):
    """Multiplicative z-score for strictly positive parameters.

    ``std`` is a dispersion *in log units*: 0.30 means "one sigma is a factor of
    ``exp(0.30)``", i.e. about +35%/-26%.

    >>> s = LogStandardizer()
    >>> round(s.to_z_score(value=9.0, mean=5.0, std=0.30), 4)
    1.9591
    >>> round(s.from_z_score(z=2.0, mean=5.0, std=0.30), 4)
    9.1106
    >>> s.from_z_score(z=-40.0, mean=5.0, std=0.30) > 0   # never leaves (0, inf)
    True
    """

    name = "log"

    def to_z_score(self, value: float, mean: float, std: float) -> float:
        _check_std(std)
        value, mean = float(value), float(mean)
        if value <= 0.0 or mean <= 0.0:
            raise ValueError(
                f"log standardisation needs strictly positive values, "
                f"got value={value!r}, mean={mean!r}"
            )
        return math.log(value / mean) / float(std)

    def from_z_score(self, z: float, mean: float, std: float) -> float:
        _check_std(std)
        mean = float(mean)
        if mean <= 0.0:
            raise ValueError(f"log standardisation needs mean > 0, got {mean!r}")
        return mean * math.exp(float(z) * float(std))


def _check_std(std: float) -> None:
    if not math.isfinite(std) or std <= 0.0:
        raise ValueError(f"std must be finite and > 0, got {std!r}")


#: The default standardiser, so the module-level helpers below match the
#: textbook definition.
_DEFAULT = LinearStandardizer()


def to_z_score(value: float, mean: float, std: float) -> float:
    """Linear z-score, as a plain function.

    >>> to_z_score(value=9, mean=5, std=2)
    2.0
    """
    return _DEFAULT.to_z_score(value, mean, std)


def from_z_score(z: float, mean: float, std: float) -> float:
    """Inverse linear z-score, as a plain function.

    >>> from_z_score(z=2, mean=5, std=2)
    9.0
    """
    return _DEFAULT.from_z_score(z, mean, std)
