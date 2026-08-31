r"""The perturbation registry: every shockable dimension and its calibration.

Choosing :math:`\sigma` for each dimension is the single most consequential
modelling decision in MarketError, because it *defines the unit* that every
reported severity is quoted in.  The dispersions live together in one table
rather than in a file each, because calibration is inherently comparative: the
question "is one sigma of spread comparable to one sigma of volatility?" can
only be answered by reading them side by side.

What sigma means here
---------------------
These are **cross-regime dispersion priors**: how much each parameter varies
between plausible market environments.  They are *not* the sampling error of a
statistic estimated from one path.  The distinction matters enormously.  The
standard error of realised volatility measured over 252 days is roughly
:math:`\sigma/\sqrt{2T} \approx 0.9` percentage points, so a "+2 sigma
volatility shock" in sampling-error units would move 20% volatility to 21.8% --
a shock no strategy would notice.  In cross-regime units the same +2 sigma
reaches 44%, which is a genuine stress.

The priors below are anchored on two things: published dispersion of the
corresponding quantities in equity markets, and internal consistency with the
regimes in :mod:`marketerror.market.regimes`.  Each entry records where its
regime anchor lands, so the scale can be audited:

=================  ===============  =========================================
Dimension          sigma            Anchor
=================  ===============  =========================================
volatility         0.40 (log)       CRISIS 60% vol sits at +2.75 sigma
spread             0.30 (log)       9 bps vs 5 bps baseline = +1.96 sigma (§6)
liquidity          0.35 (log)       CRISIS liquidity 0.25 sits at -3.96 sigma
trend              0.10 (linear)    CRISIS persistence 0.30 sits at +3.00 sigma
jump               1.00 (log)       CRISIS intensity 0.010 sits at +2.30 sigma
=================  ===============  =========================================

``marketerror regimes`` prints this table from live code, so it cannot drift out
of date.  ``--sigma-source empirical`` swaps the priors for measured path-level
dispersion when you want the other interpretation; see
:mod:`marketerror.perturbations.calibration`.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from .base import PerturbationDimension, PerturbationSpace
from .standardization import LinearStandardizer, LogStandardizer

__all__ = [
    "ALL_DIMENSION_NAMES",
    "DEFAULT_DIMENSION_NAMES",
    "DIMENSIONS",
    "build_space",
    "get_dimension",
]

_LOG = LogStandardizer()
_LINEAR = LinearStandardizer()


VOLATILITY = PerturbationDimension(
    name="volatility",
    parameter="annualized_volatility",
    std=0.40,
    standardizer=_LOG,
    adverse_sign=+1,
    lower=1e-4,
    label="Volatility",
    units="%",
    description=(
        "Diffusive volatility of returns. One sigma is a factor of exp(0.40), so "
        "20% becomes 29.8% at +1 sigma and 44.5% at +2 sigma. Hurts strategies "
        "whose position size is fixed in shares rather than in risk."
    ),
)

SPREAD = PerturbationDimension(
    name="spread",
    parameter="spread_bps",
    std=0.30,
    standardizer=_LOG,
    adverse_sign=+1,
    lower=1e-4,
    label="Spread",
    units="bps",
    description=(
        "Quoted bid/ask spread before the liquidity adjustment. Calibrated so "
        "that the specification's worked example -- 9 bps against a 5 bps "
        "baseline -- lands at +1.96 sigma. Taxes turnover, so it hurts fast "
        "signals far more than slow ones."
    ),
)

LIQUIDITY = PerturbationDimension(
    name="liquidity",
    parameter="liquidity",
    std=0.35,
    standardizer=_LOG,
    adverse_sign=-1,
    lower=1e-4,
    label="Liquidity",
    description=(
        "Dimensionless depth multiplier; LOWER is worse, so the adverse "
        "direction is negative z. A -2 sigma shock halves depth, which widens "
        "the quote, raises market impact and starts rejecting size outright."
    ),
)

TREND = PerturbationDimension(
    name="trend",
    parameter="trend_persistence",
    std=0.10,
    standardizer=_LINEAR,
    adverse_sign=-1,
    lower=-0.95,
    upper=0.95,
    label="Trend",
    description=(
        "AR(1) coefficient on returns. Linear rather than log because it is "
        "legitimately negative: a mean-reverting market. The adverse direction "
        "is signed per strategy -- trend followers die as it falls, mean "
        "reverters as it rises -- so the nominal adverse_sign here is only a "
        "labelling default."
    ),
)

JUMP = PerturbationDimension(
    name="jump",
    parameter="jump_probability",
    std=1.00,
    standardizer=_LOG,
    adverse_sign=+1,
    lower=1e-9,
    upper=0.5,
    label="Jump intensity",
    description=(
        "Per-period probability of a discrete price jump. One sigma is a factor "
        "of e, because jump intensity varies over orders of magnitude rather "
        "than percentages. Note the baseline is deliberately tiny (0.25 jumps "
        "per year), so small jump shocks legitimately do very little; this shows "
        "up as a flat response near the origin rather than as a bug."
    ),
)

DRIFT = PerturbationDimension(
    name="drift",
    parameter="drift",
    std=0.10,
    standardizer=_LINEAR,
    adverse_sign=-1,
    label="Drift",
    units="%",
    description=(
        "Annualised expected return of the market itself. Linear, because it is "
        "legitimately negative. Not one of the five defaults -- specification §7 "
        "does not list it -- but it is the only dimension that can break a "
        "buy-and-hold strategy, because the generator deliberately holds expected "
        "return fixed under volatility and jump shocks. Add it via --dims to "
        "compare a signal strategy against its passive control on equal terms. "
        "CRISIS drift of -35% sits at -4.0 sigma."
    ),
)

JUMP_SIZE = PerturbationDimension(
    name="jump_size",
    parameter="jump_size",
    std=0.35,
    standardizer=_LOG,
    adverse_sign=+1,
    lower=1e-6,
    label="Jump size",
    units="%",
    description=(
        "Standard deviation of a jump's log size. Off by default: paired with "
        "`jump` it would double-count jump risk along two nearly collinear axes."
    ),
)

SLIPPAGE = PerturbationDimension(
    name="slippage",
    parameter="slippage_coefficient",
    std=0.40,
    standardizer=_LOG,
    adverse_sign=+1,
    lower=0.0,
    label="Slippage",
    description=(
        "The Y coefficient of the square-root impact law. An execution-side "
        "dimension: invisible to a strategy that trades in small size relative to "
        "market volume, and brutal to one that does not."
    ),
)

LATENCY = PerturbationDimension(
    name="latency",
    parameter="latency_periods",
    std=1.0,
    standardizer=_LINEAR,
    adverse_sign=+1,
    lower=0.0,
    upper=20.0,
    integer=True,
    label="Latency",
    units="bars",
    description=(
        "Bars between decision and execution. Discrete, so realised z is the "
        "rounded value -- the only default-adjacent dimension where requested "
        "and realised severity can differ. Off by default for that reason."
    ),
)


#: Every dimension, by name.
DIMENSIONS: Mapping[str, PerturbationDimension] = {
    d.name: d
    for d in (
        VOLATILITY,
        SPREAD,
        LIQUIDITY,
        TREND,
        JUMP,
        DRIFT,
        JUMP_SIZE,
        SLIPPAGE,
        LATENCY,
    )
}

#: The five dimensions searched unless ``--dims`` says otherwise.  This is the
#: list specification §7 nominates for version 1, and it matches the columns of
#: the results table in §15.
DEFAULT_DIMENSION_NAMES: tuple[str, ...] = (
    "volatility",
    "spread",
    "liquidity",
    "trend",
    "jump",
)

ALL_DIMENSION_NAMES: tuple[str, ...] = tuple(DIMENSIONS)


def get_dimension(name: str) -> PerturbationDimension:
    """Look up a dimension by name, with a helpful error for typos."""
    key = str(name).strip().lower().replace("-", "_")
    try:
        return DIMENSIONS[key]
    except KeyError:
        raise KeyError(
            f"unknown perturbation dimension {name!r}; available: "
            f"{', '.join(ALL_DIMENSION_NAMES)}"
        ) from None


def build_space(
    names: Iterable[str] | None = None,
    overrides: Mapping[str, float] | None = None,
) -> PerturbationSpace:
    """Build a :class:`PerturbationSpace` from dimension names.

    ``overrides`` replaces individual dispersions, which is how
    ``--sigma-source empirical`` and any user recalibration are applied without
    mutating the module-level registry.
    """
    selected: Sequence[str] = tuple(names) if names is not None else DEFAULT_DIMENSION_NAMES
    if not selected:
        raise ValueError("at least one perturbation dimension is required")
    dimensions = []
    for name in selected:
        dimension = get_dimension(name)
        if overrides and dimension.name in overrides:
            std = float(overrides[dimension.name])
            if std <= 0.0:
                raise ValueError(f"{dimension.name}: overridden std must be > 0")
            dimension = PerturbationDimension(
                name=dimension.name,
                parameter=dimension.parameter,
                std=std,
                standardizer=dimension.standardizer,
                adverse_sign=dimension.adverse_sign,
                lower=dimension.lower,
                upper=dimension.upper,
                integer=dimension.integer,
                label=dimension.label,
                units=dimension.units,
                description=dimension.description,
            )
        dimensions.append(dimension)
    return PerturbationSpace(tuple(dimensions))
