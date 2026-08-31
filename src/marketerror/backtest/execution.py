r"""Order execution: quotes, market impact, partial fills and leverage limits.

This is where a market disruption actually reaches the strategy's P&L.  A spread
shock, a liquidity shock and a volatility shock all arrive here, by different
routes:

* a wider **spread** raises the price paid on every trade, so it taxes turnover;
* thinner **liquidity** shrinks both the volume impact is measured against and
  the top-of-book depth, so it raises slippage *and* starts rejecting size;
* higher **volatility** raises impact directly, because impact is proportional
  to it.

Market impact follows the empirical square-root law

.. math::

    \text{impact} = Y\,\sigma_{period}\sqrt{Q / V}

where :math:`Q` is the order size, :math:`V` the period's volume,
:math:`\sigma_{period}` the volatility of one period's return, and :math:`Y` the
dimensionless ``slippage_coefficient`` (empirically around 0.5-1.5).  The
volatility factor is what makes the formula dimensionally sensible and is the
reason impact is quoted as a fraction of price.

For scale: a 1,000-share order into 1,000,000 shares of volume at 20% annual
volatility costs about 4 bps.  Impact only becomes a first-order cost when the
strategy is large relative to the market, so raising ``--capital`` is the way to
make execution-side fragility matter.

Partial fills are a separate channel from impact: an order may take at most
``max_participation`` of *top-of-book depth*, and the excess is simply not
filled.  That is how a liquidity shock rations size rather than merely repricing
it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..data.schema import MarketData
from .orders import Fill, Order, Side
from .portfolio import Portfolio

__all__ = ["ExecutionConfig", "ExecutionModel"]

#: Fallback impact coefficient for data carrying no market parameters.
#: Matches ``MarketParameters.slippage_coefficient``.
_DEFAULT_SLIPPAGE_COEFFICIENT = 1.0

#: Window for the trailing volatility estimate used when the data does not
#: declare its own volatility (i.e. historical input).
_VOLATILITY_WINDOW = 20

#: Floor on the volatility estimate, so a flat stretch cannot make trading free.
_MIN_PERIOD_VOLATILITY = 1e-5


@dataclass(frozen=True)
class ExecutionConfig:
    """Broker-side assumptions.

    Attributes
    ----------
    commission_bps
        Commission per trade, in basis points of traded notional.
    use_spread
        Cross the spread on entry (buy at the ask, sell at the bid).  Turning
        this off is only useful for isolating other cost sources.
    max_participation
        Largest fraction of top-of-book depth a single order may take.  Excess
        size is not filled, which is how a liquidity shock becomes a *partial
        execution* rather than an infinitely elastic fill.
    max_leverage
        Cap on gross exposure as a multiple of equity.  ``1.0`` means a strategy
        can be fully long or fully short but never geared.  This is what stops a
        target-weight strategy from compounding into an unbounded position.
        The cap is applied at order time against pre-trade equity, so *realised*
        gross exposure can drift above it by the frictions subsequently paid --
        a handful of basis points.  It is exactly ``max_leverage`` when
        frictions are zero.
    allow_short
        Global short-selling switch, independent of any strategy's own setting.
    latency_periods
        Bars between a decision and its execution.  ``None`` inherits the market
        data's own ``latency_periods``, so the latency perturbation dimension
        flows through; an explicit value overrides it.

        ``0`` (the inherited default) means an order decided from bar ``t``'s
        information executes against bar ``t``'s quotes, crossing the spread.
        This is the standard bar-data convention and introduces no look-ahead:
        only information from bars up to ``t`` is used, and the spread is paid.
        ``1`` models a genuine one-bar execution delay, which is more
        conservative -- and materially so, because it pushes the first return the
        position earns from ``t+1`` to ``t+2``.
    slippage_coefficient
        The ``Y`` coefficient of the square-root impact law.  ``None`` means "use
        the value from the market data's own parameters", so that a slippage
        perturbation flows through automatically.
    interest_rate
        Annual rate applied to cash balances.  Defaults to zero; note that this
        means leveraged configurations do not pay borrowing costs, a documented
        simplification.
    """

    commission_bps: float = 1.0
    use_spread: bool = True
    max_participation: float = 0.25
    max_leverage: float = 1.0
    allow_short: bool = True
    latency_periods: int | None = None
    slippage_coefficient: float | None = None
    interest_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.commission_bps < 0.0:
            raise ValueError("commission_bps must be >= 0")
        if not 0.0 < self.max_participation <= 1.0:
            raise ValueError("max_participation must lie in (0, 1]")
        if self.max_leverage <= 0.0:
            raise ValueError("max_leverage must be > 0")
        if self.latency_periods is not None and self.latency_periods < 0:
            raise ValueError("latency_periods must be >= 0")
        if self.slippage_coefficient is not None and self.slippage_coefficient < 0.0:
            raise ValueError("slippage_coefficient must be >= 0")

    def to_dict(self) -> dict[str, object]:
        return {
            "commission_bps": self.commission_bps,
            "use_spread": self.use_spread,
            "max_participation": self.max_participation,
            "max_leverage": self.max_leverage,
            "allow_short": self.allow_short,
            "latency_periods": self.latency_periods,
            "slippage_coefficient": self.slippage_coefficient,
            "interest_rate": self.interest_rate,
        }


class ExecutionModel:
    """Turns an :class:`Order` into a :class:`Fill` against one bar of the book."""

    __slots__ = ("config", "_impact", "_volatility", "_latency")

    def __init__(self, config: ExecutionConfig, data: MarketData) -> None:
        self.config = config
        self._impact = self._resolve_impact(config, data)
        self._volatility = self._resolve_volatility(data)
        self._latency = self._resolve_latency(config, data)

    @staticmethod
    def _resolve_latency(config: ExecutionConfig, data: MarketData) -> int:
        """Execution delay, preferring the config, then the market's own value.

        Reading it from the market parameters is what makes ``latency`` a
        perturbable *market* dimension rather than a fixed broker assumption.
        """
        if config.latency_periods is not None:
            return int(config.latency_periods)
        parameters = data.metadata.get("parameters")
        if isinstance(parameters, dict):
            value = parameters.get("latency_periods")
            if value is not None:
                return max(0, int(round(float(value))))
        return 0

    @staticmethod
    def _resolve_impact(config: ExecutionConfig, data: MarketData) -> float:
        if config.slippage_coefficient is not None:
            return float(config.slippage_coefficient)
        parameters = data.metadata.get("parameters")
        if isinstance(parameters, dict):
            value = parameters.get("slippage_coefficient")
            if value is not None:
                return float(value)
        return _DEFAULT_SLIPPAGE_COEFFICIENT

    @staticmethod
    def _resolve_volatility(data: MarketData) -> np.ndarray:
        """Per-bar volatility used to scale market impact.

        Synthetic data declares its own volatility, which is used directly: that
        keeps the volatility perturbation an exact dial on impact rather than a
        noisy one.  Historical data gets a *causal* trailing estimate -- bar
        ``t`` uses returns up to ``t`` and never beyond, so this cannot introduce
        look-ahead.
        """
        parameters = data.metadata.get("parameters")
        if isinstance(parameters, dict):
            annualized = parameters.get("annualized_volatility")
            if annualized is not None:
                period = float(annualized) / math.sqrt(data.periods_per_year)
                return np.full(len(data), max(period, _MIN_PERIOD_VOLATILITY))

        returns = data.returns
        estimate = np.empty(len(returns))
        # Expanding window until the trailing window is available, so early bars
        # get the best causal estimate rather than a NaN.
        for t in range(len(returns)):
            start = max(1, t - _VOLATILITY_WINDOW + 1)
            window = returns[start : t + 1]
            estimate[t] = window.std(ddof=1) if len(window) > 1 else 0.0
        np.maximum(estimate, _MIN_PERIOD_VOLATILITY, out=estimate)
        return estimate

    @property
    def impact_coefficient(self) -> float:
        return self._impact

    @property
    def latency(self) -> int:
        return self._latency

    def execute(
        self, order: Order, index: int, data: MarketData, portfolio: Portfolio
    ) -> Fill:
        """Execute ``order`` against bar ``index``.

        The size is reduced in three stages -- available depth, then the
        leverage cap, then the short-selling switch -- and each binding
        constraint is recorded in ``Fill.reasons`` so a report can distinguish
        "the strategy chose not to trade" from "the market would not let it".
        """
        config = self.config
        mid = float(data.price[index])
        side = order.side
        if not order.is_actionable:
            return _empty_fill(index, side, 0.0, mid)

        sign = side.sign
        requested = order.quantity
        reasons: list[str] = []

        quote = mid
        if config.use_spread:
            quote = float(data.ask[index]) if sign > 0 else float(data.bid[index])

        depth = float(data.ask_size[index] if sign > 0 else data.bid_size[index])
        capacity = config.max_participation * depth
        quantity = requested
        if quantity > capacity:
            quantity = capacity
            reasons.append("depth")

        quantity = self._apply_position_limits(
            quantity, sign, mid, portfolio, reasons
        )
        if quantity <= 0.0:
            return _empty_fill(index, side, requested, mid, tuple(reasons))

        participation = quantity / depth if depth > 0.0 else 1.0
        impact = self._impact_fraction(quantity, index, data)
        price = quote * (1.0 + sign * impact)

        if order.limit_price is not None:
            crossed = price <= order.limit_price if sign > 0 else price >= order.limit_price
            if not crossed:
                reasons.append("limit")
                return _empty_fill(index, side, requested, mid, tuple(reasons))

        commission = config.commission_bps / 1e4 * quantity * price
        return Fill(
            index=index,
            side=side,
            requested_quantity=requested,
            filled_quantity=quantity,
            price=price,
            reference_price=mid,
            commission=commission,
            spread_cost=quantity * abs(quote - mid),
            slippage_cost=quantity * abs(price - quote),
            reasons=tuple(reasons),
        )

    # ------------------------------------------------------------------ impact
    def _impact_fraction(self, quantity: float, index: int, data: MarketData) -> float:
        r"""Square-root-law market impact, as a fraction of price.

        .. math::  \text{impact} = Y\,\sigma_{period}\sqrt{Q / V}

        Volume ``V`` already carries the liquidity multiplier, so a liquidity
        shock raises impact without any extra term.  Capped at 100% of price,
        which the formula only approaches for an order many times the period's
        entire volume.
        """
        volume = float(data.volume[index])
        if volume <= 0.0 or quantity <= 0.0:  # pragma: no cover - volume is > 0
            return 0.0
        fraction = self._impact * self._volatility[index] * math.sqrt(quantity / volume)
        return min(fraction, 1.0)

    # ------------------------------------------------------------------ limits
    def _apply_position_limits(
        self,
        quantity: float,
        sign: int,
        mid: float,
        portfolio: Portfolio,
        reasons: list[str],
    ) -> float:
        """Clip ``quantity`` so the resulting position respects the risk limits."""
        config = self.config
        position = portfolio.position
        target = position + sign * quantity

        if not config.allow_short and target < 0.0:
            quantity = max(0.0, -position / sign) if sign < 0 else quantity
            target = position + sign * quantity
            reasons.append("short_disabled")

        equity = portfolio.equity(mid)
        if equity <= 0.0:
            reasons.append("no_equity")
            return 0.0

        allowed_units = config.max_leverage * equity / mid
        if abs(target) > allowed_units * (1.0 + 1e-12):
            clipped_target = math.copysign(allowed_units, target)
            headroom = (clipped_target - position) / sign
            quantity = min(quantity, max(0.0, headroom))
            reasons.append("leverage")
        return quantity


def _empty_fill(
    index: int,
    side: Side,
    requested: float,
    mid: float,
    reasons: tuple[str, ...] = (),
) -> Fill:
    return Fill(
        index=index,
        side=side,
        requested_quantity=requested,
        filled_quantity=0.0,
        price=mid,
        reference_price=mid,
        reasons=reasons,
    )
