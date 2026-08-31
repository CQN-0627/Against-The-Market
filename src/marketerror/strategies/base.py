"""The strategy interface.

The framework knows three things about a strategy: which features it wants,
what it does with a bar, and how to reset it.  It knows nothing about its
internal logic -- that is what makes MarketError strategy-agnostic, and it is
why a user-supplied ``strategy.py`` can be dropped in without touching the
engine.

Minimal implementation::

    from marketerror import Order, Strategy

    class MyStrategy(Strategy):
        def requires(self):
            return ("return_20",)

        def on_data(self, data):
            if data["return_20"] > 0:
                return Order("BUY", 100)
            return Order.hold()

Contract
--------
``on_data`` is called once per bar, in order, and receives a
:class:`~marketerror.data.schema.MarketView` that can only see the current bar
and earlier ones.  Returning ``None`` is equivalent to ``Order.hold()``.

Strategies must be **stateless across runs**: any internal state has to be
cleared in :meth:`reset`, which the engine calls before every path.  A strategy
that leaks state between paths would make the Monte Carlo layer measure the
wrong thing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, fields, is_dataclass
from typing import Any, Sequence

from ..backtest.orders import Order
from ..data.schema import MarketView

__all__ = ["Strategy"]


class Strategy(ABC):
    """Base class for all trading strategies."""

    #: Fraction of equity a full-size position represents.  ``1.0`` is fully
    #: invested; ``0.5`` uses half the capital.  Only used by the position
    #: helpers below -- a strategy is free to ignore them and size in shares.
    allocation: float = 1.0

    #: Rebalances smaller than this fraction of equity are suppressed.  Without
    #: a deadband, a target-weight strategy re-trades every bar as equity drifts
    #: and pays spread on noise, which would confound genuine turnover costs.
    rebalance_tolerance: float = 0.02

    # ---------------------------------------------------------------- interface
    @abstractmethod
    def on_data(self, market_data: MarketView) -> Order | None:
        """React to one bar.  Return an :class:`Order` (or ``None`` to hold)."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clear per-run state.  Called by the engine before every path."""

    def requires(self) -> Sequence[str]:
        """Names of the rolling features this strategy reads.

        Declaring them lets the engine precompute them vectorised, and makes a
        typo fail at setup instead of silently yielding ``NaN``.
        """
        return ()

    # ----------------------------------------------------------------- identity
    @property
    def name(self) -> str:
        return type(self).__name__

    def parameters(self) -> dict[str, Any]:
        """Strategy parameters, recorded in every experiment for reproducibility.

        Dataclass strategies get this for free; others fall back to their public
        instance attributes.
        """
        if is_dataclass(self):
            return {f.name: getattr(self, f.name) for f in fields(self)}
        return {
            key: value
            for key, value in vars(self).items()
            if not key.startswith("_") and isinstance(value, (int, float, str, bool))
        }

    def describe(self) -> str:
        params = self.parameters()
        if not params:
            return self.name
        rendered = ", ".join(f"{k}={v!r}" for k, v in sorted(params.items()))
        return f"{self.name}({rendered})"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.describe()

    # ------------------------------------------------------- position helpers
    def order_to_shares(self, view: MarketView, target_shares: float) -> Order:
        """Order that moves the position toward ``target_shares``.

        Suppresses rebalances whose notional is below
        ``rebalance_tolerance * equity``.
        """
        delta = target_shares - view.position
        equity = view.equity
        if equity > 0.0 and abs(delta) * view.price < self.rebalance_tolerance * equity:
            return Order.hold()
        return Order.signed(delta)

    def order_to_weight(self, view: MarketView, weight: float) -> Order:
        """Order that moves the position toward ``weight`` of current equity.

        ``weight=1.0`` is fully long, ``-1.0`` fully short, ``0.0`` flat.  The
        strategy's :attr:`allocation` scales the result.
        """
        equity = view.equity
        if equity <= 0.0:  # wiped out; stop trading rather than trade on debt
            return Order.hold()
        target = weight * self.allocation * equity / view.price
        return self.order_to_shares(view, target)
