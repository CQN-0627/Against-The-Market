"""Orders and fills: the vocabulary between a strategy and the backtester.

A strategy's only output is an :class:`Order`.  It never touches cash,
positions or prices directly -- that separation is what lets the same strategy
object be replayed against thousands of different market paths without carrying
state between them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = ["Fill", "Order", "Side"]


class Side(str, Enum):
    """Direction of an order.  ``HOLD`` means "do nothing this bar"."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

    @classmethod
    def parse(cls, value: "str | Side") -> "Side":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().upper())
        except ValueError as exc:
            raise ValueError(
                f"unknown order side {value!r}; expected BUY, SELL or HOLD"
            ) from exc

    @property
    def sign(self) -> int:
        return {Side.BUY: 1, Side.SELL: -1, Side.HOLD: 0}[self]


@dataclass(frozen=True)
class Order:
    """A market or limit order for ``quantity`` units.

    ``quantity`` is always non-negative; direction lives in ``side``.  The
    constructor accepts plain strings so the documented shorthand works::

        Order("BUY", 100)
        Order(Side.SELL, 50, limit_price=99.5)
        Order.hold()
    """

    side: Side = Side.HOLD
    quantity: float = 0.0
    limit_price: float | None = None
    tag: str = ""

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        set_(self, "side", Side.parse(self.side))
        quantity = float(self.quantity)
        if not math.isfinite(quantity):
            raise ValueError("order quantity must be finite")
        if quantity < 0.0:
            raise ValueError(
                f"order quantity must be non-negative (direction is in `side`), "
                f"got {quantity!r}"
            )
        if self.side is Side.HOLD:
            quantity = 0.0
        set_(self, "quantity", quantity)
        if self.limit_price is not None:
            limit = float(self.limit_price)
            if not math.isfinite(limit) or limit <= 0.0:
                raise ValueError("limit_price must be finite and > 0")
            set_(self, "limit_price", limit)

    # ------------------------------------------------------------ constructors
    @classmethod
    def buy(cls, quantity: float, **kwargs: Any) -> "Order":
        return cls(Side.BUY, quantity, **kwargs)

    @classmethod
    def sell(cls, quantity: float, **kwargs: Any) -> "Order":
        return cls(Side.SELL, quantity, **kwargs)

    @classmethod
    def hold(cls) -> "Order":
        return cls(Side.HOLD, 0.0)

    @classmethod
    def signed(cls, quantity: float, **kwargs: Any) -> "Order":
        """Build from a signed quantity: positive buys, negative sells."""
        if quantity > 0:
            return cls.buy(quantity, **kwargs)
        if quantity < 0:
            return cls.sell(-quantity, **kwargs)
        return cls.hold()

    # ---------------------------------------------------------------- queries
    @property
    def signed_quantity(self) -> float:
        return self.side.sign * self.quantity

    @property
    def is_actionable(self) -> bool:
        return self.side is not Side.HOLD and self.quantity > 0.0

    def __bool__(self) -> bool:
        return self.is_actionable


@dataclass(frozen=True)
class Fill:
    """The outcome of submitting an order, including everything it cost.

    Costs are decomposed rather than netted so a robustness report can say
    *why* a strategy died: a strategy killed by spread widening and one killed
    by market impact look identical in the P&L alone.

    ``reference_price``
        The mid at execution time -- the price a frictionless backtest would
        have used.
    ``spread_cost``
        Half-spread paid, in currency.
    ``slippage_cost``
        Market impact beyond the quote, in currency.
    ``unfilled``
        Quantity rejected because of depth, leverage or cash limits.
    """

    index: int
    side: Side
    requested_quantity: float
    filled_quantity: float
    price: float
    reference_price: float
    commission: float = 0.0
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def signed_quantity(self) -> float:
        return self.side.sign * self.filled_quantity

    @property
    def notional(self) -> float:
        return self.filled_quantity * self.price

    @property
    def unfilled(self) -> float:
        return self.requested_quantity - self.filled_quantity

    @property
    def total_cost(self) -> float:
        """All frictions attributable to this fill."""
        return self.commission + self.spread_cost + self.slippage_cost

    @property
    def is_partial(self) -> bool:
        return self.unfilled > 1e-9 and self.filled_quantity > 0.0
