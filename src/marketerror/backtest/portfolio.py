"""Cash, position and cost accounting.

The portfolio is the only mutable state in a backtest.  It is deliberately
dumb: it applies fills, values itself at the mid, and keeps a decomposition of
what frictions have been paid.  All decisions about *whether* a fill is allowed
live in :mod:`marketerror.backtest.execution`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .orders import Fill, Side

__all__ = ["Portfolio"]


@dataclass
class Portfolio:
    """Cash and a single-instrument position, marked to the mid price.

    Cost bookkeeping note: ``fill.price`` already contains the half-spread and
    the market impact, so cash is only debited for the notional and the
    commission.  ``spread_paid`` and ``slippage_paid`` are *attributions* -- the
    difference between what was paid and what a frictionless mid-price
    execution would have cost -- and must not be subtracted from cash a second
    time.  They exist so a robustness report can say which friction killed a
    strategy, which the net P&L alone cannot.
    """

    initial_cash: float = 100_000.0
    cash: float = field(init=False)
    position: float = field(init=False, default=0.0)

    commission_paid: float = field(init=False, default=0.0)
    spread_paid: float = field(init=False, default=0.0)
    slippage_paid: float = field(init=False, default=0.0)
    traded_notional: float = field(init=False, default=0.0)
    n_trades: int = field(init=False, default=0)
    n_partial_fills: int = field(init=False, default=0)
    n_rejected: int = field(init=False, default=0)
    ruined: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if self.initial_cash <= 0.0:
            raise ValueError("initial_cash must be > 0")
        self.reset()

    def reset(self) -> None:
        self.cash = float(self.initial_cash)
        self.position = 0.0
        self.commission_paid = 0.0
        self.spread_paid = 0.0
        self.slippage_paid = 0.0
        self.traded_notional = 0.0
        self.n_trades = 0
        self.n_partial_fills = 0
        self.n_rejected = 0
        self.ruined = False

    # ------------------------------------------------------------------ valuation
    def equity(self, price: float) -> float:
        """Mark-to-market value of the portfolio at ``price``."""
        return self.cash + self.position * price

    def exposure(self, price: float) -> float:
        """Gross position value as a fraction of equity (leverage actually used)."""
        equity = self.equity(price)
        if equity <= 0.0:
            return 0.0
        return abs(self.position * price) / equity

    # --------------------------------------------------------------------- fills
    def apply(self, fill: Fill) -> None:
        """Settle a fill into cash, position and the cost accumulators."""
        if fill.filled_quantity <= 0.0:
            if fill.requested_quantity > 0.0:
                self.n_rejected += 1
            return

        self.cash -= fill.signed_quantity * fill.price
        self.cash -= fill.commission
        self.position += fill.signed_quantity

        self.commission_paid += fill.commission
        self.spread_paid += fill.spread_cost
        self.slippage_paid += fill.slippage_cost
        self.traded_notional += abs(fill.notional)
        self.n_trades += 1
        if fill.is_partial:
            self.n_partial_fills += 1

    def liquidate(self, price: float) -> None:
        """Flatten the position at ``price`` and mark the account as ruined.

        Called when equity reaches zero.  A short position has unbounded loss,
        so without this a wiped-out path would produce negative equity and
        poison every downstream return calculation with NaNs.  Flattening at the
        mid is the optimistic version of a margin call, which is the
        conservative choice here: it cannot manufacture a failure.
        """
        self.cash = self.equity(price)
        self.position = 0.0
        self.ruined = True

    @property
    def total_costs(self) -> float:
        """All frictions paid, relative to frictionless mid-price execution."""
        return self.commission_paid + self.spread_paid + self.slippage_paid
