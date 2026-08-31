"""The market-data interface every other component agrees on.

Two objects live here:

``MarketData``
    An immutable, column-oriented container for one market path.  Both the
    synthetic generator and the historical CSV loader produce this type, which
    is what keeps the backtester independent of where its data came from
    (see ``docs/methodology.md``).

``MarketView``
    The *causal* window onto a ``MarketData`` that a strategy is handed on each
    bar.  It exposes bar ``t`` and everything before it, and structurally has
    no way to reach bar ``t + 1``.  Look-ahead bias is prevented by
    construction rather than by convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Protocol, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

__all__ = [
    "FRAME_COLUMNS",
    "LookAheadError",
    "MarketData",
    "MarketView",
    "PortfolioState",
    "make_timestamps",
]

#: Column names of the tabular representation, in the order the specification
#: lists them.  ``return`` is a Python keyword, so the dataclass field is named
#: ``returns`` while the DataFrame column keeps the documented name.
FRAME_COLUMNS = (
    "timestamp",
    "price",
    "return",
    "volume",
    "bid",
    "ask",
    "spread",
    "bid_size",
    "ask_size",
)

_NUMERIC_FIELDS = (
    "price",
    "returns",
    "volume",
    "bid",
    "ask",
    "spread",
    "bid_size",
    "ask_size",
)


class LookAheadError(RuntimeError):
    """Raised when a strategy tries to read data it cannot legally know yet."""


class PortfolioState(Protocol):
    """The read-only slice of portfolio state a strategy may consult."""

    @property
    def position(self) -> float: ...

    @property
    def cash(self) -> float: ...

    def equity(self, price: float) -> float: ...


def make_timestamps(
    periods: int,
    start: str = "2020-01-01",
    periods_per_year: int = 252,
) -> np.ndarray:
    """Build a ``datetime64[ns]`` index of ``periods`` observations.

    252 periods per year is treated as business-daily, 365 as calendar-daily,
    and anything else as an evenly spaced intraday-style grid.  The timestamps
    are cosmetic -- every annualisation in the codebase uses
    ``periods_per_year`` rather than wall-clock differences -- but they make
    exported results readable.
    """
    if periods < 1:
        raise ValueError("periods must be >= 1")
    import pandas as pd

    if periods_per_year == 252:
        return pd.bdate_range(start=start, periods=periods).to_numpy()
    if periods_per_year == 365:
        return pd.date_range(start=start, periods=periods, freq="D").to_numpy()
    step = np.timedelta64(int(round(365.25 * 24 * 3600 / periods_per_year)), "s")
    return np.datetime64(start, "ns") + np.arange(periods) * step


@dataclass(frozen=True, eq=False)
class MarketData:
    """One market path: prices, returns and top-of-book liquidity.

    Every array has the same length ``n``.  ``returns[0]`` is zero by
    convention because ``price[0]`` is the starting price and no move has
    happened yet; statistical estimators therefore skip the first observation
    (see ``analysis.statistics.realized_statistics``).
    """

    timestamp: np.ndarray
    price: np.ndarray
    returns: np.ndarray
    volume: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    spread: np.ndarray
    bid_size: np.ndarray
    ask_size: np.ndarray
    periods_per_year: int = 252
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        set_(self, "timestamp", np.asarray(self.timestamp))
        for name in _NUMERIC_FIELDS:
            set_(self, name, np.ascontiguousarray(getattr(self, name), dtype=np.float64))
        set_(self, "metadata", dict(self.metadata))
        self.validate()

    # ------------------------------------------------------------------ checks
    def validate(self) -> None:
        """Assert the invariants downstream components rely on.

        These are the economic validity constraints of specification §9: a
        perturbation that violates any of them is not a market, it is a bug.
        """
        n = len(self.price)
        if n < 2:
            raise ValueError("MarketData needs at least 2 observations")
        for name in ("timestamp",) + _NUMERIC_FIELDS:
            got = len(getattr(self, name))
            if got != n:
                raise ValueError(f"field {name!r} has length {got}, expected {n}")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
        for name in ("price", "bid", "ask", "volume", "bid_size", "ask_size"):
            arr = getattr(self, name)
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"field {name!r} contains non-finite values")
            if np.any(arr <= 0.0):
                raise ValueError(f"field {name!r} must be strictly positive")
        if not np.all(np.isfinite(self.returns)):
            raise ValueError("field 'returns' contains non-finite values")
        if np.any(self.spread < 0.0):
            raise ValueError("spread must be non-negative")
        if np.any(self.ask < self.bid):
            raise ValueError("crossed book: ask < bid")

    # -------------------------------------------------------------- properties
    def __len__(self) -> int:
        return len(self.price)

    @property
    def mid(self) -> np.ndarray:
        """Mid price.  ``price`` *is* the mid; bid/ask are quoted around it."""
        return self.price

    @property
    def spread_bps(self) -> np.ndarray:
        """Quoted spread in basis points of the mid price."""
        return self.spread / self.price * 1e4

    @property
    def half_spread(self) -> np.ndarray:
        return 0.5 * self.spread

    # ----------------------------------------------------------------- reshape
    def slice(self, start: int = 0, stop: int | None = None) -> "MarketData":
        """Return a contiguous sub-path.  Metadata is carried over unchanged."""
        stop = len(self) if stop is None else stop
        sl = slice(start, stop)
        return MarketData(
            timestamp=self.timestamp[sl],
            price=self.price[sl],
            returns=self.returns[sl],
            volume=self.volume[sl],
            bid=self.bid[sl],
            ask=self.ask[sl],
            spread=self.spread[sl],
            bid_size=self.bid_size[sl],
            ask_size=self.ask_size[sl],
            periods_per_year=self.periods_per_year,
            metadata=dict(self.metadata),
        )

    def with_metadata(self, **extra: Any) -> "MarketData":
        merged = dict(self.metadata)
        merged.update(extra)
        return MarketData(
            timestamp=self.timestamp,
            price=self.price,
            returns=self.returns,
            volume=self.volume,
            bid=self.bid,
            ask=self.ask,
            spread=self.spread,
            bid_size=self.bid_size,
            ask_size=self.ask_size,
            periods_per_year=self.periods_per_year,
            metadata=merged,
        )

    def to_frame(self) -> "pd.DataFrame":
        import pandas as pd

        return pd.DataFrame(
            {
                "timestamp": self.timestamp,
                "price": self.price,
                "return": self.returns,
                "volume": self.volume,
                "bid": self.bid,
                "ask": self.ask,
                "spread": self.spread,
                "bid_size": self.bid_size,
                "ask_size": self.ask_size,
            },
            columns=list(FRAME_COLUMNS),
        )

    @classmethod
    def from_frame(
        cls,
        frame: "pd.DataFrame",
        periods_per_year: int = 252,
        metadata: Mapping[str, Any] | None = None,
    ) -> "MarketData":
        missing = [c for c in FRAME_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"frame is missing columns: {missing}")
        return cls(
            timestamp=frame["timestamp"].to_numpy(),
            price=frame["price"].to_numpy(),
            returns=frame["return"].to_numpy(),
            volume=frame["volume"].to_numpy(),
            bid=frame["bid"].to_numpy(),
            ask=frame["ask"].to_numpy(),
            spread=frame["spread"].to_numpy(),
            bid_size=frame["bid_size"].to_numpy(),
            ask_size=frame["ask_size"].to_numpy(),
            periods_per_year=periods_per_year,
            metadata=dict(metadata or {}),
        )


class MarketView:
    """A strategy's causal window onto one bar of a ``MarketData``.

    The view holds an index ``t`` and only ever indexes at or before it.  It
    intentionally does *not* expose the underlying arrays, so a strategy has no
    route to future observations even by accident::

        def on_data(self, data):
            if data["return_20"] > 0:        # scalar feature at bar t
                return Order("BUY", 100)
            recent = data.history("price", 5)   # last 5 closes, inclusive of t
    """

    __slots__ = ("_data", "_features", "_t", "_state")

    def __init__(
        self,
        data: MarketData,
        features: Mapping[str, np.ndarray] | None = None,
        index: int = 0,
        state: PortfolioState | None = None,
    ) -> None:
        self._data = data
        self._features = dict(features or {})
        self._t = int(index)
        self._state = state

    # ------------------------------------------------------------- positioning
    def _advance(self, index: int) -> None:
        """Move the window forward.  Only the engine calls this."""
        if index < self._t:
            raise ValueError("MarketView only moves forward")
        self._t = int(index)

    @property
    def t(self) -> int:
        """Index of the current bar."""
        return self._t

    @property
    def periods_per_year(self) -> int:
        return self._data.periods_per_year

    @property
    def n_seen(self) -> int:
        """How many bars the strategy has observed, including this one."""
        return self._t + 1

    # ----------------------------------------------------- scalars at bar ``t``
    @property
    def timestamp(self) -> Any:
        return self._data.timestamp[self._t]

    @property
    def price(self) -> float:
        return float(self._data.price[self._t])

    #: ``mid`` and ``price`` are the same quantity; both spellings are common.
    mid = price

    @property
    def bid(self) -> float:
        return float(self._data.bid[self._t])

    @property
    def ask(self) -> float:
        return float(self._data.ask[self._t])

    @property
    def spread(self) -> float:
        return float(self._data.spread[self._t])

    @property
    def spread_bps(self) -> float:
        return float(self._data.spread[self._t] / self._data.price[self._t] * 1e4)

    @property
    def volume(self) -> float:
        return float(self._data.volume[self._t])

    @property
    def bid_size(self) -> float:
        return float(self._data.bid_size[self._t])

    @property
    def ask_size(self) -> float:
        return float(self._data.ask_size[self._t])

    @property
    def ret(self) -> float:
        """The log return realised *into* the current bar."""
        return float(self._data.returns[self._t])

    # -------------------------------------------------------- portfolio state
    @property
    def position(self) -> float:
        return 0.0 if self._state is None else float(self._state.position)

    @property
    def cash(self) -> float:
        return 0.0 if self._state is None else float(self._state.cash)

    @property
    def equity(self) -> float:
        return 0.0 if self._state is None else float(self._state.equity(self.price))

    # -------------------------------------------------------- mapping access
    def _column(self, key: str) -> np.ndarray:
        if key in self._features:
            return self._features[key]
        if key == "return":
            return self._data.returns
        if key in _NUMERIC_FIELDS:
            return getattr(self._data, key)
        raise KeyError(
            f"unknown field or feature {key!r}; declare it in Strategy.requires()"
        )

    def __getitem__(self, key: str) -> float:
        """Value of a field or declared feature at the current bar."""
        value = self._column(key)[self._t]
        return float(value)

    def __contains__(self, key: str) -> bool:
        try:
            self._column(key)
        except KeyError:
            return False
        return True

    def __iter__(self) -> Iterator[str]:
        return iter((*_NUMERIC_FIELDS, "return", *self._features))

    def get(self, key: str, default: float = float("nan")) -> float:
        """Like ``view[key]`` but tolerant of unknown keys and warm-up NaNs."""
        try:
            value = self[key]
        except KeyError:
            return default
        return default if np.isnan(value) else value

    def ready(self, *keys: str) -> bool:
        """True when every named feature has left its warm-up window."""
        return all(not np.isnan(self[key]) for key in keys)

    def history(self, key: str, periods: int | None = None) -> np.ndarray:
        """The last ``periods`` values of ``key``, ending at the current bar.

        Never returns data from beyond bar ``t``; asking for more history than
        exists yields a shorter array rather than an error.
        """
        column = self._column(key)
        stop = self._t + 1
        if periods is None:
            start = 0
        else:
            if periods < 1:
                raise ValueError("periods must be >= 1")
            start = max(0, stop - periods)
        return column[start:stop].copy()

    def features(self) -> Sequence[str]:
        return tuple(self._features)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"MarketView(t={self._t}, price={self.price:.4f})"
