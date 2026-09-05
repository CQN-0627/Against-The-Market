"""The multi-asset analogue of :mod:`marketerror.data.schema`.

``UniverseData`` holds ``n_assets`` aligned :class:`MarketData` paths and
presents them as ``(time, asset)`` matrices.  ``UniverseView`` is the causal
window a multi-asset strategy is handed on each bar: it can read bar ``t`` and
earlier for *every* symbol, and has no route to ``t + 1`` for any of them.

The single-asset guarantee is preserved verbatim -- look-ahead is prevented by
construction rather than by convention -- and one property is added:
``UniverseView`` exposes cross-sectional helpers (``rank``, ``top``, ``bottom``)
because a universe strategy's decisions are almost always *relative*, and
recomputing a ranking by hand in every strategy invites off-by-one bugs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Protocol, Sequence

import numpy as np

from .schema import MarketData

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

__all__ = ["UniverseData", "UniverseView", "UniversePortfolioState"]

#: Per-asset columns exposed as ``(time, asset)`` matrices.
_MATRIX_FIELDS = (
    "price",
    "returns",
    "volume",
    "bid",
    "ask",
    "spread",
    "bid_size",
    "ask_size",
)


class UniversePortfolioState(Protocol):
    """The read-only portfolio slice a multi-asset strategy may consult."""

    @property
    def cash(self) -> float: ...

    def position(self, symbol: str) -> float: ...

    def equity(self, prices: Mapping[str, float]) -> float: ...


@dataclass(frozen=True, eq=False)
class UniverseData:
    """Several aligned market paths, addressed by symbol.

    Every asset shares the ``timestamp`` axis and ``periods_per_year``, so bar
    ``t`` means the same instant for all of them.  Per-asset arrays are stored
    as ``(n_periods, n_assets)`` matrices because every consumer either wants a
    whole bar across assets (the engine) or a whole history for one asset (the
    feature builder), and a matrix serves both without copying.
    """

    symbols: tuple[str, ...]
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
        set_(self, "symbols", tuple(str(s) for s in self.symbols))
        set_(self, "timestamp", np.asarray(self.timestamp))
        for name in _MATRIX_FIELDS:
            set_(
                self,
                name,
                np.ascontiguousarray(getattr(self, name), dtype=np.float64),
            )
        set_(self, "metadata", dict(self.metadata))
        set_(self, "_index", {s: i for i, s in enumerate(self.symbols)})
        self.validate()

    # ------------------------------------------------------------------ checks
    def validate(self) -> None:
        """Assert the invariants the engine relies on, for every asset."""
        if not self.symbols:
            raise ValueError("a universe needs at least one asset")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("symbols must be unique")

        n_periods = len(self.timestamp)
        if n_periods < 2:
            raise ValueError("UniverseData needs at least 2 observations")
        expected = (n_periods, len(self.symbols))
        for name in _MATRIX_FIELDS:
            shape = getattr(self, name).shape
            if shape != expected:
                raise ValueError(
                    f"field {name!r} has shape {shape}, expected {expected}"
                )
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
        return len(self.timestamp)

    @property
    def n_assets(self) -> int:
        return len(self.symbols)

    def index_of(self, symbol: str) -> int:
        try:
            return self._index[symbol]  # type: ignore[attr-defined]
        except KeyError:
            raise KeyError(
                f"unknown symbol {symbol!r}; universe has {list(self.symbols)}"
            ) from None

    # ----------------------------------------------------------------- reshape
    @classmethod
    def from_paths(
        cls,
        paths: Mapping[str, MarketData] | Sequence[tuple[str, MarketData]],
        metadata: Mapping[str, Any] | None = None,
    ) -> "UniverseData":
        """Stack per-symbol :class:`MarketData` paths into one universe.

        This is the bridge that lets existing single-asset machinery -- the
        synthetic generator, the historical CSV loader -- feed the multi-asset
        engine without either of them knowing it exists.
        """
        items = list(paths.items() if isinstance(paths, Mapping) else paths)
        if not items:
            raise ValueError("need at least one path")

        symbols = [str(symbol) for symbol, _ in items]
        first = items[0][1]
        n_periods = len(first)
        for symbol, data in items:
            if len(data) != n_periods:
                raise ValueError(
                    f"path {symbol!r} has {len(data)} bars, expected {n_periods}"
                )
            if data.periods_per_year != first.periods_per_year:
                raise ValueError(
                    f"path {symbol!r} has periods_per_year "
                    f"{data.periods_per_year}, expected {first.periods_per_year}"
                )

        columns = {
            name: np.column_stack([getattr(d, name) for _, d in items])
            for name in _MATRIX_FIELDS
        }
        merged = dict(metadata or {})
        merged.setdefault(
            "assets", {symbol: dict(data.metadata) for symbol, data in items}
        )
        return cls(
            symbols=tuple(symbols),
            timestamp=first.timestamp,
            periods_per_year=first.periods_per_year,
            metadata=merged,
            **columns,
        )

    def asset(self, symbol: str) -> MarketData:
        """Extract one asset as a standalone :class:`MarketData`.

        Lets any single-asset tool -- the existing backtester, the metrics, the
        plots -- operate on one member of a universe unchanged.
        """
        j = self.index_of(symbol)
        return MarketData(
            timestamp=self.timestamp,
            price=self.price[:, j],
            returns=self.returns[:, j],
            volume=self.volume[:, j],
            bid=self.bid[:, j],
            ask=self.ask[:, j],
            spread=self.spread[:, j],
            bid_size=self.bid_size[:, j],
            ask_size=self.ask_size[:, j],
            periods_per_year=self.periods_per_year,
            metadata={
                **dict(self.metadata.get("assets", {}).get(symbol, {})),
                "symbol": symbol,
            },
        )

    def slice(self, start: int = 0, stop: int | None = None) -> "UniverseData":
        stop = len(self) if stop is None else stop
        sl = slice(start, stop)
        return UniverseData(
            symbols=self.symbols,
            timestamp=self.timestamp[sl],
            periods_per_year=self.periods_per_year,
            metadata=dict(self.metadata),
            **{name: getattr(self, name)[sl] for name in _MATRIX_FIELDS},
        )

    def realized_correlation(self) -> np.ndarray:
        """Sample correlation matrix of returns, skipping the leading zero bar."""
        return np.corrcoef(self.returns[1:], rowvar=False)

    def to_frame(self) -> "pd.DataFrame":
        """Long-format frame: one row per (timestamp, symbol)."""
        import pandas as pd

        n_periods = len(self)
        frames = []
        for j, symbol in enumerate(self.symbols):
            frames.append(
                pd.DataFrame(
                    {
                        "timestamp": self.timestamp,
                        "symbol": [symbol] * n_periods,
                        "price": self.price[:, j],
                        "return": self.returns[:, j],
                        "volume": self.volume[:, j],
                        "bid": self.bid[:, j],
                        "ask": self.ask[:, j],
                        "spread": self.spread[:, j],
                        "bid_size": self.bid_size[:, j],
                        "ask_size": self.ask_size[:, j],
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)


class UniverseView:
    """A strategy's causal window onto one bar of a :class:`UniverseData`.

    Scalar access is by symbol::

        view.price("AAPL")
        view["return_20", "AAPL"]

    and cross-sectional access returns one value per symbol, in universe order::

        view.cross_section("return_20")   # -> np.ndarray
        view.rank("return_20")            # 0 = smallest
        view.top("return_20", 3)          # -> ("NVDA", "AAPL", "MSFT")
    """

    __slots__ = ("_data", "_features", "_t", "_state")

    def __init__(
        self,
        data: UniverseData,
        features: Mapping[str, np.ndarray] | None = None,
        index: int = 0,
        state: UniversePortfolioState | None = None,
    ) -> None:
        self._data = data
        self._features = dict(features or {})
        self._t = int(index)
        self._state = state

    # ------------------------------------------------------------- positioning
    def _advance(self, index: int) -> None:
        """Move the window forward.  Only the engine calls this."""
        if index < self._t:
            raise ValueError("UniverseView only moves forward")
        self._t = int(index)

    @property
    def t(self) -> int:
        return self._t

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._data.symbols

    @property
    def n_assets(self) -> int:
        return self._data.n_assets

    @property
    def periods_per_year(self) -> int:
        return self._data.periods_per_year

    @property
    def timestamp(self) -> Any:
        return self._data.timestamp[self._t]

    @property
    def n_seen(self) -> int:
        return self._t + 1

    # ----------------------------------------------------- scalars at bar ``t``
    def _at(self, name: str, symbol: str) -> float:
        return float(getattr(self._data, name)[self._t, self._data.index_of(symbol)])

    def price(self, symbol: str) -> float:
        return self._at("price", symbol)

    mid = price

    def bid(self, symbol: str) -> float:
        return self._at("bid", symbol)

    def ask(self, symbol: str) -> float:
        return self._at("ask", symbol)

    def spread(self, symbol: str) -> float:
        return self._at("spread", symbol)

    def volume(self, symbol: str) -> float:
        return self._at("volume", symbol)

    def bid_size(self, symbol: str) -> float:
        return self._at("bid_size", symbol)

    def ask_size(self, symbol: str) -> float:
        return self._at("ask_size", symbol)

    def ret(self, symbol: str) -> float:
        """The log return realised *into* the current bar."""
        return self._at("returns", symbol)

    def prices(self) -> dict[str, float]:
        """Every symbol's current mid, as the portfolio wants it."""
        row = self._data.price[self._t]
        return {symbol: float(row[j]) for j, symbol in enumerate(self._data.symbols)}

    # -------------------------------------------------------- portfolio state
    @property
    def cash(self) -> float:
        return 0.0 if self._state is None else float(self._state.cash)

    @property
    def equity(self) -> float:
        if self._state is None:
            return 0.0
        return float(self._state.equity(self.prices()))

    def position(self, symbol: str) -> float:
        self._data.index_of(symbol)  # reject unknown symbols consistently
        return 0.0 if self._state is None else float(self._state.position(symbol))

    def weight(self, symbol: str) -> float:
        """Current position value as a fraction of equity."""
        equity = self.equity
        if equity <= 0.0:
            return 0.0
        return self.position(symbol) * self.price(symbol) / equity

    # -------------------------------------------------------- feature access
    def _column(self, key: str) -> np.ndarray:
        if key in self._features:
            return self._features[key]
        if key == "return":
            return self._data.returns
        if key in _MATRIX_FIELDS:
            return getattr(self._data, key)
        raise KeyError(
            f"unknown field or feature {key!r}; declare it in Strategy.requires()"
        )

    def __getitem__(self, key: tuple[str, str]) -> float:
        """``view["return_20", "AAPL"]`` -- feature value for one symbol."""
        name, symbol = key
        return float(self._column(name)[self._t, self._data.index_of(symbol)])

    def get(
        self, name: str, symbol: str, default: float = float("nan")
    ) -> float:
        """Like ``view[name, symbol]`` but tolerant of unknown keys and warm-up."""
        try:
            value = self[name, symbol]
        except KeyError:
            return default
        return default if np.isnan(value) else value

    def ready(self, name: str, symbol: str | None = None) -> bool:
        """True once ``name`` has left its warm-up window.

        With no symbol, requires *every* asset to be ready -- the condition a
        cross-sectional decision actually needs.
        """
        row = self._column(name)[self._t]
        if symbol is None:
            return not bool(np.isnan(row).any())
        return not bool(np.isnan(row[self._data.index_of(symbol)]))

    def history(
        self, name: str, symbol: str, periods: int | None = None
    ) -> np.ndarray:
        """Trailing values of ``name`` for one symbol, ending at the current bar."""
        column = self._column(name)[:, self._data.index_of(symbol)]
        stop = self._t + 1
        if periods is None:
            start = 0
        else:
            if periods < 1:
                raise ValueError("periods must be >= 1")
            start = max(0, stop - periods)
        return column[start:stop].copy()

    # ------------------------------------------------------- cross-section
    def cross_section(self, name: str) -> np.ndarray:
        """One value per symbol at the current bar, in universe order."""
        return np.asarray(self._column(name)[self._t], dtype=np.float64).copy()

    def series(self, name: str) -> dict[str, float]:
        """The cross-section as a ``{symbol: value}`` mapping."""
        row = self._column(name)[self._t]
        return {symbol: float(row[j]) for j, symbol in enumerate(self._data.symbols)}

    def rank(self, name: str, ascending: bool = True) -> dict[str, float]:
        """Cross-sectional rank per symbol, ``0`` for the smallest value.

        Symbols still warming up (``NaN``) are excluded rather than sorted to
        one end, where they would masquerade as extreme signals.
        """
        row = self._column(name)[self._t]
        valid = [
            (float(row[j]), symbol)
            for j, symbol in enumerate(self._data.symbols)
            if not np.isnan(row[j])
        ]
        valid.sort(key=lambda pair: pair[0], reverse=not ascending)
        return {symbol: float(position) for position, (_, symbol) in enumerate(valid)}

    def top(self, name: str, count: int) -> tuple[str, ...]:
        """The ``count`` symbols with the highest value of ``name``."""
        if count < 0:
            raise ValueError("count must be >= 0")
        ordered = self.rank(name, ascending=False)
        return tuple(list(ordered)[:count])

    def bottom(self, name: str, count: int) -> tuple[str, ...]:
        """The ``count`` symbols with the lowest value of ``name``."""
        if count < 0:
            raise ValueError("count must be >= 0")
        ordered = self.rank(name, ascending=True)
        return tuple(list(ordered)[:count])

    def __iter__(self) -> Iterator[str]:
        return iter(self._data.symbols)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"UniverseView(t={self._t}, assets={self._data.n_assets})"
