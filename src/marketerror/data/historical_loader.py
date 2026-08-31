"""Load real market data into the same :class:`MarketData` container.

The synthetic generator is the primary environment for version 1, but the
backtester never sees it: it consumes ``MarketData``.  That indirection is the
whole point of this module -- it demonstrates that the second arrow in

.. code-block:: text

    SyntheticMarketGenerator ──> MarketData ──> Backtester
    HistoricalDataLoader     ──┘

already works, so historical validation is a data-loading problem rather than an
architectural one.

What historical data cannot do is answer counterfactuals: you cannot ask a CSV
what would have happened at +2 sigma of volatility.  Perturbation search
therefore stays on the synthetic side, while historical paths are useful for
checking that a strategy's *baseline* behaviour is not an artefact of the
generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .schema import MarketData

__all__ = ["HistoricalDataLoader", "load_csv"]

#: Column name synonyms accepted on input, mapped to canonical names.
_ALIASES: Mapping[str, str] = {
    "date": "timestamp",
    "datetime": "timestamp",
    "time": "timestamp",
    "close": "price",
    "adj_close": "price",
    "adj close": "price",
    "mid": "price",
    "last": "price",
    "vol": "volume",
    "qty": "volume",
    "bid_price": "bid",
    "ask_price": "ask",
    "bidsize": "bid_size",
    "asksize": "ask_size",
    "bid_qty": "bid_size",
    "ask_qty": "ask_size",
}


@dataclass(frozen=True)
class HistoricalDataLoader:
    """Turn a price file into ``MarketData``, filling in what is missing.

    Only ``price`` (or a recognised synonym such as ``close``) is mandatory.
    Anything absent is reconstructed from the assumptions below, which are
    recorded in ``MarketData.metadata`` so a reader can tell which columns were
    observed and which were imputed:

    ``return``
        Log difference of price; the first observation is zero.
    ``volume``
        Constant ``assumed_volume``.
    ``bid``/``ask``/``spread``
        A symmetric quote of ``assumed_spread_bps`` around the price.
    ``bid_size``/``ask_size``
        ``depth_fraction`` of the period's volume.
    """

    periods_per_year: int = 252
    assumed_spread_bps: float = 5.0
    assumed_volume: float = 1_000_000.0
    depth_fraction: float = 0.02

    def load(self, path: str | Path, **read_kwargs: Any) -> MarketData:
        """Load a CSV or Parquet file."""
        import pandas as pd

        file = Path(path)
        if not file.exists():
            raise FileNotFoundError(file)
        if file.suffix.lower() in {".parquet", ".pq"}:
            frame = pd.read_parquet(file, **read_kwargs)
        else:
            frame = pd.read_csv(file, **read_kwargs)
        return self.from_frame(frame, source=str(file))

    def from_frame(self, frame: Any, source: str = "<frame>") -> MarketData:
        import pandas as pd

        frame = frame.rename(columns=lambda c: str(c).strip().lower())
        frame = frame.rename(columns={k: v for k, v in _ALIASES.items()})
        if "price" not in frame.columns:
            raise ValueError(
                "input needs a 'price' column (or one of: "
                + ", ".join(sorted(k for k, v in _ALIASES.items() if v == "price"))
                + ")"
            )

        price = np.asarray(frame["price"], dtype=np.float64)
        if len(price) < 2:
            raise ValueError("need at least 2 observations")
        if np.any(price <= 0.0) or not np.all(np.isfinite(price)):
            raise ValueError("prices must be finite and strictly positive")

        imputed: list[str] = []

        if "timestamp" in frame.columns:
            timestamp = pd.to_datetime(frame["timestamp"]).to_numpy()
        else:
            imputed.append("timestamp")
            from .schema import make_timestamps

            timestamp = make_timestamps(len(price), periods_per_year=self.periods_per_year)

        if "return" in frame.columns:
            returns = np.asarray(frame["return"], dtype=np.float64)
        else:
            imputed.append("return")
            returns = np.zeros_like(price)
            returns[1:] = np.diff(np.log(price))

        if "volume" in frame.columns:
            volume = np.maximum(np.asarray(frame["volume"], dtype=np.float64), 1.0)
        else:
            imputed.append("volume")
            volume = np.full_like(price, self.assumed_volume)

        if {"bid", "ask"} <= set(frame.columns):
            bid = np.asarray(frame["bid"], dtype=np.float64)
            ask = np.asarray(frame["ask"], dtype=np.float64)
        else:
            imputed += ["bid", "ask"]
            half = price * (self.assumed_spread_bps / 1e4) / 2.0
            bid, ask = price - half, price + half
        spread = ask - bid

        if "bid_size" in frame.columns:
            bid_size = np.maximum(np.asarray(frame["bid_size"], dtype=np.float64), 1.0)
        else:
            imputed.append("bid_size")
            bid_size = np.maximum(volume * self.depth_fraction, 1.0)
        if "ask_size" in frame.columns:
            ask_size = np.maximum(np.asarray(frame["ask_size"], dtype=np.float64), 1.0)
        else:
            imputed.append("ask_size")
            ask_size = np.maximum(volume * self.depth_fraction, 1.0)

        return MarketData(
            timestamp=timestamp,
            price=price,
            returns=returns,
            volume=volume,
            bid=bid,
            ask=ask,
            spread=spread,
            bid_size=bid_size,
            ask_size=ask_size,
            periods_per_year=self.periods_per_year,
            metadata={
                "source": "historical",
                "path": source,
                "imputed_columns": imputed,
                "assumed_spread_bps": self.assumed_spread_bps,
            },
        )


def load_csv(path: str | Path, periods_per_year: int = 252, **kwargs: Any) -> MarketData:
    """Convenience wrapper around :class:`HistoricalDataLoader`."""
    return HistoricalDataLoader(periods_per_year=periods_per_year).load(path, **kwargs)
