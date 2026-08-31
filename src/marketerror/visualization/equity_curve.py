"""Baseline vs stressed equity curves."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..backtest.engine import BacktestResult
from .plots import COLORS, save_figure

__all__ = ["equity_curve", "equity_fan"]


def equity_curve(
    baseline: BacktestResult,
    stressed: BacktestResult | None = None,
    labels: tuple[str, str] = ("baseline", "stressed"),
    loss_periods: int = 0,
    failure_threshold: float = 0.0,
    title: str = "Equity curve: baseline vs stressed market",
    path: str | Path | None = None,
) -> Any:
    """Plot one baseline path against the same seed under a stressed market.

    Because the two runs use the same seed and therefore the same random draws,
    the divergence between the curves is attributable to the perturbation rather
    than to different luck -- the visual counterpart of the common-random-numbers
    design in the Monte Carlo layer.

    When ``loss_periods`` is set, the longest contiguous stretch below the failure
    threshold is shaded on the stressed curve, so the ``--losstime`` criterion can
    be seen rather than merely asserted.
    """
    import matplotlib.pyplot as plt

    figure, (axis, lower) = plt.subplots(
        2, 1, figsize=(8.0, 5.4), sharex=True, height_ratios=(2.4, 1.0)
    )

    x = np.arange(len(baseline.equity))
    axis.plot(
        x,
        baseline.cumulative_return,
        color=COLORS["baseline"],
        lw=1.5,
        label=f"{labels[0]} ({baseline.metrics.total_return:+.1%})",
    )
    if stressed is not None:
        axis.plot(
            np.arange(len(stressed.equity)),
            stressed.cumulative_return,
            color=COLORS["stressed"],
            lw=1.5,
            label=f"{labels[1]} ({stressed.metrics.total_return:+.1%})",
        )
        if loss_periods > 0:
            span = _longest_span_below(stressed.cumulative_return, failure_threshold)
            if span is not None and (span[1] - span[0]) > 0:
                axis.axvspan(
                    span[0],
                    span[1],
                    color=COLORS["failure"],
                    alpha=0.12,
                    label=(
                        f"longest loss run: {span[1] - span[0]} bars "
                        f"(needs {loss_periods})"
                    ),
                )

    axis.axhline(failure_threshold, color="black", lw=0.9, ls="--")
    axis.set_ylabel("cumulative return")
    axis.set_title(title)
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axis.legend(loc="best")

    lower.plot(x, baseline.exposure, color=COLORS["baseline"], lw=1.0)
    if stressed is not None:
        lower.plot(
            np.arange(len(stressed.exposure)),
            stressed.exposure,
            color=COLORS["stressed"],
            lw=1.0,
        )
    lower.set_ylabel("gross exposure")
    lower.set_xlabel("period")
    return save_figure(figure, path)


def equity_fan(
    results: Sequence[BacktestResult],
    stressed: Sequence[BacktestResult] | None = None,
    failure_threshold: float = 0.0,
    title: str = "Cumulative return across Monte Carlo paths",
    path: str | Path | None = None,
) -> Any:
    """Median and 5th-95th percentile band of cumulative return across paths.

    A single path can mislead in either direction; the band shows the range the
    failure decision is actually being made on.
    """
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.0, 4.4))
    for group, colour, label in (
        (results, COLORS["baseline"], "baseline"),
        (stressed, COLORS["stressed"], "stressed"),
    ):
        if not group:
            continue
        curves = np.vstack([r.cumulative_return for r in group])
        x = np.arange(curves.shape[1])
        median = np.median(curves, axis=0)
        low, high = np.percentile(curves, [5, 95], axis=0)
        axis.fill_between(x, low, high, color=colour, alpha=0.18)
        axis.plot(x, median, color=colour, lw=1.6, label=f"{label} median ({len(group)} paths)")

    axis.axhline(failure_threshold, color="black", lw=0.9, ls="--")
    axis.set_xlabel("period")
    axis.set_ylabel("cumulative return")
    axis.set_title(title)
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axis.legend(loc="best")
    return save_figure(figure, path)


def _longest_span_below(
    values: np.ndarray, threshold: float
) -> tuple[int, int] | None:
    """``(start, end)`` of the longest contiguous run below ``threshold``."""
    below = np.asarray(values) < threshold
    best = current = None
    best_length = 0
    for index, flag in enumerate(below):
        if flag:
            current = index if current is None else current
            if index - current + 1 > best_length:
                best_length = index - current + 1
                best = (current, index + 1)
        else:
            current = None
    return best
