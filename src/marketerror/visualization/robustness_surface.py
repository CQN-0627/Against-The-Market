"""The robustness surface: mean return as a function of two shocks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..optimization.failure_boundary import slice_2d
from ..optimization.objective import ScenarioEvaluation
from ..perturbations.base import PerturbationSpace
from .plots import COLORS, save_figure

__all__ = ["robustness_surface", "axis_sensitivity_plot"]


def robustness_surface(
    evaluations: Iterable[ScenarioEvaluation],
    space: PerturbationSpace,
    dim_x: str = "volatility",
    dim_y: str = "spread",
    failure_threshold: float = 0.0,
    title: str | None = None,
    path: str | Path | None = None,
) -> Any:
    r"""Contour :math:`Return = f(z_x, z_y)` with the break-even line marked.

    Filled contours of mean return, plus a heavy line at the failure threshold:
    that line *is* the failure boundary in this projection, and where it sits
    relative to the severity rings is the whole result.

    Requires an exhaustive search to be meaningful; an early-stopped one has only
    visited a low-severity sliver of the plane.
    """
    import matplotlib.pyplot as plt

    evaluations = list(evaluations)
    figure, axis = plt.subplots(figsize=(6.6, 5.2))
    xs, ys, returns, _failures = slice_2d(
        evaluations, space, dim_x, dim_y, aggregate="mean"
    )
    if len(xs) < 2 or len(ys) < 2 or np.all(np.isnan(returns)):
        axis.text(
            0.5,
            0.5,
            "need a 2-D grid to draw a surface\n(run with --exhaustive)",
            ha="center",
            va="center",
        )
        return save_figure(figure, path)

    # Contouring cannot span gaps; fill unvisited cells by nearest-neighbour so
    # the surface is drawn only where data exists in each row.
    filled = _fill_nans(returns)
    grid_x, grid_y = np.meshgrid(xs, ys)

    levels = 14
    surface = axis.contourf(grid_x, grid_y, filled, levels=levels, cmap="RdYlGn")
    figure.colorbar(surface, ax=axis, label="mean return", format=lambda v, _: f"{v:.0%}")
    breakeven = axis.contour(
        grid_x,
        grid_y,
        filled,
        levels=[failure_threshold],
        colors="black",
        linewidths=1.8,
    )
    axis.clabel(breakeven, fmt={failure_threshold: "break-even"}, fontsize=8)

    axis.plot(0, 0, marker="o", color=COLORS["baseline"], ms=7, label="baseline")
    axis.set_xlabel(f"{space[dim_x].label} shock ($\\sigma$)")
    axis.set_ylabel(f"{space[dim_y].label} shock ($\\sigma$)")
    axis.set_title(title or f"Robustness surface: return vs {dim_x} and {dim_y}")
    axis.legend(loc="upper left", fontsize=8)
    return save_figure(figure, path)


def axis_sensitivity_plot(
    results: Iterable[Any],
    max_abs_z: float = 4.0,
    title: str = "Single-axis failure thresholds",
    path: str | Path | None = None,
) -> Any:
    """Horizontal bars of each dimension's one-at-a-time failure severity.

    ``results`` are :class:`~marketerror.optimization.directional_search.BisectionResult`
    objects.  Directions that never fail inside the constraint box are drawn as
    open bars at the box edge and labelled, so "no failure found" is visually
    distinct from "fails at exactly 4 sigma".
    """
    import matplotlib.pyplot as plt

    results = list(results)
    figure, axis = plt.subplots(figsize=(6.8, max(2.6, 0.32 * len(results) + 1.4)))
    if not results:
        axis.text(0.5, 0.5, "no axis scan performed", ha="center", va="center")
        return save_figure(figure, path)

    order = sorted(
        results, key=lambda r: (not r.found, r.severity if r.found else max_abs_z)
    )
    labels = [r.label for r in order]
    positions = np.arange(len(order))
    for y, result in zip(positions, order):
        if result.found:
            axis.barh(y, result.severity, color=COLORS["failure"], alpha=0.8, height=0.62)
            axis.text(
                result.severity + 0.06,
                y,
                f"{result.severity:.2f}$\\sigma$",
                va="center",
                fontsize=8,
            )
        else:
            axis.barh(
                y,
                result.max_radius,
                facecolor="none",
                edgecolor=COLORS["survive"],
                hatch="///",
                height=0.62,
            )
            axis.text(
                result.max_radius + 0.06, y, "no failure", va="center", fontsize=8
            )

    axis.set_yticks(positions, labels)
    axis.set_xlabel("failure severity along this axis alone ($\\sigma$)")
    axis.set_xlim(0, max_abs_z * 1.22)
    axis.set_title(title)
    axis.invert_yaxis()
    axis.grid(axis="y", visible=False)
    return save_figure(figure, path)


def _fill_nans(grid: np.ndarray) -> np.ndarray:
    """Nearest-neighbour fill along each row, then each column."""
    filled = np.array(grid, dtype=np.float64, copy=True)
    for _ in range(2):
        for row in range(filled.shape[0]):
            filled[row] = _fill_1d(filled[row])
        filled = filled.T
    if np.any(np.isnan(filled)):
        filled = np.nan_to_num(filled, nan=float(np.nanmin(grid)))
    return filled


def _fill_1d(values: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(values)
    if not valid.any():
        return values
    indices = np.arange(len(values))
    return np.interp(indices, indices[valid], values[valid])
