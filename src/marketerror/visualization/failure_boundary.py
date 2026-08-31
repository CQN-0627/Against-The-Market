"""The failure-boundary plot: which region of a 2-D slice breaks the strategy."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..optimization.failure_boundary import slice_2d
from ..optimization.objective import ScenarioEvaluation
from ..perturbations.base import PerturbationSpace
from .plots import COLORS, save_figure

__all__ = ["failure_boundary_plot"]


def failure_boundary_plot(
    evaluations: Iterable[ScenarioEvaluation],
    space: PerturbationSpace,
    dim_x: str = "volatility",
    dim_y: str = "spread",
    minimum: ScenarioEvaluation | None = None,
    title: str | None = None,
    path: str | Path | None = None,
) -> Any:
    """Shade the failing region of the ``(dim_x, dim_y)`` plane.

    The other dimensions are collapsed pessimistically -- each cell shows the
    *worst* outcome found at that combination -- so the shaded region answers
    "can the strategy fail here?" rather than "does it fail on average here?".

    Cells never visited are left blank rather than interpolated.  An
    early-stopped search leaves most of the plane empty, and drawing a smooth
    boundary through data that was never collected would be the single most
    misleading thing this module could do.  Concentric severity rings are drawn
    for reference, since the minimum failure is by definition the failing cell
    closest to the origin.
    """
    import matplotlib.pyplot as plt

    evaluations = list(evaluations)
    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    if not evaluations:
        axis.text(0.5, 0.5, "no scenarios evaluated", ha="center", va="center")
        return save_figure(figure, path)

    xs, ys, _returns, failures = slice_2d(evaluations, space, dim_x, dim_y)
    if len(xs) < 2 or len(ys) < 2:
        axis.text(
            0.5,
            0.5,
            "not enough distinct levels on these axes\n"
            "(run with --exhaustive for a full surface)",
            ha="center",
            va="center",
        )
        return save_figure(figure, path)

    mesh = axis.pcolormesh(
        _edges(xs),
        _edges(ys),
        np.ma.masked_invalid(failures),
        cmap="RdYlGn_r",
        vmin=0.0,
        vmax=1.0,
        shading="flat",
    )
    figure.colorbar(mesh, ax=axis, label="fraction of paths failing")

    # Severity rings: distance from the origin in this 2-D projection.
    limit = max(abs(xs).max(), abs(ys).max())
    grid = np.linspace(-limit * 1.05, limit * 1.05, 200)
    gx, gy = np.meshgrid(grid, grid)
    rings = axis.contour(
        gx,
        gy,
        np.sqrt(gx**2 + gy**2),
        levels=[1, 2, 3, 4],
        colors=COLORS["neutral"],
        linewidths=0.7,
        linestyles=":",
    )
    axis.clabel(rings, fmt=lambda v: f"{v:.0f}$\\sigma$", fontsize=7)

    axis.plot(0, 0, marker="o", color=COLORS["baseline"], ms=7, label="baseline")
    if minimum is not None:
        ix, iy = space.index(dim_x), space.index(dim_y)
        axis.plot(
            minimum.realised.z[ix],
            minimum.realised.z[iy],
            marker="*",
            color=COLORS["boundary"],
            ms=17,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=f"minimum failure ({minimum.severity:.2f}$\\sigma$)",
        )

    axis.set_xlabel(f"{space[dim_x].label} shock ($\\sigma$)")
    axis.set_ylabel(f"{space[dim_y].label} shock ($\\sigma$)")
    axis.set_title(title or f"Failure boundary: {dim_x} vs {dim_y}")
    axis.legend(loc="upper left", fontsize=8)
    axis.set_aspect("equal", adjustable="box")
    return save_figure(figure, path)


def _edges(centres: np.ndarray) -> np.ndarray:
    """Cell edges from cell centres, for ``shading="flat"``."""
    centres = np.asarray(centres, dtype=np.float64)
    if len(centres) == 1:
        return np.array([centres[0] - 0.5, centres[0] + 0.5])
    midpoints = 0.5 * (centres[:-1] + centres[1:])
    first = centres[0] - (midpoints[0] - centres[0])
    last = centres[-1] + (centres[-1] - midpoints[-1])
    return np.concatenate(([first], midpoints, [last]))
