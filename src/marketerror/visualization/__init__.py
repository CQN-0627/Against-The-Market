"""Figures for robustness experiments.

Four views, each answering a different question:

``equity_curve``
    What did the perturbation do to one path?  (And where was the loss run?)
``failure_boundary``
    Which region of a two-dimensional shock plane breaks the strategy?
``robustness_surface``
    How does return vary continuously across that plane?
``plots.severity_vs_return``
    Where is the minimum failure, and does return actually decay with severity?

All functions accept an optional ``path`` and return the matplotlib ``Figure``.
The ``Agg`` backend is selected on import so experiments can plot headless.
"""

from __future__ import annotations

from .equity_curve import equity_curve, equity_fan
from .failure_boundary import failure_boundary_plot
from .plots import COLORS, return_distribution, save_figure, setup_style, severity_vs_return
from .robustness_surface import axis_sensitivity_plot, robustness_surface

__all__ = [
    "COLORS",
    "axis_sensitivity_plot",
    "equity_curve",
    "equity_fan",
    "failure_boundary_plot",
    "return_distribution",
    "robustness_surface",
    "save_figure",
    "setup_style",
    "severity_vs_return",
]
