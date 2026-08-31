"""Shared plotting setup and the two plots that summarise a whole search.

Every figure function follows the same contract: it takes data and an optional
``path``, draws onto a fresh figure, saves if a path was given, and returns the
matplotlib ``Figure``.  Nothing here calls ``plt.show()`` -- these are meant to
run headless inside experiments as much as interactively.

The ``Agg`` backend is selected on import for the same reason: a robustness sweep
that writes 40 figures should never try to open a window.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend selection)
import numpy as np  # noqa: E402

__all__ = [
    "COLORS",
    "save_figure",
    "severity_vs_return",
    "return_distribution",
    "setup_style",
]

#: One palette, used consistently: baseline is always blue, stressed always red.
COLORS = {
    "baseline": "#1f6fb4",
    "stressed": "#c0392b",
    "boundary": "#e67e22",
    "failure": "#c0392b",
    "survive": "#2e8b57",
    "neutral": "#7f8c8d",
    "grid": "#d5d8dc",
}


def setup_style() -> None:
    """Apply a plain, readable style suitable for a research report."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 8,
        }
    )


setup_style()


def save_figure(figure: Any, path: str | Path | None) -> Any:
    """Save ``figure`` to ``path`` (creating parent directories) and return it."""
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target)
    return figure


def _percent_axis(axis: Any, which: str = "y") -> None:
    from matplotlib.ticker import FuncFormatter

    formatter = FuncFormatter(lambda v, _: f"{v:.0%}")
    if which == "y":
        axis.yaxis.set_major_formatter(formatter)
    else:
        axis.xaxis.set_major_formatter(formatter)


def severity_vs_return(
    profile: Sequence[tuple[float, float, bool]],
    minimum_severity: float | None = None,
    baseline_return: float | None = None,
    failure_threshold: float = 0.0,
    title: str = "Severity vs strategy return",
    path: str | Path | None = None,
) -> Any:
    r"""Scatter :math:`D(x)` against mean return, colouring failures.

    This is the plot that makes the result legible at a glance: return should
    decay as severity grows, and the minimum failure is the leftmost red point.
    A cloud with no downward trend is itself informative -- it means the strategy
    is insensitive to the dimensions being searched, and the boundary (if any) is
    being found in a corner rather than along a gradient.

    ``profile`` is the output of
    :func:`marketerror.optimization.failure_boundary.severity_profile`.
    """
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    if not profile:
        axis.text(0.5, 0.5, "no scenarios evaluated", ha="center", va="center")
        return save_figure(figure, path)

    severity = np.array([row[0] for row in profile])
    returns = np.array([row[1] for row in profile])
    failed = np.array([row[2] for row in profile], dtype=bool)

    axis.scatter(
        severity[~failed],
        returns[~failed],
        s=18,
        c=COLORS["survive"],
        alpha=0.65,
        label=f"survived ({int((~failed).sum())})",
        edgecolors="none",
    )
    axis.scatter(
        severity[failed],
        returns[failed],
        s=22,
        c=COLORS["failure"],
        alpha=0.8,
        label=f"failed ({int(failed.sum())})",
        edgecolors="none",
    )

    axis.axhline(failure_threshold, color=COLORS["neutral"], lw=0.9, ls="--")
    if baseline_return is not None:
        axis.axhline(
            baseline_return,
            color=COLORS["baseline"],
            lw=1.1,
            ls=":",
            label=f"baseline {baseline_return:+.1%}",
        )
    if minimum_severity is not None and np.isfinite(minimum_severity):
        axis.axvline(
            minimum_severity,
            color=COLORS["boundary"],
            lw=1.4,
            label=f"minimum failure {minimum_severity:.2f}$\\sigma$",
        )

    axis.set_xlabel("perturbation severity  $D(x) = \\|x\\|_2$  ($\\sigma$)")
    axis.set_ylabel("mean return across paths")
    axis.set_title(title)
    _percent_axis(axis)
    axis.legend(loc="best")
    return save_figure(figure, path)


def return_distribution(
    baseline_returns: Iterable[float],
    stressed_returns: Iterable[float] | None = None,
    failure_threshold: float = 0.0,
    title: str = "Return distribution across Monte Carlo paths",
    path: str | Path | None = None,
) -> Any:
    """Overlaid histograms of per-path returns, baseline against stressed.

    Shows *how* a perturbation broke a strategy: a distribution that shifted
    left is a lost edge, one that merely widened is added risk.  The two have
    very different remedies, and the mean return alone cannot tell them apart.
    """
    baseline = np.asarray(list(baseline_returns), dtype=np.float64)
    figure, axis = plt.subplots(figsize=(7.0, 4.0))

    everything = [baseline]
    stressed = None
    if stressed_returns is not None:
        stressed = np.asarray(list(stressed_returns), dtype=np.float64)
        everything.append(stressed)
    combined = np.concatenate(everything)
    bins = np.linspace(combined.min(), combined.max(), max(12, min(40, len(combined) // 3)))

    axis.hist(
        baseline,
        bins=bins,
        color=COLORS["baseline"],
        alpha=0.55,
        label=f"baseline (mean {baseline.mean():+.1%})",
    )
    if stressed is not None:
        axis.hist(
            stressed,
            bins=bins,
            color=COLORS["stressed"],
            alpha=0.55,
            label=f"stressed (mean {stressed.mean():+.1%})",
        )
    axis.axvline(failure_threshold, color="black", lw=1.0, ls="--", label="failure threshold")
    axis.set_xlabel("total return")
    axis.set_ylabel("paths")
    axis.set_title(title)
    _percent_axis(axis, which="x")
    axis.legend(loc="best")
    return save_figure(figure, path)
