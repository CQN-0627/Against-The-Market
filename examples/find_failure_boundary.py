"""Example: the full failure-boundary experiment across all four strategies.

    python examples/find_failure_boundary.py

This is the demonstration of specification §28.  Each strategy is tested *in the
market regime where it has an edge* -- there is no single synthetic market on
which a trend follower and a mean reverter are both profitable, because they want
opposite signs of return autocorrelation.  The question the table answers is
therefore:

    Starting from a market where this strategy works, how large a standardised
    adverse move -- measured as the Euclidean norm of the z-scored perturbation
    vector -- does it take to push it to its failure condition?

Every number printed comes from actual simulation.  Nothing is fabricated; if you
change a seed the numbers will move, which is the point of the Monte Carlo layer.
"""

from __future__ import annotations

from dataclasses import dataclass

from marketerror.analysis.robustness import comparison_table, regime_reference
from marketerror.market.parameters import MarketParameters
from marketerror.optimization.objective import FailureCriteria, parse_loss_time
from marketerror.simulation.experiment import run_experiment
from marketerror.simulation.scenarios import ExperimentRecord, ExperimentSpec
from marketerror.strategies.loader import StrategySpec

PERIODS = 252
PATHS = 64
SEED = 42


@dataclass
class Case:
    """A strategy paired with the regime in which it has an edge to lose."""

    reference: str
    params: dict
    market: MarketParameters
    regime_note: str


#: Each strategy in its home regime. See the module docstring for why this is
#: the honest framing rather than one shared baseline.
CASES = [
    Case(
        "momentum",
        {"lookback": 5},
        MarketParameters(trend_persistence=0.15, drift=0.06),
        "trending (phi=+0.15): momentum needs positive return autocorrelation",
    ),
    Case(
        "mean_reversion",
        {"lookback": 5},
        MarketParameters(trend_persistence=-0.20, drift=0.05),
        "mean-reverting (phi=-0.20): reversion needs negative autocorrelation",
    ),
    Case(
        "moving_average",
        {"fast": 10, "slow": 50},
        MarketParameters(annualized_volatility=0.12, drift=0.20, trend_persistence=0.10),
        "strong low-vol drift: a slow crossover needs a sustained direction",
    ),
    Case(
        "buy_and_hold",
        {},
        MarketParameters(annualized_volatility=0.12, drift=0.20),
        "positive drift: the passive control, broken only by drift/vol shocks",
    ),
]


def run_case(case: Case, losstime: str = "0") -> ExperimentRecord:
    spec = ExperimentSpec(
        strategy=StrategySpec(case.reference, case.params),
        market=case.market,
        periods=PERIODS,
        paths=PATHS,
        seed=SEED,
        losstime=losstime,
        criteria=FailureCriteria(
            loss_periods=parse_loss_time(losstime, PERIODS),
            min_loss_probability=0.60,
            minimum_paths=PATHS,
        ),
    )
    return run_experiment(spec, reporter=lambda m: print(f"    {m}"))


def main() -> None:
    records = []
    for case in CASES:
        print(f"\n=== {case.reference} -- {case.regime_note} ===")
        record = run_case(case)
        records.append(record)
        boundary = record.boundary
        if boundary.found:
            print(
                f"    baseline {record.baseline.summary.mean_return:+.2%}  ->  "
                f"minimum failure {boundary.severity:.2f}s "
                f"({boundary.minimum.realised.label()})"
            )
        else:
            print(
                f"    baseline {record.baseline.summary.mean_return:+.2%}  ->  "
                f"no failure within {boundary.max_severity_searched:.1f}s"
            )

    print("\n")
    print(comparison_table(records))
    print()
    print(regime_reference(records[0]))


if __name__ == "__main__":
    main()
