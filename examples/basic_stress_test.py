"""Example: apply one explicit shock and measure what it does.

    python examples/basic_stress_test.py

Where ``find_failure_boundary.py`` *searches* for the breaking point, this script
applies a single, named perturbation and shows the before/after -- useful for
building intuition about how a given shock flows through to P&L, and for
demonstrating the ``--losstime`` sustained-loss criterion.
"""

from __future__ import annotations

from marketerror.backtest import BacktestConfig
from marketerror.market.parameters import MarketParameters
from marketerror.optimization.objective import FailureCriteria, parse_loss_time
from marketerror.perturbations.dimensions import build_space
from marketerror.perturbations.vector import PerturbationVector
from marketerror.simulation.monte_carlo import run_monte_carlo
from marketerror.strategies.loader import StrategySpec

PERIODS = 252
PATHS = 64
SEED = 42


def main() -> None:
    # Momentum on a trending market: an edge that a trend shock can erase.
    baseline = MarketParameters(trend_persistence=0.15, drift=0.06)
    strategy = StrategySpec("momentum", {"lookback": 5})
    space = build_space(("volatility", "spread", "liquidity", "trend", "jump"))

    # A -1 sigma trend shock plus a +1 sigma spread shock. Severity = sqrt(2).
    shock = PerturbationVector.from_mapping(space.names, {"trend": -1.0, "spread": 1.0})
    stressed_parameters = space.apply(baseline, shock.z)

    # Fail only if underwater for a contiguous *month* (the --losstime idea).
    criteria = FailureCriteria(
        loss_periods=parse_loss_time("1m", PERIODS), min_loss_probability=0.60
    )

    from marketerror.data.distributions import path_seeds

    seeds = path_seeds(SEED, PATHS)
    config = BacktestConfig(record_fills=False)

    base = run_monte_carlo(baseline, strategy, seeds, PERIODS, config, criteria)
    stress = run_monte_carlo(stressed_parameters, strategy, seeds, PERIODS, config, criteria)

    print(f"Momentum(5) on a trending market, shocked by {shock.label()}")
    print(f"severity D(x) = {shock.severity:.3f} sigma\n")
    print(f"{'':<22}{'baseline':>12}{'stressed':>12}")
    print("  " + "-" * 44)
    for label, key, fmt in (
        ("mean return", "mean_return", "{:+.2%}"),
        ("loss probability", "loss_probability", "{:.0%}"),
        ("failure probability", "failure_probability", "{:.0%}"),
        ("mean longest loss run", "mean_longest_loss_run", "{:.0f}"),
    ):
        print(
            f"  {label:<20}{fmt.format(getattr(base, key)):>12}"
            f"{fmt.format(getattr(stress, key)):>12}"
        )

    verdict = criteria.evaluate(stress)
    print(f"\n  failure criterion: {criteria.describe()}")
    print(f"  {verdict.explain()}")


if __name__ == "__main__":
    main()
