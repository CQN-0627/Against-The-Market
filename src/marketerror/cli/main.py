"""The ``marketerror`` command-line interface.

Six subcommands, in the order you would normally use them:

``strategies``  list the built-ins and how to point at your own
``regimes``     show what a sigma means, and where named regimes sit
``market``      generate a market and check its realised statistics
``run``         one backtest on the unperturbed market
``stress``      apply an explicit shock and compare against the baseline
``optimize``    search for the minimum disruption that causes failure
``compare``     run ``optimize`` for several strategies and tabulate

Run ``marketerror <command> --help`` for the flags of each.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Sequence

from .. import __version__

__all__ = ["build_parser", "main"]

_EPILOG = """\
examples:
  marketerror run --strategy momentum --days 252 --seed 42
  marketerror stress --strategy momentum --z volatility=1,spread=1,liquidity=-1
  marketerror optimize --strategy momentum --paths 100 --losstime 3m
  marketerror optimize --strategy ./my_strategy.py:Reverter --dims volatility,trend
  marketerror compare --strategy momentum,mean_reversion,moving_average,buy_and_hold
"""


# --------------------------------------------------------------------- arguments
def _add_strategy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--strategy",
        default="momentum",
        help=(
            "built-in name (momentum, mean_reversion, moving_average, "
            "buy_and_hold), a path to a .py file, or 'file.py:ClassName'"
        ),
    )
    parser.add_argument(
        "--strategy-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="strategy parameter; repeatable (e.g. --strategy-arg lookback=40)",
    )


def _add_market_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("market")
    group.add_argument("--days", type=int, default=252, help="periods per path (default 252)")
    group.add_argument("--seed", type=int, default=42, help="root random seed (default 42)")
    group.add_argument(
        "--regime",
        default="normal",
        help="baseline regime: normal, trending, mean_reverting, high_volatility, "
        "low_liquidity, crisis",
    )
    group.add_argument(
        "--market-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a market parameter (e.g. --market-arg trend_persistence=0.15)",
    )


def _add_backtest_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("backtest")
    group.add_argument("--capital", type=float, default=100_000.0, help="initial capital")
    group.add_argument("--commission-bps", type=float, default=1.0, help="commission in bps")
    group.add_argument("--max-leverage", type=float, default=1.0, help="gross exposure cap")
    group.add_argument(
        "--latency",
        type=int,
        default=None,
        help="bars between decision and execution (default: the market's own value, 0)",
    )
    group.add_argument("--no-short", action="store_true", help="disable short selling")


def _add_failure_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("failure criterion")
    group.add_argument(
        "--losstime",
        default="0",
        help=(
            "how long the strategy must be continuously unprofitable to count as "
            "failed: bars (60), a percentage of the window (25%%), or a calendar "
            "span (10d, 4w, 3m, 1y). 0 (default) tests the final return only."
        ),
    )
    group.add_argument(
        "--failure-return",
        type=float,
        default=0.0,
        help="cumulative return below which the strategy is unprofitable (default 0)",
    )
    group.add_argument(
        "--mean-return-threshold",
        type=float,
        default=0.0,
        help="mean return across paths must fall below this (default 0)",
    )
    group.add_argument(
        "--min-loss-prob",
        type=float,
        default=0.60,
        help="fraction of paths that must individually fail (default 0.60)",
    )
    group.add_argument(
        "--ignore-mean-return",
        action="store_true",
        help="judge on the loss probability alone",
    )


def _add_search_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("search")
    group.add_argument(
        "--dims",
        default=None,
        help="comma-separated perturbation dimensions (default: volatility,spread,"
        "liquidity,trend,jump)",
    )
    group.add_argument("--max-z", type=float, default=4.0, help="box half-width in sigma")
    group.add_argument(
        "--levels",
        default=None,
        help="comma-separated grid levels (default -2,-1,0,1,2)",
    )
    group.add_argument(
        "--max-severity", type=float, default=None, help="optional cap on D(x)"
    )
    group.add_argument(
        "--sigma-source",
        choices=("prior", "empirical"),
        default="prior",
        help="declared cross-regime dispersions (default) or dispersions estimated "
        "from baseline paths",
    )
    group.add_argument(
        "--exhaustive",
        action="store_true",
        help="evaluate every grid point instead of stopping at the first failure "
        "(needed for full surface plots; implied by --plots)",
    )
    group.add_argument("--no-refine", action="store_true", help="skip radial bisection")
    group.add_argument("--no-axis-scan", action="store_true", help="skip single-axis scan")


def _add_simulation_arguments(parser: argparse.ArgumentParser, default_paths: int = 32) -> None:
    group = parser.add_argument_group("simulation")
    group.add_argument(
        "--paths", type=int, default=default_paths, help=f"Monte Carlo paths (default {default_paths})"
    )
    group.add_argument(
        "--validation-paths",
        type=int,
        default=0,
        help="paths for out-of-sample confirmation (default: same as --paths)",
    )
    group.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="worker processes (default 1; use 0 for one per CPU)",
    )


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("output")
    group.add_argument(
        "--out",
        default="results",
        help="directory for figures and exported results (default ./results)",
    )
    group.add_argument("--plots", action="store_true", help="write figures")
    group.add_argument("--save", action="store_true", help="write JSON and CSV results")
    group.add_argument("--json", action="store_true", help="print machine-readable JSON")
    group.add_argument("--quiet", action="store_true", help="suppress progress messages")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="marketerror",
        description=(
            "Find the smallest statistically standardised market disruption that "
            "makes a trading strategy unprofitable."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"marketerror {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    sub.add_parser("strategies", help="list the built-in strategies")

    regimes = sub.add_parser("regimes", help="show sigma calibration and named regimes")
    _add_market_arguments(regimes)
    _add_search_arguments(regimes)

    market = sub.add_parser("market", help="generate a market and verify its statistics")
    _add_market_arguments(market)
    market.add_argument("--paths", type=int, default=1, help="paths to average over")
    market.add_argument("--csv", default=None, help="write the first path to this CSV")

    run = sub.add_parser("run", help="one backtest on the unperturbed market")
    _add_strategy_arguments(run)
    _add_market_arguments(run)
    _add_backtest_arguments(run)
    _add_failure_arguments(run)
    _add_simulation_arguments(run, default_paths=1)
    _add_output_arguments(run)

    stress = sub.add_parser("stress", help="apply an explicit shock and compare")
    _add_strategy_arguments(stress)
    _add_market_arguments(stress)
    _add_backtest_arguments(stress)
    _add_failure_arguments(stress)
    _add_search_arguments(stress)
    _add_simulation_arguments(stress)
    _add_output_arguments(stress)
    stress.add_argument(
        "--z",
        default=None,
        help="shock as comma-separated assignments, e.g. "
        "volatility=1,spread=1.5,liquidity=-0.5",
    )
    stress.add_argument(
        "--stress-regime",
        default=None,
        help="instead of --z, stress to a named regime's parameters",
    )

    optimize = sub.add_parser("optimize", help="find the minimum failure disruption")
    _add_strategy_arguments(optimize)
    _add_market_arguments(optimize)
    _add_backtest_arguments(optimize)
    _add_failure_arguments(optimize)
    _add_search_arguments(optimize)
    _add_simulation_arguments(optimize)
    _add_output_arguments(optimize)

    compare = sub.add_parser("compare", help="optimize several strategies and tabulate")
    _add_strategy_arguments(compare)
    _add_market_arguments(compare)
    _add_backtest_arguments(compare)
    _add_failure_arguments(compare)
    _add_search_arguments(compare)
    _add_simulation_arguments(compare)
    _add_output_arguments(compare)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.  Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0

    from . import commands

    handlers = {
        "strategies": commands.cmd_strategies,
        "regimes": commands.cmd_regimes,
        "market": commands.cmd_market,
        "run": commands.cmd_run,
        "stress": commands.cmd_stress,
        "optimize": commands.cmd_optimize,
        "compare": commands.cmd_compare,
    }
    handler = handlers[args.command]

    with ExitStack() as stack:
        executor = None
        jobs = getattr(args, "jobs", 1)
        if jobs != 1:
            import os

            workers = jobs if jobs > 0 else (os.cpu_count() or 1)
            if workers > 1:
                executor = stack.enter_context(ProcessPoolExecutor(max_workers=workers))
        try:
            return handler(args, executor)
        except KeyboardInterrupt:  # pragma: no cover - interactive
            print("\ninterrupted", file=sys.stderr)
            return 130
        except (ValueError, KeyError, FileNotFoundError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
