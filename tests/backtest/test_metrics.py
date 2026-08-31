"""Metrics arithmetic, checked against values computed by hand."""

from __future__ import annotations

import math

import numpy as np
import pytest

from marketerror.analysis.statistics import (
    drawdown_series,
    first_run_below,
    longest_run_below,
    max_drawdown,
    sharpe_ratio,
)
from marketerror.backtest.metrics import compute_metrics, period_returns
from marketerror.backtest.portfolio import Portfolio


@pytest.fixture
def empty_portfolio() -> Portfolio:
    return Portfolio(initial_cash=100.0)


class TestPeriodReturns:
    def test_simple_case(self):
        assert period_returns(np.array([100.0, 110.0, 99.0])) == pytest.approx(
            [0.10, -0.10]
        )

    def test_floors_negative_equity(self):
        """A negative equity value must not produce a nonsense positive return."""
        returns = period_returns(np.array([100.0, -50.0, -20.0]))
        assert returns[0] == pytest.approx(-1.0)
        assert returns[1] == 0.0  # nothing left to earn a return on

    def test_short_series(self):
        assert len(period_returns(np.array([100.0]))) == 0


class TestDrawdown:
    def test_known_drawdown(self):
        equity = np.array([100.0, 120.0, 90.0, 150.0])
        assert max_drawdown(equity) == pytest.approx(0.25)  # 120 -> 90

    def test_monotonic_has_no_drawdown(self):
        assert max_drawdown(np.array([1.0, 2.0, 3.0])) == 0.0

    def test_series_tracks_the_running_peak(self):
        series = drawdown_series(np.array([100.0, 50.0, 75.0, 200.0]))
        assert series == pytest.approx([0.0, 0.5, 0.25, 0.0])

    def test_total_loss(self):
        assert max_drawdown(np.array([100.0, 0.0])) == pytest.approx(1.0)


class TestSharpe:
    def test_zero_variance_is_nan_not_infinity(self):
        """Exactly flat returns have no variance, so the Sharpe is undefined.

        Uses an exactly-zero series: ``np.full(n, c)`` carries a ~1e-18 residual
        variance from floating point, which is a genuinely different (non-flat)
        input rather than the degenerate case the guard is for.
        """
        assert math.isnan(sharpe_ratio(np.zeros(10), 252))

    def test_annualisation(self):
        rng = np.random.default_rng(0)
        returns = rng.normal(0.001, 0.01, 100_000)
        expected = 0.001 / 0.01 * math.sqrt(252)
        assert sharpe_ratio(returns, 252) == pytest.approx(expected, rel=0.05)

    def test_risk_free_rate_is_subtracted(self):
        returns = np.array([0.01, 0.02, 0.005, 0.015])
        gross = sharpe_ratio(returns, 252, risk_free_rate=0.0)
        net = sharpe_ratio(returns, 252, risk_free_rate=0.05)
        assert net < gross

    def test_sign(self):
        rng = np.random.default_rng(1)
        assert sharpe_ratio(rng.normal(-0.001, 0.01, 10_000), 252) < 0


class TestLossRuns:
    """The estimator behind ``--losstime``."""

    def test_specification_example(self):
        assert longest_run_below(np.array([1.0, -1.0, -2.0, 0.5, -1.0])) == 2

    def test_no_losses(self):
        assert longest_run_below(np.array([1.0, 2.0, 3.0])) == 0

    def test_all_losses(self):
        assert longest_run_below(np.array([-1.0, -2.0, -3.0])) == 3

    def test_requires_contiguity(self):
        """Five scattered losing bars are not a five-bar loss run.

        This is the whole point of the criterion: it distinguishes a sustained
        regime of unprofitability from ordinary noise.
        """
        alternating = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        assert (alternating < 0).sum() == 5
        assert longest_run_below(alternating) == 1

    def test_run_at_the_end(self):
        assert longest_run_below(np.array([1.0, 1.0, -1.0, -1.0, -1.0])) == 3

    def test_custom_threshold(self):
        values = np.array([0.05, 0.02, 0.01, 0.08])
        assert longest_run_below(values, threshold=0.0) == 0
        assert longest_run_below(values, threshold=0.03) == 2

    def test_first_run_index(self):
        values = np.array([1.0, -1.0, -1.0, -1.0, 1.0])
        assert first_run_below(values, 0.0, 3) == 3
        assert first_run_below(values, 0.0, 4) is None

    def test_first_run_rejects_zero_length(self):
        with pytest.raises(ValueError):
            first_run_below(np.array([1.0]), 0.0, 0)


class TestComputeMetrics:
    def test_total_return_uses_initial_capital(self, empty_portfolio):
        equity = np.array([99.0, 110.0])
        metrics = compute_metrics(
            equity, empty_portfolio, periods_per_year=252, initial_capital=100.0
        )
        assert metrics.total_return == pytest.approx(0.10)
        assert metrics.initial_equity == 100.0

    def test_annualisation_over_multiple_years(self, empty_portfolio):
        """Doubling over exactly two years annualises to sqrt(2) - 1."""
        equity = np.full(504, 200.0)
        metrics = compute_metrics(
            equity, empty_portfolio, periods_per_year=252, initial_capital=100.0
        )
        assert metrics.years == pytest.approx(2.0)
        assert metrics.annualized_return == pytest.approx(math.sqrt(2.0) - 1.0)

    def test_total_loss_floors_annualised_return(self, empty_portfolio):
        equity = np.array([50.0, 0.0])
        metrics = compute_metrics(
            equity, empty_portfolio, periods_per_year=252, initial_capital=100.0
        )
        assert metrics.total_return == pytest.approx(-1.0)
        assert metrics.annualized_return == -1.0

    def test_cost_accounting_is_summed(self):
        from marketerror.backtest.orders import Fill, Side

        portfolio = Portfolio(initial_cash=100_000.0)
        portfolio.apply(
            Fill(
                index=0,
                side=Side.BUY,
                requested_quantity=100.0,
                filled_quantity=100.0,
                price=101.0,
                reference_price=100.0,
                commission=5.0,
                spread_cost=50.0,
                slippage_cost=45.0,
            )
        )
        assert portfolio.total_costs == pytest.approx(100.0)
        assert portfolio.position == 100.0
        # Cash pays notional and commission only: spread and slippage are already
        # inside the fill price and must not be double counted.
        assert portfolio.cash == pytest.approx(100_000.0 - 100.0 * 101.0 - 5.0)
        assert portfolio.n_trades == 1

    def test_rejected_fill_counts_but_does_not_move_cash(self):
        from marketerror.backtest.orders import Fill, Side

        portfolio = Portfolio(initial_cash=1_000.0)
        portfolio.apply(
            Fill(
                index=0,
                side=Side.BUY,
                requested_quantity=100.0,
                filled_quantity=0.0,
                price=10.0,
                reference_price=10.0,
                reasons=("depth",),
            )
        )
        assert portfolio.cash == 1_000.0
        assert portfolio.n_trades == 0
        assert portfolio.n_rejected == 1

    def test_turnover_is_annualised(self, empty_portfolio):
        empty_portfolio.traded_notional = 200.0
        metrics = compute_metrics(
            np.full(252, 100.0),
            empty_portfolio,
            periods_per_year=252,
            initial_capital=100.0,
        )
        assert metrics.years == pytest.approx(1.0)
        assert metrics.turnover == pytest.approx(2.0)

    def test_requires_two_observations(self, empty_portfolio):
        with pytest.raises(ValueError):
            compute_metrics(np.array([100.0]), empty_portfolio, 252)

    def test_time_underwater(self, empty_portfolio):
        equity = np.array([90.0, 90.0, 110.0, 110.0])
        metrics = compute_metrics(
            equity, empty_portfolio, periods_per_year=252, initial_capital=100.0
        )
        assert metrics.time_underwater == pytest.approx(0.5)
        assert metrics.longest_loss_run == 2


class TestPortfolio:
    def test_equity_and_exposure(self):
        portfolio = Portfolio(initial_cash=1_000.0)
        portfolio.cash = 500.0
        portfolio.position = 10.0
        assert portfolio.equity(50.0) == pytest.approx(1_000.0)
        assert portfolio.exposure(50.0) == pytest.approx(0.5)

    def test_reset_restores_the_initial_state(self):
        portfolio = Portfolio(initial_cash=1_000.0)
        portfolio.cash = 0.0
        portfolio.position = 5.0
        portfolio.n_trades = 3
        portfolio.reset()
        assert portfolio.cash == 1_000.0
        assert portfolio.position == 0.0
        assert portfolio.n_trades == 0
        assert not portfolio.ruined

    def test_liquidation_flattens(self):
        portfolio = Portfolio(initial_cash=1_000.0)
        portfolio.position = -100.0
        portfolio.cash = 1_100.0
        portfolio.liquidate(20.0)
        assert portfolio.position == 0.0
        assert portfolio.ruined
        assert portfolio.cash == pytest.approx(1_100.0 - 100.0 * 20.0)

    def test_rejects_non_positive_capital(self):
        with pytest.raises(ValueError):
            Portfolio(initial_cash=0.0)
