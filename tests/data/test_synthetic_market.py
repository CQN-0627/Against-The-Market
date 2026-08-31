"""The generator must produce the statistics it was asked for, reproducibly."""

from __future__ import annotations

import math

import numpy as np
import pytest

from marketerror.analysis.statistics import ar1_coefficient, realized_statistics
from marketerror.data.synthetic_market import SyntheticMarketGenerator
from marketerror.market.parameters import MarketParameters, ParameterError
from marketerror.market.regimes import Regime, apply_regime

#: Long paths, so realised statistics resolve sharply enough to assert on.
LONG = 20_000


@pytest.fixture(scope="module")
def long_baseline():
    return realized_statistics(
        SyntheticMarketGenerator(MarketParameters()).generate(LONG, seed=7)
    )


class TestReproducibility:
    """Specification §26: same seed -> identical market; different seed -> not."""

    def test_same_seed_is_identical(self, baseline_parameters):
        generator = SyntheticMarketGenerator(baseline_parameters)
        a = generator.generate(252, seed=42)
        b = generator.generate(252, seed=42)
        for field in ("price", "returns", "volume", "bid", "ask", "spread", "bid_size"):
            assert np.array_equal(getattr(a, field), getattr(b, field))

    def test_fresh_generator_same_seed_is_identical(self, baseline_parameters):
        a = SyntheticMarketGenerator(baseline_parameters).generate(252, seed=42)
        b = SyntheticMarketGenerator(baseline_parameters).generate(252, seed=42)
        assert np.array_equal(a.price, b.price)

    def test_different_seed_differs(self, baseline_parameters):
        generator = SyntheticMarketGenerator(baseline_parameters)
        a = generator.generate(252, seed=42)
        b = generator.generate(252, seed=43)
        assert not np.array_equal(a.price, b.price)

    def test_stream_isolation(self, baseline_parameters):
        """A shock to one quantity must not disturb an unrelated one.

        Perturbing the jump rate leaves the *spread* path identical, because the
        two are drawn from independent named streams.  Without this, a scenario
        would differ from its baseline by more than the parameter under study.
        """
        a = SyntheticMarketGenerator(baseline_parameters).generate(252, seed=1)
        shocked = baseline_parameters.replace(jump_probability=0.05)
        b = SyntheticMarketGenerator(shocked).generate(252, seed=1)
        assert np.allclose(a.spread / a.price, b.spread / b.price, atol=1e-15)
        assert not np.array_equal(a.price, b.price)  # the jumps did land


class TestRequestedStatistics:
    """Phase 2: verify the market has approximately the properties requested."""

    def test_volatility(self, long_baseline):
        assert long_baseline["annualized_volatility"] == pytest.approx(0.20, rel=0.03)

    def test_spread(self, long_baseline):
        assert long_baseline["mean_spread_bps"] == pytest.approx(5.0, rel=0.02)

    def test_volume(self, long_baseline):
        assert long_baseline["mean_volume"] == pytest.approx(1_000_000, rel=0.02)

    def test_no_autocorrelation_at_baseline(self, long_baseline):
        assert long_baseline["ar1_coefficient"] == pytest.approx(0.0, abs=0.02)

    @pytest.mark.parametrize("target", [0.10, 0.20, 0.40, 0.60])
    def test_volatility_tracks_the_parameter(self, target):
        data = SyntheticMarketGenerator(
            MarketParameters(annualized_volatility=target)
        ).generate(LONG, seed=3)
        assert realized_statistics(data)["annualized_volatility"] == pytest.approx(
            target, rel=0.04
        )

    @pytest.mark.parametrize("phi", [-0.30, -0.10, 0.10, 0.30])
    def test_trend_persistence_tracks_the_parameter(self, phi):
        data = SyntheticMarketGenerator(
            MarketParameters(trend_persistence=phi)
        ).generate(LONG, seed=3)
        assert realized_statistics(data)["ar1_coefficient"] == pytest.approx(phi, abs=0.02)

    @pytest.mark.parametrize("phi", [-0.40, 0.0, 0.40])
    def test_volatility_is_invariant_to_trend_persistence(self, phi):
        """The AR(1) innovation rescaling: changing phi must not change vol.

        Without it, a trend shock would silently also be a volatility shock and
        the severity metric would be measuring two things on one axis.
        """
        data = SyntheticMarketGenerator(
            MarketParameters(trend_persistence=phi)
        ).generate(LONG, seed=11)
        assert realized_statistics(data)["annualized_volatility"] == pytest.approx(
            0.20, rel=0.04
        )

    def test_spread_widens_as_liquidity_falls(self):
        parameters = MarketParameters(liquidity=0.25)
        data = SyntheticMarketGenerator(parameters).generate(5_000, seed=5)
        # spread_bps / liquidity, with the default unit elasticity
        assert realized_statistics(data)["mean_spread_bps"] == pytest.approx(20.0, rel=0.03)

    def test_depth_and_volume_fall_with_liquidity(self):
        full = SyntheticMarketGenerator(MarketParameters()).generate(2_000, seed=5)
        thin = SyntheticMarketGenerator(
            MarketParameters(liquidity=0.25)
        ).generate(2_000, seed=5)
        assert np.mean(thin.volume) < 0.3 * np.mean(full.volume)
        assert np.mean(thin.bid_size) < 0.3 * np.mean(full.bid_size)

    def test_jumps_add_kurtosis(self):
        calm = SyntheticMarketGenerator(MarketParameters()).generate(LONG, seed=9)
        jumpy = SyntheticMarketGenerator(
            MarketParameters(jump_probability=0.02, jump_size=0.06)
        ).generate(LONG, seed=9)
        assert realized_statistics(calm)["excess_kurtosis"] < 1.0
        assert realized_statistics(jumpy)["excess_kurtosis"] > 5.0

    def test_garch_preserves_unconditional_volatility(self):
        """Clustering changes the shape of the volatility path, not its level."""
        data = SyntheticMarketGenerator(
            MarketParameters(garch_alpha=0.06, garch_beta=0.90)
        ).generate(LONG, seed=13)
        assert realized_statistics(data)["annualized_volatility"] == pytest.approx(
            0.20, rel=0.10
        )

    def test_drift_is_calibrated(self):
        """E[log(P_T/P_0)] must equal drift - sigma^2/2 - jump compensation.

        At phi = 0 this holds identically rather than approximately, so a tight
        tolerance is legitimate: the only error is the sample mean of the shocks.
        """
        from marketerror.data.distributions import jump_drift_compensator

        parameters = MarketParameters()
        generator = SyntheticMarketGenerator(parameters)
        logs = np.array(
            [
                math.log(generator.generate(253, seed=s).price[-1] / 100.0)
                for s in range(4_000)
            ]
        )
        expected = (
            parameters.drift
            - 0.5 * parameters.annualized_volatility**2
            - jump_drift_compensator(parameters.jump_probability, parameters.jump_size)
            * 252
        )
        standard_error = logs.std(ddof=1) / math.sqrt(len(logs))
        assert abs(logs.mean() - expected) < 4.0 * standard_error

    def test_jump_shock_does_not_change_expected_price_growth(self):
        """The jump compensator: more jump risk must not mean more return.

        If it did, a jump-intensity shock could make a strategy look *better*
        and the optimiser would be chasing an artefact.
        """
        from marketerror.data.distributions import jump_drift_compensator

        parameters = MarketParameters(jump_probability=0.05, jump_size=0.08)
        generator = SyntheticMarketGenerator(parameters)
        logs = np.array(
            [
                math.log(generator.generate(253, seed=s).price[-1] / 100.0)
                for s in range(4_000)
            ]
        )
        expected = (
            parameters.drift
            - 0.5 * parameters.annualized_volatility**2
            - jump_drift_compensator(parameters.jump_probability, parameters.jump_size)
            * 252
        )
        standard_error = logs.std(ddof=1) / math.sqrt(len(logs))
        assert abs(logs.mean() - expected) < 4.0 * standard_error


class TestStructure:
    def test_starts_at_initial_price(self):
        data = SyntheticMarketGenerator(MarketParameters(initial_price=250.0)).generate(
            100, seed=1
        )
        assert data.price[0] == 250.0
        assert data.returns[0] == 0.0

    def test_returns_reconstruct_prices(self):
        data = SyntheticMarketGenerator(MarketParameters()).generate(500, seed=2)
        rebuilt = data.price[0] * np.exp(np.cumsum(data.returns))
        assert np.allclose(rebuilt, data.price)

    def test_book_is_never_crossed(self):
        data = SyntheticMarketGenerator(MarketParameters()).generate(1_000, seed=4)
        assert np.all(data.ask >= data.bid)
        assert np.all(data.bid > 0.0)
        assert np.allclose(data.price, 0.5 * (data.bid + data.ask))

    def test_everything_stays_positive_under_extreme_parameters(self):
        parameters = MarketParameters(
            annualized_volatility=2.5, liquidity=0.01, spread_bps=200.0, jump_probability=0.2
        )
        data = SyntheticMarketGenerator(parameters).generate(2_000, seed=6)
        data.validate()

    def test_length_and_metadata(self):
        data = SyntheticMarketGenerator(MarketParameters()).generate(321, seed=8)
        assert len(data) == 321
        assert data.metadata["source"] == "synthetic"
        assert data.metadata["seed"] == 8
        assert data.metadata["parameters"]["annualized_volatility"] == 0.20

    def test_rejects_too_few_periods(self):
        with pytest.raises(ValueError):
            SyntheticMarketGenerator(MarketParameters()).generate(1, seed=1)


class TestRegimes:
    def test_crisis_matches_its_documented_targets(self):
        parameters = apply_regime(MarketParameters(), Regime.CRISIS)
        assert parameters.annualized_volatility == 0.60
        assert parameters.liquidity == 0.25
        # 7.5 bps at liquidity 0.25 quotes at 30 bps, per specification §5.
        assert parameters.effective_spread_bps == pytest.approx(30.0)

    def test_normal_is_unchanged(self):
        parameters = MarketParameters()
        assert apply_regime(parameters, "normal") is parameters

    def test_unknown_regime_rejected(self):
        with pytest.raises(ValueError):
            apply_regime(MarketParameters(), "apocalypse")

    def test_regimes_are_realisable(self):
        for regime in Regime:
            parameters = apply_regime(MarketParameters(), regime)
            SyntheticMarketGenerator(parameters).generate(500, seed=1).validate()


class TestParameterValidation:
    @pytest.mark.parametrize(
        "changes",
        [
            {"annualized_volatility": 0.0},
            {"annualized_volatility": -0.1},
            {"spread_bps": 0.0},
            {"liquidity": -1.0},
            {"initial_price": 0.0},
            {"trend_persistence": 1.0},
            {"trend_persistence": -1.5},
            {"jump_probability": 1.5},
            {"garch_alpha": 0.6, "garch_beta": 0.6},
            {"periods_per_year": 0},
        ],
    )
    def test_invalid_parameters_rejected(self, changes):
        with pytest.raises(ParameterError):
            MarketParameters(**changes)

    def test_unknown_parameter_rejected(self):
        with pytest.raises(ParameterError):
            MarketParameters().replace(not_a_parameter=1.0)

    def test_derived_quantities(self):
        parameters = MarketParameters(annualized_volatility=0.20, periods_per_year=252)
        assert parameters.period_volatility == pytest.approx(0.20 / math.sqrt(252))
        assert parameters.annual_jump_count == pytest.approx(0.001 * 252)


class TestAr1Estimator:
    def test_recovers_a_known_coefficient(self):
        rng = np.random.default_rng(0)
        phi = 0.4
        series = np.zeros(200_000)
        noise = rng.standard_normal(len(series))
        for t in range(1, len(series)):
            series[t] = phi * series[t - 1] + noise[t]
        assert ar1_coefficient(series) == pytest.approx(phi, abs=0.01)

    def test_degenerate_series(self):
        assert math.isnan(ar1_coefficient(np.zeros(10)))
        assert math.isnan(ar1_coefficient(np.array([1.0])))
