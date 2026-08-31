"""Standardisation must be exact and invertible -- specification §26."""

from __future__ import annotations

import math

import pytest

from marketerror.market.parameters import MarketParameters
from marketerror.perturbations.dimensions import ALL_DIMENSION_NAMES, build_space
from marketerror.perturbations.standardization import (
    LinearStandardizer,
    LogStandardizer,
    from_z_score,
    to_z_score,
)
from marketerror.perturbations.vector import PerturbationVector, severity


class TestLinearStandardizer:
    """The worked example the specification fixes: mu=5, sigma=2, x=9 -> z=2."""

    def test_z_score_of_nine(self):
        assert to_z_score(value=9, mean=5, std=2) == 2.0

    def test_inverse_returns_nine(self):
        assert from_z_score(z=2, mean=5, std=2) == 9.0

    def test_round_trip(self):
        standardizer = LinearStandardizer()
        for value in (-10.0, 0.0, 3.7, 1e6):
            z = standardizer.to_z_score(value, mean=5.0, std=2.0)
            assert standardizer.from_z_score(z, mean=5.0, std=2.0) == pytest.approx(value)

    def test_zero_z_is_the_mean(self):
        assert from_z_score(z=0, mean=5, std=2) == 5.0

    def test_handles_negative_values(self):
        """Linear scale must accept negatives: trend persistence can be < 0."""
        assert to_z_score(value=-0.2, mean=0.0, std=0.1) == pytest.approx(-2.0)

    @pytest.mark.parametrize("std", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_invalid_std(self, std):
        with pytest.raises(ValueError):
            to_z_score(value=1.0, mean=0.0, std=std)


class TestLogStandardizer:
    def test_round_trip(self):
        standardizer = LogStandardizer()
        for value in (1e-8, 0.2, 5.0, 1e6):
            z = standardizer.to_z_score(value, mean=5.0, std=0.30)
            assert standardizer.from_z_score(z, mean=5.0, std=0.30) == pytest.approx(value)

    def test_zero_z_is_the_mean(self):
        assert LogStandardizer().from_z_score(0.0, mean=5.0, std=0.3) == 5.0

    def test_stays_positive_for_any_z(self):
        """The property that removes the need to clip: image is (0, inf)."""
        standardizer = LogStandardizer()
        for z in (-1e3, -100.0, -10.0, 10.0, 100.0):
            assert standardizer.from_z_score(z, mean=5.0, std=0.3) > 0.0

    def test_is_multiplicative(self):
        """One sigma is a constant *factor*, independent of the level."""
        standardizer = LogStandardizer()
        a = standardizer.from_z_score(1.0, mean=5.0, std=0.4)
        b = standardizer.from_z_score(1.0, mean=50.0, std=0.4)
        assert a / 5.0 == pytest.approx(b / 50.0) == pytest.approx(math.exp(0.4))

    def test_rejects_non_positive_input(self):
        with pytest.raises(ValueError):
            LogStandardizer().to_z_score(0.0, mean=5.0, std=0.3)
        with pytest.raises(ValueError):
            LogStandardizer().to_z_score(-1.0, mean=5.0, std=0.3)


class TestSeverity:
    """D(x) = ||x||_2 -- specification §8 and §26."""

    def test_specification_example(self):
        assert severity([1.0, 1.0, -1.0]) == pytest.approx(math.sqrt(3))

    def test_single_axis(self):
        assert severity([2.0, 0.0, 0.0]) == 2.0

    def test_many_small_shocks_beat_one_large_one(self):
        """§8: scenario B (three 1-sigma shocks) is *smaller* than A (one 2-sigma)."""
        scenario_a = severity([2.0, 0.0, 0.0])
        scenario_b = severity([1.0, 1.0, -1.0])
        assert scenario_b < scenario_a

    def test_sign_blind(self):
        assert severity([1.0, -2.0]) == severity([-1.0, 2.0])

    def test_baseline_is_zero(self):
        assert severity([0.0, 0.0, 0.0]) == 0.0

    def test_weighted(self):
        assert severity([1.0, 1.0], weights=[2.0, 0.0]) == pytest.approx(2.0)

    def test_rejects_non_finite(self):
        with pytest.raises(ValueError):
            severity([1.0, float("nan")])


class TestPerturbationVector:
    def test_severity_matches_function(self):
        vector = PerturbationVector(("a", "b", "c"), (1.0, 1.0, -1.0))
        assert vector.severity == pytest.approx(math.sqrt(3))

    def test_lookup_by_name(self):
        vector = PerturbationVector(("volatility", "spread"), (1.5, -0.5))
        assert vector["volatility"] == 1.5
        assert vector["spread"] == -0.5
        with pytest.raises(KeyError):
            vector["liquidity"]

    def test_active_excludes_zeros(self):
        vector = PerturbationVector(("a", "b", "c"), (1.0, 0.0, -2.0))
        assert vector.active == (("a", 1.0), ("c", -2.0))

    def test_scaling_scales_severity(self):
        vector = PerturbationVector(("a", "b"), (3.0, 4.0))
        assert vector.severity == pytest.approx(5.0)
        assert vector.scaled(0.5).severity == pytest.approx(2.5)

    def test_unit_has_severity_one(self):
        vector = PerturbationVector(("a", "b"), (3.0, 4.0))
        assert vector.unit().severity == pytest.approx(1.0)

    def test_from_mapping_fills_zeros(self):
        vector = PerturbationVector.from_mapping(("a", "b", "c"), {"b": 2.0})
        assert vector.z == (0.0, 2.0, 0.0)

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError):
            PerturbationVector(("a", "b"), (1.0,))


class TestDimensionCalibration:
    """The dispersions must reproduce the anchors documented for them."""

    def test_spread_example_is_two_sigma(self):
        """§6: a 9 bps spread against a 5 bps baseline is about +2 sigma."""
        space = build_space(("spread",))
        z = space["spread"].z_of(9.0, MarketParameters())
        assert z == pytest.approx(2.0, abs=0.05)

    def test_liquidity_adverse_direction_is_negative(self):
        """§6: '-2 sigma liquidity' must mean unusually *low* liquidity."""
        space = build_space(("liquidity",))
        dimension = space["liquidity"]
        assert dimension.adverse_sign == -1
        assert dimension.value_at(MarketParameters(), -2.0) < 1.0
        assert dimension.value_at(MarketParameters(), +2.0) > 1.0

    @pytest.mark.parametrize("name", ALL_DIMENSION_NAMES)
    def test_realised_z_is_exact_within_the_search_box(self, name):
        """Requested and realised z must agree for continuous dimensions.

        ``latency`` is exempt: it is integer-valued, so rounding is inherent and
        the framework reports the rounded severity on purpose.
        """
        space = build_space((name,))
        dimension = space[name]
        parameters = MarketParameters()
        if dimension.integer:
            pytest.skip("latency is discrete by design; see realised_z")
        for step in range(-16, 17):
            z = step * 0.25
            assert dimension.realised_z(parameters, z) == pytest.approx(z, abs=1e-9)

    @pytest.mark.parametrize("name", ALL_DIMENSION_NAMES)
    def test_extreme_z_still_yields_a_valid_market(self, name):
        """§9: no z may produce an economically invalid market."""
        space = build_space((name,))
        parameters = MarketParameters()
        for z in (-50.0, -4.0, 4.0, 50.0):
            space.apply(parameters, [z]).validate()  # raises if invalid

    def test_clamped_z_is_reported_honestly(self):
        """A bound that binds must lower the *reported* severity, never inflate it.

        Otherwise a scenario could claim to be a 40-sigma shock while simulating
        a 19-sigma market -- or worse, claim a milder shock than it simulated.
        """
        space = build_space(("volatility",))
        dimension = space["volatility"]
        parameters = MarketParameters()
        requested = -40.0
        realised = dimension.realised_z(parameters, requested)
        assert abs(realised) < abs(requested)
        # And the realised z genuinely corresponds to the applied value.
        applied = dimension.value_at(parameters, requested)
        assert dimension.value_at(parameters, realised) == pytest.approx(applied)


class TestPerturbationSpace:
    def test_shocks_are_order_independent(self):
        """Each dimension standardises against the baseline, not against a
        partially perturbed market, so simultaneous application is well defined."""
        space = build_space(("volatility", "spread", "liquidity"))
        parameters = MarketParameters()
        forward = space.apply(parameters, [1.0, -1.0, 2.0])
        # Apply one at a time, in the other order, always relative to baseline.
        manual = parameters
        for name, z in (("liquidity", 2.0), ("spread", -1.0), ("volatility", 1.0)):
            manual = manual.replace(
                **{space[name].parameter: space[name].value_at(parameters, z)}
            )
        assert forward.to_dict() == manual.to_dict()

    def test_locates_a_parameter_set_in_z_space(self):
        space = build_space(("volatility", "spread"))
        baseline = MarketParameters()
        stressed = space.apply(baseline, [1.5, -0.5])
        assert space.z_of(stressed, baseline) == pytest.approx((1.5, -0.5))

    def test_rejects_wrong_length(self):
        space = build_space(("volatility", "spread"))
        with pytest.raises(ValueError):
            space.apply(MarketParameters(), [1.0])

    def test_rejects_unknown_dimension(self):
        with pytest.raises(KeyError):
            build_space(("not_a_dimension",))
