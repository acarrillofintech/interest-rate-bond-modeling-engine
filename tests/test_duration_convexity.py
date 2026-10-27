"""Tests for bond duration and convexity calculations."""

from dataclasses import FrozenInstanceError
from math import inf, nan

import pytest

from src.bond_pricing import (
    BondParameters,
    price_coupon_bond,
)
from src.duration_convexity import (
    DurationConvexityMetrics,
    calculate_convexity,
    calculate_duration_convexity_metrics,
    calculate_dv01,
    calculate_macaulay_duration,
    calculate_modified_duration,
    estimate_new_bond_price,
    estimate_percentage_price_change,
    estimate_price_change,
)


@pytest.fixture
def bond_parameters() -> BondParameters:
    """Return the baseline fixed-coupon bond."""

    return BondParameters(
        face_value=1_000.0,
        annual_coupon_rate=0.05,
        years_to_maturity=5.0,
        payments_per_year=2,
    )


def test_macaulay_duration_is_within_maturity(
    bond_parameters: BondParameters,
) -> None:
    """Coupon-bond duration should lie between zero and maturity."""

    duration = calculate_macaulay_duration(
        parameters=bond_parameters,
        annual_yield=0.04,
    )

    assert 0.0 < duration < 5.0


def test_zero_coupon_macaulay_duration_equals_maturity() -> None:
    """Zero-coupon Macaulay duration should equal maturity."""

    parameters = BondParameters(
        face_value=1_000.0,
        annual_coupon_rate=0.0,
        years_to_maturity=5.0,
        payments_per_year=2,
    )

    duration = calculate_macaulay_duration(
        parameters=parameters,
        annual_yield=0.04,
    )

    assert duration == pytest.approx(5.0)


def test_modified_duration_relationship(
    bond_parameters: BondParameters,
) -> None:
    """Modified duration should derive from Macaulay duration."""

    annual_yield = 0.04

    macaulay_duration = calculate_macaulay_duration(
        parameters=bond_parameters,
        annual_yield=annual_yield,
    )

    modified_duration = calculate_modified_duration(
        parameters=bond_parameters,
        annual_yield=annual_yield,
    )

    expected = macaulay_duration / (
        1.0
        + annual_yield
        / bond_parameters.payments_per_year
    )

    assert modified_duration == pytest.approx(expected)


def test_zero_coupon_modified_duration() -> None:
    """Zero-coupon modified duration should follow its formula."""

    parameters = BondParameters(
        face_value=1_000.0,
        annual_coupon_rate=0.0,
        years_to_maturity=5.0,
        payments_per_year=2,
    )

    result = calculate_modified_duration(
        parameters=parameters,
        annual_yield=0.04,
    )

    expected = 5.0 / 1.02

    assert result == pytest.approx(expected)


def test_convexity_is_positive(
    bond_parameters: BondParameters,
) -> None:
    """A standard option-free bond should have positive convexity."""

    convexity = calculate_convexity(
        parameters=bond_parameters,
        annual_yield=0.04,
    )

    assert convexity > 0.0


def test_modified_duration_matches_numerical_derivative(
    bond_parameters: BondParameters,
) -> None:
    """Modified duration should match a central price derivative."""

    annual_yield = 0.04
    yield_step = 0.000001

    current_price = price_coupon_bond(
        parameters=bond_parameters,
        annual_yield=annual_yield,
    )

    lower_price = price_coupon_bond(
        parameters=bond_parameters,
        annual_yield=annual_yield - yield_step,
    )

    upper_price = price_coupon_bond(
        parameters=bond_parameters,
        annual_yield=annual_yield + yield_step,
    )

    numerical_duration = -(
        upper_price - lower_price
    ) / (
        2.0 * yield_step * current_price
    )

    analytical_duration = calculate_modified_duration(
        parameters=bond_parameters,
        annual_yield=annual_yield,
    )

    assert analytical_duration == pytest.approx(
        numerical_duration,
        rel=1e-6,
    )


def test_convexity_matches_numerical_second_derivative(
    bond_parameters: BondParameters,
) -> None:
    """Convexity should match a numerical second derivative."""

    annual_yield = 0.04
    yield_step = 0.0001

    current_price = price_coupon_bond(
        parameters=bond_parameters,
        annual_yield=annual_yield,
    )

    lower_price = price_coupon_bond(
        parameters=bond_parameters,
        annual_yield=annual_yield - yield_step,
    )

    upper_price = price_coupon_bond(
        parameters=bond_parameters,
        annual_yield=annual_yield + yield_step,
    )

    numerical_convexity = (
        upper_price
        + lower_price
        - 2.0 * current_price
    ) / (
        current_price * yield_step**2
    )

    analytical_convexity = calculate_convexity(
        parameters=bond_parameters,
        annual_yield=annual_yield,
    )

    assert analytical_convexity == pytest.approx(
        numerical_convexity,
        rel=1e-6,
    )


def test_dv01_relationship(
    bond_parameters: BondParameters,
) -> None:
    """DV01 should equal price times duration times one basis point."""

    annual_yield = 0.04

    price = price_coupon_bond(
        parameters=bond_parameters,
        annual_yield=annual_yield,
    )

    modified_duration = calculate_modified_duration(
        parameters=bond_parameters,
        annual_yield=annual_yield,
    )

    result = calculate_dv01(
        parameters=bond_parameters,
        annual_yield=annual_yield,
    )

    expected = price * modified_duration * 0.0001

    assert result == pytest.approx(expected)


def test_metrics_returns_dataclass(
    bond_parameters: BondParameters,
) -> None:
    """The combined function should return the metrics dataclass."""

    result = calculate_duration_convexity_metrics(
        parameters=bond_parameters,
        annual_yield=0.04,
    )

    assert isinstance(result, DurationConvexityMetrics)


def test_metrics_values_are_consistent(
    bond_parameters: BondParameters,
) -> None:
    """Combined metrics should match individual calculations."""

    result = calculate_duration_convexity_metrics(
        parameters=bond_parameters,
        annual_yield=0.04,
    )

    assert result.bond_price == pytest.approx(
        price_coupon_bond(bond_parameters, 0.04)
    )
    assert result.macaulay_duration == pytest.approx(
        calculate_macaulay_duration(
            bond_parameters,
            0.04,
        )
    )
    assert result.modified_duration == pytest.approx(
        calculate_modified_duration(
            bond_parameters,
            0.04,
        )
    )
    assert result.convexity == pytest.approx(
        calculate_convexity(
            bond_parameters,
            0.04,
        )
    )
    assert result.dv01 == pytest.approx(
        calculate_dv01(
            bond_parameters,
            0.04,
        )
    )


def test_metrics_are_immutable(
    bond_parameters: BondParameters,
) -> None:
    """Sensitivity metrics should be immutable."""

    result = calculate_duration_convexity_metrics(
        parameters=bond_parameters,
        annual_yield=0.04,
    )

    with pytest.raises(FrozenInstanceError):
        result.dv01 = 100.0  # type: ignore[misc]


def test_yield_increase_reduces_estimated_price(
    bond_parameters: BondParameters,
) -> None:
    """A yield increase should reduce the bond price."""

    change = estimate_price_change(
        parameters=bond_parameters,
        annual_yield=0.04,
        yield_change=0.01,
    )

    assert change < 0.0


def test_yield_decrease_increases_estimated_price(
    bond_parameters: BondParameters,
) -> None:
    """A yield decrease should increase the bond price."""

    change = estimate_price_change(
        parameters=bond_parameters,
        annual_yield=0.04,
        yield_change=-0.01,
    )

    assert change > 0.0


def test_zero_yield_change_produces_zero_price_change(
    bond_parameters: BondParameters,
) -> None:
    """No yield movement should produce no estimated change."""

    result = estimate_price_change(
        parameters=bond_parameters,
        annual_yield=0.04,
        yield_change=0.0,
    )

    assert result == pytest.approx(0.0)


def test_estimated_new_price_relationship(
    bond_parameters: BondParameters,
) -> None:
    """New price should equal current price plus estimated change."""

    current_price = price_coupon_bond(
        parameters=bond_parameters,
        annual_yield=0.04,
    )

    price_change = estimate_price_change(
        parameters=bond_parameters,
        annual_yield=0.04,
        yield_change=0.01,
    )

    estimated_price = estimate_new_bond_price(
        parameters=bond_parameters,
        annual_yield=0.04,
        yield_change=0.01,
    )

    assert estimated_price == pytest.approx(
        current_price + price_change
    )


def test_convexity_improves_price_approximation(
    bond_parameters: BondParameters,
) -> None:
    """Adding convexity should improve the duration approximation."""

    annual_yield = 0.04
    yield_change = 0.01

    exact_price = price_coupon_bond(
        parameters=bond_parameters,
        annual_yield=annual_yield + yield_change,
    )

    duration_only_price = estimate_new_bond_price(
        parameters=bond_parameters,
        annual_yield=annual_yield,
        yield_change=yield_change,
        include_convexity=False,
    )

    duration_convexity_price = estimate_new_bond_price(
        parameters=bond_parameters,
        annual_yield=annual_yield,
        yield_change=yield_change,
        include_convexity=True,
    )

    duration_error = abs(
        duration_only_price - exact_price
    )
    convexity_error = abs(
        duration_convexity_price - exact_price
    )

    assert convexity_error < duration_error


def test_duration_only_percentage_change(
    bond_parameters: BondParameters,
) -> None:
    """Without convexity, change should equal negative duration times yield."""

    yield_change = 0.01

    result = estimate_percentage_price_change(
        parameters=bond_parameters,
        annual_yield=0.04,
        yield_change=yield_change,
        include_convexity=False,
    )

    expected = -calculate_modified_duration(
        parameters=bond_parameters,
        annual_yield=0.04,
    ) * yield_change

    assert result == pytest.approx(expected)


def test_invalid_include_convexity_type(
    bond_parameters: BondParameters,
) -> None:
    """The convexity flag must be a boolean."""

    with pytest.raises(TypeError):
        estimate_percentage_price_change(
            parameters=bond_parameters,
            annual_yield=0.04,
            yield_change=0.01,
            include_convexity="yes",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "yield_change",
    [nan, inf, -inf],
)
def test_non_finite_yield_change_raises_value_error(
    bond_parameters: BondParameters,
    yield_change: float,
) -> None:
    """Yield changes must contain finite values."""

    with pytest.raises(ValueError):
        estimate_price_change(
            parameters=bond_parameters,
            annual_yield=0.04,
            yield_change=yield_change,
        )


@pytest.mark.parametrize(
    "annual_yield",
    [-2.0, -3.0],
)
def test_invalid_annual_yield_raises_value_error(
    bond_parameters: BondParameters,
    annual_yield: float,
) -> None:
    """Invalid discount factors should be rejected."""

    with pytest.raises(ValueError):
        calculate_macaulay_duration(
            parameters=bond_parameters,
            annual_yield=annual_yield,
        )


def test_invalid_parameters_object_raises_type_error() -> None:
    """Duration calculations require BondParameters."""

    with pytest.raises(TypeError):
        calculate_macaulay_duration(
            parameters="invalid",  # type: ignore[arg-type]
            annual_yield=0.04,
        )