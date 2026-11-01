"""Tests for fixed-income yield calculations."""

from dataclasses import FrozenInstanceError

import pytest

from src.bond_pricing import (
    BondParameters,
    price_coupon_bond,
    price_zero_coupon_bond,
)
from src.yield_metrics import (
    BondYieldMetrics,
    calculate_current_yield,
    calculate_yield_metrics,
    calculate_yield_to_maturity,
    effective_to_nominal_annual_rate,
    nominal_to_effective_annual_rate,
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


def test_current_yield(
    bond_parameters: BondParameters,
) -> None:
    """Current yield should equal annual coupon divided by price."""

    result = calculate_current_yield(
        parameters=bond_parameters,
        market_price=950.0,
    )

    assert result == pytest.approx(50.0 / 950.0)


def test_ytm_recovers_known_market_yield(
    bond_parameters: BondParameters,
) -> None:
    """YTM should recover the rate used to produce a price."""

    known_yield = 0.04

    market_price = price_coupon_bond(
        parameters=bond_parameters,
        annual_yield=known_yield,
    )

    calculated_yield = calculate_yield_to_maturity(
        parameters=bond_parameters,
        market_price=market_price,
    )

    assert calculated_yield == pytest.approx(
        known_yield,
        abs=1e-10,
    )


def test_par_bond_ytm_equals_coupon_rate(
    bond_parameters: BondParameters,
) -> None:
    """A bond priced at par should have YTM equal to its coupon."""

    result = calculate_yield_to_maturity(
        parameters=bond_parameters,
        market_price=1_000.0,
    )

    assert result == pytest.approx(0.05, abs=1e-10)


def test_discount_bond_ytm_exceeds_coupon_rate(
    bond_parameters: BondParameters,
) -> None:
    """A discount bond should have YTM above its coupon rate."""

    result = calculate_yield_to_maturity(
        parameters=bond_parameters,
        market_price=950.0,
    )

    assert result > bond_parameters.annual_coupon_rate


def test_premium_bond_ytm_is_below_coupon_rate(
    bond_parameters: BondParameters,
) -> None:
    """A premium bond should have YTM below its coupon rate."""

    result = calculate_yield_to_maturity(
        parameters=bond_parameters,
        market_price=1_050.0,
    )

    assert result < bond_parameters.annual_coupon_rate


def test_ytm_reconstructs_market_price(
    bond_parameters: BondParameters,
) -> None:
    """Calculated YTM should reproduce the observed price."""

    market_price = 950.0

    calculated_yield = calculate_yield_to_maturity(
        parameters=bond_parameters,
        market_price=market_price,
    )

    reconstructed_price = price_coupon_bond(
        parameters=bond_parameters,
        annual_yield=calculated_yield,
    )

    assert reconstructed_price == pytest.approx(
        market_price,
        abs=1e-8,
    )


def test_zero_coupon_bond_ytm() -> None:
    """YTM should also work for a zero-coupon bond."""

    parameters = BondParameters(
        face_value=1_000.0,
        annual_coupon_rate=0.0,
        years_to_maturity=5.0,
        payments_per_year=2,
    )

    market_price = price_zero_coupon_bond(
        face_value=1_000.0,
        annual_yield=0.04,
        years_to_maturity=5.0,
        compounding_frequency=2,
    )

    result = calculate_yield_to_maturity(
        parameters=parameters,
        market_price=market_price,
    )

    assert result == pytest.approx(0.04, abs=1e-10)


def test_nominal_to_effective_annual_rate() -> None:
    """A 6% nominal semiannual rate should become 6.09%."""

    result = nominal_to_effective_annual_rate(
        nominal_rate=0.06,
        compounding_frequency=2,
    )

    assert result == pytest.approx(0.0609)


def test_effective_to_nominal_annual_rate() -> None:
    """A 6.09% effective rate should become 6% nominal."""

    result = effective_to_nominal_annual_rate(
        effective_rate=0.0609,
        compounding_frequency=2,
    )

    assert result == pytest.approx(0.06)


@pytest.mark.parametrize(
    "compounding_frequency",
    [1, 2, 4, 12],
)
def test_rate_conversion_round_trip(
    compounding_frequency: int,
) -> None:
    """Converting a rate in both directions should recover it."""

    nominal_rate = 0.075

    effective_rate = nominal_to_effective_annual_rate(
        nominal_rate=nominal_rate,
        compounding_frequency=compounding_frequency,
    )

    reconstructed_nominal_rate = (
        effective_to_nominal_annual_rate(
            effective_rate=effective_rate,
            compounding_frequency=compounding_frequency,
        )
    )

    assert reconstructed_nominal_rate == pytest.approx(
        nominal_rate
    )


def test_calculate_yield_metrics_returns_dataclass(
    bond_parameters: BondParameters,
) -> None:
    """The combined calculation should return BondYieldMetrics."""

    result = calculate_yield_metrics(
        parameters=bond_parameters,
        market_price=950.0,
    )

    assert isinstance(result, BondYieldMetrics)


def test_calculate_yield_metrics_values(
    bond_parameters: BondParameters,
) -> None:
    """The combined calculation should return consistent values."""

    result = calculate_yield_metrics(
        parameters=bond_parameters,
        market_price=950.0,
    )

    expected_effective_yield = (
        1.0
        + result.yield_to_maturity
        / bond_parameters.payments_per_year
    ) ** bond_parameters.payments_per_year - 1.0

    assert result.current_yield == pytest.approx(
        50.0 / 950.0
    )
    assert result.yield_to_maturity > 0.05
    assert result.effective_annual_yield == pytest.approx(
        expected_effective_yield
    )


def test_yield_metrics_are_immutable(
    bond_parameters: BondParameters,
) -> None:
    """Calculated yield metrics should be immutable."""

    result = calculate_yield_metrics(
        parameters=bond_parameters,
        market_price=950.0,
    )

    with pytest.raises(FrozenInstanceError):
        result.current_yield = 0.10  # type: ignore[misc]


@pytest.mark.parametrize("market_price", [0.0, -1.0])
def test_invalid_market_price_for_current_yield(
    bond_parameters: BondParameters,
    market_price: float,
) -> None:
    """Current yield should reject non-positive prices."""

    with pytest.raises(ValueError):
        calculate_current_yield(
            parameters=bond_parameters,
            market_price=market_price,
        )


@pytest.mark.parametrize("market_price", [0.0, -1.0])
def test_invalid_market_price_for_ytm(
    bond_parameters: BondParameters,
    market_price: float,
) -> None:
    """YTM should reject non-positive market prices."""

    with pytest.raises(ValueError):
        calculate_yield_to_maturity(
            parameters=bond_parameters,
            market_price=market_price,
        )


@pytest.mark.parametrize(
    "compounding_frequency",
    [0, -1],
)
def test_invalid_compounding_frequency(
    compounding_frequency: int,
) -> None:
    """Compounding frequency must be positive."""

    with pytest.raises(ValueError):
        nominal_to_effective_annual_rate(
            nominal_rate=0.05,
            compounding_frequency=compounding_frequency,
        )


def test_non_integer_compounding_frequency() -> None:
    """Compounding frequency must be an integer."""

    with pytest.raises(TypeError):
        nominal_to_effective_annual_rate(
            nominal_rate=0.05,
            compounding_frequency=2.5,  # type: ignore[arg-type]
        )


def test_invalid_nominal_rate() -> None:
    """A nominal rate cannot create a non-positive factor."""

    with pytest.raises(ValueError):
        nominal_to_effective_annual_rate(
            nominal_rate=-2.0,
            compounding_frequency=2,
        )


def test_invalid_effective_rate() -> None:
    """An effective rate must be greater than negative 100%."""

    with pytest.raises(ValueError):
        effective_to_nominal_annual_rate(
            effective_rate=-1.0,
            compounding_frequency=2,
        )


def test_invalid_tolerance(
    bond_parameters: BondParameters,
) -> None:
    """Numerical tolerance must be positive."""

    with pytest.raises(ValueError):
        calculate_yield_to_maturity(
            parameters=bond_parameters,
            market_price=950.0,
            tolerance=0.0,
        )


@pytest.mark.parametrize(
    "maximum_iterations",
    [0, -1],
)
def test_invalid_maximum_iterations(
    bond_parameters: BondParameters,
    maximum_iterations: int,
) -> None:
    """The root solver requires a positive iteration limit."""

    with pytest.raises(ValueError):
        calculate_yield_to_maturity(
            parameters=bond_parameters,
            market_price=950.0,
            maximum_iterations=maximum_iterations,
        )


def test_non_integer_maximum_iterations(
    bond_parameters: BondParameters,
) -> None:
    """Maximum iterations must be an integer."""

    with pytest.raises(TypeError):
        calculate_yield_to_maturity(
            parameters=bond_parameters,
            market_price=950.0,
            maximum_iterations=100.5,  # type: ignore[arg-type]
        )


def test_invalid_parameters_object() -> None:
    """Yield calculations require BondParameters."""

    with pytest.raises(TypeError):
        calculate_current_yield(
            parameters="invalid",  # type: ignore[arg-type]
            market_price=950.0,
        )