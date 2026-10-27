"""Tests for the fixed-income bond pricing module."""

from dataclasses import FrozenInstanceError

import pytest

from src.bond_pricing import (
    BondCashFlow,
    BondParameters,
    calculate_bond_price,
    generate_cash_flows,
    price_coupon_bond,
    price_zero_coupon_bond,
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


def test_number_of_payments(
    bond_parameters: BondParameters,
) -> None:
    """A five-year semiannual bond should have ten payments."""

    assert bond_parameters.number_of_payments == 10


def test_coupon_payment(
    bond_parameters: BondParameters,
) -> None:
    """Each semiannual coupon should equal twenty-five dollars."""

    assert bond_parameters.coupon_payment == pytest.approx(25.0)


def test_generate_cash_flows_returns_tuple(
    bond_parameters: BondParameters,
) -> None:
    """Cash flows should be returned as an immutable tuple."""

    cash_flows = generate_cash_flows(bond_parameters)

    assert isinstance(cash_flows, tuple)


def test_generate_correct_number_of_cash_flows(
    bond_parameters: BondParameters,
) -> None:
    """The schedule should contain one cash flow per period."""

    cash_flows = generate_cash_flows(bond_parameters)

    assert len(cash_flows) == 10


def test_cash_flow_is_dataclass(
    bond_parameters: BondParameters,
) -> None:
    """Each scheduled payment should be a BondCashFlow."""

    cash_flows = generate_cash_flows(bond_parameters)

    assert isinstance(cash_flows[0], BondCashFlow)


def test_first_cash_flow(
    bond_parameters: BondParameters,
) -> None:
    """The first payment should contain only the coupon."""

    first_cash_flow = generate_cash_flows(bond_parameters)[0]

    assert first_cash_flow.payment_number == 1
    assert first_cash_flow.time_in_years == pytest.approx(0.5)
    assert first_cash_flow.coupon_payment == pytest.approx(25.0)
    assert first_cash_flow.principal_payment == pytest.approx(0.0)
    assert first_cash_flow.total_payment == pytest.approx(25.0)


def test_final_cash_flow(
    bond_parameters: BondParameters,
) -> None:
    """The final payment should include coupon and principal."""

    final_cash_flow = generate_cash_flows(bond_parameters)[-1]

    assert final_cash_flow.payment_number == 10
    assert final_cash_flow.time_in_years == pytest.approx(5.0)
    assert final_cash_flow.coupon_payment == pytest.approx(25.0)
    assert final_cash_flow.principal_payment == pytest.approx(
        1_000.0
    )
    assert final_cash_flow.total_payment == pytest.approx(
        1_025.0
    )


def test_zero_coupon_bond_price() -> None:
    """The zero-coupon price should match its known present value."""

    price = price_zero_coupon_bond(
        face_value=1_000.0,
        annual_yield=0.04,
        years_to_maturity=5.0,
        compounding_frequency=2,
    )

    assert price == pytest.approx(820.3482998752)


def test_coupon_bond_price(
    bond_parameters: BondParameters,
) -> None:
    """The coupon bond should match its known present value."""

    price = price_coupon_bond(
        parameters=bond_parameters,
        annual_yield=0.04,
    )

    assert price == pytest.approx(1_044.9129250312)


def test_bond_prices_at_par_when_coupon_equals_yield() -> None:
    """A bond should trade at par when coupon rate equals YTM."""

    parameters = BondParameters(
        face_value=1_000.0,
        annual_coupon_rate=0.05,
        years_to_maturity=5.0,
        payments_per_year=2,
    )

    price = price_coupon_bond(
        parameters=parameters,
        annual_yield=0.05,
    )

    assert price == pytest.approx(1_000.0)


def test_bond_trades_at_premium(
    bond_parameters: BondParameters,
) -> None:
    """The price should exceed par when coupon exceeds YTM."""

    price = price_coupon_bond(
        parameters=bond_parameters,
        annual_yield=0.04,
    )

    assert price > bond_parameters.face_value


def test_bond_trades_at_discount(
    bond_parameters: BondParameters,
) -> None:
    """The price should be below par when YTM exceeds coupon."""

    price = price_coupon_bond(
        parameters=bond_parameters,
        annual_yield=0.06,
    )

    assert price < bond_parameters.face_value


def test_generic_price_function(
    bond_parameters: BondParameters,
) -> None:
    """The generic function should return the coupon-bond price."""

    direct_price = price_coupon_bond(
        parameters=bond_parameters,
        annual_yield=0.04,
    )

    generic_price = calculate_bond_price(
        parameters=bond_parameters,
        annual_yield=0.04,
    )

    assert generic_price == pytest.approx(direct_price)


def test_zero_coupon_bond_matches_coupon_function() -> None:
    """Both pricing approaches should agree for a zero-coupon bond."""

    parameters = BondParameters(
        face_value=1_000.0,
        annual_coupon_rate=0.0,
        years_to_maturity=5.0,
        payments_per_year=2,
    )

    zero_coupon_price = price_zero_coupon_bond(
        face_value=1_000.0,
        annual_yield=0.04,
        years_to_maturity=5.0,
        compounding_frequency=2,
    )

    coupon_function_price = price_coupon_bond(
        parameters=parameters,
        annual_yield=0.04,
    )

    assert coupon_function_price == pytest.approx(
        zero_coupon_price
    )


def test_bond_parameters_are_immutable(
    bond_parameters: BondParameters,
) -> None:
    """Bond parameters should not change after creation."""

    with pytest.raises(FrozenInstanceError):
        bond_parameters.face_value = 2_000.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("face_value", 0.0),
        ("face_value", -1_000.0),
        ("annual_coupon_rate", -0.01),
        ("years_to_maturity", 0.0),
        ("years_to_maturity", -1.0),
    ],
)
def test_invalid_financial_parameters_raise_value_error(
    field_name: str,
    invalid_value: float,
) -> None:
    """Invalid financial values should raise ValueError."""

    values = {
        "face_value": 1_000.0,
        "annual_coupon_rate": 0.05,
        "years_to_maturity": 5.0,
        "payments_per_year": 2,
    }

    values[field_name] = invalid_value

    with pytest.raises(ValueError):
        BondParameters(**values)


@pytest.mark.parametrize("payments_per_year", [0, -1])
def test_invalid_payment_frequency_raises_value_error(
    payments_per_year: int,
) -> None:
    """Non-positive payment frequencies should be rejected."""

    with pytest.raises(ValueError):
        BondParameters(
            face_value=1_000.0,
            annual_coupon_rate=0.05,
            years_to_maturity=5.0,
            payments_per_year=payments_per_year,
        )


def test_non_integer_payment_frequency_raises_type_error() -> None:
    """Payment frequency must be represented by an integer."""

    with pytest.raises(TypeError):
        BondParameters(
            face_value=1_000.0,
            annual_coupon_rate=0.05,
            years_to_maturity=5.0,
            payments_per_year=2.5,  # type: ignore[arg-type]
        )


def test_fractional_coupon_period_raises_value_error() -> None:
    """Maturity must contain a whole number of coupon periods."""

    with pytest.raises(ValueError):
        BondParameters(
            face_value=1_000.0,
            annual_coupon_rate=0.05,
            years_to_maturity=1.2,
            payments_per_year=2,
        )


def test_invalid_yield_raises_value_error(
    bond_parameters: BondParameters,
) -> None:
    """A yield producing a non-positive discount factor is invalid."""

    with pytest.raises(ValueError):
        price_coupon_bond(
            parameters=bond_parameters,
            annual_yield=-2.0,
        )


def test_invalid_parameters_object_raises_type_error() -> None:
    """Pricing requires a valid BondParameters object."""

    with pytest.raises(TypeError):
        price_coupon_bond(
            parameters="invalid",  # type: ignore[arg-type]
            annual_yield=0.04,
        )