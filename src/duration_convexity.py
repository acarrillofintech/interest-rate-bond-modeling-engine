"""Interest-rate sensitivity measures for fixed-rate bonds.

This module calculates Macaulay duration, modified duration,
convexity, DV01, and estimated bond-price changes.
"""

from dataclasses import dataclass
from math import isfinite

from src.bond_pricing import (
    BondParameters,
    generate_cash_flows,
    price_coupon_bond,
)


@dataclass(frozen=True)
class DurationConvexityMetrics:
    """Interest-rate sensitivity measures for a bond."""

    bond_price: float
    macaulay_duration: float
    modified_duration: float
    convexity: float
    dv01: float


def _validate_real_number(
    value: float,
    name: str,
) -> None:
    """Validate that a value is a finite real number."""

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(f"{name} must be a real number.")

    if not isfinite(float(value)):
        raise ValueError(f"{name} must be finite.")


def _validate_annual_yield(
    annual_yield: float,
    payments_per_year: int,
) -> None:
    """Validate an annual nominal yield."""

    _validate_real_number(
        value=annual_yield,
        name="annual_yield",
    )

    if annual_yield <= -payments_per_year:
        raise ValueError(
            "annual_yield produces a non-positive "
            "periodic discount factor."
        )


def calculate_macaulay_duration(
    parameters: BondParameters,
    annual_yield: float,
) -> float:
    """Calculate Macaulay duration measured in years."""

    if not isinstance(parameters, BondParameters):
        raise TypeError(
            "parameters must be a BondParameters instance."
        )

    _validate_annual_yield(
        annual_yield=annual_yield,
        payments_per_year=parameters.payments_per_year,
    )

    bond_price = price_coupon_bond(
        parameters=parameters,
        annual_yield=annual_yield,
    )

    periodic_yield = (
        annual_yield / parameters.payments_per_year
    )

    weighted_present_value = 0.0

    for cash_flow in generate_cash_flows(parameters):
        discount_factor = (
            1.0 + periodic_yield
        ) ** cash_flow.payment_number

        present_value = (
            cash_flow.total_payment / discount_factor
        )

        weighted_present_value += (
            cash_flow.time_in_years * present_value
        )

    return weighted_present_value / bond_price


def calculate_modified_duration(
    parameters: BondParameters,
    annual_yield: float,
) -> float:
    """Calculate modified duration measured in years."""

    macaulay_duration = calculate_macaulay_duration(
        parameters=parameters,
        annual_yield=annual_yield,
    )

    periodic_yield = (
        annual_yield / parameters.payments_per_year
    )

    return macaulay_duration / (
        1.0 + periodic_yield
    )


def calculate_convexity(
    parameters: BondParameters,
    annual_yield: float,
) -> float:
    """Calculate annualized bond convexity."""

    if not isinstance(parameters, BondParameters):
        raise TypeError(
            "parameters must be a BondParameters instance."
        )

    _validate_annual_yield(
        annual_yield=annual_yield,
        payments_per_year=parameters.payments_per_year,
    )

    bond_price = price_coupon_bond(
        parameters=parameters,
        annual_yield=annual_yield,
    )

    payments_per_year = parameters.payments_per_year
    periodic_yield = annual_yield / payments_per_year

    convexity_numerator = 0.0

    for cash_flow in generate_cash_flows(parameters):
        period = cash_flow.payment_number

        convexity_numerator += (
            cash_flow.total_payment
            * period
            * (period + 1)
            / (
                (1.0 + periodic_yield)
                ** (period + 2)
            )
        )

    return convexity_numerator / (
        bond_price * payments_per_year**2
    )


def calculate_dv01(
    parameters: BondParameters,
    annual_yield: float,
) -> float:
    """Calculate the approximate price change for one basis point."""

    bond_price = price_coupon_bond(
        parameters=parameters,
        annual_yield=annual_yield,
    )

    modified_duration = calculate_modified_duration(
        parameters=parameters,
        annual_yield=annual_yield,
    )

    one_basis_point = 0.0001

    return (
        bond_price
        * modified_duration
        * one_basis_point
    )


def estimate_percentage_price_change(
    parameters: BondParameters,
    annual_yield: float,
    yield_change: float,
    include_convexity: bool = True,
) -> float:
    """Estimate the proportional price change from a yield movement."""

    _validate_real_number(
        value=yield_change,
        name="yield_change",
    )

    if not isinstance(include_convexity, bool):
        raise TypeError(
            "include_convexity must be a boolean."
        )

    modified_duration = calculate_modified_duration(
        parameters=parameters,
        annual_yield=annual_yield,
    )

    estimated_change = (
        -modified_duration * yield_change
    )

    if include_convexity:
        convexity = calculate_convexity(
            parameters=parameters,
            annual_yield=annual_yield,
        )

        estimated_change += (
            0.5 * convexity * yield_change**2
        )

    return estimated_change


def estimate_price_change(
    parameters: BondParameters,
    annual_yield: float,
    yield_change: float,
    include_convexity: bool = True,
) -> float:
    """Estimate the monetary change in the bond price."""

    bond_price = price_coupon_bond(
        parameters=parameters,
        annual_yield=annual_yield,
    )

    percentage_change = (
        estimate_percentage_price_change(
            parameters=parameters,
            annual_yield=annual_yield,
            yield_change=yield_change,
            include_convexity=include_convexity,
        )
    )

    return bond_price * percentage_change


def estimate_new_bond_price(
    parameters: BondParameters,
    annual_yield: float,
    yield_change: float,
    include_convexity: bool = True,
) -> float:
    """Estimate the bond price after a change in yield."""

    current_price = price_coupon_bond(
        parameters=parameters,
        annual_yield=annual_yield,
    )

    price_change = estimate_price_change(
        parameters=parameters,
        annual_yield=annual_yield,
        yield_change=yield_change,
        include_convexity=include_convexity,
    )

    return current_price + price_change


def calculate_duration_convexity_metrics(
    parameters: BondParameters,
    annual_yield: float,
) -> DurationConvexityMetrics:
    """Calculate all principal interest-rate sensitivity metrics."""

    bond_price = price_coupon_bond(
        parameters=parameters,
        annual_yield=annual_yield,
    )

    macaulay_duration = calculate_macaulay_duration(
        parameters=parameters,
        annual_yield=annual_yield,
    )

    modified_duration = calculate_modified_duration(
        parameters=parameters,
        annual_yield=annual_yield,
    )

    convexity = calculate_convexity(
        parameters=parameters,
        annual_yield=annual_yield,
    )

    dv01 = calculate_dv01(
        parameters=parameters,
        annual_yield=annual_yield,
    )

    return DurationConvexityMetrics(
        bond_price=bond_price,
        macaulay_duration=macaulay_duration,
        modified_duration=modified_duration,
        convexity=convexity,
        dv01=dv01,
    )


def main() -> None:
    """Run an example duration and convexity analysis."""

    parameters = BondParameters(
        face_value=1_000.0,
        annual_coupon_rate=0.05,
        years_to_maturity=5.0,
        payments_per_year=2,
    )

    annual_yield = 0.04
    yield_change = 0.01

    metrics = calculate_duration_convexity_metrics(
        parameters=parameters,
        annual_yield=annual_yield,
    )

    estimated_price = estimate_new_bond_price(
        parameters=parameters,
        annual_yield=annual_yield,
        yield_change=yield_change,
        include_convexity=True,
    )

    exact_price = price_coupon_bond(
        parameters=parameters,
        annual_yield=(
            annual_yield + yield_change
        ),
    )

    print("Bond duration and convexity")
    print(
        f"Current yield: "
        f"{annual_yield:.2%}"
    )
    print(
        f"Yield change: "
        f"{yield_change:+.2%}"
    )

    print("\nInterest-rate sensitivity")
    print(
        f"Bond price: "
        f"${metrics.bond_price:,.2f}"
    )
    print(
        f"Macaulay duration: "
        f"{metrics.macaulay_duration:.6f} years"
    )
    print(
        f"Modified duration: "
        f"{metrics.modified_duration:.6f}"
    )
    print(
        f"Convexity: "
        f"{metrics.convexity:.6f}"
    )
    print(
        f"DV01: "
        f"${metrics.dv01:,.4f}"
    )

    print("\nPrice impact of the yield change")
    print(
        f"Estimated price: "
        f"${estimated_price:,.2f}"
    )
    print(
        f"Exact repriced value: "
        f"${exact_price:,.2f}"
    )
    print(
        f"Approximation error: "
        f"${estimated_price - exact_price:,.4f}"
    )


if __name__ == "__main__":
    main()