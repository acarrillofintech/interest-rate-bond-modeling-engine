"""Yield calculations for fixed-income securities.

This module calculates current yield, yield to maturity, and effective
annual yield for fixed-rate bonds.
"""

from dataclasses import dataclass
from math import isfinite

from scipy.optimize import brentq

from src.bond_pricing import (
    BondParameters,
    price_coupon_bond,
)


@dataclass(frozen=True)
class BondYieldMetrics:
    """Calculated yield measures for a fixed-rate bond."""

    current_yield: float
    yield_to_maturity: float
    effective_annual_yield: float


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


def _validate_frequency(
    compounding_frequency: int,
) -> None:
    """Validate the number of compounding periods per year."""

    if isinstance(
        compounding_frequency,
        bool,
    ) or not isinstance(
        compounding_frequency,
        int,
    ):
        raise TypeError(
            "compounding_frequency must be an integer."
        )

    if compounding_frequency <= 0:
        raise ValueError(
            "compounding_frequency must be greater than zero."
        )


def calculate_current_yield(
    parameters: BondParameters,
    market_price: float,
) -> float:
    """Calculate annual coupon income divided by market price."""

    if not isinstance(parameters, BondParameters):
        raise TypeError(
            "parameters must be a BondParameters instance."
        )

    _validate_real_number(
        value=market_price,
        name="market_price",
    )

    if market_price <= 0.0:
        raise ValueError(
            "market_price must be greater than zero."
        )

    annual_coupon = (
        parameters.face_value
        * parameters.annual_coupon_rate
    )

    return annual_coupon / market_price


def nominal_to_effective_annual_rate(
    nominal_rate: float,
    compounding_frequency: int,
) -> float:
    """Convert a nominal annual rate to an effective annual rate."""

    _validate_real_number(
        value=nominal_rate,
        name="nominal_rate",
    )
    _validate_frequency(compounding_frequency)

    if nominal_rate <= -compounding_frequency:
        raise ValueError(
            "nominal_rate produces a non-positive "
            "periodic growth factor."
        )

    periodic_rate = (
        nominal_rate / compounding_frequency
    )

    return (
        1.0 + periodic_rate
    ) ** compounding_frequency - 1.0


def effective_to_nominal_annual_rate(
    effective_rate: float,
    compounding_frequency: int,
) -> float:
    """Convert an effective annual rate to a nominal annual rate."""

    _validate_real_number(
        value=effective_rate,
        name="effective_rate",
    )
    _validate_frequency(compounding_frequency)

    if effective_rate <= -1.0:
        raise ValueError(
            "effective_rate must be greater than -100%."
        )

    periodic_rate = (
        (1.0 + effective_rate)
        ** (1.0 / compounding_frequency)
        - 1.0
    )

    return periodic_rate * compounding_frequency


def calculate_yield_to_maturity(
    parameters: BondParameters,
    market_price: float,
    tolerance: float = 1e-12,
    maximum_iterations: int = 1_000,
) -> float:
    """Calculate nominal annual YTM using a numerical root solver."""

    if not isinstance(parameters, BondParameters):
        raise TypeError(
            "parameters must be a BondParameters instance."
        )

    _validate_real_number(
        value=market_price,
        name="market_price",
    )
    _validate_real_number(
        value=tolerance,
        name="tolerance",
    )

    if market_price <= 0.0:
        raise ValueError(
            "market_price must be greater than zero."
        )

    if tolerance <= 0.0:
        raise ValueError(
            "tolerance must be greater than zero."
        )

    if isinstance(
        maximum_iterations,
        bool,
    ) or not isinstance(
        maximum_iterations,
        int,
    ):
        raise TypeError(
            "maximum_iterations must be an integer."
        )

    if maximum_iterations <= 0:
        raise ValueError(
            "maximum_iterations must be greater than zero."
        )

    def pricing_error(annual_yield: float) -> float:
        calculated_price = price_coupon_bond(
            parameters=parameters,
            annual_yield=annual_yield,
        )

        return calculated_price - market_price

    lower_bound = (
        -0.99 * parameters.payments_per_year
    )
    upper_bound = 1.0

    while pricing_error(upper_bound) > 0.0:
        upper_bound = upper_bound * 2.0 + 0.10

        if upper_bound > 1_000_000.0:
            raise RuntimeError(
                "Unable to find an upper bound for YTM."
            )

    try:
        return float(
            brentq(
                pricing_error,
                lower_bound,
                upper_bound,
                xtol=tolerance,
                maxiter=maximum_iterations,
            )
        )
    except ValueError as error:
        raise RuntimeError(
            "Yield to maturity could not be calculated "
            "for the supplied bond and market price."
        ) from error


def calculate_yield_metrics(
    parameters: BondParameters,
    market_price: float,
) -> BondYieldMetrics:
    """Calculate the principal yield measures of a bond."""

    yield_to_maturity = calculate_yield_to_maturity(
        parameters=parameters,
        market_price=market_price,
    )

    current_yield = calculate_current_yield(
        parameters=parameters,
        market_price=market_price,
    )

    effective_annual_yield = (
        nominal_to_effective_annual_rate(
            nominal_rate=yield_to_maturity,
            compounding_frequency=(
                parameters.payments_per_year
            ),
        )
    )

    return BondYieldMetrics(
        current_yield=current_yield,
        yield_to_maturity=yield_to_maturity,
        effective_annual_yield=effective_annual_yield,
    )


def main() -> None:
    """Run an example bond-yield analysis."""

    parameters = BondParameters(
        face_value=1_000.0,
        annual_coupon_rate=0.05,
        years_to_maturity=5.0,
        payments_per_year=2,
    )

    market_price = 950.0

    metrics = calculate_yield_metrics(
        parameters=parameters,
        market_price=market_price,
    )

    reconstructed_price = price_coupon_bond(
        parameters=parameters,
        annual_yield=metrics.yield_to_maturity,
    )

    print("Fixed-income yield analysis")
    print(
        f"Face value: "
        f"${parameters.face_value:,.2f}"
    )
    print(
        f"Market price: "
        f"${market_price:,.2f}"
    )
    print(
        f"Annual coupon rate: "
        f"{parameters.annual_coupon_rate:.2%}"
    )
    print(
        f"Years to maturity: "
        f"{parameters.years_to_maturity:.2f}"
    )

    print("\nYield metrics")
    print(
        f"Current yield: "
        f"{metrics.current_yield:.4%}"
    )
    print(
        f"Yield to maturity: "
        f"{metrics.yield_to_maturity:.4%}"
    )
    print(
        f"Effective annual yield: "
        f"{metrics.effective_annual_yield:.4%}"
    )

    print("\nPricing verification")
    print(
        f"Price reconstructed from YTM: "
        f"${reconstructed_price:,.2f}"
    )
    print(
        f"Pricing difference: "
        f"${reconstructed_price - market_price:,.10f}"
    )


if __name__ == "__main__":
    main()