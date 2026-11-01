"""Fixed-income bond pricing tools.

This module provides functions for generating bond cash flows and
calculating the present value of zero-coupon and fixed-coupon bonds.
"""

from dataclasses import dataclass
from math import isclose, isfinite


@dataclass(frozen=True)
class BondParameters:
    """Financial parameters of a fixed-rate bond."""

    face_value: float
    annual_coupon_rate: float
    years_to_maturity: float
    payments_per_year: int = 2

    def __post_init__(self) -> None:
        """Validate the bond parameters."""

        numeric_values = {
            "face_value": self.face_value,
            "annual_coupon_rate": self.annual_coupon_rate,
            "years_to_maturity": self.years_to_maturity,
        }

        for name, value in numeric_values.items():
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise TypeError(f"{name} must be a real number.")

            if not isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")

        if self.face_value <= 0.0:
            raise ValueError("face_value must be greater than zero.")

        if self.annual_coupon_rate < 0.0:
            raise ValueError(
                "annual_coupon_rate cannot be negative."
            )

        if self.years_to_maturity <= 0.0:
            raise ValueError(
                "years_to_maturity must be greater than zero."
            )

        if isinstance(self.payments_per_year, bool) or not isinstance(
            self.payments_per_year,
            int,
        ):
            raise TypeError("payments_per_year must be an integer.")

        if self.payments_per_year <= 0:
            raise ValueError(
                "payments_per_year must be greater than zero."
            )

        periods = (
            self.years_to_maturity
            * self.payments_per_year
        )

        if not isclose(
            periods,
            round(periods),
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise ValueError(
                "years_to_maturity must produce a whole number "
                "of coupon periods."
            )

    @property
    def number_of_payments(self) -> int:
        """Return the total number of coupon payments."""

        return round(
            self.years_to_maturity
            * self.payments_per_year
        )

    @property
    def coupon_payment(self) -> float:
        """Return the coupon paid in each period."""

        return (
            self.face_value
            * self.annual_coupon_rate
            / self.payments_per_year
        )


@dataclass(frozen=True)
class BondCashFlow:
    """One scheduled bond cash flow."""

    payment_number: int
    time_in_years: float
    coupon_payment: float
    principal_payment: float

    @property
    def total_payment(self) -> float:
        """Return coupon plus principal for this payment."""

        return self.coupon_payment + self.principal_payment


def _validate_annual_yield(
    annual_yield: float,
    payments_per_year: int,
) -> None:
    """Validate an annual yield used for discounting."""

    if isinstance(annual_yield, bool) or not isinstance(
        annual_yield,
        (int, float),
    ):
        raise TypeError("annual_yield must be a real number.")

    if not isfinite(float(annual_yield)):
        raise ValueError("annual_yield must be finite.")

    if annual_yield <= -payments_per_year:
        raise ValueError(
            "annual_yield produces a non-positive "
            "periodic discount factor."
        )


def generate_cash_flows(
    parameters: BondParameters,
) -> tuple[BondCashFlow, ...]:
    """Generate all scheduled cash flows for a fixed-rate bond."""

    if not isinstance(parameters, BondParameters):
        raise TypeError(
            "parameters must be a BondParameters instance."
        )

    cash_flows: list[BondCashFlow] = []

    for payment_number in range(
        1,
        parameters.number_of_payments + 1,
    ):
        is_final_payment = (
            payment_number
            == parameters.number_of_payments
        )

        principal_payment = (
            parameters.face_value
            if is_final_payment
            else 0.0
        )

        cash_flows.append(
            BondCashFlow(
                payment_number=payment_number,
                time_in_years=(
                    payment_number
                    / parameters.payments_per_year
                ),
                coupon_payment=parameters.coupon_payment,
                principal_payment=principal_payment,
            )
        )

    return tuple(cash_flows)


def price_zero_coupon_bond(
    face_value: float,
    annual_yield: float,
    years_to_maturity: float,
    compounding_frequency: int = 1,
) -> float:
    """Calculate the price of a zero-coupon bond."""

    parameters = BondParameters(
        face_value=face_value,
        annual_coupon_rate=0.0,
        years_to_maturity=years_to_maturity,
        payments_per_year=compounding_frequency,
    )

    _validate_annual_yield(
        annual_yield=annual_yield,
        payments_per_year=compounding_frequency,
    )

    periodic_yield = (
        annual_yield / compounding_frequency
    )

    return face_value / (
        (1.0 + periodic_yield)
        ** parameters.number_of_payments
    )


def price_coupon_bond(
    parameters: BondParameters,
    annual_yield: float,
) -> float:
    """Calculate the present value of a fixed-coupon bond."""

    if not isinstance(parameters, BondParameters):
        raise TypeError(
            "parameters must be a BondParameters instance."
        )

    _validate_annual_yield(
        annual_yield=annual_yield,
        payments_per_year=parameters.payments_per_year,
    )

    periodic_yield = (
        annual_yield / parameters.payments_per_year
    )

    price = 0.0

    for cash_flow in generate_cash_flows(parameters):
        discount_factor = (
            1.0 + periodic_yield
        ) ** cash_flow.payment_number

        price += (
            cash_flow.total_payment / discount_factor
        )

    return price


def calculate_bond_price(
    parameters: BondParameters,
    annual_yield: float,
) -> float:
    """Calculate a bond price using its scheduled cash flows."""

    return price_coupon_bond(
        parameters=parameters,
        annual_yield=annual_yield,
    )


def main() -> None:
    """Run an example fixed-income valuation."""

    parameters = BondParameters(
        face_value=1_000.0,
        annual_coupon_rate=0.05,
        years_to_maturity=5.0,
        payments_per_year=2,
    )

    annual_yield = 0.04

    coupon_bond_price = price_coupon_bond(
        parameters=parameters,
        annual_yield=annual_yield,
    )

    zero_coupon_price = price_zero_coupon_bond(
        face_value=1_000.0,
        annual_yield=annual_yield,
        years_to_maturity=5.0,
        compounding_frequency=2,
    )

    print("Fixed-income bond pricing")
    print(f"Face value: ${parameters.face_value:,.2f}")
    print(
        f"Annual coupon rate: "
        f"{parameters.annual_coupon_rate:.2%}"
    )
    print(
        f"Yield to maturity: {annual_yield:.2%}"
    )
    print(
        f"Years to maturity: "
        f"{parameters.years_to_maturity:.2f}"
    )
    print(
        f"Payments per year: "
        f"{parameters.payments_per_year}"
    )

    print("\nBond values")
    print(
        f"Fixed-coupon bond: "
        f"${coupon_bond_price:,.2f}"
    )
    print(
        f"Zero-coupon bond: "
        f"${zero_coupon_price:,.2f}"
    )

    print("\nCash-flow schedule")

    for cash_flow in generate_cash_flows(parameters):
        print(
            f"Period {cash_flow.payment_number:>2}: "
            f"time={cash_flow.time_in_years:>4.1f} years, "
            f"coupon=${cash_flow.coupon_payment:>6,.2f}, "
            f"principal=${cash_flow.principal_payment:>9,.2f}, "
            f"total=${cash_flow.total_payment:>9,.2f}"
        )


if __name__ == "__main__":
    main()