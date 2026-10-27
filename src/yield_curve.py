"""Yield-curve construction and interest-rate term-structure tools."""

from dataclasses import dataclass
from math import isclose, isfinite
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class YieldCurvePoint:
    """One point on a zero-coupon yield curve."""

    maturity_years: float
    spot_rate: float
    discount_factor: float


@dataclass(frozen=True)
class ParBondQuote:
    """Market quote used for zero-curve bootstrapping."""

    maturity_years: float
    annual_coupon_rate: float
    market_price: float
    face_value: float = 100.0
    payments_per_year: int = 2

    def __post_init__(self) -> None:
        """Validate the quoted bond."""

        values = {
            "maturity_years": self.maturity_years,
            "annual_coupon_rate": self.annual_coupon_rate,
            "market_price": self.market_price,
            "face_value": self.face_value,
        }

        for name, value in values.items():
            _validate_real_number(value, name)

        if self.maturity_years <= 0.0:
            raise ValueError(
                "maturity_years must be greater than zero."
            )

        if self.annual_coupon_rate < 0.0:
            raise ValueError(
                "annual_coupon_rate cannot be negative."
            )

        if self.market_price <= 0.0:
            raise ValueError(
                "market_price must be greater than zero."
            )

        if self.face_value <= 0.0:
            raise ValueError(
                "face_value must be greater than zero."
            )

        _validate_frequency(self.payments_per_year)

        periods = (
            self.maturity_years * self.payments_per_year
        )

        if not isclose(
            periods,
            round(periods),
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise ValueError(
                "maturity_years must produce a whole "
                "number of payment periods."
            )

    @property
    def number_of_payments(self) -> int:
        """Return the total number of coupon periods."""

        return round(
            self.maturity_years
            * self.payments_per_year
        )

    @property
    def coupon_payment(self) -> float:
        """Return the coupon paid during each period."""

        return (
            self.face_value
            * self.annual_coupon_rate
            / self.payments_per_year
        )


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
    """Validate the annual compounding frequency."""

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


def calculate_discount_factor(
    spot_rate: float,
    maturity_years: float,
    compounding_frequency: int = 2,
) -> float:
    """Calculate a discount factor from a nominal spot rate."""

    _validate_real_number(spot_rate, "spot_rate")
    _validate_real_number(
        maturity_years,
        "maturity_years",
    )
    _validate_frequency(compounding_frequency)

    if maturity_years <= 0.0:
        raise ValueError(
            "maturity_years must be greater than zero."
        )

    if spot_rate <= -compounding_frequency:
        raise ValueError(
            "spot_rate produces a non-positive "
            "periodic discount factor."
        )

    number_of_periods = (
        maturity_years * compounding_frequency
    )

    return 1.0 / (
        1.0
        + spot_rate / compounding_frequency
    ) ** number_of_periods


def calculate_spot_rate(
    discount_factor: float,
    maturity_years: float,
    compounding_frequency: int = 2,
) -> float:
    """Recover a nominal spot rate from a discount factor."""

    _validate_real_number(
        discount_factor,
        "discount_factor",
    )
    _validate_real_number(
        maturity_years,
        "maturity_years",
    )
    _validate_frequency(compounding_frequency)

    if discount_factor <= 0.0:
        raise ValueError(
            "discount_factor must be greater than zero."
        )

    if maturity_years <= 0.0:
        raise ValueError(
            "maturity_years must be greater than zero."
        )

    number_of_periods = (
        maturity_years * compounding_frequency
    )

    periodic_rate = (
        discount_factor
        ** (-1.0 / number_of_periods)
        - 1.0
    )

    return periodic_rate * compounding_frequency


def discount_cash_flow(
    cash_flow: float,
    spot_rate: float,
    maturity_years: float,
    compounding_frequency: int = 2,
) -> float:
    """Calculate the present value of one future cash flow."""

    _validate_real_number(cash_flow, "cash_flow")

    discount_factor = calculate_discount_factor(
        spot_rate=spot_rate,
        maturity_years=maturity_years,
        compounding_frequency=compounding_frequency,
    )

    return cash_flow * discount_factor


def calculate_forward_rate(
    start_spot_rate: float,
    end_spot_rate: float,
    start_maturity_years: float,
    end_maturity_years: float,
    compounding_frequency: int = 2,
) -> float:
    """Calculate the nominal forward rate between two maturities."""

    _validate_real_number(
        start_maturity_years,
        "start_maturity_years",
    )
    _validate_real_number(
        end_maturity_years,
        "end_maturity_years",
    )
    _validate_frequency(compounding_frequency)

    if start_maturity_years <= 0.0:
        raise ValueError(
            "start_maturity_years must be greater than zero."
        )

    if end_maturity_years <= start_maturity_years:
        raise ValueError(
            "end_maturity_years must be greater than "
            "start_maturity_years."
        )

    start_discount_factor = calculate_discount_factor(
        spot_rate=start_spot_rate,
        maturity_years=start_maturity_years,
        compounding_frequency=compounding_frequency,
    )

    end_discount_factor = calculate_discount_factor(
        spot_rate=end_spot_rate,
        maturity_years=end_maturity_years,
        compounding_frequency=compounding_frequency,
    )

    forward_periods = (
        (end_maturity_years - start_maturity_years)
        * compounding_frequency
    )

    periodic_forward_rate = (
        start_discount_factor / end_discount_factor
    ) ** (1.0 / forward_periods) - 1.0

    return (
        periodic_forward_rate
        * compounding_frequency
    )


def build_yield_curve(
    maturities: Sequence[float],
    spot_rates: Sequence[float],
    compounding_frequency: int = 2,
) -> tuple[YieldCurvePoint, ...]:
    """Build yield-curve points from maturities and spot rates."""

    _validate_frequency(compounding_frequency)

    if len(maturities) == 0:
        raise ValueError(
            "At least one maturity is required."
        )

    if len(maturities) != len(spot_rates):
        raise ValueError(
            "maturities and spot_rates must have "
            "the same length."
        )

    points: list[YieldCurvePoint] = []
    previous_maturity = 0.0

    for maturity, spot_rate in zip(
        maturities,
        spot_rates,
        strict=True,
    ):
        _validate_real_number(
            maturity,
            "maturity",
        )
        _validate_real_number(
            spot_rate,
            "spot_rate",
        )

        if maturity <= previous_maturity:
            raise ValueError(
                "maturities must be positive and "
                "strictly increasing."
            )

        discount_factor = calculate_discount_factor(
            spot_rate=spot_rate,
            maturity_years=maturity,
            compounding_frequency=compounding_frequency,
        )

        points.append(
            YieldCurvePoint(
                maturity_years=float(maturity),
                spot_rate=float(spot_rate),
                discount_factor=discount_factor,
            )
        )

        previous_maturity = float(maturity)

    return tuple(points)


def interpolate_spot_rate(
    curve: Sequence[YieldCurvePoint],
    target_maturity_years: float,
) -> float:
    """Linearly interpolate a spot rate inside the curve."""

    _validate_real_number(
        target_maturity_years,
        "target_maturity_years",
    )

    if len(curve) < 2:
        raise ValueError(
            "At least two curve points are required."
        )

    maturities = np.array(
        [point.maturity_years for point in curve],
        dtype=float,
    )

    spot_rates = np.array(
        [point.spot_rate for point in curve],
        dtype=float,
    )

    if np.any(np.diff(maturities) <= 0.0):
        raise ValueError(
            "Curve maturities must be strictly increasing."
        )

    if (
        target_maturity_years < maturities[0]
        or target_maturity_years > maturities[-1]
    ):
        raise ValueError(
            "target_maturity_years must lie within "
            "the yield curve."
        )

    return float(
        np.interp(
            target_maturity_years,
            maturities,
            spot_rates,
        )
    )


def bootstrap_par_bond_curve(
    quotes: Sequence[ParBondQuote],
) -> tuple[YieldCurvePoint, ...]:
    """Bootstrap a zero-coupon curve from consecutive bond quotes."""

    if len(quotes) == 0:
        raise ValueError(
            "At least one bond quote is required."
        )

    sorted_quotes = sorted(
        quotes,
        key=lambda quote: quote.number_of_payments,
    )

    frequency = sorted_quotes[0].payments_per_year
    discount_factors: dict[int, float] = {}
    curve_points: list[YieldCurvePoint] = []

    for expected_period, quote in enumerate(
        sorted_quotes,
        start=1,
    ):
        if not isinstance(quote, ParBondQuote):
            raise TypeError(
                "All quotes must be ParBondQuote instances."
            )

        if quote.payments_per_year != frequency:
            raise ValueError(
                "All quotes must use the same "
                "payments_per_year."
            )

        if quote.number_of_payments != expected_period:
            raise ValueError(
                "Quotes must provide consecutive maturities "
                "starting with the first payment period."
            )

        known_coupon_value = sum(
            quote.coupon_payment
            * discount_factors[period]
            for period in range(1, expected_period)
        )

        final_cash_flow = (
            quote.face_value + quote.coupon_payment
        )

        final_discount_factor = (
            quote.market_price - known_coupon_value
        ) / final_cash_flow

        if final_discount_factor <= 0.0:
            raise ValueError(
                "The supplied quotes produce a non-positive "
                "discount factor."
            )

        discount_factors[expected_period] = (
            final_discount_factor
        )

        spot_rate = calculate_spot_rate(
            discount_factor=final_discount_factor,
            maturity_years=quote.maturity_years,
            compounding_frequency=frequency,
        )

        curve_points.append(
            YieldCurvePoint(
                maturity_years=quote.maturity_years,
                spot_rate=spot_rate,
                discount_factor=final_discount_factor,
            )
        )

    return tuple(curve_points)


def main() -> None:
    """Run an example yield-curve analysis."""

    maturities = (0.5, 1.0, 2.0, 3.0, 5.0)
    spot_rates = (0.035, 0.037, 0.040, 0.042, 0.045)
    frequency = 2

    curve = build_yield_curve(
        maturities=maturities,
        spot_rates=spot_rates,
        compounding_frequency=frequency,
    )

    interpolated_rate = interpolate_spot_rate(
        curve=curve,
        target_maturity_years=4.0,
    )

    forward_rate = calculate_forward_rate(
        start_spot_rate=0.040,
        end_spot_rate=0.045,
        start_maturity_years=2.0,
        end_maturity_years=5.0,
        compounding_frequency=frequency,
    )

    five_year_present_value = discount_cash_flow(
        cash_flow=1_000.0,
        spot_rate=0.045,
        maturity_years=5.0,
        compounding_frequency=frequency,
    )

    print("Zero-coupon yield curve")
    print(
        "Maturity       Spot rate       "
        "Discount factor"
    )

    for point in curve:
        print(
            f"{point.maturity_years:>6.1f} years"
            f"{point.spot_rate:>15.4%}"
            f"{point.discount_factor:>20.6f}"
        )

    print("\nInterpolated rate")
    print(
        f"Four-year spot rate: "
        f"{interpolated_rate:.4%}"
    )

    print("\nImplied forward rate")
    print(
        f"Two-to-five-year forward rate: "
        f"{forward_rate:.4%}"
    )

    print("\nPresent value")
    print(
        f"Present value of $1,000 in five years: "
        f"${five_year_present_value:,.2f}"
    )


if __name__ == "__main__":
    main()