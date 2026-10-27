"""Tests for yield-curve and term-structure calculations."""

from dataclasses import FrozenInstanceError

import pytest

from src.yield_curve import (
    ParBondQuote,
    YieldCurvePoint,
    bootstrap_par_bond_curve,
    build_yield_curve,
    calculate_discount_factor,
    calculate_forward_rate,
    calculate_spot_rate,
    discount_cash_flow,
    interpolate_spot_rate,
)


@pytest.fixture
def yield_curve() -> tuple[YieldCurvePoint, ...]:
    """Return a baseline zero-coupon yield curve."""

    return build_yield_curve(
        maturities=(0.5, 1.0, 2.0, 3.0, 5.0),
        spot_rates=(0.035, 0.037, 0.040, 0.042, 0.045),
        compounding_frequency=2,
    )


def test_discount_factor() -> None:
    """The discount factor should match its financial formula."""

    result = calculate_discount_factor(
        spot_rate=0.04,
        maturity_years=5.0,
        compounding_frequency=2,
    )

    expected = 1.0 / 1.02**10

    assert result == pytest.approx(expected)


def test_spot_rate_from_discount_factor() -> None:
    """A discount factor should reproduce its original spot rate."""

    discount_factor = 1.0 / 1.02**10

    result = calculate_spot_rate(
        discount_factor=discount_factor,
        maturity_years=5.0,
        compounding_frequency=2,
    )

    assert result == pytest.approx(0.04)


@pytest.mark.parametrize(
    "compounding_frequency",
    [1, 2, 4, 12],
)
def test_spot_discount_round_trip(
    compounding_frequency: int,
) -> None:
    """Rate-to-discount conversion should be reversible."""

    original_rate = 0.0475

    discount_factor = calculate_discount_factor(
        spot_rate=original_rate,
        maturity_years=3.0,
        compounding_frequency=compounding_frequency,
    )

    reconstructed_rate = calculate_spot_rate(
        discount_factor=discount_factor,
        maturity_years=3.0,
        compounding_frequency=compounding_frequency,
    )

    assert reconstructed_rate == pytest.approx(
        original_rate
    )


def test_discount_cash_flow() -> None:
    """Present value should equal cash flow times discount factor."""

    result = discount_cash_flow(
        cash_flow=1_000.0,
        spot_rate=0.04,
        maturity_years=5.0,
        compounding_frequency=2,
    )

    expected = 1_000.0 / 1.02**10

    assert result == pytest.approx(expected)


def test_forward_rate_is_positive() -> None:
    """An upward curve should imply a positive forward rate."""

    result = calculate_forward_rate(
        start_spot_rate=0.04,
        end_spot_rate=0.045,
        start_maturity_years=2.0,
        end_maturity_years=5.0,
        compounding_frequency=2,
    )

    assert result > 0.0


def test_forward_rate_compounding_identity() -> None:
    """Spot and forward rates should satisfy no-arbitrage identity."""

    start_rate = 0.04
    end_rate = 0.045
    start_maturity = 2.0
    end_maturity = 5.0
    frequency = 2

    forward_rate = calculate_forward_rate(
        start_spot_rate=start_rate,
        end_spot_rate=end_rate,
        start_maturity_years=start_maturity,
        end_maturity_years=end_maturity,
        compounding_frequency=frequency,
    )

    end_growth = (
        1.0 + end_rate / frequency
    ) ** (frequency * end_maturity)

    start_growth = (
        1.0 + start_rate / frequency
    ) ** (frequency * start_maturity)

    forward_growth = (
        1.0 + forward_rate / frequency
    ) ** (
        frequency
        * (end_maturity - start_maturity)
    )

    assert end_growth == pytest.approx(
        start_growth * forward_growth
    )


def test_build_yield_curve_returns_tuple(
    yield_curve: tuple[YieldCurvePoint, ...],
) -> None:
    """The curve should be returned as an immutable tuple."""

    assert isinstance(yield_curve, tuple)


def test_build_yield_curve_length(
    yield_curve: tuple[YieldCurvePoint, ...],
) -> None:
    """The curve should contain one point per maturity."""

    assert len(yield_curve) == 5


def test_yield_curve_points(
    yield_curve: tuple[YieldCurvePoint, ...],
) -> None:
    """Each curve entry should contain the supplied information."""

    first_point = yield_curve[0]
    final_point = yield_curve[-1]

    assert isinstance(first_point, YieldCurvePoint)
    assert first_point.maturity_years == pytest.approx(0.5)
    assert first_point.spot_rate == pytest.approx(0.035)
    assert first_point.discount_factor > 0.0

    assert final_point.maturity_years == pytest.approx(5.0)
    assert final_point.spot_rate == pytest.approx(0.045)


def test_yield_curve_points_are_immutable(
    yield_curve: tuple[YieldCurvePoint, ...],
) -> None:
    """Curve points should not change after construction."""

    with pytest.raises(FrozenInstanceError):
        yield_curve[0].spot_rate = 0.10  # type: ignore[misc]


def test_interpolation_at_existing_maturity(
    yield_curve: tuple[YieldCurvePoint, ...],
) -> None:
    """Interpolation at a known maturity should return its rate."""

    result = interpolate_spot_rate(
        curve=yield_curve,
        target_maturity_years=3.0,
    )

    assert result == pytest.approx(0.042)


def test_linear_interpolation_between_points(
    yield_curve: tuple[YieldCurvePoint, ...],
) -> None:
    """The four-year rate should lie halfway between 3Y and 5Y."""

    result = interpolate_spot_rate(
        curve=yield_curve,
        target_maturity_years=4.0,
    )

    assert result == pytest.approx(0.0435)


def test_bootstrap_recovers_known_spot_rates() -> None:
    """Bootstrapping should recover rates used to price the bonds."""

    frequency = 2
    maturities = (0.5, 1.0, 1.5)
    known_spot_rates = (0.03, 0.035, 0.04)
    coupon_rates = (0.03, 0.035, 0.04)
    face_value = 100.0

    discount_factors = [
        calculate_discount_factor(
            spot_rate=spot_rate,
            maturity_years=maturity,
            compounding_frequency=frequency,
        )
        for maturity, spot_rate in zip(
            maturities,
            known_spot_rates,
            strict=True,
        )
    ]

    quotes: list[ParBondQuote] = []

    for period, (
        maturity,
        coupon_rate,
    ) in enumerate(
        zip(
            maturities,
            coupon_rates,
            strict=True,
        ),
        start=1,
    ):
        coupon_payment = (
            face_value * coupon_rate / frequency
        )

        market_price = (
            coupon_payment
            * sum(discount_factors[: period - 1])
            + (
                face_value + coupon_payment
            )
            * discount_factors[period - 1]
        )

        quotes.append(
            ParBondQuote(
                maturity_years=maturity,
                annual_coupon_rate=coupon_rate,
                market_price=market_price,
                face_value=face_value,
                payments_per_year=frequency,
            )
        )

    bootstrapped_curve = bootstrap_par_bond_curve(
        quotes
    )

    assert len(bootstrapped_curve) == 3

    for point, expected_rate in zip(
        bootstrapped_curve,
        known_spot_rates,
        strict=True,
    ):
        assert point.spot_rate == pytest.approx(
            expected_rate,
            abs=1e-10,
        )


def test_bootstrap_recovers_discount_factors() -> None:
    """Bootstrapped factors should reproduce known values."""

    quote = ParBondQuote(
        maturity_years=0.5,
        annual_coupon_rate=0.04,
        market_price=101.0,
        face_value=100.0,
        payments_per_year=2,
    )

    curve = bootstrap_par_bond_curve([quote])

    expected_factor = 101.0 / 102.0

    assert curve[0].discount_factor == pytest.approx(
        expected_factor
    )


def test_empty_curve_inputs_raise_value_error() -> None:
    """A curve requires at least one observation."""

    with pytest.raises(ValueError):
        build_yield_curve(
            maturities=(),
            spot_rates=(),
        )


def test_different_input_lengths_raise_value_error() -> None:
    """Maturities and rates must have equal lengths."""

    with pytest.raises(ValueError):
        build_yield_curve(
            maturities=(1.0, 2.0),
            spot_rates=(0.04,),
        )


@pytest.mark.parametrize(
    "maturities",
    [
        (1.0, 1.0, 2.0),
        (2.0, 1.0, 3.0),
    ],
)
def test_non_increasing_maturities_raise_value_error(
    maturities: tuple[float, ...],
) -> None:
    """Curve maturities must be strictly increasing."""

    with pytest.raises(ValueError):
        build_yield_curve(
            maturities=maturities,
            spot_rates=(0.03, 0.04, 0.05),
        )


@pytest.mark.parametrize(
    "target_maturity",
    [0.25, 6.0],
)
def test_interpolation_outside_curve_raises_value_error(
    yield_curve: tuple[YieldCurvePoint, ...],
    target_maturity: float,
) -> None:
    """Interpolation should not extrapolate beyond the curve."""

    with pytest.raises(ValueError):
        interpolate_spot_rate(
            curve=yield_curve,
            target_maturity_years=target_maturity,
        )


def test_interpolation_requires_two_points() -> None:
    """Interpolation requires at least two curve points."""

    point = YieldCurvePoint(
        maturity_years=1.0,
        spot_rate=0.04,
        discount_factor=1.0 / 1.04,
    )

    with pytest.raises(ValueError):
        interpolate_spot_rate(
            curve=(point,),
            target_maturity_years=1.0,
        )


@pytest.mark.parametrize(
    "maturity_years",
    [0.0, -1.0],
)
def test_invalid_discount_maturity(
    maturity_years: float,
) -> None:
    """Discount-factor maturity must be positive."""

    with pytest.raises(ValueError):
        calculate_discount_factor(
            spot_rate=0.04,
            maturity_years=maturity_years,
        )


@pytest.mark.parametrize(
    "discount_factor",
    [0.0, -1.0],
)
def test_invalid_discount_factor(
    discount_factor: float,
) -> None:
    """A discount factor must be positive."""

    with pytest.raises(ValueError):
        calculate_spot_rate(
            discount_factor=discount_factor,
            maturity_years=5.0,
        )


@pytest.mark.parametrize(
    ("start_maturity", "end_maturity"),
    [
        (0.0, 2.0),
        (2.0, 2.0),
        (3.0, 2.0),
    ],
)
def test_invalid_forward_maturities(
    start_maturity: float,
    end_maturity: float,
) -> None:
    """Forward-rate maturities must define a positive interval."""

    with pytest.raises(ValueError):
        calculate_forward_rate(
            start_spot_rate=0.03,
            end_spot_rate=0.04,
            start_maturity_years=start_maturity,
            end_maturity_years=end_maturity,
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
        calculate_discount_factor(
            spot_rate=0.04,
            maturity_years=5.0,
            compounding_frequency=compounding_frequency,
        )


def test_non_integer_compounding_frequency() -> None:
    """Compounding frequency must be an integer."""

    with pytest.raises(TypeError):
        calculate_discount_factor(
            spot_rate=0.04,
            maturity_years=5.0,
            compounding_frequency=2.5,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("maturity_years", 0.0),
        ("annual_coupon_rate", -0.01),
        ("market_price", 0.0),
        ("face_value", 0.0),
    ],
)
def test_invalid_par_bond_quote(
    field_name: str,
    invalid_value: float,
) -> None:
    """Invalid market quotes should be rejected."""

    values = {
        "maturity_years": 1.0,
        "annual_coupon_rate": 0.04,
        "market_price": 100.0,
        "face_value": 100.0,
        "payments_per_year": 2,
    }

    values[field_name] = invalid_value

    with pytest.raises(ValueError):
        ParBondQuote(**values)


def test_fractional_quote_period_raises_value_error() -> None:
    """Quoted maturity must align with its payment frequency."""

    with pytest.raises(ValueError):
        ParBondQuote(
            maturity_years=1.2,
            annual_coupon_rate=0.04,
            market_price=100.0,
            payments_per_year=2,
        )


def test_empty_bootstrap_quotes_raise_value_error() -> None:
    """Bootstrapping requires at least one quote."""

    with pytest.raises(ValueError):
        bootstrap_par_bond_curve([])


def test_nonconsecutive_quotes_raise_value_error() -> None:
    """Bootstrapping requires consecutive coupon maturities."""

    quotes = [
        ParBondQuote(
            maturity_years=0.5,
            annual_coupon_rate=0.04,
            market_price=100.0,
        ),
        ParBondQuote(
            maturity_years=1.5,
            annual_coupon_rate=0.04,
            market_price=100.0,
        ),
    ]

    with pytest.raises(ValueError):
        bootstrap_par_bond_curve(quotes)


def test_mixed_quote_frequencies_raise_value_error() -> None:
    """All bootstrapping quotes must use one frequency."""

    quotes = [
        ParBondQuote(
            maturity_years=0.5,
            annual_coupon_rate=0.04,
            market_price=100.0,
            payments_per_year=2,
        ),
        ParBondQuote(
            maturity_years=1.0,
            annual_coupon_rate=0.04,
            market_price=100.0,
            payments_per_year=1,
        ),
    ]

    with pytest.raises(ValueError):
        bootstrap_par_bond_curve(quotes)