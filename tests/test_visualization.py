"""Tests for fixed-income visualization functions."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.bond_pricing import BondParameters
from src.rate_models import (
    CIRParameters,
    InterestRateSimulation,
    VasicekParameters,
    simulate_cir,
    simulate_vasicek,
)
from src.visualization import (
    generate_all_figures,
    plot_duration_convexity_approximation,
    plot_price_yield_relationship,
    plot_rate_paths,
    plot_rate_percentile_bands,
    plot_yield_curve,
)
from src.yield_curve import (
    YieldCurvePoint,
    build_yield_curve,
)


@pytest.fixture
def bond_parameters() -> BondParameters:
    """Return baseline bond parameters."""

    return BondParameters(
        face_value=1_000.0,
        annual_coupon_rate=0.05,
        years_to_maturity=5.0,
        payments_per_year=2,
    )


@pytest.fixture
def yield_curve() -> tuple[YieldCurvePoint, ...]:
    """Return a baseline zero-coupon curve."""

    return build_yield_curve(
        maturities=(0.5, 1.0, 2.0, 3.0, 5.0),
        spot_rates=(0.035, 0.037, 0.040, 0.042, 0.045),
        compounding_frequency=2,
    )


@pytest.fixture
def vasicek_simulation() -> InterestRateSimulation:
    """Return a small Vasicek simulation."""

    return simulate_vasicek(
        parameters=VasicekParameters(
            initial_rate=0.04,
            mean_reversion_speed=0.60,
            long_term_mean=0.05,
            volatility=0.015,
        ),
        years=2.0,
        steps_per_year=12,
        simulations=200,
        seed=42,
    )


@pytest.fixture
def cir_simulation() -> InterestRateSimulation:
    """Return a small CIR simulation."""

    return simulate_cir(
        parameters=CIRParameters(
            initial_rate=0.04,
            mean_reversion_speed=0.60,
            long_term_mean=0.05,
            volatility=0.10,
        ),
        years=2.0,
        steps_per_year=12,
        simulations=200,
        seed=42,
    )


def test_plot_yield_curve_returns_figure_and_axis(
    yield_curve: tuple[YieldCurvePoint, ...],
) -> None:
    """Yield-curve plotting should return Matplotlib objects."""

    figure, axis = plot_yield_curve(yield_curve)

    assert isinstance(figure, Figure)
    assert isinstance(axis, Axes)
    assert len(axis.lines) == 1

    plt.close(figure)


def test_plot_yield_curve_labels(
    yield_curve: tuple[YieldCurvePoint, ...],
) -> None:
    """Yield-curve chart should contain professional labels."""

    figure, axis = plot_yield_curve(yield_curve)

    assert axis.get_title() == "Zero-Coupon Yield Curve"
    assert axis.get_xlabel() == "Maturity (years)"
    assert axis.get_ylabel() == "Spot rate (%)"

    plt.close(figure)


def test_plot_yield_curve_saves_file(
    yield_curve: tuple[YieldCurvePoint, ...],
    tmp_path: Path,
) -> None:
    """Yield-curve chart should be saved as an image."""

    output_path = tmp_path / "yield_curve.png"

    figure, _ = plot_yield_curve(
        curve=yield_curve,
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    plt.close(figure)


def test_plot_rate_paths_returns_objects(
    vasicek_simulation: InterestRateSimulation,
) -> None:
    """Rate-path plotting should return a figure and axis."""

    figure, axis = plot_rate_paths(
        simulation=vasicek_simulation,
        paths_to_show=50,
    )

    assert isinstance(figure, Figure)
    assert isinstance(axis, Axes)
    assert "Vasicek" in axis.get_title()

    plt.close(figure)


def test_plot_rate_paths_contains_average_path(
    vasicek_simulation: InterestRateSimulation,
) -> None:
    """Rate-path chart should include its average trajectory."""

    figure, axis = plot_rate_paths(
        simulation=vasicek_simulation,
        paths_to_show=20,
    )

    labels = [
        line.get_label()
        for line in axis.lines
    ]

    assert "Average path" in labels

    plt.close(figure)


def test_plot_more_paths_than_available(
    vasicek_simulation: InterestRateSimulation,
) -> None:
    """Requested paths should be limited to those available."""

    figure, axis = plot_rate_paths(
        simulation=vasicek_simulation,
        paths_to_show=1_000,
    )

    assert isinstance(axis, Axes)

    plt.close(figure)


def test_plot_rate_paths_saves_file(
    cir_simulation: InterestRateSimulation,
    tmp_path: Path,
) -> None:
    """CIR paths should be saved successfully."""

    output_path = tmp_path / "cir_paths.png"

    figure, _ = plot_rate_paths(
        simulation=cir_simulation,
        paths_to_show=25,
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    plt.close(figure)


def test_percentile_band_plot(
    vasicek_simulation: InterestRateSimulation,
) -> None:
    """Percentile chart should contain median and mean lines."""

    figure, axis = plot_rate_percentile_bands(
        simulation=vasicek_simulation,
    )

    labels = {
        line.get_label()
        for line in axis.lines
    }

    assert "Median" in labels
    assert "Mean" in labels
    assert len(axis.collections) >= 1

    plt.close(figure)


def test_percentile_band_saves_file(
    vasicek_simulation: InterestRateSimulation,
    tmp_path: Path,
) -> None:
    """Percentile-band chart should be saved."""

    output_path = tmp_path / "percentiles.png"

    figure, _ = plot_rate_percentile_bands(
        simulation=vasicek_simulation,
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    plt.close(figure)


def test_price_yield_plot_returns_objects(
    bond_parameters: BondParameters,
) -> None:
    """Price-yield plotting should return Matplotlib objects."""

    figure, axis = plot_price_yield_relationship(
        parameters=bond_parameters,
        annual_yields=np.linspace(
            0.01,
            0.10,
            20,
        ),
    )

    assert isinstance(figure, Figure)
    assert isinstance(axis, Axes)
    assert axis.get_title() == (
        "Bond Price–Yield Relationship"
    )

    plt.close(figure)


def test_price_decreases_as_yield_increases(
    bond_parameters: BondParameters,
) -> None:
    """The plotted price-yield relationship should slope downward."""

    figure, axis = plot_price_yield_relationship(
        parameters=bond_parameters,
        annual_yields=(0.02, 0.04, 0.06, 0.08),
    )

    prices = axis.lines[0].get_ydata()

    assert np.all(np.diff(prices) < 0.0)

    plt.close(figure)


def test_price_yield_plot_saves_file(
    bond_parameters: BondParameters,
    tmp_path: Path,
) -> None:
    """Price-yield chart should be saved."""

    output_path = tmp_path / "price_yield.png"

    figure, _ = plot_price_yield_relationship(
        parameters=bond_parameters,
        annual_yields=np.linspace(
            0.0,
            0.10,
            25,
        ),
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    plt.close(figure)


def test_duration_convexity_plot(
    bond_parameters: BondParameters,
) -> None:
    """Approximation chart should contain all three models."""

    figure, axis = (
        plot_duration_convexity_approximation(
            parameters=bond_parameters,
            annual_yield=0.04,
            yield_changes=np.linspace(
                -0.02,
                0.02,
                21,
            ),
        )
    )

    labels = {
        line.get_label()
        for line in axis.lines
    }

    assert "Exact repricing" in labels
    assert "Duration only" in labels
    assert "Duration + convexity" in labels

    plt.close(figure)


def test_duration_convexity_plot_saves_file(
    bond_parameters: BondParameters,
    tmp_path: Path,
) -> None:
    """Approximation chart should be saved."""

    output_path = tmp_path / "duration_convexity.png"

    figure, _ = (
        plot_duration_convexity_approximation(
            parameters=bond_parameters,
            annual_yield=0.04,
            yield_changes=np.linspace(
                -0.02,
                0.02,
                21,
            ),
            output_path=output_path,
        )
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    plt.close(figure)


def test_generate_all_figures(
    tmp_path: Path,
) -> None:
    """The project generator should produce seven images."""

    generated_paths = generate_all_figures(
        output_directory=tmp_path
    )

    assert len(generated_paths) == 7

    for path in generated_paths:
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0


def test_empty_yield_curve_raises_value_error() -> None:
    """A yield-curve chart requires at least one point."""

    with pytest.raises(ValueError):
        plot_yield_curve(())


def test_invalid_yield_curve_object_raises_type_error() -> None:
    """Yield-curve chart requires proper curve points."""

    with pytest.raises(TypeError):
        plot_yield_curve(
            ("invalid",)  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "paths_to_show",
    [0, -1],
)
def test_invalid_paths_to_show(
    vasicek_simulation: InterestRateSimulation,
    paths_to_show: int,
) -> None:
    """The number of visible paths must be positive."""

    with pytest.raises(ValueError):
        plot_rate_paths(
            simulation=vasicek_simulation,
            paths_to_show=paths_to_show,
        )


def test_non_integer_paths_to_show(
    vasicek_simulation: InterestRateSimulation,
) -> None:
    """The number of paths must be an integer."""

    with pytest.raises(TypeError):
        plot_rate_paths(
            simulation=vasicek_simulation,
            paths_to_show=10.5,  # type: ignore[arg-type]
        )


def test_invalid_simulation_object() -> None:
    """Rate charts require a simulation object."""

    with pytest.raises(TypeError):
        plot_rate_paths(
            simulation="invalid",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        (-1.0, 95.0),
        (5.0, 101.0),
        (95.0, 5.0),
        (50.0, 50.0),
    ],
)
def test_invalid_percentile_limits(
    vasicek_simulation: InterestRateSimulation,
    lower: float,
    upper: float,
) -> None:
    """Percentile limits must be ordered inside 0 to 100."""

    with pytest.raises(ValueError):
        plot_rate_percentile_bands(
            simulation=vasicek_simulation,
            lower_percentile=lower,
            upper_percentile=upper,
        )


def test_invalid_bond_parameters_object() -> None:
    """Price-yield charts require BondParameters."""

    with pytest.raises(TypeError):
        plot_price_yield_relationship(
            parameters="invalid",  # type: ignore[arg-type]
            annual_yields=(0.03, 0.04),
        )


def test_empty_yield_values(
    bond_parameters: BondParameters,
) -> None:
    """Price-yield charts require yield observations."""

    with pytest.raises(ValueError):
        plot_price_yield_relationship(
            parameters=bond_parameters,
            annual_yields=(),
        )


def test_non_finite_yield_values(
    bond_parameters: BondParameters,
) -> None:
    """Price-yield inputs must contain finite values."""

    with pytest.raises(ValueError):
        plot_price_yield_relationship(
            parameters=bond_parameters,
            annual_yields=(0.03, np.nan, 0.05),
        )


def test_empty_yield_changes(
    bond_parameters: BondParameters,
) -> None:
    """Duration chart requires yield-change observations."""

    with pytest.raises(ValueError):
        plot_duration_convexity_approximation(
            parameters=bond_parameters,
            annual_yield=0.04,
            yield_changes=(),
        )


def test_invalid_show_type(
    yield_curve: tuple[YieldCurvePoint, ...],
) -> None:
    """The show argument must be a boolean."""

    with pytest.raises(TypeError):
        plot_yield_curve(
            curve=yield_curve,
            show="yes",  # type: ignore[arg-type]
        )

    plt.close("all")