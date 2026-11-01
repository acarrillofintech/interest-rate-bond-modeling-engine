"""Professional visualizations for fixed-income analytics."""

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.bond_pricing import (
    BondParameters,
    price_coupon_bond,
)
from src.duration_convexity import (
    estimate_new_bond_price,
)
from src.rate_models import (
    CIRParameters,
    InterestRateSimulation,
    VasicekParameters,
    simulate_cir,
    simulate_vasicek,
)
from src.yield_curve import (
    YieldCurvePoint,
    build_yield_curve,
)


COLORS = {
    "blue": "#2C6EBA",
    "orange": "#E58C3A",
    "green": "#3A9D78",
    "red": "#C94C4C",
    "purple": "#7768AE",
    "gray": "#64748B",
    "light_blue": "#A8D3F0",
}


def _validate_boolean(
    value: bool,
    name: str,
) -> None:
    """Validate a boolean argument."""

    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")


def _save_figure(
    figure: Figure,
    output_path: str | Path | None,
) -> Path | None:
    """Save a figure and create its parent directory."""

    if output_path is None:
        return None

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    return path


def _finish_figure(
    figure: Figure,
    output_path: str | Path | None,
    show: bool,
) -> Path | None:
    """Apply layout, save, and optionally display a figure."""

    _validate_boolean(show, "show")

    figure.tight_layout()
    saved_path = _save_figure(
        figure=figure,
        output_path=output_path,
    )

    if show:
        plt.show()

    return saved_path


def plot_yield_curve(
    curve: Sequence[YieldCurvePoint],
    output_path: str | Path | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot the zero-coupon spot-rate curve."""

    if len(curve) == 0:
        raise ValueError(
            "curve must contain at least one point."
        )

    if not all(
        isinstance(point, YieldCurvePoint)
        for point in curve
    ):
        raise TypeError(
            "curve must contain YieldCurvePoint instances."
        )

    maturities = np.array(
        [point.maturity_years for point in curve],
        dtype=float,
    )

    rates = np.array(
        [point.spot_rate for point in curve],
        dtype=float,
    )

    figure, axis = plt.subplots(figsize=(11, 6))

    axis.plot(
        maturities,
        rates * 100.0,
        color=COLORS["blue"],
        marker="o",
        markersize=7,
        linewidth=2.2,
        label="Spot rate",
    )

    axis.fill_between(
        maturities,
        rates * 100.0,
        alpha=0.12,
        color=COLORS["blue"],
    )

    axis.set_title(
        "Zero-Coupon Yield Curve",
        fontsize=15,
        fontweight="bold",
    )
    axis.set_xlabel("Maturity (years)")
    axis.set_ylabel("Spot rate (%)")
    axis.grid(alpha=0.25)
    axis.legend()

    _finish_figure(
        figure=figure,
        output_path=output_path,
        show=show,
    )

    return figure, axis


def plot_rate_paths(
    simulation: InterestRateSimulation,
    paths_to_show: int = 100,
    output_path: str | Path | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot a sample of simulated short-rate paths."""

    if not isinstance(
        simulation,
        InterestRateSimulation,
    ):
        raise TypeError(
            "simulation must be an "
            "InterestRateSimulation instance."
        )

    if isinstance(paths_to_show, bool) or not isinstance(
        paths_to_show,
        int,
    ):
        raise TypeError(
            "paths_to_show must be an integer."
        )

    if paths_to_show <= 0:
        raise ValueError(
            "paths_to_show must be greater than zero."
        )

    number_to_plot = min(
        paths_to_show,
        simulation.number_of_simulations,
    )

    figure, axis = plt.subplots(figsize=(12, 6))

    axis.plot(
        simulation.time_grid,
        simulation.rate_paths[
            :, :number_to_plot
        ] * 100.0,
        linewidth=0.75,
        alpha=0.28,
        color=COLORS["blue"],
    )

    axis.plot(
        simulation.time_grid,
        simulation.average_path * 100.0,
        linewidth=2.5,
        color=COLORS["orange"],
        label="Average path",
    )

    axis.set_title(
        f"{simulation.model_name} Simulated Interest-Rate Paths",
        fontsize=15,
        fontweight="bold",
    )
    axis.set_xlabel("Time (years)")
    axis.set_ylabel("Short rate (%)")
    axis.grid(alpha=0.22)
    axis.legend()

    _finish_figure(
        figure=figure,
        output_path=output_path,
        show=show,
    )

    return figure, axis


def plot_rate_percentile_bands(
    simulation: InterestRateSimulation,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
    output_path: str | Path | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot median and percentile bands of simulated rates."""

    if not isinstance(
        simulation,
        InterestRateSimulation,
    ):
        raise TypeError(
            "simulation must be an "
            "InterestRateSimulation instance."
        )

    for name, value in {
        "lower_percentile": lower_percentile,
        "upper_percentile": upper_percentile,
    }.items():
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{name} must be a real number."
            )

    if not (
        0.0
        <= lower_percentile
        < upper_percentile
        <= 100.0
    ):
        raise ValueError(
            "Percentiles must satisfy "
            "0 <= lower < upper <= 100."
        )

    lower_band = np.percentile(
        simulation.rate_paths,
        lower_percentile,
        axis=1,
    )

    median_path = np.median(
        simulation.rate_paths,
        axis=1,
    )

    upper_band = np.percentile(
        simulation.rate_paths,
        upper_percentile,
        axis=1,
    )

    mean_path = simulation.average_path

    figure, axis = plt.subplots(figsize=(12, 6))

    axis.fill_between(
        simulation.time_grid,
        lower_band * 100.0,
        upper_band * 100.0,
        color=COLORS["light_blue"],
        alpha=0.50,
        label=(
            f"{lower_percentile:.0f}th–"
            f"{upper_percentile:.0f}th percentile"
        ),
    )

    axis.plot(
        simulation.time_grid,
        median_path * 100.0,
        color=COLORS["blue"],
        linewidth=2.2,
        label="Median",
    )

    axis.plot(
        simulation.time_grid,
        mean_path * 100.0,
        color=COLORS["orange"],
        linewidth=1.8,
        linestyle="--",
        label="Mean",
    )

    axis.set_title(
        f"{simulation.model_name} Interest-Rate "
        "Percentile Bands",
        fontsize=15,
        fontweight="bold",
    )
    axis.set_xlabel("Time (years)")
    axis.set_ylabel("Short rate (%)")
    axis.grid(alpha=0.22)
    axis.legend()

    _finish_figure(
        figure=figure,
        output_path=output_path,
        show=show,
    )

    return figure, axis


def plot_price_yield_relationship(
    parameters: BondParameters,
    annual_yields: Sequence[float],
    output_path: str | Path | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot the inverse relationship between price and yield."""

    if not isinstance(parameters, BondParameters):
        raise TypeError(
            "parameters must be a BondParameters instance."
        )

    yields = np.asarray(
        tuple(annual_yields),
        dtype=float,
    )

    if yields.ndim != 1 or yields.size == 0:
        raise ValueError(
            "annual_yields must contain at least one value."
        )

    if not np.all(np.isfinite(yields)):
        raise ValueError(
            "annual_yields must contain finite values."
        )

    prices = np.array(
        [
            price_coupon_bond(
                parameters=parameters,
                annual_yield=float(annual_yield),
            )
            for annual_yield in yields
        ],
        dtype=float,
    )

    figure, axis = plt.subplots(figsize=(11, 6))

    axis.plot(
        yields * 100.0,
        prices,
        color=COLORS["green"],
        linewidth=2.4,
    )

    axis.axhline(
        parameters.face_value,
        color=COLORS["gray"],
        linestyle="--",
        linewidth=1.4,
        label="Face value",
    )

    axis.axvline(
        parameters.annual_coupon_rate * 100.0,
        color=COLORS["orange"],
        linestyle=":",
        linewidth=1.7,
        label="Coupon rate",
    )

    axis.set_title(
        "Bond Price–Yield Relationship",
        fontsize=15,
        fontweight="bold",
    )
    axis.set_xlabel("Yield to maturity (%)")
    axis.set_ylabel("Bond price ($)")
    axis.grid(alpha=0.25)
    axis.legend()

    _finish_figure(
        figure=figure,
        output_path=output_path,
        show=show,
    )

    return figure, axis


def plot_duration_convexity_approximation(
    parameters: BondParameters,
    annual_yield: float,
    yield_changes: Sequence[float],
    output_path: str | Path | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Compare exact repricing with duration approximations."""

    if not isinstance(parameters, BondParameters):
        raise TypeError(
            "parameters must be a BondParameters instance."
        )

    changes = np.asarray(
        tuple(yield_changes),
        dtype=float,
    )

    if changes.ndim != 1 or changes.size == 0:
        raise ValueError(
            "yield_changes must contain at least one value."
        )

    if not np.all(np.isfinite(changes)):
        raise ValueError(
            "yield_changes must contain finite values."
        )

    exact_prices = np.array(
        [
            price_coupon_bond(
                parameters=parameters,
                annual_yield=(
                    annual_yield + float(change)
                ),
            )
            for change in changes
        ]
    )

    duration_prices = np.array(
        [
            estimate_new_bond_price(
                parameters=parameters,
                annual_yield=annual_yield,
                yield_change=float(change),
                include_convexity=False,
            )
            for change in changes
        ]
    )

    convexity_prices = np.array(
        [
            estimate_new_bond_price(
                parameters=parameters,
                annual_yield=annual_yield,
                yield_change=float(change),
                include_convexity=True,
            )
            for change in changes
        ]
    )

    figure, axis = plt.subplots(figsize=(11, 6))

    axis.plot(
        changes * 10_000.0,
        exact_prices,
        color=COLORS["blue"],
        linewidth=2.5,
        label="Exact repricing",
    )

    axis.plot(
        changes * 10_000.0,
        duration_prices,
        color=COLORS["red"],
        linewidth=1.8,
        linestyle="--",
        label="Duration only",
    )

    axis.plot(
        changes * 10_000.0,
        convexity_prices,
        color=COLORS["green"],
        linewidth=1.8,
        linestyle="-.",
        label="Duration + convexity",
    )

    axis.axvline(
        0.0,
        color=COLORS["gray"],
        linewidth=1.0,
    )

    axis.set_title(
        "Duration and Convexity Approximation",
        fontsize=15,
        fontweight="bold",
    )
    axis.set_xlabel("Yield change (basis points)")
    axis.set_ylabel("Estimated bond price ($)")
    axis.grid(alpha=0.25)
    axis.legend()

    _finish_figure(
        figure=figure,
        output_path=output_path,
        show=show,
    )

    return figure, axis


def generate_all_figures(
    output_directory: str | Path = "results/figures",
) -> tuple[Path, ...]:
    """Generate and save all project figures."""

    output_directory = Path(output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    bond_parameters = BondParameters(
        face_value=1_000.0,
        annual_coupon_rate=0.05,
        years_to_maturity=5.0,
        payments_per_year=2,
    )

    yield_curve = build_yield_curve(
        maturities=(0.5, 1.0, 2.0, 3.0, 5.0),
        spot_rates=(0.035, 0.037, 0.040, 0.042, 0.045),
        compounding_frequency=2,
    )

    vasicek_result = simulate_vasicek(
        parameters=VasicekParameters(
            initial_rate=0.04,
            mean_reversion_speed=0.60,
            long_term_mean=0.05,
            volatility=0.015,
        ),
        years=5.0,
        steps_per_year=12,
        simulations=2_000,
        seed=42,
    )

    cir_result = simulate_cir(
        parameters=CIRParameters(
            initial_rate=0.04,
            mean_reversion_speed=0.60,
            long_term_mean=0.05,
            volatility=0.10,
        ),
        years=5.0,
        steps_per_year=12,
        simulations=2_000,
        seed=42,
    )

    paths = (
        output_directory / "yield_curve.png",
        output_directory / "price_yield_relationship.png",
        output_directory / "vasicek_paths.png",
        output_directory / "cir_paths.png",
        output_directory / "vasicek_percentile_bands.png",
        output_directory / "cir_percentile_bands.png",
        output_directory / "duration_convexity.png",
    )

    figures: list[Figure] = []

    figure, _ = plot_yield_curve(
        curve=yield_curve,
        output_path=paths[0],
    )
    figures.append(figure)

    figure, _ = plot_price_yield_relationship(
        parameters=bond_parameters,
        annual_yields=np.linspace(
            0.0,
            0.10,
            101,
        ),
        output_path=paths[1],
    )
    figures.append(figure)

    figure, _ = plot_rate_paths(
        simulation=vasicek_result,
        paths_to_show=100,
        output_path=paths[2],
    )
    figures.append(figure)

    figure, _ = plot_rate_paths(
        simulation=cir_result,
        paths_to_show=100,
        output_path=paths[3],
    )
    figures.append(figure)

    figure, _ = plot_rate_percentile_bands(
        simulation=vasicek_result,
        output_path=paths[4],
    )
    figures.append(figure)

    figure, _ = plot_rate_percentile_bands(
        simulation=cir_result,
        output_path=paths[5],
    )
    figures.append(figure)

    figure, _ = plot_duration_convexity_approximation(
        parameters=bond_parameters,
        annual_yield=0.04,
        yield_changes=np.linspace(
            -0.02,
            0.02,
            81,
        ),
        output_path=paths[6],
    )
    figures.append(figure)

    for figure in figures:
        plt.close(figure)

    return paths


def main() -> None:
    """Generate every project visualization."""

    generated_paths = generate_all_figures()

    print("Generated fixed-income figures")

    for path in generated_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()