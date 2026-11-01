"""Stochastic short-rate models for fixed-income analysis.

This module implements the Vasicek and Cox-Ingersoll-Ross models
using Monte Carlo simulation with Euler discretization.
"""

from dataclasses import dataclass
from math import exp, isfinite, sqrt

import numpy as np


@dataclass(frozen=True)
class VasicekParameters:
    """Parameters of the Vasicek short-rate model."""

    initial_rate: float
    mean_reversion_speed: float
    long_term_mean: float
    volatility: float

    def __post_init__(self) -> None:
        """Validate model parameters."""

        _validate_model_parameters(
            initial_rate=self.initial_rate,
            mean_reversion_speed=self.mean_reversion_speed,
            long_term_mean=self.long_term_mean,
            volatility=self.volatility,
            require_non_negative_rates=False,
        )


@dataclass(frozen=True)
class CIRParameters:
    """Parameters of the Cox-Ingersoll-Ross short-rate model."""

    initial_rate: float
    mean_reversion_speed: float
    long_term_mean: float
    volatility: float

    def __post_init__(self) -> None:
        """Validate model parameters."""

        _validate_model_parameters(
            initial_rate=self.initial_rate,
            mean_reversion_speed=self.mean_reversion_speed,
            long_term_mean=self.long_term_mean,
            volatility=self.volatility,
            require_non_negative_rates=True,
        )


@dataclass(frozen=True)
class InterestRateSimulation:
    """Simulated interest-rate paths and their time grid."""

    model_name: str
    time_grid: np.ndarray
    rate_paths: np.ndarray

    def __post_init__(self) -> None:
        """Validate the simulation arrays."""

        if not isinstance(self.model_name, str):
            raise TypeError("model_name must be a string.")

        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty.")

        if not isinstance(self.time_grid, np.ndarray):
            raise TypeError(
                "time_grid must be a NumPy array."
            )

        if not isinstance(self.rate_paths, np.ndarray):
            raise TypeError(
                "rate_paths must be a NumPy array."
            )

        if self.time_grid.ndim != 1:
            raise ValueError(
                "time_grid must be one-dimensional."
            )

        if self.rate_paths.ndim != 2:
            raise ValueError(
                "rate_paths must be two-dimensional."
            )

        if (
            self.rate_paths.shape[0]
            != self.time_grid.size
        ):
            raise ValueError(
                "rate_paths rows must match time_grid."
            )

        if self.rate_paths.shape[1] == 0:
            raise ValueError(
                "rate_paths must contain simulations."
            )

        if not np.all(np.isfinite(self.time_grid)):
            raise ValueError(
                "time_grid must contain finite values."
            )

        if not np.all(np.isfinite(self.rate_paths)):
            raise ValueError(
                "rate_paths must contain finite values."
            )

    @property
    def terminal_rates(self) -> np.ndarray:
        """Return the final rate from every simulation."""

        return self.rate_paths[-1].copy()

    @property
    def average_path(self) -> np.ndarray:
        """Return the mean interest rate at each time step."""

        return np.mean(self.rate_paths, axis=1)

    @property
    def number_of_simulations(self) -> int:
        """Return the number of simulated paths."""

        return int(self.rate_paths.shape[1])

    @property
    def number_of_steps(self) -> int:
        """Return the number of simulated time intervals."""

        return int(self.rate_paths.shape[0] - 1)


@dataclass(frozen=True)
class RateSimulationSummary:
    """Descriptive statistics for terminal short rates."""

    mean_terminal_rate: float
    median_terminal_rate: float
    standard_deviation: float
    fifth_percentile: float
    ninety_fifth_percentile: float
    minimum_rate: float
    maximum_rate: float


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


def _validate_model_parameters(
    initial_rate: float,
    mean_reversion_speed: float,
    long_term_mean: float,
    volatility: float,
    require_non_negative_rates: bool,
) -> None:
    """Validate common short-rate model parameters."""

    values = {
        "initial_rate": initial_rate,
        "mean_reversion_speed": mean_reversion_speed,
        "long_term_mean": long_term_mean,
        "volatility": volatility,
    }

    for name, value in values.items():
        _validate_real_number(value, name)

    if mean_reversion_speed <= 0.0:
        raise ValueError(
            "mean_reversion_speed must be greater than zero."
        )

    if volatility < 0.0:
        raise ValueError(
            "volatility cannot be negative."
        )

    if require_non_negative_rates:
        if initial_rate < 0.0:
            raise ValueError(
                "CIR initial_rate cannot be negative."
            )

        if long_term_mean < 0.0:
            raise ValueError(
                "CIR long_term_mean cannot be negative."
            )


def _validate_simulation_inputs(
    years: float,
    steps_per_year: int,
    simulations: int,
    seed: int | None,
) -> None:
    """Validate Monte Carlo simulation settings."""

    _validate_real_number(years, "years")

    if years <= 0.0:
        raise ValueError(
            "years must be greater than zero."
        )

    if isinstance(
        steps_per_year,
        bool,
    ) or not isinstance(
        steps_per_year,
        int,
    ):
        raise TypeError(
            "steps_per_year must be an integer."
        )

    if steps_per_year <= 0:
        raise ValueError(
            "steps_per_year must be greater than zero."
        )

    if isinstance(
        simulations,
        bool,
    ) or not isinstance(
        simulations,
        int,
    ):
        raise TypeError(
            "simulations must be an integer."
        )

    if simulations <= 0:
        raise ValueError(
            "simulations must be greater than zero."
        )

    if seed is not None and (
        isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise TypeError(
            "seed must be an integer or None."
        )


def _build_time_grid(
    years: float,
    steps_per_year: int,
) -> tuple[np.ndarray, int, float]:
    """Construct an evenly spaced simulation time grid."""

    number_of_steps = round(
        years * steps_per_year
    )

    if number_of_steps <= 0:
        raise ValueError(
            "Simulation horizon must contain "
            "at least one time step."
        )

    time_grid = np.linspace(
        0.0,
        years,
        number_of_steps + 1,
        dtype=float,
    )

    time_step = years / number_of_steps

    return time_grid, number_of_steps, time_step


def calculate_expected_short_rate(
    initial_rate: float,
    mean_reversion_speed: float,
    long_term_mean: float,
    time_years: float,
) -> float:
    """Calculate the expected rate under mean reversion."""

    values = {
        "initial_rate": initial_rate,
        "mean_reversion_speed": mean_reversion_speed,
        "long_term_mean": long_term_mean,
        "time_years": time_years,
    }

    for name, value in values.items():
        _validate_real_number(value, name)

    if mean_reversion_speed <= 0.0:
        raise ValueError(
            "mean_reversion_speed must be greater than zero."
        )

    if time_years < 0.0:
        raise ValueError(
            "time_years cannot be negative."
        )

    return (
        long_term_mean
        + (initial_rate - long_term_mean)
        * exp(-mean_reversion_speed * time_years)
    )


def simulate_vasicek(
    parameters: VasicekParameters,
    years: float,
    steps_per_year: int = 252,
    simulations: int = 10_000,
    seed: int | None = None,
) -> InterestRateSimulation:
    """Simulate short rates using the Vasicek model."""

    if not isinstance(parameters, VasicekParameters):
        raise TypeError(
            "parameters must be a VasicekParameters instance."
        )

    _validate_simulation_inputs(
        years=years,
        steps_per_year=steps_per_year,
        simulations=simulations,
        seed=seed,
    )

    (
        time_grid,
        number_of_steps,
        time_step,
    ) = _build_time_grid(
        years=years,
        steps_per_year=steps_per_year,
    )

    random_generator = np.random.default_rng(seed)

    rate_paths = np.empty(
        (number_of_steps + 1, simulations),
        dtype=float,
    )

    rate_paths[0] = parameters.initial_rate

    random_shocks = random_generator.standard_normal(
        size=(number_of_steps, simulations)
    )

    diffusion_scale = (
        parameters.volatility * sqrt(time_step)
    )

    for step in range(1, number_of_steps + 1):
        previous_rates = rate_paths[step - 1]

        drift = (
            parameters.mean_reversion_speed
            * (
                parameters.long_term_mean
                - previous_rates
            )
            * time_step
        )

        diffusion = (
            diffusion_scale
            * random_shocks[step - 1]
        )

        rate_paths[step] = (
            previous_rates + drift + diffusion
        )

    return InterestRateSimulation(
        model_name="Vasicek",
        time_grid=time_grid,
        rate_paths=rate_paths,
    )


def simulate_cir(
    parameters: CIRParameters,
    years: float,
    steps_per_year: int = 252,
    simulations: int = 10_000,
    seed: int | None = None,
) -> InterestRateSimulation:
    """Simulate short rates using the CIR model."""

    if not isinstance(parameters, CIRParameters):
        raise TypeError(
            "parameters must be a CIRParameters instance."
        )

    _validate_simulation_inputs(
        years=years,
        steps_per_year=steps_per_year,
        simulations=simulations,
        seed=seed,
    )

    (
        time_grid,
        number_of_steps,
        time_step,
    ) = _build_time_grid(
        years=years,
        steps_per_year=steps_per_year,
    )

    random_generator = np.random.default_rng(seed)

    rate_paths = np.empty(
        (number_of_steps + 1, simulations),
        dtype=float,
    )

    rate_paths[0] = parameters.initial_rate

    random_shocks = random_generator.standard_normal(
        size=(number_of_steps, simulations)
    )

    square_root_time_step = sqrt(time_step)

    for step in range(1, number_of_steps + 1):
        previous_rates = rate_paths[step - 1]
        truncated_rates = np.maximum(
            previous_rates,
            0.0,
        )

        drift = (
            parameters.mean_reversion_speed
            * (
                parameters.long_term_mean
                - truncated_rates
            )
            * time_step
        )

        diffusion = (
            parameters.volatility
            * np.sqrt(truncated_rates)
            * square_root_time_step
            * random_shocks[step - 1]
        )

        next_rates = (
            previous_rates + drift + diffusion
        )

        rate_paths[step] = np.maximum(
            next_rates,
            0.0,
        )

    return InterestRateSimulation(
        model_name="CIR",
        time_grid=time_grid,
        rate_paths=rate_paths,
    )


def cir_feller_condition(
    parameters: CIRParameters,
) -> bool:
    """Return whether the CIR Feller condition is satisfied."""

    if not isinstance(parameters, CIRParameters):
        raise TypeError(
            "parameters must be a CIRParameters instance."
        )

    return (
        2.0
        * parameters.mean_reversion_speed
        * parameters.long_term_mean
        >= parameters.volatility**2
    )


def summarize_simulation(
    simulation: InterestRateSimulation,
) -> RateSimulationSummary:
    """Calculate descriptive statistics of terminal rates."""

    if not isinstance(
        simulation,
        InterestRateSimulation,
    ):
        raise TypeError(
            "simulation must be an "
            "InterestRateSimulation instance."
        )

    terminal_rates = simulation.terminal_rates

    return RateSimulationSummary(
        mean_terminal_rate=float(
            np.mean(terminal_rates)
        ),
        median_terminal_rate=float(
            np.median(terminal_rates)
        ),
        standard_deviation=float(
            np.std(terminal_rates, ddof=1)
            if terminal_rates.size > 1
            else 0.0
        ),
        fifth_percentile=float(
            np.percentile(terminal_rates, 5.0)
        ),
        ninety_fifth_percentile=float(
            np.percentile(terminal_rates, 95.0)
        ),
        minimum_rate=float(
            np.min(terminal_rates)
        ),
        maximum_rate=float(
            np.max(terminal_rates)
        ),
    )


def main() -> None:
    """Run example Vasicek and CIR simulations."""

    vasicek_parameters = VasicekParameters(
        initial_rate=0.04,
        mean_reversion_speed=0.60,
        long_term_mean=0.05,
        volatility=0.015,
    )

    cir_parameters = CIRParameters(
        initial_rate=0.04,
        mean_reversion_speed=0.60,
        long_term_mean=0.05,
        volatility=0.10,
    )

    years = 5.0
    steps_per_year = 12
    simulations = 10_000
    seed = 42

    vasicek_result = simulate_vasicek(
        parameters=vasicek_parameters,
        years=years,
        steps_per_year=steps_per_year,
        simulations=simulations,
        seed=seed,
    )

    cir_result = simulate_cir(
        parameters=cir_parameters,
        years=years,
        steps_per_year=steps_per_year,
        simulations=simulations,
        seed=seed,
    )

    vasicek_summary = summarize_simulation(
        vasicek_result
    )
    cir_summary = summarize_simulation(cir_result)

    print("Stochastic interest-rate simulation")
    print(f"Horizon: {years:.2f} years")
    print(
        f"Number of simulations: "
        f"{simulations:,}"
    )

    print("\nVasicek model")
    print(
        f"Mean terminal rate: "
        f"{vasicek_summary.mean_terminal_rate:.4%}"
    )
    print(
        f"Median terminal rate: "
        f"{vasicek_summary.median_terminal_rate:.4%}"
    )
    print(
        f"Standard deviation: "
        f"{vasicek_summary.standard_deviation:.4%}"
    )
    print(
        f"5th percentile: "
        f"{vasicek_summary.fifth_percentile:.4%}"
    )
    print(
        f"95th percentile: "
        f"{vasicek_summary.ninety_fifth_percentile:.4%}"
    )

    print("\nCIR model")
    print(
        f"Mean terminal rate: "
        f"{cir_summary.mean_terminal_rate:.4%}"
    )
    print(
        f"Median terminal rate: "
        f"{cir_summary.median_terminal_rate:.4%}"
    )
    print(
        f"Standard deviation: "
        f"{cir_summary.standard_deviation:.4%}"
    )
    print(
        f"5th percentile: "
        f"{cir_summary.fifth_percentile:.4%}"
    )
    print(
        f"95th percentile: "
        f"{cir_summary.ninety_fifth_percentile:.4%}"
    )
    print(
        f"Minimum simulated rate: "
        f"{cir_summary.minimum_rate:.4%}"
    )
    print(
        f"Feller condition satisfied: "
        f"{cir_feller_condition(cir_parameters)}"
    )


if __name__ == "__main__":
    main()