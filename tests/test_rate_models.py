"""Tests for stochastic short-rate models."""

from dataclasses import FrozenInstanceError
from math import exp, inf, nan

import numpy as np
import pytest

from src.rate_models import (
    CIRParameters,
    InterestRateSimulation,
    RateSimulationSummary,
    VasicekParameters,
    calculate_expected_short_rate,
    cir_feller_condition,
    simulate_cir,
    simulate_vasicek,
    summarize_simulation,
)


@pytest.fixture
def vasicek_parameters() -> VasicekParameters:
    """Return baseline Vasicek parameters."""

    return VasicekParameters(
        initial_rate=0.04,
        mean_reversion_speed=0.60,
        long_term_mean=0.05,
        volatility=0.015,
    )


@pytest.fixture
def cir_parameters() -> CIRParameters:
    """Return baseline CIR parameters."""

    return CIRParameters(
        initial_rate=0.04,
        mean_reversion_speed=0.60,
        long_term_mean=0.05,
        volatility=0.10,
    )


def test_expected_rate_at_time_zero() -> None:
    """Expected rate at time zero should equal the initial rate."""

    result = calculate_expected_short_rate(
        initial_rate=0.04,
        mean_reversion_speed=0.60,
        long_term_mean=0.05,
        time_years=0.0,
    )

    assert result == pytest.approx(0.04)


def test_expected_rate_formula() -> None:
    """Expected rate should follow the mean-reversion formula."""

    result = calculate_expected_short_rate(
        initial_rate=0.04,
        mean_reversion_speed=0.60,
        long_term_mean=0.05,
        time_years=5.0,
    )

    expected = (
        0.05
        + (0.04 - 0.05)
        * exp(-0.60 * 5.0)
    )

    assert result == pytest.approx(expected)


def test_expected_rate_converges_toward_long_term_mean() -> None:
    """Long-horizon expectation should approach its mean."""

    result = calculate_expected_short_rate(
        initial_rate=0.01,
        mean_reversion_speed=0.60,
        long_term_mean=0.05,
        time_years=100.0,
    )

    assert result == pytest.approx(0.05, abs=1e-10)


def test_vasicek_simulation_shape(
    vasicek_parameters: VasicekParameters,
) -> None:
    """Vasicek output should have steps plus initial row."""

    result = simulate_vasicek(
        parameters=vasicek_parameters,
        years=2.0,
        steps_per_year=12,
        simulations=500,
        seed=42,
    )

    assert result.rate_paths.shape == (25, 500)
    assert result.time_grid.shape == (25,)


def test_cir_simulation_shape(
    cir_parameters: CIRParameters,
) -> None:
    """CIR output should have steps plus initial row."""

    result = simulate_cir(
        parameters=cir_parameters,
        years=2.0,
        steps_per_year=12,
        simulations=500,
        seed=42,
    )

    assert result.rate_paths.shape == (25, 500)
    assert result.time_grid.shape == (25,)


def test_vasicek_initial_rates(
    vasicek_parameters: VasicekParameters,
) -> None:
    """Every Vasicek path should begin at the initial rate."""

    result = simulate_vasicek(
        parameters=vasicek_parameters,
        years=1.0,
        steps_per_year=12,
        simulations=100,
        seed=42,
    )

    assert np.allclose(
        result.rate_paths[0],
        vasicek_parameters.initial_rate,
    )


def test_cir_initial_rates(
    cir_parameters: CIRParameters,
) -> None:
    """Every CIR path should begin at the initial rate."""

    result = simulate_cir(
        parameters=cir_parameters,
        years=1.0,
        steps_per_year=12,
        simulations=100,
        seed=42,
    )

    assert np.allclose(
        result.rate_paths[0],
        cir_parameters.initial_rate,
    )


def test_time_grid_endpoints(
    vasicek_parameters: VasicekParameters,
) -> None:
    """Time grid should begin at zero and end at horizon."""

    result = simulate_vasicek(
        parameters=vasicek_parameters,
        years=5.0,
        steps_per_year=12,
        simulations=10,
        seed=42,
    )

    assert result.time_grid[0] == pytest.approx(0.0)
    assert result.time_grid[-1] == pytest.approx(5.0)


def test_vasicek_is_reproducible(
    vasicek_parameters: VasicekParameters,
) -> None:
    """Equal seeds should produce equal Vasicek paths."""

    first_result = simulate_vasicek(
        parameters=vasicek_parameters,
        years=1.0,
        steps_per_year=12,
        simulations=100,
        seed=42,
    )

    second_result = simulate_vasicek(
        parameters=vasicek_parameters,
        years=1.0,
        steps_per_year=12,
        simulations=100,
        seed=42,
    )

    assert np.array_equal(
        first_result.rate_paths,
        second_result.rate_paths,
    )


def test_cir_is_reproducible(
    cir_parameters: CIRParameters,
) -> None:
    """Equal seeds should produce equal CIR paths."""

    first_result = simulate_cir(
        parameters=cir_parameters,
        years=1.0,
        steps_per_year=12,
        simulations=100,
        seed=42,
    )

    second_result = simulate_cir(
        parameters=cir_parameters,
        years=1.0,
        steps_per_year=12,
        simulations=100,
        seed=42,
    )

    assert np.array_equal(
        first_result.rate_paths,
        second_result.rate_paths,
    )


def test_different_seeds_produce_different_paths(
    vasicek_parameters: VasicekParameters,
) -> None:
    """Different random seeds should change simulated paths."""

    first_result = simulate_vasicek(
        parameters=vasicek_parameters,
        years=1.0,
        steps_per_year=12,
        simulations=100,
        seed=1,
    )

    second_result = simulate_vasicek(
        parameters=vasicek_parameters,
        years=1.0,
        steps_per_year=12,
        simulations=100,
        seed=2,
    )

    assert not np.array_equal(
        first_result.rate_paths,
        second_result.rate_paths,
    )


def test_cir_rates_are_non_negative(
    cir_parameters: CIRParameters,
) -> None:
    """Full truncation should keep CIR rates non-negative."""

    result = simulate_cir(
        parameters=cir_parameters,
        years=5.0,
        steps_per_year=12,
        simulations=2_000,
        seed=42,
    )

    assert np.all(result.rate_paths >= 0.0)


def test_vasicek_can_produce_negative_rates() -> None:
    """Vasicek should permit negative simulated rates."""

    parameters = VasicekParameters(
        initial_rate=0.01,
        mean_reversion_speed=0.30,
        long_term_mean=0.02,
        volatility=0.20,
    )

    result = simulate_vasicek(
        parameters=parameters,
        years=1.0,
        steps_per_year=12,
        simulations=5_000,
        seed=42,
    )

    assert np.any(result.rate_paths < 0.0)


def test_simulated_vasicek_mean_is_near_expectation(
    vasicek_parameters: VasicekParameters,
) -> None:
    """Monte Carlo mean should approximate analytical expectation."""

    years = 2.0

    result = simulate_vasicek(
        parameters=vasicek_parameters,
        years=years,
        steps_per_year=12,
        simulations=15_000,
        seed=42,
    )

    analytical_mean = calculate_expected_short_rate(
        initial_rate=vasicek_parameters.initial_rate,
        mean_reversion_speed=(
            vasicek_parameters.mean_reversion_speed
        ),
        long_term_mean=(
            vasicek_parameters.long_term_mean
        ),
        time_years=years,
    )

    assert np.mean(
        result.terminal_rates
    ) == pytest.approx(
        analytical_mean,
        abs=0.002,
    )


def test_zero_volatility_paths_are_identical() -> None:
    """Without volatility, every simulated path should match."""

    parameters = VasicekParameters(
        initial_rate=0.04,
        mean_reversion_speed=0.60,
        long_term_mean=0.05,
        volatility=0.0,
    )

    result = simulate_vasicek(
        parameters=parameters,
        years=2.0,
        steps_per_year=12,
        simulations=100,
        seed=42,
    )

    first_path = result.rate_paths[:, [0]]

    assert np.allclose(
        result.rate_paths,
        first_path,
    )


def test_simulation_properties(
    vasicek_parameters: VasicekParameters,
) -> None:
    """Simulation convenience properties should be consistent."""

    result = simulate_vasicek(
        parameters=vasicek_parameters,
        years=2.0,
        steps_per_year=12,
        simulations=500,
        seed=42,
    )

    assert result.model_name == "Vasicek"
    assert result.number_of_steps == 24
    assert result.number_of_simulations == 500
    assert result.terminal_rates.shape == (500,)
    assert result.average_path.shape == (25,)


def test_terminal_rates_returns_copy(
    vasicek_parameters: VasicekParameters,
) -> None:
    """Changing terminal-rate output should not alter paths."""

    result = simulate_vasicek(
        parameters=vasicek_parameters,
        years=1.0,
        steps_per_year=12,
        simulations=10,
        seed=42,
    )

    terminal_rates = result.terminal_rates
    original_value = result.rate_paths[-1, 0]

    terminal_rates[0] = 999.0

    assert result.rate_paths[-1, 0] == pytest.approx(
        original_value
    )


def test_feller_condition_is_satisfied(
    cir_parameters: CIRParameters,
) -> None:
    """Baseline CIR parameters should satisfy Feller."""

    assert cir_feller_condition(cir_parameters) is True


def test_feller_condition_is_not_satisfied() -> None:
    """High volatility may violate the Feller condition."""

    parameters = CIRParameters(
        initial_rate=0.04,
        mean_reversion_speed=0.60,
        long_term_mean=0.05,
        volatility=0.30,
    )

    assert cir_feller_condition(parameters) is False


def test_simulation_summary(
    vasicek_parameters: VasicekParameters,
) -> None:
    """Summary should contain consistent terminal statistics."""

    result = simulate_vasicek(
        parameters=vasicek_parameters,
        years=2.0,
        steps_per_year=12,
        simulations=1_000,
        seed=42,
    )

    summary = summarize_simulation(result)

    assert isinstance(summary, RateSimulationSummary)
    assert summary.minimum_rate <= summary.fifth_percentile
    assert (
        summary.fifth_percentile
        <= summary.median_terminal_rate
    )
    assert (
        summary.median_terminal_rate
        <= summary.ninety_fifth_percentile
    )
    assert (
        summary.ninety_fifth_percentile
        <= summary.maximum_rate
    )
    assert summary.standard_deviation > 0.0


def test_parameter_dataclasses_are_immutable(
    vasicek_parameters: VasicekParameters,
) -> None:
    """Model parameters should not change after creation."""

    with pytest.raises(FrozenInstanceError):
        vasicek_parameters.initial_rate = 0.10  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("mean_reversion_speed", 0.0),
        ("mean_reversion_speed", -0.50),
        ("volatility", -0.01),
    ],
)
def test_invalid_vasicek_parameters(
    field_name: str,
    invalid_value: float,
) -> None:
    """Invalid Vasicek parameters should be rejected."""

    values = {
        "initial_rate": 0.04,
        "mean_reversion_speed": 0.60,
        "long_term_mean": 0.05,
        "volatility": 0.015,
    }

    values[field_name] = invalid_value

    with pytest.raises(ValueError):
        VasicekParameters(**values)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("initial_rate", -0.01),
        ("long_term_mean", -0.01),
        ("mean_reversion_speed", 0.0),
        ("volatility", -0.01),
    ],
)
def test_invalid_cir_parameters(
    field_name: str,
    invalid_value: float,
) -> None:
    """Invalid CIR parameters should be rejected."""

    values = {
        "initial_rate": 0.04,
        "mean_reversion_speed": 0.60,
        "long_term_mean": 0.05,
        "volatility": 0.10,
    }

    values[field_name] = invalid_value

    with pytest.raises(ValueError):
        CIRParameters(**values)


@pytest.mark.parametrize(
    "invalid_value",
    [nan, inf, -inf],
)
def test_non_finite_model_parameters(
    invalid_value: float,
) -> None:
    """Model parameters must contain finite values."""

    with pytest.raises(ValueError):
        VasicekParameters(
            initial_rate=invalid_value,
            mean_reversion_speed=0.60,
            long_term_mean=0.05,
            volatility=0.015,
        )


@pytest.mark.parametrize("years", [0.0, -1.0])
def test_invalid_simulation_years(
    vasicek_parameters: VasicekParameters,
    years: float,
) -> None:
    """Simulation horizon must be positive."""

    with pytest.raises(ValueError):
        simulate_vasicek(
            parameters=vasicek_parameters,
            years=years,
        )


@pytest.mark.parametrize(
    "steps_per_year",
    [0, -1],
)
def test_invalid_steps_per_year(
    vasicek_parameters: VasicekParameters,
    steps_per_year: int,
) -> None:
    """Steps per year must be positive."""

    with pytest.raises(ValueError):
        simulate_vasicek(
            parameters=vasicek_parameters,
            years=1.0,
            steps_per_year=steps_per_year,
        )


@pytest.mark.parametrize(
    "simulations",
    [0, -1],
)
def test_invalid_simulation_count(
    vasicek_parameters: VasicekParameters,
    simulations: int,
) -> None:
    """Simulation count must be positive."""

    with pytest.raises(ValueError):
        simulate_vasicek(
            parameters=vasicek_parameters,
            years=1.0,
            simulations=simulations,
        )


def test_non_integer_simulation_count(
    vasicek_parameters: VasicekParameters,
) -> None:
    """Simulation count must be an integer."""

    with pytest.raises(TypeError):
        simulate_vasicek(
            parameters=vasicek_parameters,
            years=1.0,
            simulations=100.5,  # type: ignore[arg-type]
        )


def test_invalid_seed_type(
    vasicek_parameters: VasicekParameters,
) -> None:
    """Random seed must be an integer or None."""

    with pytest.raises(TypeError):
        simulate_vasicek(
            parameters=vasicek_parameters,
            years=1.0,
            seed="42",  # type: ignore[arg-type]
        )


def test_invalid_vasicek_parameter_object() -> None:
    """Vasicek simulation requires its parameter dataclass."""

    with pytest.raises(TypeError):
        simulate_vasicek(
            parameters="invalid",  # type: ignore[arg-type]
            years=1.0,
        )


def test_invalid_cir_parameter_object() -> None:
    """CIR simulation requires its parameter dataclass."""

    with pytest.raises(TypeError):
        simulate_cir(
            parameters="invalid",  # type: ignore[arg-type]
            years=1.0,
        )


def test_invalid_summary_object() -> None:
    """Summary calculation requires a simulation result."""

    with pytest.raises(TypeError):
        summarize_simulation("invalid")  # type: ignore[arg-type]


def test_invalid_feller_parameter_object() -> None:
    """Feller test requires CIR parameters."""

    with pytest.raises(TypeError):
        cir_feller_condition(
            "invalid"  # type: ignore[arg-type]
        )


def test_invalid_simulation_array_dimensions() -> None:
    """Simulation paths must be a two-dimensional array."""

    with pytest.raises(ValueError):
        InterestRateSimulation(
            model_name="Invalid",
            time_grid=np.array([0.0, 1.0]),
            rate_paths=np.array([0.04, 0.05]),
        )


def test_mismatched_simulation_dimensions() -> None:
    """Time grid and path rows must have equal lengths."""

    with pytest.raises(ValueError):
        InterestRateSimulation(
            model_name="Invalid",
            time_grid=np.array([0.0, 1.0]),
            rate_paths=np.zeros((3, 10)),
        )