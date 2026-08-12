import numpy as np

from mantishrimp.validation import (
    gamma_shape_rate_from_mean_sd,
    sample_lambda,
    sample_rates,
    simulate_Population,
    simulate_population,
)


def test_gamma_parameter_conversion():
    shape, rate = gamma_shape_rate_from_mean_sd(2.0, 1.0)
    assert shape == 4.0
    assert rate == 2.0


def test_synthetic_population_is_reproducible():
    first = simulate_population(
        20,
        3.0,
        mode="heterogeneous",
        mean_rate=1.2,
        rate_sd=0.4,
        zero_fraction=0.2,
        seed=17,
    )
    second = simulate_population(
        20,
        3.0,
        mode="heterogeneous",
        mean_rate=1.2,
        rate_sd=0.4,
        zero_fraction=0.2,
        seed=17,
    )
    assert first.equals(second)


def test_orca_compatibility_wrappers_preserve_schema():
    rates = sample_lambda(5, mu_lambda=0.5, seed=2)
    np.testing.assert_allclose(rates, 0.5)

    population = simulate_Population(5, T=2.0, rates=rates, seed=2)
    assert set(population) == {"n_cells", "max_time", "rates", "n_events"}
    assert population["n_events"].shape == (5,)


def test_zero_fraction_can_create_inactive_cells():
    rates = sample_rates(
        10,
        mode="heterogeneous",
        mean_rate=1.0,
        rate_sd=0.5,
        zero_fraction=1.0,
        seed=1,
    )
    np.testing.assert_array_equal(rates, np.zeros(10))
