from dataclasses import replace

import pytest

from mantishrimp import DomainConfig, SimulationConfig


def test_szonja_baseline_is_valid_and_serialisable():
    config = SimulationConfig.szonja_baseline(seed=7)
    config.validate()

    payload = config.to_dict()
    assert payload["seed"] == 7
    assert payload["duration"] == 80.0
    assert payload["mechanics"]["repulsion_kt"] > 0
    assert payload["domain"]["boundary"] == "periodic"
    assert SimulationConfig.szonja_baseline(duration=5.0).duration == 5.0


def test_invalid_domain_is_rejected():
    config = replace(
        SimulationConfig(),
        domain=DomainConfig(x_min=1.0, x_max=0.0),
    )
    with pytest.raises(ValueError, match="maxima"):
        config.validate()
