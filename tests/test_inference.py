import numpy as np
import pytest

pytest.importorskip("pymc")

from mantishrimp.inference import MODEL_ORDER, build_count_model


@pytest.mark.parametrize("model_name", MODEL_ORDER)
def test_all_orca_count_models_build(model_name):
    model = build_count_model(model_name, [0, 1, 2, 0], exposure=2.0)
    assert "eta" in model.named_vars
    assert "counts" in model.named_vars
    if model_name in {"Z2P", "hetero3"}:
        assert "p_zero" in model.named_vars
    if model_name in {"Dis2P", "hetero3"}:
        assert "sigma_lambda" in model.named_vars


def test_count_validation_rejects_fractional_values():
    with pytest.raises(ValueError, match="integers"):
        build_count_model("homo", np.array([0.0, 1.5]), exposure=1.0)
