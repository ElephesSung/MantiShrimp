import numpy as np

from mantishrimp.analysis import (
    ground_truth_for_model,
    scenario_directory,
    scenario_ground_truth_common,
    summarize_replicates,
)


def test_scenario_helpers_map_parameters(tmp_path):
    scenario = {
        "number": "01",
        "name": "mixed population",
        "mu_lambda": 1.2,
        "sigma_lambda": 0.3,
        "p_zero": 0.25,
    }
    assert scenario_directory(tmp_path, scenario).name == "01_mixed-population"
    assert ground_truth_for_model(scenario, "Z2P") == {
        "lambda": 1.2,
        "p_zero": 0.25,
    }
    assert scenario_ground_truth_common(scenario)["sigma_lambda"] == 0.3


def test_replicate_summary_supports_sem():
    import pandas as pd

    data = pd.DataFrame({"model": ["a", "a"], "x": [1, 1], "value": [1.0, 3.0]})
    summary = summarize_replicates(
        data,
        group_columns=["model", "x"],
        value_column="value",
        error_style="sem",
    )
    assert summary.loc[0, "mean"] == 2.0
    assert np.isclose(summary.loc[0, "error"], 1.0)
