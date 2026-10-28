import numpy as np

from portfolio_pilot.config import PROJECT_ROOT, Results, load_config, to_plain


def test_config_paths_are_absolute_and_inside_the_project():
    cfg = load_config()
    for path in cfg["paths"].values():
        assert path.is_absolute()
        assert PROJECT_ROOT in path.parents
    assert cfg["seed"] == 7275
    assert abs(sum(cfg["scf"]["split"].values()) - 1) < 1e-9


def test_to_plain_handles_numpy_values_and_nan():
    plain = to_plain({1: np.int64(3), "x": [np.float64(0.123456789), np.nan], "a": np.array([1, 2])})
    assert plain == {"1": 3, "x": [0.123457, None], "a": [1, 2]}


def test_results_are_saved_and_extended_by_the_next_session(tmp_path):
    path = tmp_path / "numbers.json"
    first = Results(path)
    first.note("raw", "rows", np.int64(21800))
    first.save()

    second = Results(path)
    second.note("split", "n_strata", 18)
    second.save()

    third = Results(path)
    assert third["raw"]["rows"] == 21800 and third["split"]["n_strata"] == 18
