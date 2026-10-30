import numpy as np
import pandas as pd

from portfolio_pilot.config import Results
from portfolio_pilot.split import load_partitions, save_split_ids, split_households, split_returns_by_time

SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}


def test_household_split_is_disjoint_and_complete(households):
    parts = split_households(households, SPLIT, seed=7275)
    ids = [set(d["household_id"]) for d in parts.values()]
    assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2])
    assert set().union(*ids) == set(households["household_id"])


def test_household_split_sizes_follow_the_ratios(households):
    parts = split_households(households, SPLIT, seed=7275)
    n = len(households)
    assert abs(len(parts["train"]) - 0.70 * n) <= 2
    assert abs(len(parts["val"]) - 0.15 * n) <= 2
    assert abs(len(parts["test"]) - 0.15 * n) <= 2


def test_household_split_is_reproducible_and_seed_dependent(households):
    a = split_households(households, SPLIT, seed=7275)
    b = split_households(households, SPLIT, seed=7275)
    c = split_households(households, SPLIT, seed=1)
    assert a["train"]["household_id"].tolist() == b["train"]["household_id"].tolist()
    assert a["train"]["household_id"].tolist() != c["train"]["household_id"].tolist()


def test_household_split_keeps_strata_balanced(households):
    parts = split_households(households, SPLIT, seed=7275)
    share = lambda d: (d["risk_willingness"].value_counts(normalize=True)).sort_index()
    for name in ["val", "test"]:
        assert (share(parts["train"]) - share(parts[name])).abs().max() < 0.05


def test_tiny_strata_are_merged(households):
    households = households.copy()
    households.loc[households.index[:3], "age_band"] = "rare"       # 3 households < min_stratum_size
    results = Results()
    parts = split_households(households, SPLIT, seed=7275, min_stratum_size=10, results=results)
    counts = (households["age_band"] + "|" + households["risk_willingness"].astype(str)).value_counts()
    assert (counts < 10).any()
    assert results["split"]["n_strata"] == int((counts >= 10).sum()) + 1     # small strata become one "other"
    assert sum(len(d) for d in parts.values()) == len(households)           # no household is lost


def test_saved_ids_rebuild_the_same_partitions(households, tmp_path):
    parts = split_households(households, SPLIT, seed=7275)
    save_split_ids(parts, tmp_path / "ids.csv")
    rebuilt = load_partitions(households, tmp_path / "ids.csv")
    for name in parts:
        assert rebuilt[name]["household_id"].tolist() == parts[name]["household_id"].tolist()
        assert np.allclose(rebuilt[name]["income"], parts[name]["income"])


def test_time_split_has_no_overlap_and_no_look_ahead():
    idx = pd.bdate_range("2016-01-04", "2025-12-31")
    ret = pd.DataFrame({"A": np.arange(len(idx), dtype=float)}, index=idx)
    results = Results()
    parts = split_returns_by_time(ret, "2021-12-31", "2023-12-31", results)
    assert parts["train"].index.max() <= pd.Timestamp("2021-12-31")
    assert parts["val"].index.min() > parts["train"].index.max()
    assert parts["val"].index.max() <= pd.Timestamp("2023-12-31")
    assert parts["test"].index.min() > parts["val"].index.max()
    assert sum(len(d) for d in parts.values()) == len(ret)
    assert results["etf_split"]["test"]["to"] == "2025-12-31"
