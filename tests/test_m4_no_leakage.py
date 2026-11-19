"""M4 leakage checks (run with: pytest). Uses the real data, so it is skipped if the SCF file is missing."""
import numpy as np
import pytest

from portfolio_pilot import dataset
from portfolio_pilot.config import load_config
from portfolio_pilot.evaluate import cross_validate
from portfolio_pilot.features import CLASSIFIER_FEATURES
from portfolio_pilot.models.baselines import make_logreg


@pytest.fixture(scope="module")
def built():
    cfg = load_config()
    if not cfg["paths"]["scf_file"].exists():
        pytest.skip("SCF file not downloaded")
    return cfg, dataset.build(cfg)


def test_test_set_not_loaded(built):
    _, b = built
    assert "test" not in b["data"]


def test_no_household_in_two_partitions(built):
    _, b = built
    assert not set(b["data"]["train"]["household_id"]) & set(b["data"]["val"]["household_id"])


def test_scaler_learns_from_training_rows_only(built):
    cfg, b = built
    tr = b["data"]["train"]
    m = make_logreg(seed=cfg["seed"]).fit(tr[CLASSIFIER_FEATURES], tr["archetype"])
    assert np.allclose(m.named_steps["scale"].mean_, tr[CLASSIFIER_FEATURES].mean().to_numpy())


def test_cross_validation_runs(built):
    cfg, b = built
    tr = b["data"]["train"]
    cv = cross_validate(make_logreg(seed=cfg["seed"]), tr[CLASSIFIER_FEATURES], tr["archetype"].to_numpy(),
                        tr["weight"].to_numpy(float), n_folds=3, seed=cfg["seed"])
    assert len(cv) == 3
