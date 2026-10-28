import numpy as np
import pandas as pd
import pytest

from portfolio_pilot.config import Results
from portfolio_pilot.prepare import fit_caps, fit_scaler, ratio, weighted_median, weighted_resample


def test_weighted_median_uses_the_weights():
    assert weighted_median([1, 2, 3], [1, 1, 1]) == 2
    assert weighted_median([1, 2, 3], [1, 1, 10]) == 3
    assert weighted_median([3, 1, 2], [10, 1, 1]) == 3        # input order does not matter


def test_ratio_is_undefined_when_the_denominator_is_not_positive():
    out = ratio(pd.Series([10.0, 10.0, 10.0]), pd.Series([5.0, 0.0, -2.0]))
    assert out.iloc[0] == 2.0 and out.iloc[1:].isna().all()


def test_weighted_resample_is_reproducible_and_follows_the_weights():
    df = pd.DataFrame({"id": range(100), "weight": [1.0] * 99 + [10_000.0]})
    a, b = weighted_resample(df, 5), weighted_resample(df, 5)
    assert a["id"].tolist() == b["id"].tolist()
    assert len(a) == len(df)
    assert (a["id"] == 99).mean() > 0.9


def test_caps_come_from_the_training_households_only(households):
    train, other = households.iloc[:700], households.iloc[700:].copy()
    caps = fit_caps(train, 0.99)
    other["debt"] = other["debt"] * 1000                       # extreme values elsewhere must not matter
    assert fit_caps(train, 0.99) == caps
    expected = float(ratio(train["debt"], train["income"]).quantile(0.99))
    assert caps["dti"] == round(expected, 2)                   # caps are rounded to two decimals


def test_fit_caps_records_training_counts(households):
    results = Results()
    caps = fit_caps(households, 0.99, results)
    n_above = int((ratio(households["debt"], households["income"]) > caps["dti"]).sum())
    assert results["features"]["train_dti_above_cap"] == n_above
    assert set(results["features"]["caps_from_train"]) == {"dti", "liq_to_income"}


def test_scaler_is_fitted_on_the_training_resample(households):
    cols = ["age", "income"]
    scaler, base = fit_scaler(households, cols, seed=7275)
    assert np.allclose(scaler.transform(base[cols]).mean(axis=0), 0, atol=1e-9)
    assert np.allclose(scaler.transform(base[cols]).std(axis=0), 1, atol=1e-9)
