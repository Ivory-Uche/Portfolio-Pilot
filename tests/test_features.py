import numpy as np
import pandas as pd
import pytest

from portfolio_pilot.features import (CLASSIFIER_FEATURES, CLUSTER_FEATURES, build_features,
                                      build_partition_features, daily_log_returns)

CAPS = {"dti": 2.0, "liq_to_income": 3.0}


def _row(**kw):
    base = {"income": 100.0, "networth": 500.0, "liq": 50.0, "debt": 100.0, "equity": 20.0, "fin": 100.0}
    base.update(kw)
    return pd.DataFrame([base])


def test_ratios_are_capped_at_the_given_caps():
    out = build_features(_row(debt=10_000.0, liq=10_000.0), CAPS)
    assert out.loc[0, "dti"] == CAPS["dti"]
    assert out.loc[0, "liq_to_income"] == CAPS["liq_to_income"]


def test_ratios_below_the_cap_are_unchanged():
    out = build_features(_row(debt=100.0, liq=50.0), CAPS)
    assert out.loc[0, "dti"] == 1.0 and out.loc[0, "liq_to_income"] == 0.5


def test_non_positive_income_is_flagged_and_filled():
    out = build_features(pd.concat([_row(income=0.0, debt=50.0, liq=10.0),
                                    _row(income=-5.0, debt=0.0, liq=0.0)], ignore_index=True), CAPS)
    assert out["income_nonpositive"].tolist() == [1, 1]
    assert out.loc[0, "dti"] == CAPS["dti"] and out.loc[0, "liq_to_income"] == CAPS["liq_to_income"]
    assert out.loc[1, "dti"] == 0.0 and out.loc[1, "liq_to_income"] == 0.0


def test_equity_share_is_between_zero_and_one_and_zero_without_financial_assets():
    out = build_features(pd.concat([_row(equity=500.0, fin=100.0), _row(fin=0.0, equity=0.0)],
                                   ignore_index=True), CAPS)
    assert out.loc[0, "equity_share"] == 1.0
    assert out.loc[1, "equity_share"] == 0.0


def test_asinh_columns_are_added():
    out = build_features(_row(networth=-1000.0), CAPS)
    assert out.loc[0, "asinh_networth"] == pytest.approx(np.arcsinh(-1000.0))
    assert {"asinh_income", "asinh_liq", "asinh_dti", "asinh_liq_to_income"} <= set(out.columns)


def test_build_features_does_not_change_its_input():
    df = _row()
    build_features(df, CAPS)
    assert list(df.columns) == ["income", "networth", "liq", "debt", "equity", "fin"]


def test_each_partition_uses_the_same_training_caps(households):
    parts = {"train": households.iloc[:700], "val": households.iloc[700:]}
    feats = build_partition_features(parts, CAPS)
    for d in feats.values():
        assert d["dti"].max() <= CAPS["dti"]
        assert d["liq_to_income"].max() <= CAPS["liq_to_income"]
    assert set(CLUSTER_FEATURES + CLASSIFIER_FEATURES) <= set(feats["val"].columns)


def test_missing_model_features_in_training_are_rejected(households):
    broken = households.copy()
    broken.loc[0, "age"] = np.nan
    with pytest.raises(ValueError, match="Missing values"):
        build_partition_features({"train": broken, "val": households}, CAPS)


def test_daily_log_returns():
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]}, index=pd.bdate_range("2020-01-01", periods=3))
    ret = daily_log_returns(prices)
    assert len(ret) == 2
    assert ret["A"].iloc[0] == pytest.approx(np.log(1.1))
    assert ret["A"].iloc[1] == pytest.approx(np.log(0.9))
