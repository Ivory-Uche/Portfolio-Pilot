import numpy as np
import pandas as pd
import pytest

from conftest import make_scf_file
from portfolio_pilot.clean import (clean_etf_prices, clean_scf, etf_wide_to_long, iqr_outlier_table,
                                   load_etf_prices, load_scf)
from portfolio_pilot.config import Results


def test_load_scf_keeps_mapped_columns_and_implicate(tmp_path, columns):
    path = make_scf_file(tmp_path / "scf.csv", n_households=10)
    results = Results()
    full = load_scf(path, columns, results)
    assert list(full.columns) == list(columns) + ["implicate"]
    assert results["raw"]["households"] == 10
    assert results["raw"]["implicates"] == [1, 2, 3, 4, 5]
    assert (full["record_id"] == full["household_id"] * 10 + full["implicate"]).all()


def test_load_scf_reports_missing_columns(tmp_path, columns):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"YY1": [1], "Y1": [11]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Columns not in the file"):
        load_scf(path, columns)


def _full(tmp_path, columns, **overrides):
    full = load_scf(make_scf_file(tmp_path / "scf.csv", n_households=20), columns)
    for col, (rows, value) in overrides.items():
        full.loc[full["record_id"].isin(rows), col] = value
    return full


def test_clean_scf_drops_only_invalid_households(tmp_path, columns):
    # household 1: age 150, household 2: weight 0, household 3: both risk flags, household 4: negative debt
    full = _full(tmp_path, columns,
                 age=([11], 150), weight=([21], 0.0), yes_risk=([31], 1), no_risk=([31], 1), debt=([41], -5.0))
    results = Results()
    _, hh = clean_scf(full, implicate=1, results=results)
    assert set(range(1, 21)) - set(hh["household_id"]) == {1, 2, 3}      # negative debt is only reported
    assert results["errors"]["rows_dropped"] == 3
    assert results["errors"]["debt_negative"] == 1
    assert len(hh) == 17


def test_clean_scf_one_row_per_household_and_derived_columns(tmp_path, columns):
    full = _full(tmp_path, columns, age=([11, 21, 31], [30, 40, 80]))
    _, hh = clean_scf(full, implicate=1)
    assert hh["household_id"].is_unique
    assert hh.set_index("household_id").loc[1, "age_band"] == "<35"
    assert hh.set_index("household_id").loc[2, "age_band"] == "35-44"
    assert hh.set_index("household_id").loc[3, "age_band"] == "75+"
    assert set(hh["risk_willingness"]) <= {0, 1, 2}


def test_clean_scf_risk_willingness_mapping(tmp_path, columns):
    full = _full(tmp_path, columns, yes_risk=([11, 21, 31], [0, 1, 0]), no_risk=([11, 21, 31], [1, 0, 0]))
    _, hh = clean_scf(full, implicate=1)
    rw = hh.set_index("household_id")["risk_willingness"]
    assert (rw.loc[1], rw.loc[2], rw.loc[3]) == (0, 2, 1)


def test_clean_scf_removes_duplicate_records(tmp_path, columns):
    full = _full(tmp_path, columns)
    doubled = pd.concat([full, full.iloc[:7]])
    results = Results()
    deduped, _ = clean_scf(doubled, implicate=1, results=results)
    assert results["duplicates"]["duplicate_record_ids"] == 7
    assert len(deduped) == len(full)


def test_iqr_outlier_table_counts_values_beyond_fences():
    df = pd.DataFrame({"x": [1.0] * 20 + [2.0] * 20 + [1000.0]})
    table = iqr_outlier_table(df, ["x"])
    assert table.loc[0, "n_beyond_3iqr"] == 1
    assert table.loc[0, "max"] == 1000.0


def _prices():
    idx = pd.bdate_range("2020-01-01", periods=10, name="date")
    return pd.DataFrame({"A": np.arange(1.0, 11.0), "B": np.arange(11.0, 21.0)}, index=idx)


def test_load_etf_prices_rejects_a_ticker_with_no_prices(tmp_path):
    wide = _prices()
    wide["B"] = np.nan
    path = tmp_path / "etf.csv"
    wide.to_csv(path)
    with pytest.raises(ValueError, match=r"No prices at all for \['B'\]"):
        load_etf_prices(path, ["A", "B"])


def test_load_etf_prices_checks_structure(tmp_path):
    path = tmp_path / "etf.csv"
    _prices().to_csv(path)
    results = Results()
    wide = load_etf_prices(path, ["A", "B", "C"], results)
    assert results["etf_raw"]["tickers_missing"] == ["C"]
    assert results["etf_raw"]["rows_trading_days"] == len(wide) == 10


def test_clean_etf_prices_fixes_duplicates_and_impossible_prices():
    wide = _prices()
    long = etf_wide_to_long(wide)
    long = pd.concat([long, long.iloc[:3]])                                  # three duplicate rows
    long.loc[long.index[5], "adj_close"] = -1.0                              # one impossible price
    results = Results()
    prices = clean_etf_prices(long, ["A", "B"], results)
    assert results["etf"]["duplicate_date_ticker"] == 3
    assert results["etf"]["non_positive_prices"] == 1
    assert list(prices.columns) == ["A", "B"]
    assert prices.index.is_monotonic_increasing


def test_clean_etf_prices_fills_one_day_gaps_only():
    wide = _prices()
    wide.iloc[3, 0] = np.nan                       # isolated gap: forward filled
    wide.iloc[6:8, 1] = np.nan                     # two-day gap: those days are dropped
    results = Results()
    prices = clean_etf_prices(etf_wide_to_long(wide), ["A", "B"], results)
    assert results["etf"]["cells_forward_filled"] == 2        # the isolated cell + the first day of the long gap
    assert prices.iloc[3]["A"] == wide.iloc[2]["A"]           # previous day's price
    assert len(prices) == 10 - 1                              # only the second day of the long gap is dropped
