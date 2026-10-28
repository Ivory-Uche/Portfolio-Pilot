"""Fix errors, duplicates, and inconsistent values in the SCF and ETF data."""
import numpy as np
import pandas as pd

from .config import Results


def _results(results):
    return Results() if results is None else results


# ---------------------------------------------------------------- SCF

def load_scf(path, columns, results=None):
    """Read the SCF summary extract and keep the columns in `columns` (new name -> SCF name)."""
    results = _results(results)
    raw = pd.read_csv(path, low_memory=False)
    raw.columns = [c.upper() for c in raw.columns]
    results.note("raw", "rows", len(raw))
    results.note("raw", "columns", raw.shape[1])

    missing_cols = [c for c in columns.values() if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"Columns not in the file: {missing_cols}. Fix scf.columns in config.yaml.")

    full = raw[list(columns.values())].copy()
    full.columns = list(columns.keys())
    full["implicate"] = (full["record_id"] % 10).astype(int)    # Y1 = YY1*10 + implicate
    results.note("raw", "households", full["household_id"].nunique())
    results.note("raw", "implicates", sorted(full["implicate"].unique().tolist()))
    return full


def quality_checks(hh):
    """Boolean masks, one per check; True marks a row with a potential problem."""
    return {
        "age_under_18_or_over_100": (hh["age"] < 18) | (hh["age"] > 100),
        "weight_not_positive": hh["weight"] <= 0,
        "liq_negative": hh["liq"] < 0,
        "debt_negative": hh["debt"] < 0,
        "equity_negative": hh["equity"] < 0,
        "fin_negative": hh["fin"] < 0,
        "equity_more_than_fin": hh["equity"] > hh["fin"] + 1,
        "liq_more_than_fin": hh["liq"] > hh["fin"] + 1,
        "both_risk_flags_1": (hh["yes_risk"] == 1) & (hh["no_risk"] == 1),
    }


def clean_scf(full, implicate=1, results=None):
    """Drop duplicates and invalid households and derive risk willingness and age band.

    Returns the de-duplicated table with all implicates and the cleaned table
    with one row per household (the chosen implicate).
    """
    results = _results(results)
    results.note("duplicates", "exact_duplicate_rows", int(full.duplicated().sum()))
    results.note("duplicates", "duplicate_record_ids", int(full["record_id"].duplicated().sum()))
    full = full.drop_duplicates(subset="record_id")

    hh = full[full["implicate"] == implicate].copy()           # one row per household
    results.note("implicate_1", "rows", len(hh))

    na = hh.isna().sum()
    results.note("missing", "by_column", na[na > 0].to_dict())
    results.note("missing", "total", int(na.sum()))

    checks = quality_checks(hh)
    for name, mask in checks.items():
        results.note("errors", name, int(mask.sum()))

    # Only these conditions remove rows. Other unusual financial values may be genuine.
    bad = checks["age_under_18_or_over_100"] | checks["weight_not_positive"] | checks["both_risk_flags_1"]
    results.note("errors", "rows_dropped", int(bad.sum()))
    hh = hh[~bad].reset_index(drop=True)

    # Real values that need special handling later (not errors)
    for name, mask in {"income_zero": hh["income"] == 0, "income_negative": hh["income"] < 0,
                       "networth_negative": hh["networth"] < 0, "fin_zero": hh["fin"] == 0,
                       "debt_zero": hh["debt"] == 0}.items():
        results.note("genuine", name, int(mask.sum()))

    # Risk willingness: 0 = no risk, 1 = average, 2 = above-average / substantial
    hh["risk_willingness"] = np.select([hh["no_risk"] == 1, hh["yes_risk"] == 1], [0, 2], default=1)
    hh["age_band"] = pd.cut(hh["age"], [0, 34, 44, 54, 64, 74, 200],
                            labels=["<35", "35-44", "45-54", "55-64", "65-74", "75+"]).astype(str)
    results.note("implicate_1", "rows_after_cleaning", len(hh))
    return full, hh


def iqr_outlier_table(df, columns, k=3):
    """Count values beyond k x IQR fences, with each column's min and max."""
    rows = []
    for col in columns:
        x = df[col].astype(float)
        q1, q3 = x.quantile([0.25, 0.75])
        n = int(((x < q1 - k * (q3 - q1)) | (x > q3 + k * (q3 - q1))).sum())
        rows.append({"variable": col, "n_beyond_3iqr": n, "min": x.min(), "max": x.max()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- ETF

def load_etf_prices(path, tickers, results=None):
    """Read the wide price file (one column per ETF) and check its structure."""
    results = _results(results)
    wide = pd.read_csv(path, index_col=0, parse_dates=True)
    wide.index.name = "date"
    if len(wide) == 0:
        raise ValueError("No prices in the file: the download failed. Delete it and download again.")
    empty = [c for c in wide.columns if wide[c].isna().all()]
    if empty:
        raise ValueError(f"No prices at all for {empty}: the download was incomplete. "
                         "Delete the file and run python -m portfolio_pilot.download_etf again.")

    results.note("etf_raw", "rows_trading_days", len(wide))
    results.note("etf_raw", "columns", wide.shape[1])
    results.note("etf_raw", "date_range", [str(wide.index.min().date()), str(wide.index.max().date())])
    results.note("etf_raw", "tickers_missing", sorted(set(tickers) - set(wide.columns)))
    results.note("etf_raw", "unexpected_columns", sorted(set(wide.columns) - set(tickers)))
    results.note("etf_raw", "missing_by_ticker", wide.isna().sum().to_dict())
    results.note("etf_raw", "first_valid_date", {c: str(wide[c].first_valid_index().date()) for c in wide.columns})
    return wide


def etf_wide_to_long(wide):
    """One row per date and ticker: columns date, ticker, adj_close."""
    return wide.reset_index().melt(id_vars="date", var_name="ticker", value_name="adj_close")


def clean_etf_prices(long, tickers, results=None):
    """Remove duplicates and impossible prices; return one column per ETF, in the order of `tickers`."""
    results = _results(results)
    results.note("etf", "rows", len(long))
    results.note("etf", "tickers_missing", sorted(set(tickers) - set(long["ticker"])))
    results.note("etf", "duplicate_date_ticker", int(long.duplicated(["date", "ticker"]).sum()))
    results.note("etf", "missing_prices", int(long["adj_close"].isna().sum()))
    results.note("etf", "non_positive_prices", int((long["adj_close"] <= 0).sum()))

    long = long.drop_duplicates(["date", "ticker"], keep="last")   # keep one row per ETF per day
    long = long[long["adj_close"] > 0]                             # drop impossible prices
    prices = long.pivot(index="date", columns="ticker", values="adj_close").sort_index()[list(tickers)]

    gaps = prices.isna().sum()
    results.note("etf", "missing_days_by_ticker", gaps[gaps > 0].to_dict())

    # Fill isolated one-day gaps with the previous day's price (= zero return that day);
    # longer gaps are left empty and those days are dropped
    filled = prices.ffill(limit=1)
    results.note("etf", "cells_forward_filled", int(prices.isna().sum().sum() - filled.isna().sum().sum()))
    prices = filled.dropna()                                       # keep only days with every price

    results.note("etf", "trading_days", len(prices))
    results.note("etf", "date_range", [str(prices.index.min().date()), str(prices.index.max().date())])
    return prices
