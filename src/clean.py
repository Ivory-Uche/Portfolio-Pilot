"""Rule-based cleaning (no statistics estimated, so safe before the split) and ETF cleaning."""

import numpy as np
import pandas as pd


def clean_scf(full, implicate=1, rec=None):
    """Duplicates, missing values, impossible values and inconsistencies. Returns one row per household."""
    note = rec.note if rec else (lambda *a: None)

    note("duplicates", "exact_duplicate_rows", int(full.duplicated().sum()))
    note("duplicates", "duplicate_record_ids", int(full["record_id"].duplicated().sum()))
    full = full.drop_duplicates(subset="record_id")

    hh = full[full["implicate"] == implicate].copy()
    note("implicate_1", "rows", len(hh))

    na = hh.isna().sum()
    note("missing", "by_column", na[na > 0].to_dict())
    note("missing", "total", int(na.sum()))

    checks = {
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
    for name, mask in checks.items():
        note("errors", name, int(mask.sum()))
    bad = checks["age_under_18_or_over_100"] | checks["weight_not_positive"] | checks["both_risk_flags_1"]
    note("errors", "rows_dropped", int(bad.sum()))
    hh = hh[~bad].reset_index(drop=True)

    # Real values that need special handling later (not errors)
    for name, mask in {"income_zero": hh["income"] == 0, "income_negative": hh["income"] < 0,
                       "networth_negative": hh["networth"] < 0, "fin_zero": hh["fin"] == 0,
                       "debt_zero": hh["debt"] == 0}.items():
        note("genuine", name, int(mask.sum()))

    # Risk willingness: 0 = no risk, 1 = average, 2 = above-average / substantial
    hh["risk_willingness"] = np.select([hh["no_risk"] == 1, hh["yes_risk"] == 1], [0, 2], default=1)
    hh["age_band"] = pd.cut(hh["age"], [0, 34, 44, 54, 64, 74, 200],
                            labels=["<35", "35-44", "45-54", "55-64", "65-74", "75+"]).astype(str)
    note("implicate_1", "rows_after_cleaning", len(hh))
    return hh


def implicate_check(full, train_ids, rec=None):
    """How much do the five implicates of each TRAINING household differ?"""
    g = full[full["household_id"].isin(train_ids)].groupby("household_id")
    out = {"train_households_without_5_rows": int((g.size() != 5).sum())}
    for c in ["income", "networth", "liq", "debt"]:
        out[f"share_where_{c}_varies"] = float((g[c].nunique() > 1).mean().round(2))
    if rec:
        for k, v in out.items():
            rec.note("implicates", k, v)
    return out


def clean_etf(wide, tickers, rec=None):
    """Remove duplicate and impossible prices, fill one-day gaps, keep days with all ETFs."""
    note = rec.note if rec else (lambda *a: None)
    eraw = wide.reset_index().melt(id_vars="date", var_name="ticker", value_name="adj_close")

    note("etf", "rows", len(eraw))
    note("etf", "tickers_missing", sorted(set(tickers) - set(eraw["ticker"])))
    note("etf", "duplicate_date_ticker", int(eraw.duplicated(["date", "ticker"]).sum()))
    note("etf", "missing_prices", int(eraw["adj_close"].isna().sum()))
    note("etf", "non_positive_prices", int((eraw["adj_close"] <= 0).sum()))

    eraw = eraw.drop_duplicates(["date", "ticker"], keep="last")
    eraw = eraw[eraw["adj_close"] > 0]
    prices = eraw.pivot(index="date", columns="ticker", values="adj_close").sort_index()[tickers]

    gaps = prices.isna().sum()
    note("etf", "missing_days_by_ticker", gaps[gaps > 0].to_dict())
    filled = prices.ffill(limit=1)                      # one-day gaps only
    note("etf", "cells_forward_filled", int(prices.isna().sum().sum() - filled.isna().sum().sum()))
    prices = filled.dropna()
    note("etf", "trading_days", len(prices))
    note("etf", "date_range", [str(prices.index.min().date()), str(prices.index.max().date())])
    return prices


def log_returns(prices):
    """Daily log returns: log(today / yesterday)."""
    return np.log(prices).diff().dropna()


def extreme_returns(train_ret, n_sd=5, rec=None):
    """Daily returns beyond n_sd training standard deviations (inspected, not removed)."""
    z = (train_ret - train_ret.mean()) / train_ret.std()
    flags = z.abs() > n_sd
    if rec:
        rec.note("etf_outliers", "n_flagged_by_ticker", flags.sum().to_dict())
        rec.note("etf_outliers", "dates", [str(d.date()) for d in train_ret.index[flags.any(axis=1)]])
    table = (train_ret[flags].stack().rename("log_return").to_frame()
             .assign(z=z.stack(), pct_return=lambda d: (np.exp(d["log_return"]) - 1) * 100).round(2))
    table.index.names = ["date", "ticker"]
    return table.sort_index()
