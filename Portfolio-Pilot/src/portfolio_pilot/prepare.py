"""Load the raw data files. The raw files are only read, never changed."""

from datetime import date
from pathlib import Path

import pandas as pd


def load_scf(path, cols, implicate_digit=True, rec=None):
    """Read the SCF summary extract and keep the project columns under short names.

    Returns all five implicates; `implicate` is the last digit of the record ID (Y1 = YY1*10 + implicate).
    """
    raw = pd.read_csv(path, low_memory=False)
    raw.columns = [c.upper() for c in raw.columns]
    missing = [c for c in cols.values() if c not in raw.columns]
    if missing:
        raise KeyError(f"Columns not in the file: {missing}. Fix COLS in config.py.")
    if rec:
        rec.note("raw", "rows", len(raw))
        rec.note("raw", "columns", raw.shape[1])

    full = raw[list(cols.values())].copy()
    full.columns = list(cols.keys())
    full["implicate"] = (full["record_id"] % 10).astype(int)
    if rec:
        rec.note("raw", "households", full["household_id"].nunique())
        rec.note("raw", "implicates", sorted(full["implicate"].unique().tolist()))
    return full


def load_etf_prices(path, tickers, start, end, rec=None):
    """Load adjusted daily closes (one column per ETF). Downloads once with yfinance if the file is missing."""
    path = Path(path)
    stamp = path.parent / "etf_download_date.txt"
    if not path.exists():
        import yfinance as yf
        data = yf.download(tickers=tickers, start=start, end=end, interval="1d",
                           auto_adjust=True, progress=False)
        data["Close"].to_csv(path)
        stamp.write_text(str(date.today()))
        print("Downloaded on", date.today())
    else:
        print("Using saved file:", path)

    wide = pd.read_csv(path, index_col=0, parse_dates=True)
    wide.index.name = "date"
    if len(wide) == 0:
        raise ValueError("No prices: the download failed. Delete the CSV and try again later.")

    if rec:
        rec.note("etf_raw", "rows_trading_days", len(wide))
        rec.note("etf_raw", "columns", wide.shape[1])
        rec.note("etf_raw", "date_range", [str(wide.index.min().date()), str(wide.index.max().date())])
        rec.note("etf_raw", "tickers_missing", sorted(set(tickers) - set(wide.columns)))
        rec.note("etf_raw", "unexpected_columns", sorted(set(wide.columns) - set(tickers)))
        rec.note("etf_raw", "missing_by_ticker", wide.isna().sum().to_dict())
        rec.note("etf_raw", "first_valid_date",
                 {c: (str(wide[c].first_valid_index().date()) if wide[c].notna().any() else "no data")
                  for c in wide.columns})
        if stamp.exists():
            rec.note("etf_raw", "download_date", stamp.read_text())
    return wide
