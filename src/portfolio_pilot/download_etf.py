"""Download daily adjusted ETF prices from Yahoo Finance.

Run once from the project root:  python -m portfolio_pilot.download_etf
Writes the closing prices and a small metadata file into data/raw/.
"""
import json
from datetime import date

import yfinance as yf

from .config import load_config


def download(cfg):
    etf = cfg["etf"]
    data = yf.download(tickers=etf["tickers"], start=etf["start"], end=etf["end"], interval="1d",
                       auto_adjust=True, progress=False)
    prices = data["Close"]
    if len(prices) == 0:
        raise RuntimeError("No prices came back from Yahoo Finance; try again later.")

    out = cfg["paths"]["etf_prices_file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(out)                                  # adjusted closing prices, one column per ETF

    meta = {"source": "Yahoo Finance via yfinance", "downloaded_on": str(date.today()),
            "tickers": etf["tickers"], "start": etf["start"], "end_exclusive": etf["end"],
            "rows": len(prices)}
    cfg["paths"]["etf_meta_file"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    print("Saved", download(load_config()))
