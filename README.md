# PortfolioPilot

Data-driven investor profiling and diversified ETF allocation for self-directed retail investors
(IE 7275, Fall 2026, group Top Research: Juan P. Beltran B. and Onyinyechi Ivory Uche).

**Problem.** Questionnaire-based robo-advisor risk scores rely on platform-set thresholds. We learn
investor archetypes from household survey data, predict a new user's archetype from five intake answers
(age, risk willingness, income, net worth, liquid assets), and build an archetype-constrained portfolio
with Hierarchical Risk Parity.
**Intended user.** Self-directed retail investors who want an educational view of their risk profile.
The project gives no personalised financial or tax advice and places no trades.

## Data

| Data | Source | Version / date | License | How to obtain |
|---|---|---|---|---|
| Survey of Consumer Finances 2025, summary extract | Board of Governors of the Federal Reserve System | 2025 survey, file `SCFP2025.csv`, downloaded 2026-10-20 | Public-domain U.S. government data | Download the "Summary Extract Public Data" (CSV) for 2025 from https://www.federalreserve.gov/econres/scfindex.htm, unzip, save as `data/raw/SCFP2025.csv` |
| Daily adjusted ETF prices (SPY, VEA, VWO, AGG, TLT, TIP, VNQ, GLD, DBC), 2016-01-01 to 2025-12-31 | Yahoo Finance via `yfinance` | downloaded 2026-10-20 (see `data/raw/etf_download_meta.json`) | Yahoo terms of use: personal/research use, not redistributed here | `python scripts/download_etf.py` |

Raw data is read-only and is not committed (see `.gitignore`). All derived data is rebuilt from it.

