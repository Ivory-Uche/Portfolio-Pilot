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
| Daily adjusted ETF prices (SPY, VEA, VWO, AGG, TLT, TIP, VNQ, GLD, DBC), 2016-01-01 to 2025-12-31 | Yahoo Finance via `yfinance` | download date recorded in `data/raw/etf_download_meta.json` | Yahoo terms of use: personal/research use, not redistributed here | `python -m portfolio_pilot.download_etf` |

Raw data is read-only and is not committed (see `.gitignore`). All derived data is rebuilt from it.

## Setup

Python 3.10 or newer. From the project root:

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The second install makes `src/portfolio_pilot/` importable from the notebooks, the tests, and the app.

## Reproduce every result

1. Put the SCF file in `data/raw/` (see the table above) and run `python -m portfolio_pilot.download_etf`.
2. Run the notebooks in `notebooks/` in order, top to bottom:

| Notebook | What it does | Writes |
|---|---|---|
| `01_clean_and_split.ipynb` | Loads and cleans the SCF and ETF data, splits both before any exploration | `data/processed/` |
| `02_eda.ipynb` | Features with training-fitted caps, outlier checks, correlations, ETF exploration | `reports/figures/`, `reports/tables/` |
| `03_archetypes.ipynb` | Chooses the number of archetypes, fits them, labels every household | `data/processed/scf_*.csv`, `reports/` |
| `04_pca.ipynb` | PCA on the SCF clustering variables and on the ETF returns | `reports/` |

Every count quoted in the report is saved to `reports/numbers.json`. The figures and tables the notebooks write under `reports/` are exploratory and are not committed; the notebooks keep them as outputs.
Settings (paths, random seed, split sizes, clustering options, ETF dates) are in `config.yaml`.
The k-selection in notebook 03 takes a few minutes; set `clustering.n_bootstrap` to 3 for a quick test.

Run the tests with `pytest`.

## Project structure

```
config.yaml        paths, random seeds, and settings in one place
src/portfolio_pilot/
    clean.py       fixes errors, duplicates, and impossible values (SCF and ETF)
    split.py       train/validation/test splits, with a fixed seed
    prepare.py     caps, scaling, and weighted sampling, fitted on training data only
    features.py    engineered features and ETF returns
    models/        archetype clustering (more model code from M4 onward)
    config.py      reads config.yaml and records the numbers quoted in the report
    download_etf.py  downloads the ETF prices
notebooks/         numbered exploration notebooks
reports/figures/   the figures used in the report
tests/             tests for our own functions
app/               the microproduct (a later milestone)
```
