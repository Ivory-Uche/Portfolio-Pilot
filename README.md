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
| Survey of Consumer Finances 2022, summary extract | Board of Governors of the Federal Reserve System | 2022 survey, file `SCFP2022.csv`, downloaded YYYY-MM-DD | Public-domain U.S. government data | Download the "Summary Extract Public Data" (CSV) for 2022 from https://www.federalreserve.gov/econres/scfindex.htm, unzip, save as `data/raw/SCFP2022.csv` |
| Daily adjusted ETF prices (SPY, VEA, VWO, AGG, TLT, TIP, VNQ, GLD, DBC), 2016-01-01 to 2025-12-31 | Yahoo Finance via `yfinance` | downloaded YYYY-MM-DD (see `data/raw/etf_download_meta.json`) | Yahoo terms of use: personal/research use, not redistributed here | `python scripts/download_etf.py` |

Raw data is read-only and is not committed (see `.gitignore`). All derived data is rebuilt from it.

## Install (Python 3.12)

```bash
git clone https://github.com/<user>/portfoliopilot.git
cd portfoliopilot
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Reproduce every M3 result

```bash
python scripts/download_etf.py      # once; writes data/raw/etf_prices_raw.csv
python scripts/run_m3.py            # cleaning, split, features, target, PCA, all figures and tables
pytest                              # sanity tests on synthetic data
```

`run_m3.py` writes:
- `reports/m3_numbers.json`: every count and statistic quoted in the report;
- `reports/figures/fig01 ... fig08 *.png`: Figures 1-8 (file number = figure number in the report);
- `reports/tables/*.csv`: k selection, archetype profiles, PCA loadings, outlier counts;
- `data/processed/`: train / validation / test feature files and `scf_split_ids.csv`.

All settings (paths, seed `7275`, split proportions and dates) are in `config.yaml`.

## Repository structure

```
config.yaml            settings: paths, seed, split, clustering and PCA options
data/raw/              original files, read-only, not in Git
data/interim/          cleaned but not yet split, not in Git
data/processed/        model-ready train/val/test files, not in Git
src/portfoliopilot/    installable package
  scf.py               load + rule-based cleaning of the SCF
  etf.py               ETF price cleaning and returns
  split.py             SCF household split, ETF time split
  features.py          feature construction (statistics fitted on training data)
  target.py            archetype definition by clustering training households
  reduce.py            PCA
  figures.py           report figures
scripts/               entry points (download_etf.py, run_m3.py)
notebooks/             exploration notebooks (training data only)
tests/                 pytest tests (synthetic data only)
reports/               figures, tables and m3_numbers.json
app/                   microproduct (later milestones)
```

## Milestone tags
`m3` marks the version submitted for Milestone 3.
