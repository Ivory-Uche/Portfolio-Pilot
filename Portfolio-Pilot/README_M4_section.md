## Milestone 4: baseline models

**Reproduce the results table with one command** (from the repository root, after placing the SCF file in `data/raw/`):

```bash
pip install -r requirements.txt
pip install -e .
python -m portfolio_pilot.run_baselines
```

This rebuilds the M3 data from the raw file (cleaning, split, features, archetype labels), fits every model,
and writes to `results/`:

| File | Contents |
|---|---|
| `m4_results_table.csv` | the report's results table |
| `m4_baselines.csv` | every metric on training, 5-fold cross-validation and validation |
| `m4_val_predictions.csv` | each model's prediction for every validation household |
| `m4_knn_k_search.csv` | k-NN settings tried on the validation set |
| `m4_cv_seed_check.csv` | cross-validation repeated with other fold seeds |
| `m4_implicates.csv` | validation results on SCF implicates 2-5 |
| `m4_per_class_f1.csv` | validation F1 for each archetype |
| `m4_run_info.json` | settings and library versions used |
| `figures/` | confusion matrices and per-archetype F1 |

All seeds, model settings, the number of folds and the success criterion are in `config.yaml`.
The test set is never loaded at M4 (`evaluation.use_test: false`).

Leakage checks: `pytest` (runs `tests/test_no_leakage.py`).

In Google Colab, open `notebook/02_m4_baselines.ipynb` and run the cells in order.
