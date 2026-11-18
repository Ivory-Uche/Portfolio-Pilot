"""Survey-weighted metrics and cross-validation for the archetype classifier.

Archetypes are ordered A0 (most conservative) ... A4 (most growth-oriented), so the size and
direction of an error matter: predicting a more aggressive archetype than the true one is the costly error.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

USES_SAMPLE_WEIGHT = {"MajorityClass", "LogisticRegression"}


def metrics(y_true, y_pred, w, n_classes=5):
    """All metrics are weighted by the survey weights, so they describe the population."""
    y_true, y_pred, w = np.asarray(y_true), np.asarray(y_pred), np.asarray(w, float)
    diff = y_pred - y_true
    return {
        "macro_f1": f1_score(y_true, y_pred, labels=range(n_classes), average="macro",
                             sample_weight=w, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred, sample_weight=w),
        "within_one": float(np.average(np.abs(diff) <= 1, weights=w)),
        "over_risk": float(np.average(diff >= 1, weights=w)),          # placed too aggressive by 1+
        "severe_over_risk": float(np.average(diff >= 2, weights=w)),   # placed too aggressive by 2+
    }


def per_class_f1(y_true, y_pred, w, n_classes=5):
    return f1_score(y_true, y_pred, labels=range(n_classes), average=None, sample_weight=w, zero_division=0)


def _model_name(model):
    m = model.steps[-1][1] if hasattr(model, "steps") else model
    return type(m).__name__


def fit(model, X, y, w):
    """Fit with survey weights when the model supports them."""
    if _model_name(model) in USES_SAMPLE_WEIGHT:
        key = "model__sample_weight" if hasattr(model, "steps") else "sample_weight"
        return model.fit(X, y, **{key: w})
    return model.fit(X, y)


def cross_validate(model, X, y, w, n_folds=5, seed=7275, n_classes=5):
    """Stratified k-fold on the training households. A fresh copy of the whole pipeline
    (scaler included) is fitted inside every fold, so nothing leaks from the held-out fold."""
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows = []
    for i, (a, b) in enumerate(folds.split(X, y)):
        m = fit(clone(model), X.iloc[a], y[a], w[a])
        tr = metrics(y[a], m.predict(X.iloc[a]), w[a], n_classes)
        va = metrics(y[b], m.predict(X.iloc[b]), w[b], n_classes)
        rows.append({"fold": i, **{f"train_{k}": v for k, v in tr.items()},
                     **{f"cv_{k}": v for k, v in va.items()}})
    return pd.DataFrame(rows)


def summarise_cv(cv):
    """Mean and standard deviation across folds."""
    out = {}
    for c in cv.columns:
        if c != "fold":
            out[f"{c}_mean"] = cv[c].mean()
            out[f"{c}_sd"] = cv[c].std()
    return out
