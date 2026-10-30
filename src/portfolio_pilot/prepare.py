"""Caps, scaling, and weighted sampling. Everything is fitted on training data only."""
import numpy as np
from sklearn.preprocessing import StandardScaler

from .config import Results


def ratio(num, den):
    return (num / den.where(den > 0)).astype(float)       # undefined (NaN) when den <= 0


def fit_caps(train, quantile, results=None):
    """Caps for the debt-to-income and liquid-to-income ratios, from the training households.

    The caps are the `quantile` of each ratio, rounded to two decimals.
    """
    results = Results() if results is None else results
    dti = ratio(train["debt"], train["income"])
    lti = ratio(train["liq"], train["income"])
    caps = {"dti": float(dti.quantile(quantile).round(2)),
            "liq_to_income": float(lti.quantile(quantile).round(2))}
    results.note("features", "caps_from_train", caps)
    results.note("features", "train_ratio_undefined_income_le_0", int((train["income"] <= 0).sum()))
    results.note("features", "train_dti_above_cap", int((dti > caps["dti"]).sum()))
    results.note("features", "train_lti_above_cap", int((lti > caps["liq_to_income"]).sum()))
    results.note("features", "train_fin_zero", int((train["fin"] <= 0).sum()))
    return caps


def weighted_resample(df, seed):
    """Draw households with probability proportional to the survey weight."""
    rng = np.random.default_rng(seed)
    p = df["weight"].to_numpy(float)
    return df.iloc[rng.choice(len(df), size=len(df), replace=True, p=p / p.sum())]


def weighted_median(x, w):
    o = np.argsort(np.asarray(x))
    x, w = np.asarray(x, float)[o], np.asarray(w, float)[o]
    c = np.cumsum(w)
    return float(x[np.searchsorted(c, 0.5 * c[-1])])


def fit_scaler(train, columns, seed):
    """Fit a standard scaler on a survey-weighted resample of the training households.

    Returns the scaler and the resample it was fitted on.
    """
    base = weighted_resample(train, seed)
    return StandardScaler().fit(base[columns]), base
