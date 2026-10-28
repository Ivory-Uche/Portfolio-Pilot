"""Engineered features for the SCF households and the ETF returns."""
import numpy as np

from .prepare import ratio

# Clustering variables define the target; classifier variables are the ones a user can give
CLUSTER_FEATURES = ["age", "risk_willingness", "asinh_networth", "asinh_dti",
                    "asinh_liq_to_income", "equity_share"]
CLASSIFIER_FEATURES = ["age", "risk_willingness", "asinh_income", "asinh_networth", "asinh_liq",
                       "asinh_liq_to_income", "income_nonpositive"]


def build_features(df, caps):
    """Add the model features to a table of households.

    `caps` must come from the training households (prepare.fit_caps), so the
    same caps are used for every partition and for new users.
    """
    out = df.copy()
    nonpos = out["income"] <= 0
    out["income_nonpositive"] = nonpos.astype(int)
    dti = ratio(out["debt"], out["income"])
    dti = dti.where(~nonpos, np.where(out["debt"] > 0, caps["dti"], 0.0))
    out["dti"] = dti.clip(upper=caps["dti"])
    lti = ratio(out["liq"], out["income"])
    lti = lti.where(~nonpos, np.where(out["liq"] > 0, caps["liq_to_income"], 0.0))
    out["liq_to_income"] = lti.clip(upper=caps["liq_to_income"])
    out["equity_share"] = ratio(out["equity"], out["fin"]).fillna(0.0).clip(0, 1)
    for c in ["income", "networth", "liq", "dti", "liq_to_income"]:
        out[f"asinh_{c}"] = np.arcsinh(out[c].astype(float))
    return out


def build_partition_features(parts, caps):
    """build_features for each partition; fails if a model feature is missing."""
    feats = {name: build_features(d, caps) for name, d in parts.items()}
    cols = CLUSTER_FEATURES + CLASSIFIER_FEATURES
    if feats["train"][cols].isna().any().any():
        raise ValueError("Missing values in the model features of the training partition")
    return feats


def daily_log_returns(prices):
    """log(today's price / yesterday's price); the first day has no previous price and is dropped."""
    return np.log(prices).diff().dropna()
