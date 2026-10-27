"""Feature construction. Ratio caps are learned from TRAINING data only, then applied to every split.

CLUSTER_FEATURES define the archetype (the target). They include debt-to-income and equity share,
which an app user does not provide, so they never enter the classifier.
CLASSIFIER_FEATURES are known at prediction time: intake answers or values computed from them.
"""

import numpy as np
import pandas as pd

CLUSTER_FEATURES = ["age", "risk_willingness", "asinh_networth", "asinh_dti",
                    "asinh_liq_to_income", "equity_share"]
CLASSIFIER_FEATURES = ["age", "risk_willingness", "asinh_income", "asinh_networth", "asinh_liq",
                       "asinh_liq_to_income", "income_nonpositive"]


def ratio(num, den):
    """num / den, undefined (NaN) when den <= 0."""
    return (num / den.where(den > 0)).astype(float)


class FeatureBuilder:
    """fit() learns the caps on training data; transform() builds the features for any split."""

    def __init__(self, cap_quantile=0.99):
        self.cap_quantile = cap_quantile
        self.caps = {}

    def fit(self, train, rec=None):
        q = self.cap_quantile
        self.caps = {"dti": float(ratio(train["debt"], train["income"]).quantile(q).round(2)),
                     "liq_to_income": float(ratio(train["liq"], train["income"]).quantile(q).round(2))}
        if rec:
            rec.note("features", "caps_from_train", self.caps)
            rec.note("features", "train_ratio_undefined_income_le_0", int((train["income"] <= 0).sum()))
            rec.note("features", "train_dti_above_cap",
                     int((ratio(train["debt"], train["income"]) > self.caps["dti"]).sum()))
            rec.note("features", "train_lti_above_cap",
                     int((ratio(train["liq"], train["income"]) > self.caps["liq_to_income"]).sum()))
            rec.note("features", "train_fin_zero", int((train["fin"] <= 0).sum()))
        return self

    def transform(self, df):
        caps = self.caps
        out = df.copy()
        nonpos = out["income"] <= 0
        out["income_nonpositive"] = nonpos.astype(int)

        # Undefined ratios (income <= 0): cap if the numerator is positive, otherwise 0
        dti = ratio(out["debt"], out["income"])
        dti = dti.where(~nonpos, np.where(out["debt"] > 0, caps["dti"], 0.0))
        out["dti"] = dti.clip(upper=caps["dti"])
        lti = ratio(out["liq"], out["income"])
        lti = lti.where(~nonpos, np.where(out["liq"] > 0, caps["liq_to_income"], 0.0))
        out["liq_to_income"] = lti.clip(upper=caps["liq_to_income"])

        # No financial assets -> no equity exposure
        out["equity_share"] = ratio(out["equity"], out["fin"]).fillna(0.0).clip(0, 1)

        for c in ["income", "networth", "liq", "dti", "liq_to_income"]:
            out[f"asinh_{c}"] = np.arcsinh(out[c].astype(float))
        return out


def build_all(parts, cap_quantile=0.99, rec=None):
    """Fit on parts['train'], transform every split, and check there are no missing features."""
    fb = FeatureBuilder(cap_quantile).fit(parts["train"], rec)
    feats = {name: fb.transform(d) for name, d in parts.items()}
    assert not feats["train"][CLUSTER_FEATURES + CLASSIFIER_FEATURES].isna().any().any()
    return feats, fb


def outlier_report(train, rec=None):
    """Count values beyond 3*IQR fences, on the dollar scale and after asinh (training data)."""
    rows = []
    for col in ["income", "asinh_income", "networth", "asinh_networth", "liq", "asinh_liq"]:
        x = train[col].astype(float)
        q1, q3 = x.quantile([0.25, 0.75])
        n = int(((x < q1 - 3 * (q3 - q1)) | (x > q3 + 3 * (q3 - q1))).sum())
        rows.append({"variable": col, "n_beyond_3iqr": n, "min": x.min(), "max": x.max()})
    table = pd.DataFrame(rows)
    if rec:
        rec.note("outliers", "table", table.to_dict(orient="records"))
    return table
