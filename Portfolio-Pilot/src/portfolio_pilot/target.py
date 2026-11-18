"""Define the target (investor archetype) by clustering TRAINING households.

1. standardise CLUSTER_FEATURES on a survey-weighted resample of training households;
2. fit K-Means and Gaussian mixtures for each k and compare them (select_k);
3. pick k (choose_k): options within SIL_TOLERANCE of the best silhouette tie; the most stable wins;
4. order clusters A0 (most conservative) ... A(k-1) and label every split with the training model.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .features import CLUSTER_FEATURES
from .utils import weighted_median, weighted_resample


def make_model(method, k, seed):
    if method == "kmeans":
        return KMeans(n_clusters=k, n_init=20, random_state=seed)
    return GaussianMixture(n_components=k, covariance_type="full", n_init=5, random_state=seed)


def _base(train, seed):
    """The weighted resample, the scaler fitted on it, and the scaled matrices."""
    base = weighted_resample(train, seed)
    scaler = StandardScaler().fit(base[CLUSTER_FEATURES])
    return scaler, scaler.transform(base[CLUSTER_FEATURES]), scaler.transform(train[CLUSTER_FEATURES])


def select_k(train, k_range, seed, n_bootstrap=20, sil_sample=3000, rec=None, verbose=True):
    """Silhouette, GMM BIC, bootstrap stability (ARI) and smallest weighted cluster for each method and k."""
    scaler, Xb, Xt = _base(train, seed)
    w = train["weight"].to_numpy(float)
    sub = np.random.default_rng(seed).choice(len(Xb), size=min(sil_sample, len(Xb)), replace=False)

    results = []
    for method in ["kmeans", "gmm"]:
        for k in k_range:
            m = make_model(method, k, seed).fit(Xb)
            ref = m.predict(Xt)
            aris = []
            for b in range(n_bootstrap):
                rs = weighted_resample(train, seed + 1 + b)
                mb = make_model(method, k, seed + 1 + b).fit(scaler.transform(rs[CLUSTER_FEATURES]))
                aris.append(adjusted_rand_score(ref, mb.predict(Xt)))
            results.append({
                "method": method, "k": k,
                "silhouette": silhouette_score(Xb[sub], m.predict(Xb)[sub]),
                "bic": m.bic(Xb) if method == "gmm" else np.nan,
                "stability_ari_mean": np.mean(aris), "stability_ari_sd": np.std(aris),
                "min_cluster_share": (np.bincount(ref, weights=w, minlength=k) / w.sum()).min(),
            })
            if verbose:
                print("done:", method, k)
    ktab = pd.DataFrame(results)
    if rec:
        rec.note("target", "k_selection", ktab.round(4).to_dict(orient="records"))
    return ktab


def choose_k(ktab, method="kmeans", chosen_k=None, min_share=0.05, sil_tolerance=0.01, rec=None):
    """Near-best silhouette counts as a tie; among tied options pick the most stable."""
    if chosen_k:
        k = int(chosen_k)
    else:
        cand = ktab[(ktab["method"] == method) & (ktab["min_cluster_share"] >= min_share)]
        if cand.empty:
            cand = ktab[ktab["method"] == method]
        near_best = cand[cand["silhouette"] >= cand["silhouette"].max() - sil_tolerance]
        k = int(near_best.sort_values("stability_ari_mean", ascending=False).iloc[0]["k"])
        print("Tied on silhouette:", near_best["k"].tolist(), "-> most stable:", k)
    if rec:
        rec.note("target", "chosen", {"method": method, "k": k})
    print("Chosen:", method, "k =", k)
    return k


class Archetyper:
    """Clustering fitted on training households; labels 0..k-1 ordered conservative -> growth."""

    def fit(self, train, k, method, seed):
        self.scaler, Xb, _ = _base(train, seed)
        self.model = make_model(method, k, seed).fit(Xb)
        lab = self.model.predict(Xb)
        cols = [CLUSTER_FEATURES.index("risk_willingness"), CLUSTER_FEATURES.index("asinh_networth")]
        score = [Xb[lab == c][:, cols].sum(axis=1).mean() for c in range(k)]
        self.order = {int(old): new for new, old in enumerate(np.argsort(score))}
        return self

    def predict(self, df):
        return [self.order[c] for c in self.model.predict(self.scaler.transform(df[CLUSTER_FEATURES]))]


def label_all(feats, archetyper, rec=None):
    """Add the archetype column to every split (the model is never refitted on val/test)."""
    for d in feats.values():
        d["archetype"] = archetyper.predict(d)
    tr = feats["train"]
    if rec:
        rec.note("target", "train_weighted_share",
                 (tr.groupby("archetype")["weight"].sum() / tr["weight"].sum()).to_dict())
        rec.note("target", "train_count", tr["archetype"].value_counts().sort_index().to_dict())
    return feats


def profile(train):
    """Weighted medians (mean for risk) of the clustering inputs by archetype - Table 3."""
    return train.groupby("archetype").apply(lambda d: pd.Series({
        **{f"median_{c}": weighted_median(d[c], d["weight"])
           for c in ["age", "income", "networth", "liq", "dti", "liq_to_income", "equity_share"]},
        "mean_risk_willingness": np.average(d["risk_willingness"], weights=d["weight"]),
        "weighted_share": d["weight"].sum() / train["weight"].sum(),
        "n_households": len(d)}), include_groups=False)
