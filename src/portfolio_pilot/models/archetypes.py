"""Investor archetypes: choose the number of clusters, fit, and label households."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

from ..prepare import weighted_median, weighted_resample


def make_model(method, k, seed):
    if method == "kmeans":
        return KMeans(n_clusters=k, n_init=20, random_state=seed)
    return GaussianMixture(n_components=k, covariance_type="full", n_init=5, random_state=seed)


def select_k(train, scaler, base, columns, k_range, n_bootstrap, seed, methods=("kmeans", "gmm"),
             verbose=True):
    """Compare K-Means and Gaussian mixtures for each k.

    Columns: silhouette, BIC (mixtures only), bootstrap stability (adjusted Rand index
    against the reference labels of the training households) and the smallest group share.
    `base` is the weighted resample the scaler was fitted on (prepare.fit_scaler).
    """
    Xb = scaler.transform(base[columns])
    Xt = scaler.transform(train[columns])
    w = train["weight"].to_numpy(float)
    sub = np.random.default_rng(seed).choice(len(Xb), size=min(3000, len(Xb)), replace=False)

    results = []
    for method in methods:
        for k in k_range:
            m = make_model(method, k, seed).fit(Xb)
            ref = m.predict(Xt)
            aris = []
            for b in range(n_bootstrap):
                rs = weighted_resample(train, seed + 1 + b)
                mb = make_model(method, k, seed + 1 + b).fit(scaler.transform(rs[columns]))
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
    return pd.DataFrame(results)


def choose_k(ktab, method, min_cluster_share, tolerance=0.01):
    """Highest silhouette among groups that are not too small; ties go to the most stable k."""
    cand = ktab[(ktab["method"] == method) & (ktab["min_cluster_share"] >= min_cluster_share)]
    if cand.empty:
        cand = ktab[ktab["method"] == method]
    best_sil = cand["silhouette"].max()
    near_best = cand[cand["silhouette"] >= best_sil - tolerance]      # options tied with the best
    k = int(near_best.sort_values("stability_ari_mean", ascending=False).iloc[0]["k"])
    return k, near_best["k"].tolist()


def fit_archetypes(scaler, base, columns, method, k, seed):
    """Fit the chosen model; return it with the label order A0 (most conservative) ... A(k-1)."""
    Xb = scaler.transform(base[columns])
    model = make_model(method, k, seed).fit(Xb)

    lab_b = model.predict(Xb)
    cols = [columns.index("risk_willingness"), columns.index("asinh_networth")]
    score = [Xb[lab_b == c][:, cols].sum(axis=1).mean() for c in range(k)]
    order = {int(old): new for new, old in enumerate(np.argsort(score))}
    return model, order


def assign_archetypes(feats, model, scaler, order, columns):
    """Label every partition with the model fitted on the training data (no refitting)."""
    for d in feats.values():
        d["archetype"] = [order[c] for c in model.predict(scaler.transform(d[columns]))]
    return feats


def archetype_profile(train):
    """Weighted medians and shares by archetype, for the training households."""
    return train.groupby("archetype").apply(lambda d: pd.Series({
        **{f"median_{c}": weighted_median(d[c], d["weight"])
           for c in ["age", "income", "networth", "liq", "dti", "liq_to_income", "equity_share"]},
        "mean_risk_willingness": np.average(d["risk_willingness"], weights=d["weight"]),
        "weighted_share": d["weight"].sum() / train["weight"].sum(),
        "n_households": len(d)}), include_groups=False)
