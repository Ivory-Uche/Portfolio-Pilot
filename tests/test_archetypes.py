import numpy as np
import pandas as pd

from portfolio_pilot.models.archetypes import (archetype_profile, assign_archetypes, choose_k,
                                               fit_archetypes, select_k)
from portfolio_pilot.prepare import fit_scaler

COLUMNS = ["risk_willingness", "asinh_networth"]


def _blobs(seed=0, per_group=150):
    """Three well-separated groups: conservative, middle, growth-oriented."""
    rng = np.random.default_rng(seed)
    parts = []
    for risk, nw in [(0, 2.0), (1, 8.0), (2, 14.0)]:
        parts.append(pd.DataFrame({
            "risk_willingness": risk + rng.normal(0, 0.05, per_group),
            "asinh_networth": nw + rng.normal(0, 0.3, per_group),
            "weight": rng.uniform(500, 1500, per_group)}))
    return pd.concat(parts, ignore_index=True)


def test_choose_k_prefers_the_most_stable_among_tied_silhouettes():
    ktab = pd.DataFrame({
        "method": ["kmeans"] * 4, "k": [3, 4, 5, 6],
        "silhouette": [0.20, 0.265, 0.26, 0.264],
        "stability_ari_mean": [0.9, 0.79, 0.94, 0.65],
        "min_cluster_share": [0.2, 0.1, 0.09, 0.08]})
    k, tied = choose_k(ktab, "kmeans", min_cluster_share=0.05, tolerance=0.01)
    assert tied == [4, 5, 6] and k == 5


def test_choose_k_skips_options_with_tiny_groups():
    ktab = pd.DataFrame({
        "method": ["kmeans"] * 2, "k": [3, 4], "silhouette": [0.20, 0.50],
        "stability_ari_mean": [0.9, 0.9], "min_cluster_share": [0.2, 0.01]})
    assert choose_k(ktab, "kmeans", min_cluster_share=0.05)[0] == 3


def test_select_k_finds_the_three_groups():
    train = _blobs()
    scaler, base = fit_scaler(train, COLUMNS, seed=1)
    ktab = select_k(train, scaler, base, COLUMNS, k_range=[2, 3, 4], n_bootstrap=2, seed=1,
                    methods=("kmeans",), verbose=False)
    assert set(ktab.columns) >= {"method", "k", "silhouette", "bic", "stability_ari_mean", "min_cluster_share"}
    assert choose_k(ktab, "kmeans", min_cluster_share=0.05)[0] == 3


def test_labels_run_from_conservative_to_growth_oriented():
    train = _blobs()
    scaler, base = fit_scaler(train, COLUMNS, seed=1)
    model, order = fit_archetypes(scaler, base, COLUMNS, "kmeans", 3, seed=1)
    feats = assign_archetypes({"train": train.copy()}, model, scaler, order, COLUMNS)
    labelled = feats["train"]
    means = labelled.groupby("archetype")["asinh_networth"].mean()
    assert means.is_monotonic_increasing and list(means.index) == [0, 1, 2]


def test_assign_archetypes_does_not_refit():
    train = _blobs()
    scaler, base = fit_scaler(train, COLUMNS, seed=1)
    model, order = fit_archetypes(scaler, base, COLUMNS, "kmeans", 3, seed=1)
    other = pd.DataFrame({"risk_willingness": [0.0, 2.0], "asinh_networth": [2.0, 14.0]})
    out = assign_archetypes({"val": other}, model, scaler, order, COLUMNS)["val"]
    assert out["archetype"].tolist() == [0, 2]


def test_profile_has_one_row_per_archetype():
    train = _blobs()
    scaler, base = fit_scaler(train, COLUMNS, seed=1)
    model, order = fit_archetypes(scaler, base, COLUMNS, "kmeans", 3, seed=1)
    labelled = assign_archetypes({"train": train.copy()}, model, scaler, order, COLUMNS)["train"]
    for c in ["age", "income", "networth", "liq", "dti", "liq_to_income", "equity_share"]:
        labelled[c] = 1.0
    prof = archetype_profile(labelled)
    assert list(prof.index) == [0, 1, 2]
    assert abs(prof["weighted_share"].sum() - 1) < 1e-9
    assert prof["n_households"].sum() == len(labelled)
