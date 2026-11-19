"""Rebuild the modelling data with the team's M3 functions: clean -> split -> caps -> features -> archetypes.

The test households are dropped straight after the split unless evaluation.use_test is true,
so they are never featured, labelled or scored at M4.
"""
import numpy as np
import pandas as pd

from . import clean, prepare, split
from .config import Results
from .features import CLASSIFIER_FEATURES, CLUSTER_FEATURES, build_features, build_partition_features
from .models import archetypes

EXTRA = ["household_id", "weight", "age_band"]   # carried along; age_band is used only by the rule


def build(cfg, results=None, use_test=None):
    results = Results() if results is None else results
    use_test = cfg["evaluation"]["use_test"] if use_test is None else use_test
    seed, scf, cl = cfg["seed"], cfg["scf"], cfg["clustering"]

    full = clean.load_scf(cfg["paths"]["scf_file"], scf["columns"], results)
    full, hh = clean.clean_scf(full, scf["implicate"], results)
    parts = split.split_households(hh, scf["split"], seed, scf["min_stratum_size"], results)
    if not use_test:
        parts.pop("test")

    caps = prepare.fit_caps(parts["train"], scf["cap_quantile"], results)
    feats = build_partition_features(parts, caps)

    scaler, base = prepare.fit_scaler(feats["train"], CLUSTER_FEATURES, seed)
    k = cl["chosen_k"]
    if not k:   # same automatic rule as M3
        ktab = archetypes.select_k(feats["train"], scaler, base, CLUSTER_FEATURES, cl["k_range"],
                                   cl["n_bootstrap"], seed, verbose=False)
        k, _ = archetypes.choose_k(ktab, cl["method"], cl["min_cluster_share"], cl["silhouette_tolerance"])
    model, order = archetypes.fit_archetypes(scaler, base, CLUSTER_FEATURES, cl["method"], k, seed)
    feats = archetypes.assign_archetypes(feats, model, scaler, order, CLUSTER_FEATURES)

    data = {n: d[EXTRA + CLASSIFIER_FEATURES + ["archetype"]].reset_index(drop=True) for n, d in feats.items()}
    return {"data": data, "full": full, "caps": caps, "scaler": scaler, "model": model, "order": order,
            "k": k, "results": results}


def validation_implicate(built, implicate):
    """The same validation households with values from another implicate (2-5).

    Training caps and the training clustering are reused, so this measures only how much
    the imputed values change the labels and predictions.
    """
    full, ids = built["full"], built["data"]["val"]["household_id"]
    d = full[(full["implicate"] == implicate) & full["household_id"].isin(ids)].copy()
    # same derived columns as clean.clean_scf
    d["risk_willingness"] = np.select([d["no_risk"] == 1, d["yes_risk"] == 1], [0, 2], default=1)
    d["age_band"] = pd.cut(d["age"], [0, 34, 44, 54, 64, 74, 200],
                           labels=["<35", "35-44", "45-54", "55-64", "65-74", "75+"]).astype(str)
    d = build_features(d, built["caps"])
    d = archetypes.assign_archetypes({"v": d}, built["model"], built["scaler"], built["order"], CLUSTER_FEATURES)["v"]
    return d[EXTRA + CLASSIFIER_FEATURES + ["archetype"]].reset_index(drop=True)
