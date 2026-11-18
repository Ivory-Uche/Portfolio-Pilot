"""Reproduce every M4 result with one command (from the repository root):

    python -m portfolio_pilot.run_baselines

Writes to results/:
    m4_baselines.csv          every metric (training, cross-validation and validation)
    m4_results_table.csv      the report's results table, built from m4_baselines.csv
    m4_val_predictions.csv    each model's prediction for every validation household
    m4_knn_k_search.csv       k-NN settings tried on the validation set
    m4_cv_seed_check.csv      cross-validation repeated with other fold seeds
    m4_implicates.csv         validation macro F1 on implicates 2-5
    m4_per_class_f1.csv       validation F1 for each archetype
    m4_run_info.json          settings and library versions used
    figures/                  confusion matrices and per-archetype F1
"""

import json
import platform
import sys

import numpy as np
import pandas as pd
import sklearn

from . import config, dataset
from .evaluate import cross_validate, fit, metrics, per_class_f1, summarise_cv
from .features import CLASSIFIER_FEATURES
from .models import MajorityClass, QuestionnaireRule, make_centroid, make_knn, make_logreg
from .plots import confusion_matrices, per_class_f1_plot


def build_models(cfg, k_best):
    m = cfg["models"]
    lr = m["logreg"]
    return {
        "Reference: majority class": (MajorityClass(), "predicts the largest weighted archetype"),
        "Rule: questionnaire score": (QuestionnaireRule(m["questionnaire"]["age_points"], cfg["target"]["k"]),
                                      "risk willingness + age points, no learning"),
        "Nearest centroid": (make_centroid(), "scaled features"),
        "Logistic regression": (make_logreg(lr["C"], lr["max_iter"], lr["class_weight"], cfg["seed"]),
                                f"L2, C={lr['C']}, class_weight={lr['class_weight']}, survey weights"),
        f"k-NN (k={k_best})": (make_knn(k_best, m["knn"]["weights"]), f"k={k_best} chosen on validation, scaled"),
    }


def report_table(t):
    """The results table exactly as it appears in the report (built from m4_baselines.csv)."""
    f = lambda x: f"{x:.3f}"
    return pd.DataFrame({
        "Model": t["model"],
        "Macro F1, training": t["train_macro_f1"].map(f),
        "Macro F1, 5-fold CV (mean ± SD)": [f"{m:.3f} ± {s:.3f}"
                                            for m, s in zip(t["cv_macro_f1_mean"], t["cv_macro_f1_sd"])],
        "Macro F1, validation": t["val_macro_f1"].map(f),
        "Severe over-risk, validation": (t["val_severe_over_risk"] * 100).map(lambda x: f"{x:.1f}%"),
        "Within one archetype, validation": (t["val_within_one"] * 100).map(lambda x: f"{x:.1f}%"),
        "Meets criterion": t["meets_success_criterion"].map({True: "Yes", False: "No"}),
    })


def main():
    cfg = config.load()
    paths = config.get_paths(cfg["root"])
    out, figs = paths["results"], paths["results_figures"]
    n_classes, seed = cfg["target"]["k"], cfg["seed"]
    succ = cfg["evaluation"]["success"]
    print("Building data (clean -> split -> features -> archetypes) ...")
    built = dataset.build(cfg, paths)
    tr, va = built["data"]["train"], built["data"]["val"]
    print(f"Training households: {len(tr)} | validation households: {len(va)} | test set loaded: "
          f"{'test' in built['data']}")

    cols = CLASSIFIER_FEATURES + ["age_band"]          # age_band is read only by the questionnaire rule
    Xtr, ytr, wtr = tr[cols], tr["archetype"].to_numpy(), tr["weight"].to_numpy(float)
    Xva, yva, wva = va[cols], va["archetype"].to_numpy(), va["weight"].to_numpy(float)
    num = CLASSIFIER_FEATURES                           # the learned models see only the intake features

    # 1. k-NN: small documented search on the validation set
    ks = []
    for k in cfg["models"]["knn"]["k_grid"]:
        m = make_knn(k, cfg["models"]["knn"]["weights"]).fit(Xtr[num], ytr)
        ks.append({"k": k, **{f"val_{a}": b for a, b in metrics(yva, m.predict(Xva[num]), wva, n_classes).items()}})
    ks = pd.DataFrame(ks)
    ks.to_csv(out / "m4_knn_k_search.csv", index=False)
    k_best = int(ks.sort_values("val_macro_f1", ascending=False).iloc[0]["k"])
    print("k-NN search:", ks[["k", "val_macro_f1"]].round(3).to_dict(orient="records"), "-> k =", k_best)

    # 2. Fit every model on the full training set; cross-validate on training; score on validation
    models = build_models(cfg, k_best)
    rows, preds, pcf, seed_rows, fitted = [], va[["household_id", "weight", "archetype"]].copy(), [], [], {}
    for name, (model, settings) in models.items():
        X_tr = Xtr if "Rule" in name else Xtr[num]
        X_va = Xva if "Rule" in name else Xva[num]
        m = fit(model, X_tr, ytr, wtr)
        fitted[name] = (m, "Rule" in name)
        p_tr, p_va = m.predict(X_tr), m.predict(X_va)
        preds[name] = p_va
        cv = cross_validate(model, X_tr, ytr, wtr, cfg["evaluation"]["cv_folds"], seed, n_classes)
        mt, mv = metrics(ytr, p_tr, wtr, n_classes), metrics(yva, p_va, wva, n_classes)
        rows.append({
            "model": name, "settings": settings,
            **{f"train_{k}": v for k, v in mt.items()},
            **summarise_cv(cv),
            **{f"val_{k}": v for k, v in mv.items()},
            "meets_success_criterion": bool(mv["macro_f1"] >= succ["min_macro_f1"]
                                            and mv["severe_over_risk"] <= succ["max_severe_over_risk"]),
        })
        pcf.append({"model": name, **{f"A{i}": f for i, f in enumerate(per_class_f1(yva, p_va, wva, n_classes))}})
        for s in cfg["evaluation"]["cv_seeds"]:
            cvs = cross_validate(model, X_tr, ytr, wtr, cfg["evaluation"]["cv_folds"], s, n_classes)
            seed_rows.append({"model": name, "cv_seed": s, "cv_macro_f1_mean": cvs["cv_macro_f1"].mean(),
                              "cv_macro_f1_sd": cvs["cv_macro_f1"].std()})
        print(f"  {name:32s} train F1 {mt['macro_f1']:.3f} | CV F1 {cv['cv_macro_f1'].mean():.3f} "
              f"± {cv['cv_macro_f1'].std():.3f} | val F1 {mv['macro_f1']:.3f}")

    table = pd.DataFrame(rows)
    table.to_csv(out / "m4_baselines.csv", index=False)
    report_table(table).to_csv(out / "m4_results_table.csv", index=False)
    preds.to_csv(out / "m4_val_predictions.csv", index=False)
    pd.DataFrame(pcf).to_csv(out / "m4_per_class_f1.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(out / "m4_cv_seed_check.csv", index=False)

    # 3. Robustness: the same validation households, values from implicates 2-5
    imp_rows = []
    for imp in cfg["evaluation"]["validate_implicates"]:
        vi = dataset.validation_implicate(built, imp, cfg)
        for name, (m, is_rule) in fitted.items():
            Xi = vi[cols] if is_rule else vi[num]
            r = metrics(vi["archetype"].to_numpy(), m.predict(Xi), vi["weight"].to_numpy(float), n_classes)
            imp_rows.append({"model": name, "implicate": imp, "val_macro_f1": r["macro_f1"],
                             "label_changed_vs_implicate1": float(np.mean(
                                 vi.set_index("household_id")["archetype"].reindex(va["household_id"]).to_numpy()
                                 != yva))})
    pd.DataFrame(imp_rows).to_csv(out / "m4_implicates.csv", index=False)

    # 4. Figures from the saved predictions
    confusion_matrices(preds, ["Logistic regression", f"k-NN (k={k_best})"], n_classes, figs)
    per_class_f1_plot(pd.DataFrame(pcf), figs)

    info = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "seed": seed, "k_knn": k_best,
            "success_criterion": succ, "n_train": len(tr), "n_val": len(va), "test_loaded": "test" in built["data"]}
    (out / "m4_run_info.json").write_text(json.dumps(info, indent=2))
    print("Saved results to", out)


if __name__ == "__main__":
    main()
