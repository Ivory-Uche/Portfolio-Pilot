"""Reproduce every M4 result with one command (from the project root):

    python -m portfolio_pilot.run_baselines

Writes to results/:
    m4_results_table.csv      the report's results table, built from m4_baselines.csv
    m4_baselines.csv          every metric (training, cross-validation and validation)
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

from . import dataset
from .config import load_config
from .evaluate import cross_validate, fit, metrics, per_class_f1, summarise_cv
from .features import CLASSIFIER_FEATURES
from .models.baselines import make_centroid, make_knn, make_logreg
from .models.reference import MajorityClass, QuestionnaireRule
from .plots import confusion_matrices, per_class_f1_plot


def build_models(cfg, k_best, n_classes):
    m, lr = cfg["models"], cfg["models"]["logreg"]
    return {
        "Reference: majority class": (MajorityClass(), "predicts the largest weighted archetype"),
        "Rule: questionnaire score": (QuestionnaireRule(m["questionnaire"]["age_points"], n_classes),
                                      "risk willingness + age points, no learning"),
        "Nearest centroid": (make_centroid(), "scaled features"),
        "Logistic regression": (make_logreg(lr["C"], lr["max_iter"], lr["class_weight"], cfg["seed"]),
                                f"L2, C={lr['C']}, class_weight={lr['class_weight']}, survey weights"),
        f"k-NN (k={k_best})": (make_knn(k_best, m["knn"]["weights"]), f"k={k_best} chosen on validation, scaled"),
    }


def report_table(t):
    """The results table exactly as it appears in the report."""
    f = lambda x: f"{x:.3f}"
    pct = lambda x: f"{x * 100:.1f}%"
    return pd.DataFrame({
        "Model": t["model"],
        "Macro F1, training": t["train_macro_f1"].map(f),
        "Macro F1, 5-fold CV (mean ± SD)": [f"{a:.3f} ± {b:.3f}" for a, b in zip(t["cv_macro_f1_mean"], t["cv_macro_f1_sd"])],
        "Macro F1, validation": t["val_macro_f1"].map(f),
        "Severe over-risk, validation": t["val_severe_over_risk"].map(pct),
        "Within one archetype, validation": t["val_within_one"].map(pct),
        "Meets criterion": t["meets_success_criterion"].map({True: "Yes", False: "No"}),
    })


def main():
    cfg = load_config()
    out = cfg["paths"]["results_dir"]
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    seed, ev, succ = cfg["seed"], cfg["evaluation"], cfg["evaluation"]["success"]

    print("Building data (clean -> split -> features -> archetypes) ...")
    built = dataset.build(cfg)
    n_classes = built["k"]
    tr, va = built["data"]["train"], built["data"]["val"]
    print(f"Training households: {len(tr)} | validation: {len(va)} | archetypes: {n_classes} | "
          f"test set loaded: {'test' in built['data']}")

    cols = CLASSIFIER_FEATURES + ["age_band"]          # age_band is read only by the questionnaire rule
    num = CLASSIFIER_FEATURES                           # the learned models see only the intake features
    Xtr, ytr, wtr = tr[cols], tr["archetype"].to_numpy(), tr["weight"].to_numpy(float)
    Xva, yva, wva = va[cols], va["archetype"].to_numpy(), va["weight"].to_numpy(float)

    # 1. k-NN: small documented search on the validation set
    ks = []
    for k in cfg["models"]["knn"]["k_grid"]:
        m = make_knn(k, cfg["models"]["knn"]["weights"]).fit(Xtr[num], ytr)
        ks.append({"k": k, **{f"val_{a}": b for a, b in metrics(yva, m.predict(Xva[num]), wva, n_classes).items()}})
    ks = pd.DataFrame(ks)
    ks.to_csv(out / "m4_knn_k_search.csv", index=False)
    k_best = int(ks.sort_values("val_macro_f1", ascending=False).iloc[0]["k"])
    print("k-NN search:", ks[["k", "val_macro_f1"]].round(3).to_dict(orient="records"), "-> k =", k_best)

    # 2. Fit on the training households; cross-validate on training; score on validation
    rows, pcf, seed_rows, fitted = [], [], [], {}
    preds = va[["household_id", "weight", "archetype"]].copy()
    for name, (model, settings) in build_models(cfg, k_best, n_classes).items():
        rule = "Rule" in name
        X_tr, X_va = (Xtr, Xva) if rule else (Xtr[num], Xva[num])
        m = fit(model, X_tr, ytr, wtr)
        fitted[name] = (m, rule)
        p_tr, p_va = m.predict(X_tr), m.predict(X_va)
        preds[name] = p_va
        cv = cross_validate(model, X_tr, ytr, wtr, ev["cv_folds"], seed, n_classes)
        mt, mv = metrics(ytr, p_tr, wtr, n_classes), metrics(yva, p_va, wva, n_classes)
        rows.append({"model": name, "settings": settings,
                     **{f"train_{k}": v for k, v in mt.items()}, **summarise_cv(cv),
                     **{f"val_{k}": v for k, v in mv.items()},
                     "meets_success_criterion": bool(mv["macro_f1"] >= succ["min_macro_f1"]
                                                     and mv["severe_over_risk"] <= succ["max_severe_over_risk"])})
        pcf.append({"model": name, **{f"A{i}": f for i, f in enumerate(per_class_f1(yva, p_va, wva, n_classes))}})
        for s in ev["cv_seeds"]:
            cvs = cross_validate(model, X_tr, ytr, wtr, ev["cv_folds"], s, n_classes)
            seed_rows.append({"model": name, "cv_seed": s, "cv_macro_f1_mean": cvs["cv_macro_f1"].mean(),
                              "cv_macro_f1_sd": cvs["cv_macro_f1"].std()})
        print(f"  {name:28s} train F1 {mt['macro_f1']:.3f} | CV F1 {cv['cv_macro_f1'].mean():.3f} "
              f"± {cv['cv_macro_f1'].std():.3f} | val F1 {mv['macro_f1']:.3f}")

    table = pd.DataFrame(rows)
    table.to_csv(out / "m4_baselines.csv", index=False)
    report_table(table).to_csv(out / "m4_results_table.csv", index=False)
    preds.to_csv(out / "m4_val_predictions.csv", index=False)
    pd.DataFrame(pcf).to_csv(out / "m4_per_class_f1.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(out / "m4_cv_seed_check.csv", index=False)

    # 3. Robustness: the same validation households, values from implicates 2-5
    imp_rows = []
    for imp in ev["validate_implicates"]:
        vi = dataset.validation_implicate(built, imp)
        changed = float(np.mean(vi.set_index("household_id")["archetype"].reindex(va["household_id"]).to_numpy() != yva))
        for name, (m, rule) in fitted.items():
            r = metrics(vi["archetype"].to_numpy(), m.predict(vi[cols] if rule else vi[num]),
                        vi["weight"].to_numpy(float), n_classes)
            imp_rows.append({"model": name, "implicate": imp, "val_macro_f1": r["macro_f1"],
                             "label_changed_vs_implicate1": changed})
    pd.DataFrame(imp_rows).to_csv(out / "m4_implicates.csv", index=False)

    # 4. Figures from the saved predictions
    confusion_matrices(preds, ["Logistic regression", f"k-NN (k={k_best})"], n_classes, figs)
    per_class_f1_plot(pd.DataFrame(pcf), figs)

    info = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "seed": seed, "archetypes_k": n_classes,
            "k_knn": k_best, "success_criterion": succ, "n_train": len(tr), "n_val": len(va),
            "test_loaded": "test" in built["data"]}
    (out / "m4_run_info.json").write_text(json.dumps(info, indent=2))
    print("Saved results to", out)


if __name__ == "__main__":
    main()
