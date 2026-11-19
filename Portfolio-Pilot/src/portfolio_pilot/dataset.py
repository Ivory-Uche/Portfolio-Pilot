"""Rebuild the M3 modelling data from the raw SCF file: clean -> split -> features -> archetype labels.

Every step is the same code as in M3, so the partitions, caps and archetypes are identical.
The test households are dropped straight after the split unless evaluation.use_test is true.
"""

from . import clean, prepare, split
from .features import CLASSIFIER_FEATURES, build_all
from .target import Archetyper
from .utils import Recorder

EXTRA = ["household_id", "weight", "age_band"]   # carried along; age_band is used only by the rule


def build(cfg, paths, rec=None, use_test=None):
    rec = rec or Recorder()
    use_test = cfg["evaluation"]["use_test"] if use_test is None else use_test
    seed = cfg["seed"]

    full = prepare.load_scf(paths["scf_file"], cfg["scf"]["columns"], rec=rec)
    hh = clean.clean_scf(full, cfg["scf"]["implicate"], rec)
    parts = split.split_scf(hh, cfg["split"], seed, rec)
    if not use_test:
        parts.pop("test")                         # the test set is never featured, labelled or scored at M4

    feats, fb = build_all(parts, cfg["scf"]["cap_quantile"], rec)
    t = cfg["target"]
    arch = Archetyper().fit(feats["train"], t["k"], t["method"], seed)
    for d in feats.values():
        d["archetype"] = arch.predict(d)

    data = {name: d[EXTRA + CLASSIFIER_FEATURES + ["archetype"]].reset_index(drop=True)
            for name, d in feats.items()}
    return {"data": data, "full": full, "builder": fb, "archetyper": arch, "rec": rec}


def validation_implicate(built, implicate, cfg):
    """The same validation households, but values from another implicate (2-5).

    Features use the training caps and labels use the training clustering, so this only measures
    how much the imputed values change the predictions.
    """
    full = built["full"]
    ids = built["data"]["val"]["household_id"]
    d = full[(full["implicate"] == implicate) & full["household_id"].isin(ids)]
    d = clean.derive_columns(d)
    d = built["builder"].transform(d)
    d["archetype"] = built["archetyper"].predict(d)
    return d[EXTRA + CLASSIFIER_FEATURES + ["archetype"]].reset_index(drop=True)
