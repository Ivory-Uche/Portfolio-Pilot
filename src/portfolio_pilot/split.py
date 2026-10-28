"""Training, validation, and test splits, with a fixed seed."""
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Results


def split_households(hh, split, seed, min_stratum_size=10, results=None):
    """Stratified split by household (age band x risk willingness).

    Strata with fewer than `min_stratum_size` households are merged into "other".
    Returns {"train": ..., "val": ..., "test": ...}.
    """
    results = Results() if results is None else results
    strata = hh["age_band"] + "|" + hh["risk_willingness"].astype(str)
    counts = strata.value_counts()
    strata = strata.where(strata.map(counts) >= min_stratum_size, "other")
    results.note("split", "n_strata", strata.nunique())

    train_val, test = train_test_split(hh, test_size=split["test"], stratify=strata, random_state=seed)
    train, val = train_test_split(train_val, test_size=split["val"] / (split["train"] + split["val"]),
                                  stratify=strata.loc[train_val.index], random_state=seed)
    parts = {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True),
             "test": test.reset_index(drop=True)}

    for name, d in parts.items():
        results.note("split", f"{name}_households", len(d))
        results.note("split", f"{name}_weight_share", d["weight"].sum() / hh["weight"].sum())
    return parts


def save_split_ids(parts, path):
    pd.concat([pd.DataFrame({"household_id": d["household_id"], "partition": n}) for n, d in parts.items()]) \
      .to_csv(path, index=False)


def load_partitions(hh, ids_path):
    """Rebuild the partitions from the saved household ids, in their original order."""
    ids = pd.read_csv(ids_path)
    indexed = hh.set_index("household_id", drop=False)
    return {name: indexed.loc[g["household_id"]].reset_index(drop=True)
            for name, g in ids.groupby("partition", sort=False)}


def split_returns_by_time(ret, train_end, val_end, results=None):
    """Chronological split (no shuffling: the future must never inform the past)."""
    results = Results() if results is None else results
    t_end, v_end = pd.Timestamp(train_end), pd.Timestamp(val_end)
    parts = {"train": ret.loc[:t_end],
             "val": ret.loc[t_end + pd.Timedelta(days=1): v_end],
             "test": ret.loc[v_end + pd.Timedelta(days=1):]}
    for name, d in parts.items():
        results.note("etf_split", name, {"days": len(d), "from": str(d.index.min().date()),
                                         "to": str(d.index.max().date())})
    return parts
