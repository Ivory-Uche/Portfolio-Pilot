"""Train / validation / test partitions."""

import pandas as pd
from sklearn.model_selection import train_test_split


def split_scf(hh, split, seed, rec=None, out_dir=None):
    """Stratified household split on age band x risk willingness, done before any exploration."""
    strata = hh["age_band"] + "|" + hh["risk_willingness"].astype(str)
    counts = strata.value_counts()
    strata = strata.where(strata.map(counts) >= 10, "other")      # merge tiny strata

    train_val, test = train_test_split(hh, test_size=split["test"], stratify=strata, random_state=seed)
    train, val = train_test_split(train_val, test_size=split["val"] / (split["train"] + split["val"]),
                                  stratify=strata.loc[train_val.index], random_state=seed)
    parts = {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True),
             "test": test.reset_index(drop=True)}

    if rec:
        rec.note("split", "n_strata", strata.nunique())
        for name, d in parts.items():
            rec.note("split", f"{name}_households", len(d))
            rec.note("split", f"{name}_weight_share", d["weight"].sum() / hh["weight"].sum())
    if out_dir:   # household IDs, so implicates 2-5 can follow the same partition later
        pd.concat([pd.DataFrame({"household_id": d["household_id"], "partition": n})
                   for n, d in parts.items()]).to_csv(out_dir / "scf_split_ids.csv", index=False)
    return parts


def split_etf(ret, train_end, val_end, rec=None, out_dir=None):
    """Chronological split: the future never informs the past."""
    t_end, v_end = pd.Timestamp(train_end), pd.Timestamp(val_end)
    parts = {"train": ret.loc[:t_end],
             "val": ret.loc[t_end + pd.Timedelta(days=1): v_end],
             "test": ret.loc[v_end + pd.Timedelta(days=1):]}
    for name, d in parts.items():
        if out_dir:
            d.to_csv(out_dir / f"etf_returns_{name}.csv")
        if rec:
            rec.note("etf_split", name, {"days": len(d), "from": str(d.index.min().date()),
                                         "to": str(d.index.max().date())})
    return parts
