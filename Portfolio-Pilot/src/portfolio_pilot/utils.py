"""Run log and survey-weight helpers."""

import json

import numpy as np


class Recorder:
    """Collects every number quoted in the report; `save` writes them to JSON."""

    def __init__(self):
        self.data = {}

    def note(self, section, key, value):
        self.data.setdefault(section, {})[key] = value

    def __getitem__(self, section):
        return self.data[section]

    def save(self, path):
        path.write_text(json.dumps(to_plain(self.data), indent=2))


def to_plain(v):
    """Make numpy values JSON-friendly."""
    if isinstance(v, dict):
        return {str(k): to_plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [to_plain(x) for x in v]
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return round(float(v), 6)
    return v


def weighted_resample(df, seed):
    """Draw households with probability proportional to the survey weight."""
    rng = np.random.default_rng(seed)
    p = df["weight"].to_numpy(float)
    return df.iloc[rng.choice(len(df), size=len(df), replace=True, p=p / p.sum())]


def weighted_median(x, w):
    o = np.argsort(np.asarray(x))
    x, w = np.asarray(x, float)[o], np.asarray(w, float)[o]
    c = np.cumsum(w)
    return float(x[np.searchsorted(c, 0.5 * c[-1])])
