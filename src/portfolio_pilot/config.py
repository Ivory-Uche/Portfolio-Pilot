"""Read config.yaml and record the numbers quoted in the report."""
import json
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_ROOT / "config.yaml"


def load_config(path=CONFIG_FILE):
    """Return the settings with every path turned into an absolute Path."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: PROJECT_ROOT / v for k, v in cfg["paths"].items()}
    return cfg


def ensure_output_dirs(cfg):
    for key in ["processed_dir", "figures_dir", "tables_dir"]:
        cfg["paths"][key].mkdir(parents=True, exist_ok=True)
    cfg["paths"]["numbers_file"].parent.mkdir(parents=True, exist_ok=True)


def to_plain(v):
    """Make numpy values JSON-friendly."""
    if isinstance(v, dict):
        return {str(k): to_plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [to_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return to_plain(v.tolist())
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 6)
    return v


class Results:
    """Every count quoted in the report, by section. Saved to a JSON file.

    If the file already exists its sections are loaded first, so each notebook
    adds to the numbers written by the earlier ones.
    """

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.numbers = {}
        if self.path is not None and self.path.exists():
            self.numbers = json.loads(self.path.read_text(encoding="utf-8"))

    def note(self, section, key, value):
        self.numbers.setdefault(section, {})[key] = value

    def __getitem__(self, section):
        return self.numbers[section]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(to_plain(self.numbers), indent=2), encoding="utf-8")
