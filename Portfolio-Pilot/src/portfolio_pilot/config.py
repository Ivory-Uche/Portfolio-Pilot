"""Project settings, read from config.yaml at the repository root.

The upper-case names (SEED, COLS, ...) are kept so the M3 notebook keeps working.
`load()` returns the full settings dictionary for the M4 code.
"""

from pathlib import Path

import yaml


def find_root(start=None) -> Path:
    """Find the folder that contains config.yaml and src/portfolio_pilot (searching upward)."""
    for here in [Path(start or Path.cwd()).resolve(), Path(__file__).resolve()]:
        for p in [here, *here.parents]:
            if (p / "config.yaml").exists() and (p / "src" / "portfolio_pilot").is_dir():
                return p
    raise FileNotFoundError("Repository root not found: no config.yaml next to src/portfolio_pilot")


def load(root=None) -> dict:
    root = Path(root) if root else find_root()
    with open(root / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["root"] = root
    return cfg


def get_paths(root=None) -> dict:
    """All file and folder locations. Output folders are created."""
    cfg = load(root)
    root = cfg["root"]
    out = root / cfg["paths"]["outputs"]
    res = root / cfg["paths"]["results"]
    p = {
        "root": root,
        "raw": (root / cfg["paths"]["scf_file"]).parent,
        "scf_file": root / cfg["paths"]["scf_file"],
        "etf_file": root / cfg["paths"]["etf_file"],
        "outputs": out,
        "figures": out / "figures",
        "tables": out / "tables",
        "processed": out / "processed",
        "results": res,
        "results_figures": res / "figures",
    }
    for key in ["raw", "figures", "tables", "processed", "results", "results_figures"]:
        p[key].mkdir(parents=True, exist_ok=True)
    return p


# ---- Upper-case names used by the M3 notebook ----
_c = load()
SEED = _c["seed"]
COLS = _c["scf"]["columns"]
IMPLICATE = _c["scf"]["implicate"]
SPLIT = _c["split"]
CAP_QUANTILE = _c["scf"]["cap_quantile"]
K_RANGE = _c["target"]["k_range"]
CLUSTER_METHOD = _c["target"]["method"]
CHOSEN_K = _c["target"]["k"]
MIN_CLUSTER_SHARE = _c["target"]["min_cluster_share"]
SIL_TOLERANCE = _c["target"]["sil_tolerance"]
N_BOOTSTRAP = _c["target"]["n_bootstrap"]
SIL_SAMPLE = _c["target"]["sil_sample"]
PCA_THRESHOLD = _c["pca"]["threshold"]
TICKERS = _c["etf"]["tickers"]
ETF_START, ETF_END = _c["etf"]["start"], _c["etf"]["end"]
ETF_TRAIN_END, ETF_VAL_END = _c["etf"]["train_end"], _c["etf"]["val_end"]
EXTREME_SD = _c["etf"]["extreme_sd"]
ROLL_WINDOW = _c["etf"]["roll_window"]
