import numpy as np
import pandas as pd
import pytest

COLUMNS = {
    "household_id": "YY1", "record_id": "Y1", "weight": "WGT", "age": "AGE",
    "income": "INCOME", "networth": "NETWORTH", "liq": "LIQ", "debt": "DEBT",
    "equity": "EQUITY", "fin": "FIN", "yes_risk": "YESFINRISK", "no_risk": "NOFINRISK",
}


@pytest.fixture
def columns():
    return dict(COLUMNS)


def make_scf_file(path, n_households=40, seed=0):
    """Write a small SCF-like CSV: five implicates per household, Y1 = YY1 * 10 + implicate."""
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(1, n_households + 1):
        age = int(rng.integers(20, 90))
        income = float(rng.integers(0, 200_000))
        fin = float(rng.integers(0, 300_000))
        liq = float(min(fin, rng.integers(0, 100_000)))
        equity = float(min(fin - liq, rng.integers(0, 100_000)))
        yes = int(rng.random() < 0.3)
        no = int((not yes) and rng.random() < 0.4)
        for imp in range(1, 6):
            rows.append({"YY1": h, "Y1": h * 10 + imp, "WGT": float(rng.uniform(500, 1500)), "AGE": age,
                         "INCOME": income, "NETWORTH": fin + float(rng.integers(-5_000, 500_000)),
                         "LIQ": liq, "DEBT": float(rng.integers(0, 200_000)), "EQUITY": equity,
                         "FIN": fin, "YESFINRISK": yes, "NOFINRISK": no, "EXTRA": 1})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def households():
    """Cleaned-style table, one row per household, with the derived columns."""
    rng = np.random.default_rng(1)
    n = 1000
    df = pd.DataFrame({
        "household_id": np.arange(1, n + 1),
        "weight": rng.uniform(500, 1500, n),
        "age": rng.integers(20, 90, n),
        "income": rng.lognormal(10.8, 1.0, n),
        "networth": rng.lognormal(11.5, 1.5, n) - 20_000,
        "liq": rng.lognormal(8.5, 1.8, n),
        "debt": rng.lognormal(10.0, 1.5, n),
        "fin": rng.lognormal(10.5, 1.7, n),
    })
    df["equity"] = df["fin"] * rng.uniform(0, 0.9, n)
    df["risk_willingness"] = rng.choice([0, 1, 2], n, p=[0.3, 0.4, 0.3])
    df["age_band"] = pd.cut(df["age"], [0, 34, 44, 54, 64, 74, 200],
                            labels=["<35", "35-44", "45-54", "55-64", "65-74", "75+"]).astype(str)
    return df
