"""Report tables built from the Recorder (run log)."""

import pandas as pd


def table1_data_problems(rec, train):
    """Table 1: data problems and counts."""
    E, G, F = rec["errors"], rec["genuine"], rec["features"]
    out_counts = {r["variable"]: r["n_beyond_3iqr"] for r in rec["outliers"]["table"]}
    flags = rec["etf_outliers"]["n_flagged_by_ticker"]
    dates = rec["etf_outliers"]["dates"]
    rows = [
        ["Missing values in SCF source variables", f'{rec["missing"]["total"]}'],
        ["Debt-to-income and liquid-to-income undefined (income ≤ 0)",
         f'{int((train["income"] <= 0).sum())} train ({int((train["income"] == 0).sum())} zero, '
         f'{int((train["income"] < 0).sum())} negative)'],
        ["Equity share undefined (financial assets = 0)", f'{F["train_fin_zero"]} train'],
        ["Duplicate records (record ID / exact rows)",
         f'{rec["duplicates"]["duplicate_record_ids"]} / {rec["duplicates"]["exact_duplicate_rows"]}'],
        ["Impossible values (age < 18, weight ≤ 0, negative assets or debt)",
         f'{E["age_under_18_or_over_100"]} age, {E["weight_not_positive"]} weight, '
         f'{E["liq_negative"] + E["fin_negative"] + E["equity_negative"]} negative assets, '
         f'{E["debt_negative"]} negative debt'],
        ["Inconsistencies (equity > financial assets; both risk flags = 1)",
         f'{E["equity_more_than_fin"]} / {E["both_risk_flags_1"]}'],
        ["Negative net worth", f'{G["networth_negative"]}'],
        ["Extreme wealth and income (beyond 3×IQR, raw scale)",
         f'{out_counts["networth"]} net worth, {out_counts["income"]} income (train)'],
        ["Extreme ratios (above training 99th pct.)",
         f'{F["train_dti_above_cap"]} DTI, {F["train_lti_above_cap"]} LTI (train)'],
        ["ETF missing prices / duplicate dates",
         f'{rec["etf"]["missing_prices"]} / {rec["etf"]["duplicate_date_ticker"]} '
         f'({rec["etf"]["cells_forward_filled"]} cells forward-filled)'],
        ["ETF extreme daily returns (> 5 training SD)", f'{sum(flags.values())} (train), on {len(dates)} dates'],
    ]
    return pd.DataFrame(rows, columns=["Problem", "Count"])


def table2_partitions(rec):
    """Table 2: partition sizes."""
    rows = []
    for name, label in [("train", "Training"), ("val", "Validation"), ("test", "Test")]:
        n = rec["split"][f"{name}_households"]
        share = rec["split"][f"{name}_weight_share"] * 100
        e = rec["etf_split"][name]
        rows.append([label, f"{n:,} ({share:.1f}%)", f'{e["days"]:,}', f'{e["from"]} – {e["to"]}'])
    return pd.DataFrame(rows, columns=["Partition", "SCF households (weighted share)",
                                       "ETF trading days", "ETF date range"])


def table3_archetypes(prof):
    """Table 3: archetype profiles (weighted medians; risk is a weighted mean)."""
    return pd.DataFrame({
        "Archetype": [f"A{a}" for a in prof.index],
        "Share (wtd.)": (prof["weighted_share"] * 100).round(1).astype(str) + "%",
        "Age": prof["median_age"].round(0).astype(int),
        "Risk (mean, 0–2)": prof["mean_risk_willingness"].round(2),
        "Net worth (USD)": prof["median_networth"].round(-3).map("{:,.0f}".format),
        "Debt / income": prof["median_dti"].round(2),
        "Equity share": (prof["median_equity_share"] * 100).round(0).astype(int).astype(str) + "%",
    })


def table4_pca(rec):
    """Table 4: cumulative variance of the first four components."""
    rows = []
    for label, key in [("SCF clustering variables (6)", "pca_scf"), ("ETF daily returns (9)", "pca_etf")]:
        cum = rec[key]["cumulative"]
        rows.append([label] + [f"{cum[i] * 100:.1f}%" for i in range(4)]
                    + [f'{rec[key]["n_to_threshold"]} (not used)'])
    return pd.DataFrame(rows, columns=["Variable set", "PC1", "PC1–2", "PC1–3", "PC1–4", "Kept (80% rule)"])
