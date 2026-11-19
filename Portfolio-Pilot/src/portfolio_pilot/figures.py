"""Report figures (training data only). Each function saves a PNG in fig_dir and shows it."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 110, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def save_fig(fig, fig_dir, name):
    fig.tight_layout()
    fig.savefig(fig_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.show()


def target_balance(tr, fig_dir):
    """Figure 1: share of households per archetype, unweighted vs survey-weighted."""
    arche = sorted(tr["archetype"].unique())
    g = tr.groupby("archetype")
    x = np.arange(len(arche))
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.bar(x - 0.2, g.size() / len(tr) * 100, 0.4, label="Unweighted (sample)")
    ax.bar(x + 0.2, g["weight"].sum() / tr["weight"].sum() * 100, 0.4, label="Survey-weighted (population)")
    ax.set_xticks(x, [f"A{a}" for a in arche])
    ax.set_xlabel("Archetype (A0 = most conservative)")
    ax.set_ylabel("Share of training households (%)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    save_fig(fig, fig_dir, "fig01_target_balance")


def skew_asinh(tr, fig_dir):
    """Figure 2: net worth and income in dollars vs after asinh."""
    fig, axes = plt.subplots(2, 2, figsize=(7, 4.6))
    for row, (c, lab) in enumerate([("networth", "Net worth"), ("income", "Income")]):
        lo, hi = tr[c].quantile([0.01, 0.99])
        axes[row, 0].hist(tr[c].clip(lo, hi) / 1e3, bins=60)
        axes[row, 0].set_xlabel(f"{lab} (USD thousands, 1st-99th pct.)")
        axes[row, 1].hist(np.arcsinh(tr[c]), bins=60)
        axes[row, 1].set_xlabel(f"asinh({lab.lower()} in USD)")
        for a in axes[row]:
            a.set_ylabel("Households (count)")
    save_fig(fig, fig_dir, "fig02_skew_asinh")


def features_by_archetype(tr, fig_dir):
    """Figure 3: intake features by archetype, plus the weighted risk-willingness mix."""
    arche = sorted(tr["archetype"].unique())
    panels = [("age", "Age (years)"), ("asinh_income", "asinh(income, USD)"),
              ("asinh_networth", "asinh(net worth, USD)"), ("asinh_liq", "asinh(liquid assets, USD)"),
              ("asinh_liq_to_income", "asinh(liquid assets / income)")]
    fig, axes = plt.subplots(2, 3, figsize=(8.5, 5))
    for ax, (c, lab) in zip(axes.flat, panels):
        ax.boxplot([tr.loc[tr["archetype"] == a, c] for a in arche], showfliers=False)
        ax.set_xticks(range(1, len(arche) + 1), [f"A{a}" for a in arche])
        ax.set_xlabel("Archetype")
        ax.set_ylabel(lab)
    ax = axes.flat[-1]
    ct = pd.crosstab(tr["archetype"], tr["risk_willingness"], values=tr["weight"],
                     aggfunc="sum", normalize="index") * 100
    bottom = np.zeros(len(ct))
    for r, name in zip(ct.columns, ["No risk", "Average", "Above avg."]):
        ax.bar([f"A{a}" for a in ct.index], ct[r], bottom=bottom, label=name)
        bottom += ct[r].to_numpy()
    ax.set_xlabel("Archetype")
    ax.set_ylabel("Weighted share (%)")
    ax.legend(title="Risk willingness", frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    save_fig(fig, fig_dir, "fig03_features_by_archetype")


def corr_plot(df, cols, fig_dir, name, method):
    """Figures 4 and 6: correlation heatmap. Returns the correlation matrix."""
    c = df[cols].corr(method=method)
    fig, ax = plt.subplots(figsize=(0.6 * len(cols) + 2, 0.55 * len(cols) + 1.5))
    im = ax.imshow(c, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label=f"{method.capitalize()} correlation")
    save_fig(fig, fig_dir, name)
    return c


def k_selection(ktab, fig_dir):
    """Figure 5: silhouette, bootstrap stability and GMM BIC by k."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    for m, d in ktab.groupby("method"):
        axes[0].plot(d["k"], d["silhouette"], marker="o", label=m)
        axes[1].errorbar(d["k"], d["stability_ari_mean"], yerr=d["stability_ari_sd"],
                         marker="o", capsize=2, label=m)
    gm = ktab[ktab["method"] == "gmm"]
    axes[2].plot(gm["k"], gm["bic"], marker="o", color="C1")
    for a, lab in zip(axes, ["Silhouette score", "Bootstrap stability (ARI)", "GMM BIC (lower is better)"]):
        a.set_xlabel("Number of clusters k")
        a.set_ylabel(lab)
    axes[0].legend(frameon=False)
    save_fig(fig, fig_dir, "fig05_k_selection")


def etf_rolling(etr, tickers, fig_dir, window=252):
    """Figure 7: rolling volatility of each ETF and rolling SPY-AGG correlation."""
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.8), sharex=True)
    vol = etr.rolling(window).std() * np.sqrt(252) * 100
    for c in tickers:
        axes[0].plot(vol.index, vol[c], lw=1, label=c)
    axes[0].set_ylabel(f"Rolling {window}-day vol.\n(% per year)")
    axes[0].legend(ncol=5, frameon=False, fontsize=7)
    axes[1].plot(etr.index, etr["SPY"].rolling(window).corr(etr["AGG"]), color="k", lw=1)
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].set_ylabel(f"Rolling {window}-day corr.\nSPY vs AGG")
    axes[1].set_xlabel("Date")
    save_fig(fig, fig_dir, "fig07_etf_rolling")


def pca_scree(evr_scf, evr_etf, fig_dir, threshold=0.80):
    """Figure 8: variance explained per component and cumulatively, SCF and ETF."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for ax, (title, evr) in zip(axes, [("SCF clustering variables", evr_scf), ("ETF daily returns", evr_etf)]):
        xs = np.arange(1, len(evr) + 1)
        ax.bar(xs, evr * 100, label="Per component")
        ax.plot(xs, np.cumsum(evr) * 100, marker="o", color="C3", label="Cumulative")
        ax.axhline(threshold * 100, ls="--", color="grey", lw=1, label=f"{int(threshold * 100)}% threshold")
        ax.set_xticks(xs)
        ax.set_xlabel("Principal component")
        ax.set_ylabel("Variance explained (%)")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=7)
    save_fig(fig, fig_dir, "fig08_pca_scree")
