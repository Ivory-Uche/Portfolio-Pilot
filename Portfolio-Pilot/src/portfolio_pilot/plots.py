"""M4 figures, drawn from the saved validation predictions (no retraining)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _weighted_cm(y, p, w, n):
    cm = np.zeros((n, n))
    for t, q, ww in zip(y, p, w):
        cm[t, q] += ww
    return cm / cm.sum(axis=1, keepdims=True).clip(min=1e-12) * 100


def confusion_matrices(preds, models, n_classes, fig_dir):
    """Row-normalised, survey-weighted confusion matrices on the validation set."""
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4))
    for ax, name in zip(np.atleast_1d(axes), models):
        cm = _weighted_cm(preds["archetype"], preds[name], preds["weight"], n_classes)
        im = ax.imshow(cm, vmin=0, vmax=100, cmap="Blues")
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center",
                        color="white" if cm[i, j] > 60 else "black", fontsize=8)
        labels = [f"A{i}" for i in range(n_classes)]
        ax.set_xticks(range(n_classes), labels)
        ax.set_yticks(range(n_classes), labels)
        ax.set_xlabel("Predicted archetype")
        ax.set_ylabel("True archetype")
        ax.set_title(name, fontsize=10)
    fig.colorbar(im, ax=axes, label="Share of true archetype (%, weighted)", shrink=0.8)
    fig.savefig(fig_dir / "m4_fig1_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def per_class_f1_plot(pcf, fig_dir):
    """Validation F1 for each archetype and model."""
    cls = [c for c in pcf.columns if c.startswith("A")]
    x = np.arange(len(cls))
    width = 0.8 / len(pcf)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for i, (_, r) in enumerate(pcf.iterrows()):
        ax.bar(x + i * width - 0.4 + width / 2, r[cls].astype(float), width, label=r["model"])
    ax.set_xticks(x, cls)
    ax.set_xlabel("True archetype")
    ax.set_ylabel("F1 score (validation, weighted)")
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left", bbox_to_anchor=(1, 1))
    fig.savefig(fig_dir / "m4_fig2_per_class_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
