"""Principal component analysis, fitted on training data only."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .utils import weighted_resample


def run_pca(df, cols, name, threshold=0.80, seed=7275, weighted=False, rec=None):
    """Standardise `cols`, fit PCA, and record variance explained. weighted=True uses a survey-weighted resample."""
    X = (weighted_resample(df, seed) if weighted else df)[cols].to_numpy(float)
    pca = PCA(random_state=seed).fit(StandardScaler().fit_transform(X))
    evr = pca.explained_variance_ratio_
    loadings = pd.DataFrame(pca.components_.T, index=cols, columns=[f"PC{i + 1}" for i in range(len(cols))])
    if rec:
        rec.note(f"pca_{name}", "explained", evr.tolist())
        rec.note(f"pca_{name}", "cumulative", np.cumsum(evr).tolist())
        rec.note(f"pca_{name}", "n_to_threshold", int(np.searchsorted(np.cumsum(evr), threshold) + 1))
        if weighted:
            rec.note(f"pca_{name}", "n_eigen_gt_1", int((pca.explained_variance_ > 1).sum()))
    return pca, evr, loadings
