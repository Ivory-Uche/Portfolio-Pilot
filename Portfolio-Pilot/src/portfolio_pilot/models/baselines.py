"""Baseline classifiers. Scaling is a pipeline step, so it is learned only from the data each model is fitted on."""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logreg(C=1.0, max_iter=2000, class_weight="balanced", seed=None):
    """Multinomial logistic regression with L2 regularisation and balanced class weights.

    Survey weights are passed at fit time as model__sample_weight (multiplied by the class weights).
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=C, max_iter=max_iter, class_weight=class_weight, random_state=seed)),
    ])


def make_knn(k=15, weights="uniform"):
    """k-nearest neighbours on standardised features (k-NN cannot use survey weights when fitting)."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=k, weights=weights)),
    ])


def make_centroid():
    """Nearest centroid: each archetype is represented by its mean intake profile (scaled)."""
    return Pipeline([("scale", StandardScaler()), ("model", NearestCentroid())])
