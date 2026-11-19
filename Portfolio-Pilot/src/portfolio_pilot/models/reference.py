"""Models that learn nothing from the features: the floor every baseline must clear."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class MajorityClass(BaseEstimator, ClassifierMixin):
    """Always predicts the archetype with the largest survey-weighted share in the training data."""

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        self.classes_ = np.unique(y)
        self.majority_ = self.classes_[np.argmax([w[y == c].sum() for c in self.classes_])]
        return self

    def predict(self, X):
        return np.full(len(X), self.majority_)


class QuestionnaireRule(BaseEstimator, ClassifierMixin):
    """A fixed robo-advisor style rule, not learned from data.

    score = risk_willingness (0-2) + age points (younger = more capacity, 0-2), so the score runs 0-4.
    The score is mapped evenly onto archetypes 0..k-1 (with k = 5, score s -> archetype s).
    """

    def __init__(self, age_points=None, n_classes=5):
        self.age_points = age_points
        self.n_classes = n_classes

    def fit(self, X, y=None, sample_weight=None):
        self.classes_ = np.arange(self.n_classes)
        return self

    def predict(self, X):
        pts = X["age_band"].map(self.age_points).fillna(0).to_numpy()
        score = X["risk_willingness"].to_numpy() + pts          # 0..4
        return np.rint(score / 4 * (self.n_classes - 1)).astype(int)
