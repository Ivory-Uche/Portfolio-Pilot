"""Reference and baseline classifiers for the archetype prediction task.

Every model has the scikit-learn interface: fit(X, y, sample_weight=None) and predict(X),
where X is a DataFrame with the CLASSIFIER_FEATURES columns (plus age_band for the rule).
"""

from .baselines import make_centroid, make_knn, make_logreg
from .reference import MajorityClass, QuestionnaireRule

__all__ = ["MajorityClass", "QuestionnaireRule", "make_logreg", "make_knn", "make_centroid"]
