"""
Model wrapper for Entity Resolution Classifier.
Uses Gradient Boosted Decision Trees (HistGradientBoostingClassifier).
"""

import os
import joblib
import numpy as np
from typing import List, Optional
from sklearn.ensemble import HistGradientBoostingClassifier
from .config import MODEL_PARAMS, DEFAULT_THRESHOLD, DEFAULT_MODEL_PATH
from .features import FEATURE_NAMES


class EntityResolutionModel:
    """Wrapper around GBDT classifier with threshold calibration for Macro F_0.5."""

    def __init__(
        self,
        classifier=None,
        threshold: float = DEFAULT_THRESHOLD,
        feature_names: List[str] = None
    ):
        self.feature_names = feature_names or FEATURE_NAMES
        self.threshold = threshold
        self.clf = classifier or HistGradientBoostingClassifier(**MODEL_PARAMS)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the classifier on candidate pair features and labels."""
        self.clf.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict match probabilities (probability of class 1)."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet.")
        probs = self.clf.predict_proba(X)
        return probs[:, 1]

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Predict binary match decisions based on calibrated threshold."""
        thresh = threshold if threshold is not None else self.threshold
        probs = self.predict_proba(X)
        return (probs >= thresh).astype(int)

    def save(self, filepath: str = DEFAULT_MODEL_PATH):
        """Serialize model, calibrated threshold, and metadata to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        payload = {
            "clf": self.clf,
            "threshold": self.threshold,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted
        }
        joblib.dump(payload, filepath, compress=3)

    @classmethod
    def load(cls, filepath: str = DEFAULT_MODEL_PATH) -> "EntityResolutionModel":
        """Deserialize model artifact from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")
        payload = joblib.load(filepath)
        instance = cls(
            classifier=payload["clf"],
            threshold=payload["threshold"],
            feature_names=payload["feature_names"]
        )
        instance.is_fitted = payload.get("is_fitted", True)
        return instance
