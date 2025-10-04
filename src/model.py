"""Model utilities for exoplanet classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from .data import KOI_FEATURE_COLUMNS


def build_model(*, random_state: int = 42) -> Pipeline:
    """Create a scikit-learn pipeline for classifying KOI dispositions."""

    classifier = GradientBoostingClassifier(random_state=random_state)
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )
    return pipeline


def evaluate_model(model: Pipeline, X_test, y_test) -> Dict[str, object]:
    """Return evaluation metrics for the trained model."""

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    matrix = confusion_matrix(y_test, predictions)
    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "labels": np.unique(y_test).tolist(),
    }


def save_model(model: Pipeline, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path | str) -> Pipeline:
    return joblib.load(path)


def save_metrics(metrics: Dict[str, object], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    return path


def load_metrics(path: Path | str) -> Dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as fp:
        return json.load(fp)


__all__ = [
    "KOI_FEATURE_COLUMNS",
    "build_model",
    "evaluate_model",
    "save_model",
    "load_model",
    "save_metrics",
    "load_metrics",
]
