"""Model utilities and evaluation helpers for exoplanet classification."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import joblib
import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from .data import KOI_FEATURE_COLUMNS

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """Metadata describing a candidate ensemble model."""

    name: str
    description: str
    requires_scaling: bool
    estimator_factory: Callable[[int], ClassifierMixin]


def _stacking_factory(random_state: int) -> StackingClassifier:
    """Return a stacked ensemble inspired by ensemble comparisons in recent literature."""

    estimators = [
        (
            "gradient_boosting",
            GradientBoostingClassifier(
                random_state=random_state,
                learning_rate=0.08,
                n_estimators=150,
                max_depth=5,
                min_samples_split=20,
                subsample=0.85,
                max_features="sqrt",
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=150,
                max_depth=25,
                max_features="sqrt",
                min_samples_split=15,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        (
            "extra_trees",
            ExtraTreesClassifier(
                n_estimators=150,
                max_depth=25,
                max_features="sqrt",
                min_samples_split=15,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
    ]
    final_estimator = LogisticRegression(
        max_iter=1000, 
        class_weight="balanced", 
        C=0.5,  # Regularization
        random_state=random_state
    )
    return StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )


def _voting_factory(random_state: int) -> VotingClassifier:
    """Return a soft voting ensemble with diverse base models."""
    
    estimators = [
        (
            "gradient_boosting",
            GradientBoostingClassifier(
                random_state=random_state,
                learning_rate=0.08,
                n_estimators=150,
                max_depth=5,
                min_samples_split=20,
                subsample=0.85,
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=150,
                max_depth=25,
                max_features="sqrt",
                min_samples_split=15,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        (
            "extra_trees",
            ExtraTreesClassifier(
                n_estimators=150,
                max_depth=25,
                max_features="sqrt",
                min_samples_split=15,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
    ]
    
    return VotingClassifier(
        estimators=estimators,
        voting="soft",
        weights=[2, 1, 1],  # Slightly favor gradient boosting
        n_jobs=-1,
    )


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "gradient_boosting": ModelSpec(
        name="gradient_boosting",
        description="Maximum accuracy Gradient Boosting with extended training.",
        requires_scaling=True,
        estimator_factory=lambda random_state: GradientBoostingClassifier(
            random_state=random_state,
            learning_rate=0.05,
            max_depth=7,
            n_estimators=300,
            min_samples_split=15,
            min_samples_leaf=8,
            subsample=0.8,
            max_features="sqrt",
            validation_fraction=0.1,
            n_iter_no_change=15,
        ),
    ),
    "random_forest": ModelSpec(
        name="random_forest",
        description="Maximum accuracy Random Forest with more trees.",
        requires_scaling=False,
        estimator_factory=lambda random_state: RandomForestClassifier(
            n_estimators=250,
            max_depth=30,
            max_features="sqrt",
            min_samples_split=10,
            min_samples_leaf=3,
            max_samples=0.75,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    ),
    "extra_trees": ModelSpec(
        name="extra_trees",
        description="Optimized ExtraTrees with increased diversity.",
        requires_scaling=False,
        estimator_factory=lambda random_state: ExtraTreesClassifier(
            n_estimators=150,
            max_depth=25,
            max_features="sqrt",
            min_samples_split=15,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    ),
    "ada_boost": ModelSpec(
        name="ada_boost",
        description="AdaBoost ensemble with optimized parameters.",
        requires_scaling=True,
        estimator_factory=lambda random_state: AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=4, min_samples_split=20, random_state=random_state),
            n_estimators=200,
            learning_rate=0.4,
            random_state=random_state,
        ),
    ),
    "random_subspace": ModelSpec(
        name="random_subspace",
        description="Random Subspace method via Bagging with feature subsampling.",
        requires_scaling=False,
        estimator_factory=lambda random_state: BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=20, min_samples_split=15, random_state=random_state),
            n_estimators=120,
            max_features=0.7,
            bootstrap=True,
            n_jobs=-1,
            random_state=random_state,
        ),
    ),
    "stacking": ModelSpec(
        name="stacking",
        description="Optimized stacked ensemble with enhanced base models.",
        requires_scaling=True,
        estimator_factory=_stacking_factory,
    ),
    "voting": ModelSpec(
        name="voting",
        description="Soft voting ensemble combining diverse models.",
        requires_scaling=True,
        estimator_factory=_voting_factory,
    ),
}

# Add XGBoost if available
if XGBOOST_AVAILABLE:
    MODEL_REGISTRY["xgboost_gpu"] = ModelSpec(
        name="xgboost_gpu",
        description="XGBoost with GPU acceleration for fast training.",
        requires_scaling=True,
        estimator_factory=lambda random_state: XGBClassifier(
            random_state=random_state,
            tree_method='gpu_hist',  # Enable GPU acceleration
            gpu_id=0,
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric='logloss',
            n_jobs=-1,
        ),
    )
    
    MODEL_REGISTRY["xgboost_cpu"] = ModelSpec(
        name="xgboost_cpu",
        description="XGBoost optimized for CPU training.",
        requires_scaling=True,
        estimator_factory=lambda random_state: XGBClassifier(
            random_state=random_state,
            tree_method='hist',  # Fast CPU algorithm
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric='logloss',
            n_jobs=-1,
        ),
    )


def get_model_registry() -> Dict[str, ModelSpec]:
    """Return the registry of available ensemble configurations."""

    return MODEL_REGISTRY


def build_model(model_name: str = "gradient_boosting", *, random_state: int = 42, 
                use_feature_engineering: bool = True, use_polynomial: bool = False) -> Pipeline:
    """Create a scikit-learn pipeline for classifying KOI dispositions."""

    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{model_name}'. Available models: {sorted(MODEL_REGISTRY)}")

    spec = MODEL_REGISTRY[model_name]
    steps: List[tuple[str, object]] = []
    
    # Add feature engineering if requested (before imputation to create features)
    if use_feature_engineering:
        try:
            from .feature_engineering import ExoplanetFeatureEngineer
            steps.append(("feature_engineer", ExoplanetFeatureEngineer(
                add_polynomial=False,  # Do polynomial separately after imputation
                polynomial_degree=2,
                add_interactions=True,
                add_ratios=True
            )))
        except ImportError:
            LOGGER.warning("Feature engineering module not available, skipping")
    
    # Add imputation (must be after feature engineering but before polynomial)
    steps.append(("imputer", SimpleImputer(strategy="median")))
    
    # Add polynomial features after imputation to avoid NaN issues
    if use_polynomial:
        from sklearn.preprocessing import PolynomialFeatures
        steps.append(("polynomial", PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)))
        LOGGER.info("Added polynomial features (degree=2)")
    
    # Add scaling if needed
    if spec.requires_scaling:
        from sklearn.preprocessing import RobustScaler
        steps.append(("scaler", RobustScaler()))  # More robust to outliers
    
    # Add classifier
    steps.append(("classifier", spec.estimator_factory(random_state)))
    
    # Enable memory caching to avoid redundant preprocessing during CV
    return Pipeline(steps=steps, memory=joblib.Memory(location=".cache", verbose=0))


def compute_specificity(conf: np.ndarray, labels: Iterable[str]) -> Dict[str, float]:
    """Compute specificity (true negative rate) for each class and its macro average."""

    specificity_by_class: Dict[str, float] = {}
    conf = np.asarray(conf)
    total = conf.sum()
    labels_list = list(labels)
    for idx, label in enumerate(labels_list):
        tp = conf[idx, idx]
        fp = conf[:, idx].sum() - tp
        fn = conf[idx, :].sum() - tp
        tn = total - (tp + fp + fn)
        denom = tn + fp
        specificity_by_class[label] = float(tn / denom) if denom else float("nan")
    macro = float(np.nanmean(list(specificity_by_class.values())))
    specificity_by_class["macro"] = macro
    return specificity_by_class


def summarise_basic_metrics(y_true, y_pred, labels: Iterable[str]) -> Dict[str, float]:
    """Return fundamental classification metrics for quick comparison."""

    conf = confusion_matrix(y_true, y_pred, labels=list(labels))
    specificity = compute_specificity(conf, labels)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "specificity_macro": float(specificity["macro"]),
    }
    return metrics


def evaluate_predictions(y_true, y_pred, labels: Iterable[str]) -> Dict[str, object]:
    """Return a rich evaluation payload based on model predictions."""

    labels = list(labels)
    conf = confusion_matrix(y_true, y_pred, labels=labels)
    specificity = compute_specificity(conf, labels)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    payload: Dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "specificity_macro": float(specificity["macro"]),
        "specificity_by_class": {k: float(v) for k, v in specificity.items() if k != "macro"},
        "classification_report": report,
        "confusion_matrix": conf.tolist(),
        "labels": labels,
    }
    return payload


def cross_validate_model(
    pipeline: Pipeline,
    X,
    y,
    *,
    cv,
    labels: Iterable[str],
) -> Dict[str, object]:
    """Evaluate a pipeline using cross-validation and return fold-level metrics."""

    fold_metrics: List[Dict[str, float]] = []
    total_folds = cv.get_n_splits(X, y)
    for fold_index, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        cloned = clone(pipeline)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        start = time.perf_counter()
        LOGGER.info("Fitting fold %d/%d", fold_index, total_folds)
        cloned.fit(X_train, y_train)
        duration = float(time.perf_counter() - start)
        predictions = cloned.predict(X_test)
        metrics = summarise_basic_metrics(y_test, predictions, labels)
        metrics["duration_seconds"] = duration
        metrics["fold"] = fold_index
        fold_metrics.append(metrics)
        LOGGER.info(
            "Completed fold %d/%d in %.2f seconds (f1_macro=%.3f, accuracy=%.3f)",
            fold_index,
            total_folds,
            duration,
            metrics.get("f1_macro", float("nan")),
            metrics.get("accuracy", float("nan")),
        )

    aggregated: Dict[str, Dict[str, float]] = {}
    metric_keys = [key for key in fold_metrics[0] if key not in {"fold"}]
    for key in metric_keys:
        values = [fold[key] for fold in fold_metrics]
        aggregated[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }

    return {"per_fold": fold_metrics, "metrics": aggregated}


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
    "MODEL_REGISTRY",
    "ModelSpec",
    "build_model",
    "compute_specificity",
    "cross_validate_model",
    "evaluate_predictions",
    "get_model_registry",
    "load_metrics",
    "load_model",
    "save_metrics",
    "save_model",
    "summarise_basic_metrics",
]
