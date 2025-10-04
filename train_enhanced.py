"""Enhanced training script with feature engineering and hyperparameter optimization."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from src.data import load_koi_dataframe, split_features_and_target
from src.model import build_model, cross_validate_model, evaluate_predictions, save_model, save_metrics
from sklearn.model_selection import StratifiedKFold, cross_val_predict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


def train_enhanced_model(
    model_name: str = "gradient_boosting",
    cv_splits: int = 5,
    random_state: int = 42,
    use_feature_engineering: bool = True,
    use_polynomial: bool = True,
    model_path: str = "models/enhanced_model.joblib",
    metrics_path: str = "models/enhanced_metrics.json",
):
    """Train an enhanced model with feature engineering."""
    
    LOGGER.info("="*60)
    LOGGER.info("ENHANCED MODEL TRAINING (MAXIMUM ACCURACY)")
    LOGGER.info("="*60)
    LOGGER.info(f"Model: {model_name}")
    LOGGER.info(f"Feature Engineering: {'Enabled' if use_feature_engineering else 'Disabled'}")
    LOGGER.info(f"Polynomial Features: {'Enabled' if use_polynomial else 'Disabled'}")
    LOGGER.info(f"CV Folds: {cv_splits}")
    LOGGER.info("="*60)
    
    start_time = time.perf_counter()
    
    # Load data
    LOGGER.info("Loading dataset...")
    df = load_koi_dataframe(refresh=False)
    X, y = split_features_and_target(df)
    labels = sorted(y.unique())
    
    LOGGER.info(f"Dataset size: {len(X)} samples, {X.shape[1]} features")
    LOGGER.info(f"Class distribution:\n{y.value_counts()}")
    
    # Build enhanced pipeline
    LOGGER.info("Building enhanced pipeline...")
    pipeline = build_model(
        model_name=model_name,
        random_state=random_state,
        use_feature_engineering=use_feature_engineering,
        use_polynomial=use_polynomial
    )
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
    LOGGER.info(f"Starting {cv_splits}-fold cross-validation...")
    cv_result = cross_validate_model(pipeline, X, y, cv=cv, labels=labels)
    
    # Display CV results
    LOGGER.info("="*60)
    LOGGER.info("CROSS-VALIDATION RESULTS")
    LOGGER.info("="*60)
    for metric_name, metric_stats in cv_result['metrics'].items():
        if metric_name != 'duration_seconds':
            LOGGER.info(f"{metric_name:25s}: {metric_stats['mean']:.4f} ± {metric_stats['std']:.4f}")
    LOGGER.info("="*60)
    
    # Generate final predictions
    LOGGER.info("Generating cross-validated predictions...")
    cv_predictions = cross_val_predict(pipeline, X, y, cv=cv)
    evaluation = evaluate_predictions(y, cv_predictions, labels)
    
    # Train on full dataset
    LOGGER.info(f"Training on full dataset ({len(X)} samples)...")
    pipeline.fit(X, y)
    
    # Save model and metrics
    save_model(pipeline, model_path)
    LOGGER.info(f"Model saved to: {model_path}")
    
    metrics_payload = {
        "version": 3,
        "model": model_name,
        "feature_engineering_enabled": use_feature_engineering,
        "polynomial_features_enabled": use_polynomial,
        "cv_folds": cv_splits,
        "metrics": {
            "accuracy": evaluation["accuracy"],
            "precision_macro": evaluation["precision_macro"],
            "recall_macro": evaluation["recall_macro"],
            "f1_macro": evaluation["f1_macro"],
            "specificity_macro": evaluation["specificity_macro"],
        },
        "cross_validation_details": cv_result,
        "full_evaluation": evaluation,
        "dataset_size": len(df),
    }
    
    save_metrics(metrics_payload, metrics_path)
    LOGGER.info(f"Metrics saved to: {metrics_path}")
    
    elapsed = time.perf_counter() - start_time
    
    LOGGER.info("="*60)
    LOGGER.info("TRAINING COMPLETE")
    LOGGER.info("="*60)
    LOGGER.info(f"Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    LOGGER.info(f"Final Accuracy: {evaluation['accuracy']:.4f}")
    LOGGER.info(f"Final F1 Score: {evaluation['f1_macro']:.4f}")
    LOGGER.info("="*60)
    
    return metrics_payload


def main():
    parser = argparse.ArgumentParser(description="Train enhanced exoplanet classifier")
    parser.add_argument(
        "--model",
        default="gradient_boosting",
        choices=["gradient_boosting", "random_forest", "extra_trees", "stacking", "voting"],
        help="Model to train"
    )
    parser.add_argument("--cv-splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--no-feature-engineering",
        action="store_true",
        help="Disable feature engineering"
    )
    parser.add_argument(
        "--no-polynomial",
        action="store_true",
        help="Disable polynomial features"
    )
    parser.add_argument("--model-path", default="models/enhanced_model.joblib")
    parser.add_argument("--metrics-path", default="models/enhanced_metrics.json")
    
    args = parser.parse_args()
    
    train_enhanced_model(
        model_name=args.model,
        cv_splits=args.cv_splits,
        random_state=args.random_state,
        use_feature_engineering=not args.no_feature_engineering,
        use_polynomial=not args.no_polynomial,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
    )


if __name__ == "__main__":
    main()
