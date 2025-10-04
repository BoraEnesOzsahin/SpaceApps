"""XGBoost GPU training script with feature engineering for maximum speed."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from src.data import load_koi_dataframe, split_features_and_target
from src.model import build_model, cross_validate_model, evaluate_predictions, save_model, save_metrics, XGBOOST_AVAILABLE
from sklearn.model_selection import StratifiedKFold, cross_val_predict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


def train_xgboost_gpu(
    use_gpu: bool = True,
    cv_splits: int = 5,
    random_state: int = 42,
    use_feature_engineering: bool = True,
    use_polynomial: bool = True,
    model_path: str = "models/xgboost_gpu_model.joblib",
    metrics_path: str = "models/xgboost_gpu_metrics.json",
):
    """Train XGBoost model with GPU acceleration."""
    
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
    
    model_name = "xgboost_gpu" if use_gpu else "xgboost_cpu"
    
    LOGGER.info("="*60)
    LOGGER.info("XGBOOST GPU-ACCELERATED TRAINING")
    LOGGER.info("="*60)
    LOGGER.info(f"Model: {model_name}")
    LOGGER.info(f"GPU Enabled: {use_gpu}")
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
    
    # Build model pipeline
    LOGGER.info(f"\nBuilding {model_name} pipeline...")
    pipeline = build_model(
        model_name,
        random_state=random_state,
        use_feature_engineering=use_feature_engineering,
        use_polynomial=use_polynomial,
    )
    
    LOGGER.info(f"Pipeline steps: {list(pipeline.named_steps.keys())}")
    
    # Cross-validation
    LOGGER.info(f"\nPerforming {cv_splits}-fold cross-validation...")
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
    cv_start = time.perf_counter()
    cv_scores = cross_validate_model(pipeline, X, y, cv=cv, scoring="accuracy")
    cv_duration = time.perf_counter() - cv_start
    
    LOGGER.info(f"CV completed in {cv_duration:.1f}s")
    LOGGER.info(f"CV Accuracy: {cv_scores['test_score'].mean():.4f} (+/- {cv_scores['test_score'].std():.4f})")
    
    # Get cross-validated predictions
    LOGGER.info("\nGenerating cross-validated predictions...")
    y_pred = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=1)  # n_jobs=1 for GPU
    
    # Evaluate
    LOGGER.info("\nEvaluating model performance...")
    metrics = evaluate_predictions(y_true=y, y_pred=y_pred, labels=labels)
    
    # Train final model on full dataset
    LOGGER.info("\nTraining final model on full dataset...")
    train_start = time.perf_counter()
    pipeline.fit(X, y)
    train_duration = time.perf_counter() - train_start
    LOGGER.info(f"Final training completed in {train_duration:.1f}s")
    
    # Save model and metrics
    LOGGER.info(f"\nSaving model to: {model_path}")
    save_model(pipeline, model_path)
    
    LOGGER.info(f"Saving metrics to: {metrics_path}")
    save_metrics(metrics, metrics_path)
    
    total_duration = time.perf_counter() - start_time
    
    LOGGER.info("="*60)
    LOGGER.info("TRAINING COMPLETE")
    LOGGER.info("="*60)
    LOGGER.info(f"Total Duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
    LOGGER.info(f"CV Duration: {cv_duration:.1f}s")
    LOGGER.info(f"Final Training: {train_duration:.1f}s")
    LOGGER.info(f"Accuracy: {metrics['accuracy']:.4f}")
    LOGGER.info(f"F1 Score: {metrics['f1_score']:.4f}")
    LOGGER.info(f"Precision: {metrics['precision']:.4f}")
    LOGGER.info(f"Recall: {metrics['recall']:.4f}")
    LOGGER.info("="*60)
    
    return pipeline, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost model with GPU acceleration")
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU and use CPU training"
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )
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
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/xgboost_gpu_model.joblib",
        help="Path to save the trained model"
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default="models/xgboost_gpu_metrics.json",
        help="Path to save the evaluation metrics"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    train_xgboost_gpu(
        use_gpu=not args.no_gpu,
        cv_splits=args.cv_splits,
        random_state=args.random_state,
        use_feature_engineering=not args.no_feature_engineering,
        use_polynomial=not args.no_polynomial,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
    )
