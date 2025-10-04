"""Script to train the exoplanet classification model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

from sklearn.model_selection import train_test_split

from .data import KOI_FEATURE_COLUMNS, load_koi_dataframe, split_features_and_target
from .model import build_model, evaluate_model, save_metrics, save_model

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_PATH = Path("models/exoplanet_classifier.joblib")
DEFAULT_METRICS_PATH = Path("models/metrics.json")
DEFAULT_FEATURE_IMPORTANCE_PATH = Path("models/feature_importances.json")


def train(
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    refresh_data: bool = False,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    metrics_path: Path | str = DEFAULT_METRICS_PATH,
    feature_importance_path: Path | str = DEFAULT_FEATURE_IMPORTANCE_PATH,
) -> Dict[str, object]:
    """Train the classifier and persist the artefacts to disk."""

    df = load_koi_dataframe(refresh=refresh_data)
    X, y = split_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = build_model(random_state=random_state)
    LOGGER.info("Training model on %d samples", len(X_train))
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    LOGGER.info("Validation accuracy: %.3f", metrics["accuracy"])

    save_model(model, model_path)
    save_metrics(metrics, metrics_path)

    if hasattr(model[-1], "feature_importances_"):
        importances = model[-1].feature_importances_
        importance_payload = {
            "features": list(KOI_FEATURE_COLUMNS),
            "importances": importances.tolist(),
        }
        with Path(feature_importance_path).open("w", encoding="utf-8") as fp:
            json.dump(importance_payload, fp, indent=2)
    else:
        LOGGER.warning("Model does not expose feature importances; skipping export")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data to reserve for validation")
    parser.add_argument("--refresh-data", action="store_true", help="Force re-download of the KOI dataset")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for data splitting and model")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to store the trained model")
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH, help="Path to store evaluation metrics")
    parser.add_argument(
        "--feature-importance-path",
        type=Path,
        default=DEFAULT_FEATURE_IMPORTANCE_PATH,
        help="Path to store feature importance values",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (e.g. INFO, DEBUG)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    train(
        test_size=args.test_size,
        refresh_data=args.refresh_data,
        random_state=args.random_state,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        feature_importance_path=args.feature_importance_path,
    )


if __name__ == "__main__":
    main()
