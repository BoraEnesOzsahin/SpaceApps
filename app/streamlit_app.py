"""Streamlit application for interacting with the exoplanet classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.data import KOI_FEATURE_COLUMNS, TARGET_COLUMN, load_koi_dataframe
from src.model import load_metrics, load_model

MODEL_PATH = PROJECT_ROOT / "models" / "exoplanet_classifier.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "models" / "feature_importances.json"

st.set_page_config(page_title="Exoplanet Transit Classifier", layout="wide")
st.title("Exoplanet Transit Classification Toolkit")


@st.cache_data(show_spinner=True)
def get_dataset() -> pd.DataFrame:
    return load_koi_dataframe()


@st.cache_resource(show_spinner=True)
def get_model():
    return load_model(MODEL_PATH)


@st.cache_data(show_spinner=False)
def get_metrics() -> Dict[str, object] | None:
    if METRICS_PATH.exists():
        return load_metrics(METRICS_PATH)
    return None


@st.cache_data(show_spinner=False)
def get_feature_importances() -> pd.DataFrame | None:
    if not FEATURE_IMPORTANCE_PATH.exists():
        return None
    payload = json.loads(FEATURE_IMPORTANCE_PATH.read_text(encoding="utf-8"))
    return pd.DataFrame({"feature": payload["features"], "importance": payload["importances"]})


def render_metrics_section(metrics: Dict[str, object] | None) -> None:
    st.header("Model performance")
    if not metrics:
        st.warning("No metrics found. Run `python -m src.train` to train the model and populate metrics.")
        return

    cols = st.columns(3)
    cols[0].metric("Validation accuracy", f"{metrics['accuracy']:.3f}")
    class_report = pd.DataFrame(metrics["classification_report"]).T
    class_report = class_report.rename(columns={"precision": "Precision", "recall": "Recall", "f1-score": "F1-score"})
    st.dataframe(class_report.style.format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1-score": "{:.2f}"}))

    matrix = np.array(metrics["confusion_matrix"])
    labels = metrics.get("labels", [])
    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
    heatmap = (
        alt.Chart(matrix_df.reset_index().melt(id_vars="index", var_name="Predicted", value_name="Count"))
        .mark_rect()
        .encode(x="Predicted:O", y="index:O", color="Count:Q", tooltip=["index", "Predicted", "Count"])
        .properties(width=300, height=300, title="Confusion matrix")
    )
    st.altair_chart(heatmap, use_container_width=False)

    feature_df = get_feature_importances()
    if feature_df is not None:
        chart = (
            alt.Chart(feature_df)
            .mark_bar()
            .encode(
                x="importance:Q",
                y=alt.Y("feature:N", sort="-x"),
                tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")],
            )
            .properties(title="Feature importance", width=400, height=300)
        )
        st.altair_chart(chart, use_container_width=False)


def render_dataset_section(df: pd.DataFrame) -> None:
    st.header("Explore the Kepler Objects of Interest dataset")
    with st.expander("Dataset preview", expanded=True):
        disposition_filter = st.multiselect(
            "Filter by disposition", sorted(df[TARGET_COLUMN].unique()), default=list(sorted(df[TARGET_COLUMN].unique()))
        )
        filtered = df[df[TARGET_COLUMN].isin(disposition_filter)]
        st.write(f"Showing {len(filtered):,} of {len(df):,} records.")
        st.dataframe(filtered.head(100))

    with st.expander("Feature distributions"):
        feature = st.selectbox("Select feature", KOI_FEATURE_COLUMNS)
        chart = (
            alt.Chart(df)
            .mark_bar(opacity=0.7)
            .encode(x=alt.X(f"{feature}:Q", bin=alt.Bin(maxbins=40)), y="count()", color=f"{TARGET_COLUMN}:N")
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)


def _infer_bounds(df: pd.DataFrame, feature: str) -> tuple[float, float]:
    series = df[feature].dropna()
    if series.empty:
        return (0.0, 1.0)
    lower = float(series.quantile(0.01))
    upper = float(series.quantile(0.99))
    if lower == upper:
        upper = lower + 1.0
    return lower, upper


def render_manual_prediction(df: pd.DataFrame) -> None:
    st.subheader("Manual input")
    model = None
    if MODEL_PATH.exists():
        model = get_model()
    else:
        st.info("Model artefact not found. Train the model via `python -m src.train` to enable predictions.")
        return

    st.write("Provide feature values to obtain a disposition prediction.")
    medians = df[list(KOI_FEATURE_COLUMNS)].median(numeric_only=True)
    with st.form("manual_prediction"):
        inputs: Dict[str, float] = {}
        for feature in KOI_FEATURE_COLUMNS:
            lower, upper = _infer_bounds(df, feature)
            default = float(medians.get(feature, (lower + upper) / 2))
            inputs[feature] = st.number_input(
                feature,
                value=default,
                min_value=float(lower),
                max_value=float(upper),
                step=(upper - lower) / 200 if upper > lower else 0.01,
                format="%.4f",
            )
        submitted = st.form_submit_button("Predict disposition")

    if submitted:
        input_df = pd.DataFrame([inputs])
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        st.success(f"Predicted disposition: **{prediction}**")
        prob_df = pd.DataFrame({"disposition": model.classes_, "probability": probabilities})
        prob_chart = (
            alt.Chart(prob_df)
            .mark_bar()
            .encode(
                x="probability:Q",
                y=alt.Y("disposition:N", sort="-x"),
                tooltip=["disposition", alt.Tooltip("probability:Q", format=".2f")],
            )
        )
        st.altair_chart(prob_chart, use_container_width=True)


def render_batch_prediction(df: pd.DataFrame) -> None:
    st.subheader("Batch predictions")
    model = None
    if MODEL_PATH.exists():
        model = get_model()
    else:
        st.info("Model artefact not found. Train the model via `python -m src.train` to enable predictions.")
        return

    uploaded = st.file_uploader("Upload a CSV file with feature columns", type="csv")
    if uploaded is None:
        st.write("Your CSV must contain the following columns:")
        st.code(", ".join(KOI_FEATURE_COLUMNS))
        return

    try:
        user_df = pd.read_csv(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to read CSV: {exc}")
        return

    missing = set(KOI_FEATURE_COLUMNS) - set(user_df.columns)
    if missing:
        st.error(f"Uploaded file is missing required columns: {sorted(missing)}")
        return

    feature_frame = user_df[list(KOI_FEATURE_COLUMNS)]
    predictions = model.predict(feature_frame)
    probabilities = model.predict_proba(feature_frame)
    result_df = user_df.copy()
    result_df["predicted_disposition"] = predictions
    for idx, disposition in enumerate(model.classes_):
        result_df[f"prob_{disposition}"] = probabilities[:, idx]

    st.download_button(
        "Download predictions",
        data=result_df.to_csv(index=False).encode("utf-8"),
        file_name="exoplanet_predictions.csv",
        mime="text/csv",
    )
    st.dataframe(result_df.head(100))


def main() -> None:
    dataset = get_dataset()
    metrics = get_metrics()

    render_metrics_section(metrics)
    render_dataset_section(dataset)

    st.header("Predict new candidates")
    tabs = st.tabs(["Manual input", "Batch predictions"])
    with tabs[0]:
        render_manual_prediction(dataset)
    with tabs[1]:
        render_batch_prediction(dataset)


if __name__ == "__main__":
    main()
