# Exoplanet Transit Classification Toolkit

This project provides a reproducible machine-learning pipeline and interactive Streamlit dashboard for classifying candidates from NASA's Kepler Objects of Interest (KOI) catalogue. It automatically downloads publicly available KOI data, trains an ensemble model, and exposes an interface for exploring the dataset and predicting dispositions for new observations.

## Features

- Automated download of a curated subset of the KOI dataset directly from the NASA Exoplanet Archive `nstedAPI` service.
- Scikit-learn pipeline with preprocessing, Gradient Boosting classifier, and evaluation utilities.
- Persisted model artefacts (trained model, metrics, feature importances) for reproducible inference.
- Streamlit application with:
  - Performance dashboard (accuracy, classification report, confusion matrix, feature importance).
  - Dataset explorer with filtering and feature distribution visualisations.
  - Manual prediction form and batch CSV upload for new candidate classification.

## Getting started

1. **Install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Train the model** (downloads the latest KOI data and saves artefacts under `models/`):

   ```bash
   python -m src.train
   ```

   Optional flags:

   - `--refresh-data` — force re-download of the KOI dataset.
   - `--test-size` — change the validation split proportion (default `0.2`).

3. **Launch the Streamlit interface**:

   ```bash
   streamlit run app/streamlit_app.py
   ```

   The dashboard will be available at <http://localhost:8501> by default.

## Repository layout

```
.
├── app/                   # Streamlit front-end
│   └── streamlit_app.py
├── data/                  # Cached KOI dataset (downloaded automatically)
├── models/                # Trained model and metrics outputs
├── requirements.txt       # Python dependencies
└── src/
    ├── __init__.py
    ├── data.py            # Data download and loading helpers
    ├── model.py           # Model construction and persistence utilities
    └── train.py           # Training entry point
```

## Data source

The project queries the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) KOI catalogue via the legacy [`nstedAPI` endpoint](https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html#nstead) to retrieve a compact subset of informative features alongside the `koi_disposition` label. Downloads remain lightweight while preserving the key science context.

## Extending the project

- Experiment with alternative models (e.g., XGBoost, Random Forest) by modifying `src/model.py`.
- Incorporate additional features from the KOI table or other missions (K2, TESS) by updating `src/data.py`.
- Log training runs with MLflow or Weights & Biases for experiment tracking.
- Deploy the Streamlit app (e.g., Streamlit Cloud, Hugging Face Spaces) for easy sharing with collaborators.
