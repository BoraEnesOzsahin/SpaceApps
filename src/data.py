"""Data loading utilities for the NASA Kepler KOI dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)

KOI_FEATURE_COLUMNS: Tuple[str, ...] = (
    "koi_period",  # orbital period [days]
    "koi_time0bk",  # transit epoch [BJD-2454833]
    "koi_impact",  # impact parameter
    "koi_duration",  # transit duration [hrs]
    "koi_depth",  # transit depth [ppm]
    "koi_prad",  # planet radius [Earth radii]
    "koi_teq",  # equilibrium temperature [K]
    "koi_insol",  # insolation flux [Earth flux]
    "koi_slogg",  # log surface gravity of host star
    "koi_srad",  # stellar radius [Solar radii]
    "koi_steff",  # stellar effective temperature [K]
    "koi_kepmag",  # Kepler magnitude
)

TARGET_COLUMN = "koi_disposition"

_SELECT_COLUMNS = ",".join((TARGET_COLUMN, *KOI_FEATURE_COLUMNS))

NSTED_ENDPOINT = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_PATH = MODULE_ROOT / "data" / "kepler_koi.csv"


def _build_request_params() -> dict[str, str]:
    return {
        "table": "koi",
        "select": _SELECT_COLUMNS,
        "where": "koi_disposition is not null",
        "format": "csv",
    }


def download_koi_dataset(destination: Path | None = None, *, overwrite: bool = False) -> Path:
    """Download the KOI dataset from the NASA Exoplanet Archive.

    Parameters
    ----------
    destination:
        Where the downloaded CSV should be stored. Defaults to
        ``data/kepler_koi.csv`` within the repository.
    overwrite:
        If ``True`` the dataset will be re-downloaded even if the file already
        exists.

    Returns
    -------
    pathlib.Path
        Path to the cached CSV file.
    """

    destination = Path(destination or DEFAULT_CACHE_PATH)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        LOGGER.info("Dataset already cached at %s", destination)
        return destination

    params = _build_request_params()
    LOGGER.info("Downloading KOI dataset from %s", NSTED_ENDPOINT)
    response = requests.get(NSTED_ENDPOINT, params=params, timeout=120)
    response.raise_for_status()
    destination.write_text(response.text, encoding="utf-8")
    LOGGER.info("Saved KOI dataset to %s (%.2f MB)", destination, destination.stat().st_size / 1e6)
    return destination


def load_koi_dataframe(cache_path: Path | None = None, *, refresh: bool = False) -> pd.DataFrame:
    """Load the KOI dataset from the local cache, downloading it if necessary."""

    cache_path = download_koi_dataset(cache_path, overwrite=refresh)
    df = pd.read_csv(cache_path)
    LOGGER.debug("Loaded KOI dataframe with shape %s", df.shape)
    return df


def split_features_and_target(df: pd.DataFrame, *, features: Iterable[str] | None = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Split the dataframe into features and the target series.

    Parameters
    ----------
    df:
        A dataframe returned by :func:`load_koi_dataframe`.
    features:
        Iterable of feature column names to keep. Defaults to
        :data:`KOI_FEATURE_COLUMNS`.
    """

    feature_columns = list(features or KOI_FEATURE_COLUMNS)
    missing = set(feature_columns + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    clean_df = df.dropna(subset=[TARGET_COLUMN])
    X = clean_df[feature_columns]
    y = clean_df[TARGET_COLUMN]
    return X, y


__all__ = [
    "KOI_FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "download_koi_dataset",
    "load_koi_dataframe",
    "split_features_and_target",
]
