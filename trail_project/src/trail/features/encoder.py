"""
Tabular feature encoder for TRAIL.

Produces normalized numeric feature vectors from unified trip records,
used for:
  - prototype clustering (K-means on 2011 data)
  - retrieval similarity (cosine similarity for query vs corpus)
  - baseline model inputs (MNL, XGBoost)

No external embedding models are used.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from trail.data.schema import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, FEATURE_COLUMNS
from trail.utils.logging import get_logger

logger = get_logger(__name__)

# Default feature columns (subset of FEATURE_COLUMNS used for similarity / clustering)
DEFAULT_RETRIEVAL_FEATURES = [
    "car_availability",
    "age_group",
    "employment_status",
    "trip_purpose",
    "origin_zone_int",
    "dest_zone_int",
    "sex",
    "departure_period",
]

DEFAULT_MODEL_FEATURES = [
    "car_availability",
    "age_group",
    "employment_status",
    "trip_purpose",
    "origin_zone_int",
    "dest_zone_int",
    "journey_time",
    "sex",
    "departure_period",
    # income_group excluded: zero variance in 2022 (all records = 0)
    # hh_size excluded: very low mutual information (0.007)
]


class TabularEncoder:
    """
    Fit a scaler on training data and produce normalized feature matrices.

    Two modes:
      - 'retrieval': uses DEFAULT_RETRIEVAL_FEATURES (categorical only, no journey_time gaps)
      - 'model': uses DEFAULT_MODEL_FEATURES with numeric imputation
    """

    def __init__(self, feature_cols: Optional[list[str]] = None, mode: str = "retrieval"):
        if feature_cols is not None:
            self.feature_cols = feature_cols
        elif mode == "retrieval":
            self.feature_cols = DEFAULT_RETRIEVAL_FEATURES
        else:
            self.feature_cols = DEFAULT_MODEL_FEATURES

        self.scaler = StandardScaler()
        self._fitted = False
        self._numeric_medians: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "TabularEncoder":
        X = self._prepare(df, is_fit=True)
        self.scaler.fit(X)
        self._fitted = True
        logger.info(f"TabularEncoder fitted on {len(df):,} rows, {len(self.feature_cols)} features")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform()")
        X = self._prepare(df, is_fit=False)
        return self.scaler.transform(X)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def _prepare(self, df: pd.DataFrame, is_fit: bool) -> np.ndarray:
        """Build raw (unscaled) feature matrix with imputation."""
        # Derive integer zone columns if raw string zone columns exist
        df = df.copy()
        if "origin_zone_int" not in df.columns and "origin_zone" in df.columns:
            df["origin_zone_int"] = pd.to_numeric(df["origin_zone"], errors="coerce").fillna(0).astype(int)
        if "dest_zone_int" not in df.columns and "destination_zone" in df.columns:
            df["dest_zone_int"] = pd.to_numeric(df["destination_zone"], errors="coerce").fillna(0).astype(int)

        cols_present = [c for c in self.feature_cols if c in df.columns]
        X = df[cols_present].copy()

        for col in cols_present:
            if col in CATEGORICAL_COLUMNS:
                # Negative codes → 0 (unknown category)
                X[col] = X[col].astype(float).fillna(0.0).clip(lower=0)
            else:
                # Numeric: impute missing with median
                if is_fit:
                    median = X[col].median()
                    self._numeric_medians[col] = median if not np.isnan(median) else 0.0
                X[col] = X[col].astype(float).fillna(self._numeric_medians.get(col, 0.0))

        # Add zero columns for any missing features
        for col in self.feature_cols:
            if col not in cols_present:
                X[col] = 0.0

        return X[self.feature_cols].values.astype(float)

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)


def encode_for_retrieval(
    df: pd.DataFrame,
    encoder: Optional[TabularEncoder] = None,
    feature_cols: Optional[list[str]] = None,
) -> tuple[np.ndarray, TabularEncoder]:
    """
    Convenience function: fit encoder (if not provided) and return
    normalized feature matrix suitable for cosine similarity retrieval.

    Returns:
        (feature_matrix, fitted_encoder)
    """
    if encoder is None:
        encoder = TabularEncoder(feature_cols=feature_cols, mode="retrieval")
        X = encoder.fit_transform(df)
    else:
        X = encoder.transform(df)
    return X, encoder


def encode_for_model(
    df: pd.DataFrame,
    encoder: Optional[TabularEncoder] = None,
    feature_cols: Optional[list[str]] = None,
) -> tuple[np.ndarray, TabularEncoder]:
    """
    Convenience function: fit encoder and return feature matrix for ML baselines.
    Includes numeric features (journey_time, hh_size) with imputation.

    Returns:
        (feature_matrix, fitted_encoder)
    """
    if encoder is None:
        encoder = TabularEncoder(feature_cols=feature_cols, mode="model")
        X = encoder.fit_transform(df)
    else:
        X = encoder.transform(df)
    return X, encoder
