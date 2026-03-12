"""
Multinomial Logit (MNL) baseline using statsmodels.

We use statsmodels MNLogit (not biogeme) for simplicity and reliability.
Biogeme is better for utility-specification models; statsmodels MNLogit
is sufficient as a benchmark baseline.
"""

from typing import Optional

import numpy as np
import pandas as pd

from trail.utils.logging import get_logger

logger = get_logger(__name__)

# Avoid circular import
_MNL_FEATURES = [
    "car_availability",
    "age_group", "sex", "employment_status",
    "trip_purpose", "departure_period",
    "origin_zone_int", "dest_zone_int",
    "journey_time",
]


class MNLBaseline:
    """
    Multinomial Logit using statsmodels MNLogit.

    Fits a softmax regression on tabular features.
    """

    def __init__(self, feature_cols: Optional[list[str]] = None, max_iter: int = 500):
        self.feature_cols = feature_cols or _MNL_FEATURES
        self.max_iter = max_iter
        self.model = None
        self.result = None
        self.classes_: Optional[np.ndarray] = None
        self._feature_medians: dict[str, float] = {}

    def _prepare_X(self, df: pd.DataFrame, is_fit: bool = False) -> np.ndarray:
        import statsmodels.api as sm
        df = df.copy()
        if "origin_zone_int" not in df.columns and "origin_zone" in df.columns:
            df["origin_zone_int"] = pd.to_numeric(df["origin_zone"], errors="coerce").fillna(0).astype(int)
        if "dest_zone_int" not in df.columns and "destination_zone" in df.columns:
            df["dest_zone_int"] = pd.to_numeric(df["destination_zone"], errors="coerce").fillna(0).astype(int)
        cols = [c for c in self.feature_cols if c in df.columns]
        X = df[cols].copy()

        for col in cols:
            if is_fit:
                median = X[col].median()
                self._feature_medians[col] = median if not np.isnan(median) else 0.0
            X[col] = X[col].fillna(self._feature_medians.get(col, 0.0)).astype(float).clip(lower=0)

        # Add missing columns as zeros
        for col in self.feature_cols:
            if col not in cols:
                X[col] = 0.0

        X = X[self.feature_cols].values.astype(float)
        # Add intercept
        return sm.add_constant(X, has_constant='add')

    def fit(self, df: pd.DataFrame, target_col: str = "main_mode") -> "MNLBaseline":
        import statsmodels.api as sm

        y = df[target_col].values
        self.classes_ = np.sort(np.unique(y))
        X = self._prepare_X(df, is_fit=True)

        logger.info(f"Fitting MNL on {len(df):,} rows, {X.shape[1]} features, {len(self.classes_)} classes ...")
        self.model = sm.MNLogit(y, X)
        try:
            self.result = self.model.fit(
                method="bfgs", maxiter=self.max_iter, disp=False
            )
            logger.info("MNL fitting complete")
        except Exception as e:
            logger.warning(f"BFGS failed ({e}), retrying with Newton ...")
            self.result = self.model.fit(maxiter=self.max_iter, disp=False)

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self._prepare_X(df, is_fit=False)
        proba = self.result.predict(X)  # shape (N, n_classes)
        return np.array(proba)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(df)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]

    def score(self, df: pd.DataFrame, target_col: str = "main_mode") -> float:
        y_true = df[target_col].values
        y_pred = self.predict(df)
        return float(np.mean(y_true == y_pred))
