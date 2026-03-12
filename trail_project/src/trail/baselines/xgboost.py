"""
XGBoost baseline for mode choice prediction.

Strong tabular baseline for few-shot comparison.
"""

from typing import Optional

import numpy as np
import pandas as pd

from trail.utils.logging import get_logger

logger = get_logger(__name__)

_XGB_FEATURES = [
    "car_availability",
    "age_group", "sex", "employment_status",
    "trip_purpose", "departure_period",
    "origin_zone_int", "dest_zone_int",
    "journey_time",
]


class XGBoostBaseline:
    """XGBoost multi-class classifier baseline."""

    def __init__(
        self,
        feature_cols: Optional[list[str]] = None,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        seed: int = 42,
    ):
        self.feature_cols = feature_cols or _XGB_FEATURES
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.seed = seed
        self.model = None
        self.classes_: Optional[np.ndarray] = None
        self._label_map: dict[int, int] = {}
        self._inv_label_map: dict[int, int] = {}
        self._feature_medians: dict[str, float] = {}

    def _prepare_X(self, df: pd.DataFrame, is_fit: bool = False) -> np.ndarray:
        df = df.copy()
        if "origin_zone_int" not in df.columns and "origin_zone" in df.columns:
            df["origin_zone_int"] = pd.to_numeric(df["origin_zone"], errors="coerce").fillna(0).astype(int)
        if "dest_zone_int" not in df.columns and "destination_zone" in df.columns:
            df["dest_zone_int"] = pd.to_numeric(df["destination_zone"], errors="coerce").fillna(0).astype(int)
        cols = [c for c in self.feature_cols if c in df.columns]
        X = df[cols].copy()

        for col in cols:
            if is_fit:
                med = X[col].median()
                self._feature_medians[col] = med if not np.isnan(med) else 0.0
            X[col] = X[col].fillna(self._feature_medians.get(col, 0.0)).astype(float)

        for col in self.feature_cols:
            if col not in cols:
                X[col] = 0.0

        return X[self.feature_cols].values.astype(float)

    def fit(self, df: pd.DataFrame, target_col: str = "main_mode") -> "XGBoostBaseline":
        import xgboost as xgb

        y_raw = df[target_col].values
        self.classes_ = np.sort(np.unique(y_raw))

        # XGBoost requires 0-indexed labels
        self._label_map = {c: i for i, c in enumerate(self.classes_)}
        self._inv_label_map = {i: c for c, i in self._label_map.items()}
        y = np.array([self._label_map[c] for c in y_raw])

        X = self._prepare_X(df, is_fit=True)

        logger.info(
            f"Fitting XGBoost on {len(df):,} rows, {X.shape[1]} features, "
            f"{len(self.classes_)} classes ..."
        )
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.seed,
            use_label_encoder=False,
            eval_metric="mlogloss",
            verbosity=0,
        )
        self.model.fit(X, y)
        logger.info("XGBoost fitting complete")
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self._prepare_X(df, is_fit=False)
        return self.model.predict_proba(X)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(df)
        idx = np.argmax(proba, axis=1)
        return np.array([self._inv_label_map[i] for i in idx])

    def score(self, df: pd.DataFrame, target_col: str = "main_mode") -> float:
        y_true = df[target_col].values
        y_pred = self.predict(df)
        return float(np.mean(y_true == y_pred))
