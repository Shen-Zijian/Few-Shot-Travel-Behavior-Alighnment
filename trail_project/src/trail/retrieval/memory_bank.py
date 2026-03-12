"""
Memory bank for TRAIL retrieval system.

Three types of memory:
  1. HistoricalPriorMemory: TCS 2011 records (demographic prior)
  2. FewShotUpdateMemory: TCS 2022 few-shot labeled examples
  3. CalibrationErrorMemory: subgroups with high calibration error (Phase D+)
"""

from typing import Optional

import numpy as np
import pandas as pd

from trail.features.encoder import TabularEncoder, encode_for_retrieval
from trail.utils.logging import get_logger

logger = get_logger(__name__)


class BaseMemory:
    """Base class for a retrieval memory bank."""

    def __init__(self, name: str, feature_cols: Optional[list[str]] = None):
        self.name = name
        self.feature_cols = feature_cols
        self.df: Optional[pd.DataFrame] = None
        self.X: Optional[np.ndarray] = None
        self.encoder: Optional[TabularEncoder] = None

    def build(self, df: pd.DataFrame, encoder: Optional[TabularEncoder] = None) -> "BaseMemory":
        """Index the DataFrame for retrieval."""
        self.df = df.reset_index(drop=True)
        if encoder is not None:
            self.encoder = encoder
            self.X = encoder.transform(df)
        else:
            self.X, self.encoder = encode_for_retrieval(df, feature_cols=self.feature_cols)
        logger.info(f"[{self.name}] Indexed {len(self.df):,} records, feature_dim={self.X.shape[1]}")
        return self

    def update(self, df_new: pd.DataFrame) -> "BaseMemory":
        """Append new records to the memory bank."""
        if self.df is None or self.X is None:
            return self.build(df_new)

        X_new = self.encoder.transform(df_new)
        self.df = pd.concat([self.df, df_new.reset_index(drop=True)], ignore_index=True)
        self.X = np.vstack([self.X, X_new])
        logger.info(f"[{self.name}] Updated to {len(self.df):,} records")
        return self

    @property
    def size(self) -> int:
        return 0 if self.df is None else len(self.df)


class HistoricalPriorMemory(BaseMemory):
    """TCS 2011 records providing the demographic behavioral prior."""

    def __init__(self, feature_cols: Optional[list[str]] = None):
        super().__init__("HistoricalPrior", feature_cols)


class FewShotUpdateMemory(BaseMemory):
    """TCS 2022 few-shot labeled examples with current-year mode specificity."""

    def __init__(self, feature_cols: Optional[list[str]] = None):
        super().__init__("FewShotUpdate", feature_cols)


class CalibrationErrorMemory(BaseMemory):
    """
    Records from subgroups with high calibration error.
    Populated during the active survey loop (Phase D).
    """

    def __init__(self, feature_cols: Optional[list[str]] = None):
        super().__init__("CalibrationError", feature_cols)
        self.subgroup_errors: dict = {}

    def update_errors(self, subgroup_errors: dict) -> None:
        """Update tracked subgroup calibration errors."""
        self.subgroup_errors.update(subgroup_errors)
