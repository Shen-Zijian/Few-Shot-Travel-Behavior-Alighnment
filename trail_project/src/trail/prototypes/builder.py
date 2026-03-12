"""
Behavior prototype builder.

Clusters TCS 2011 TP24 records into demographic behavior prototypes.
Each prototype captures a representative traveler subgroup with:
  - centroid feature vector
  - demographic summary statistics
  - historical mode share on the shared 9-class mechanised mode hierarchy
  - semantic label (rule-based)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from trail.features.encoder import TabularEncoder, DEFAULT_RETRIEVAL_FEATURES
from trail.prototypes.clustering import KMeansClustering
from trail.data.schema import (
    AGE_LABELS, SEX_LABELS, EMPLOYMENT_LABELS, MODE_LABELS,
    PURPOSE_LABELS, INCOME_LABELS, CAR_LABELS
)
from trail.utils.config import load_model_config, PROJECT_ROOT
from trail.utils.io import save_pickle, load_pickle
from trail.utils.logging import get_logger

logger = get_logger(__name__)


def _rule_based_label(stats: dict) -> str:
    """Generate a human-readable label from prototype demographics."""
    age = AGE_LABELS.get(int(round(stats.get("age_group", 3))), "Middle-aged")
    emp = EMPLOYMENT_LABELS.get(int(round(stats.get("employment_status", 1))), "Employed")
    car = CAR_LABELS.get(int(round(stats.get("car_availability", 0))), "No car")
    return f"{age} {emp} ({car})"


class BehaviorPrototypeMemory:
    """
    Collection of behavior prototypes built from TCS 2011.

    Each prototype is a dict with:
      - prototype_id: int
      - centroid: np.ndarray (in feature space)
      - n_samples: int
      - demographic_summary: dict {feature: mean_value}
      - mode_share: dict {mode_code: share}
      - purpose_share: dict {purpose_code: share}
      - semantic_label: str
    """

    def __init__(
        self,
        n_clusters: int = 10,
        feature_cols: Optional[list[str]] = None,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.feature_cols = feature_cols or DEFAULT_RETRIEVAL_FEATURES
        self.encoder = TabularEncoder(feature_cols=self.feature_cols, mode="retrieval")
        self.clusterer = KMeansClustering(
            n_clusters=n_clusters, random_state=random_state
        )
        self.prototypes: list[dict] = []
        self._fitted = False

    def build(self, df_2011: pd.DataFrame) -> "BehaviorPrototypeMemory":
        """
        Build prototypes from the 2011 harmonized dataset.
        """
        logger.info(f"Building {self.n_clusters} prototypes from {len(df_2011):,} 2011 records ...")

        # Encode features
        X = self.encoder.fit_transform(df_2011)

        # Cluster
        self.clusterer.fit(X)
        cluster_labels = self.clusterer.labels_

        # Build prototype summaries
        df = df_2011.copy()
        df["_cluster"] = cluster_labels

        self.prototypes = []
        for k in range(self.n_clusters):
            mask = df["_cluster"] == k
            sub = df[mask]
            n = len(sub)

            # Demographic summary (mean of encoded features)
            demo = {}
            for col in self.feature_cols:
                if col in sub.columns:
                    demo[col] = float(sub[col].fillna(-1).mean())

            # Mode share
            mode_counts = sub["main_mode"].value_counts()
            mode_share = {m: float(mode_counts.get(m, 0)) / n for m in sorted(MODE_LABELS.keys())}

            # Purpose share
            pur_counts = sub["trip_purpose"].value_counts()
            pur_share = {p: float(pur_counts.get(p, 0)) / n for p in sorted(PURPOSE_LABELS.keys())}

            label = _rule_based_label(demo)

            prototype = {
                "prototype_id": k,
                "centroid": self.clusterer.cluster_centers_[k],
                "n_samples": n,
                "demographic_summary": demo,
                "mode_share": mode_share,
                "purpose_share": pur_share,
                "semantic_label": label,
            }
            self.prototypes.append(prototype)

        self._fitted = True
        logger.info(
            f"Built {len(self.prototypes)} prototypes. "
            f"Sizes: {sorted([p['n_samples'] for p in self.prototypes], reverse=True)}"
        )
        return self

    def assign(self, df: pd.DataFrame) -> np.ndarray:
        """Assign new records to their nearest prototype. Returns cluster IDs."""
        X = self.encoder.transform(df)
        return self.clusterer.predict(X)

    def get_prototype(self, prototype_id: int) -> Optional[dict]:
        """Retrieve a prototype by ID."""
        for p in self.prototypes:
            if p["prototype_id"] == prototype_id:
                return p
        return None

    def get_prototype_summary(self, prototype_id: int) -> str:
        """Return a human-readable summary of a prototype for LLM prompts."""
        p = self.get_prototype(prototype_id)
        if p is None:
            return "Unknown prototype"

        lines = [
            f"Behavioral Group {prototype_id}: {p['semantic_label']}",
            f"  Size: {p['n_samples']:,} historical travelers",
            f"  Mode share (2011): " + ", ".join([
                f"{MODE_LABELS.get(m, m)}={v:.1%}"
                for m, v in p["mode_share"].items() if v > 0
            ]),
        ]
        return "\n".join(lines)

    def save(self, path: Optional[Path] = None) -> None:
        path = path or PROJECT_ROOT / "data/processed/prototype_cache/prototypes.pkl"
        save_pickle(self, path)
        logger.info(f"Prototypes saved to {path}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "BehaviorPrototypeMemory":
        path = path or PROJECT_ROOT / "data/processed/prototype_cache/prototypes.pkl"
        obj = load_pickle(path)
        logger.info(f"Prototypes loaded from {path}")
        return obj
