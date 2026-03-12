"""
TRAIL retriever: unified interface for evidence retrieval.

For a query case, retrieves:
  - historical_examples: top-k from 2011 (demographic prior)
  - update_examples: top-k from 2022 few-shot (current behavior)
  - prototype_summary: the query's assigned prototype description
"""

from typing import Optional

import numpy as np
import pandas as pd

from trail.retrieval.memory_bank import (
    HistoricalPriorMemory,
    FewShotUpdateMemory,
    CalibrationErrorMemory,
)
from trail.retrieval.scorer import compute_retrieval_scores
from trail.features.encoder import TabularEncoder
from trail.data.prompt_builder import example_to_demonstration
from trail.utils.config import load_model_config
from trail.utils.logging import get_logger

logger = get_logger(__name__)


class TrailRetriever:
    """
    Retrieves relevant evidence for a query traveler profile.

    Uses cosine similarity on tabular features with temporal shift weighting.
    """

    def __init__(
        self,
        historical_memory: Optional[HistoricalPriorMemory] = None,
        fewshot_memory: Optional[FewShotUpdateMemory] = None,
        error_memory: Optional[CalibrationErrorMemory] = None,
        top_k: int = 8,
        lambda_sim: float = 0.5,
        lambda_shift: float = 0.3,
        lambda_priority: float = 0.2,
    ):
        self.historical_memory = historical_memory
        self.fewshot_memory = fewshot_memory
        self.error_memory = error_memory
        self.top_k = top_k
        self.lambda_sim = lambda_sim
        self.lambda_shift = lambda_shift
        self.lambda_priority = lambda_priority

    @classmethod
    def from_config(
        cls,
        historical_memory: HistoricalPriorMemory,
        fewshot_memory: Optional[FewShotUpdateMemory] = None,
        config_name: str = "retriever",
    ) -> "TrailRetriever":
        cfg = load_model_config(config_name)
        return cls(
            historical_memory=historical_memory,
            fewshot_memory=fewshot_memory,
            top_k=cfg.get("top_k", 8),
            lambda_sim=cfg.get("lambda_similarity", 0.5),
            lambda_shift=cfg.get("lambda_shift", 0.3),
            lambda_priority=cfg.get("lambda_priority", 0.2),
        )

    def retrieve(
        self,
        query_df: pd.DataFrame,
        prototype_memory=None,
    ) -> list[dict]:
        """
        Retrieve evidence for a batch of query records.

        Args:
            query_df: DataFrame of query trip records
            prototype_memory: BehaviorPrototypeMemory (optional)

        Returns:
            List of evidence dicts, one per query row.
        """
        encoder = self._get_encoder()
        query_X = encoder.transform(query_df)

        results = []
        for i in range(len(query_df)):
            q_X = query_X[i:i+1]
            evidence = {}

            # Historical examples from 2011
            if self.historical_memory and self.historical_memory.size > 0:
                hist_examples = self._retrieve_from_memory(
                    q_X, self.historical_memory, n=self.top_k // 2
                )
                evidence["historical_examples"] = hist_examples

            # Few-shot update examples from 2022
            if self.fewshot_memory and self.fewshot_memory.size > 0:
                fewshot_examples = self._retrieve_from_memory(
                    q_X, self.fewshot_memory, n=self.top_k
                )
                evidence["update_examples"] = fewshot_examples
            else:
                evidence["update_examples"] = []

            evidence["historical_examples"] = evidence.get("historical_examples", [])

            # Prototype summary
            if prototype_memory is not None:
                query_row = query_df.iloc[i]
                proto_id = prototype_memory.assign(query_df.iloc[i:i+1])[0]
                evidence["prototype_summary"] = prototype_memory.get_prototype_summary(proto_id)
                evidence["prototype_id"] = int(proto_id)
            else:
                evidence["prototype_summary"] = ""
                evidence["prototype_id"] = -1

            results.append(evidence)

        return results

    def _retrieve_from_memory(
        self, query_X: np.ndarray, memory, n: int
    ) -> list[dict]:
        """Retrieve top-n examples from a memory bank."""
        scores = compute_retrieval_scores(
            query_X,
            memory.X,
            memory.df,
            lambda_sim=self.lambda_sim,
            lambda_shift=self.lambda_shift,
            lambda_priority=self.lambda_priority,
        )  # (1, n_corpus)
        scores_1d = scores[0]
        top_n = min(n, len(scores_1d))
        top_indices = np.argsort(scores_1d)[::-1][:top_n]

        examples = []
        for idx in top_indices:
            row = memory.df.iloc[idx]
            demo = example_to_demonstration(row)
            demo["_score"] = float(scores_1d[idx])
            examples.append(demo)

        return examples

    def _get_encoder(self) -> TabularEncoder:
        """Get the encoder from the first available memory bank."""
        if self.historical_memory and self.historical_memory.encoder:
            return self.historical_memory.encoder
        if self.fewshot_memory and self.fewshot_memory.encoder:
            return self.fewshot_memory.encoder
        raise RuntimeError("No fitted encoder found in memory banks. Build memories first.")
