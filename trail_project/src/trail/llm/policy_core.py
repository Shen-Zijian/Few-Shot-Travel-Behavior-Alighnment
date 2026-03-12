"""
TRAIL LLM Policy Core.

Orchestrates:
  1. Retrieve evidence (prototypes + historical + few-shot)
  2. Build prompt from templates
  3. Call LLM
  4. Parse structured prediction
"""

from typing import Any, Optional

import pandas as pd

from trail.data.prompt_builder import record_to_context
from trail.llm.client import LLMClient
from trail.llm.parser import parse_mode_prediction
from trail.llm.templates import build_trail_messages
from trail.retrieval.retriever import TrailRetriever
from trail.utils.logging import get_logger

logger = get_logger(__name__)


class TrailPolicyCore:
    """
    Main TRAIL inference engine.

    For each query trip record:
      1. Build traveler context
      2. Retrieve historical + few-shot evidence
      3. Construct prompt
      4. Call LLM
      5. Parse and return prediction
    """

    def __init__(
        self,
        retriever: TrailRetriever,
        prototype_memory=None,
        client: Optional[LLMClient] = None,
        max_hist_examples: int = 3,
        max_update_examples: int = 4,
    ):
        self.retriever = retriever
        self.prototype_memory = prototype_memory
        self.client = client or LLMClient()
        self.max_hist_examples = max_hist_examples
        self.max_update_examples = max_update_examples

    def predict_one(self, row: pd.Series) -> dict[str, Any]:
        """
        Predict mode for a single trip record.

        Returns:
            dict with: predicted_mode, confidence, reasoning_brief, prototype_id
        """
        query_df = row.to_frame().T.reset_index(drop=True)

        # 1. Build context
        context = record_to_context(row)

        # 2. Retrieve evidence
        evidences = self.retriever.retrieve(query_df, prototype_memory=self.prototype_memory)
        evidence = evidences[0] if evidences else {}

        historical_examples = evidence.get("historical_examples", [])
        update_examples = evidence.get("update_examples", [])
        prototype_summary = evidence.get("prototype_summary", "")
        prototype_id = evidence.get("prototype_id", -1)

        # 3. Build prompt
        messages = build_trail_messages(
            query_context=context,
            prototype_summary=prototype_summary,
            historical_examples=historical_examples,
            update_examples=update_examples,
            max_hist=self.max_hist_examples,
            max_update=self.max_update_examples,
        )

        # 4. Call LLM
        response = self.client.chat(messages)

        # 5. Parse
        prediction = parse_mode_prediction(response)
        prediction["prototype_id"] = prototype_id
        prediction["n_hist_examples"] = len(historical_examples)
        prediction["n_update_examples"] = len(update_examples)

        return prediction

    def predict_batch(
        self,
        df: pd.DataFrame,
        progress: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Predict mode for all rows in a DataFrame.
        """
        results = []
        iterable = df.iterrows()

        if progress:
            try:
                from tqdm import tqdm
                iterable = tqdm(df.iterrows(), total=len(df), desc="TRAIL")
            except ImportError:
                pass

        for _, row in iterable:
            try:
                result = self.predict_one(row)
            except Exception as e:
                logger.warning(f"Prediction failed for {row.get('case_id', '?')}: {e}")
                result = {
                    "predicted_mode": -1,
                    "confidence": 0.0,
                    "reasoning_brief": f"error: {str(e)[:100]}",
                    "prototype_id": -1,
                }
            results.append(result)

        return results

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run batch prediction and return a DataFrame with columns:
          - predicted_mode, confidence, reasoning_brief, prototype_id
        """
        results = self.predict_batch(df)
        return pd.DataFrame(results, index=df.index)
