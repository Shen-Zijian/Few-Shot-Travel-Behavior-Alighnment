"""
Prompt-only LLM baseline: direct mode choice prediction with no retrieval.

This is the 'vanilla LLM' baseline that uses only the traveler profile
and trip context to predict mode, without any historical examples.
"""

from typing import Any, Optional

import pandas as pd

from trail.data.prompt_builder import record_to_context
from trail.data.schema import MODE_LABELS
from trail.llm.client import LLMClient
from trail.llm.parser import parse_mode_prediction
from trail.utils.logging import get_logger

logger = get_logger(__name__)

PROMPT_ONLY_SYSTEM = """You are an expert in Hong Kong travel behavior analysis.
Given a traveler's profile and trip characteristics, predict the most likely main transport mode.

Available modes: {mode_options}

You MUST respond in valid JSON with this exact structure:
{{"predicted_mode": <integer>, "confidence": <float 0-1>, "reasoning_brief": "<one sentence>"}}"""

PROMPT_ONLY_USER = """Traveler profile:
- Age group: {person[age_group]}
- Sex: {person[sex]}
- Employment status: {person[employment_status]}
- Income level: {person[income_group]}
- Car availability in household: {household[car_availability]}

Trip details:
- Purpose: {trip[trip_purpose]}
- Departure period: {trip[departure_period]}
- Journey time: {trip[journey_time]}

Predict the main transport mode for this trip."""


class PromptOnlyBaseline:
    """
    Zero-shot LLM baseline: predicts mode choice from profile only.
    No retrieved examples, no historical prior.
    """

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or LLMClient()
        self._mode_options = self._build_mode_options_str()

    def _build_mode_options_str(self) -> str:
        return ", ".join([f"{k}={v}" for k, v in MODE_LABELS.items()])

    def predict_one(self, row: pd.Series | dict) -> dict[str, Any]:
        """Predict mode for a single trip record. Returns structured prediction."""
        context = record_to_context(row)
        system_msg = PROMPT_ONLY_SYSTEM.format(mode_options=self._mode_options)
        user_msg = PROMPT_ONLY_USER.format(**context)

        response = self.client.chat(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
        )
        return parse_mode_prediction(response)

    def predict_batch(
        self,
        df: pd.DataFrame,
        progress: bool = True,
    ) -> list[dict[str, Any]]:
        """Predict mode for a DataFrame of trip records."""
        results = []
        iterable = df.iterrows()
        if progress:
            try:
                from tqdm import tqdm
                iterable = tqdm(df.iterrows(), total=len(df), desc="PromptOnly")
            except ImportError:
                pass

        for _, row in iterable:
            try:
                result = self.predict_one(row)
            except Exception as e:
                logger.warning(f"Prediction failed for case {row.get('case_id', '?')}: {e}")
                result = {"predicted_mode": -1, "confidence": 0.0, "reasoning_brief": "error"}
            results.append(result)

        return results

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run batch prediction and return a DataFrame with prediction columns:
          - predicted_mode: int
          - confidence: float
          - reasoning_brief: str
        """
        results = self.predict_batch(df)
        pred_df = pd.DataFrame(results, index=df.index)
        return pred_df
