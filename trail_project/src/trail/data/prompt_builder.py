"""
Builds structured prompt inputs from unified trip records.

Converts a UnifiedTripRecord (or DataFrame row) into a structured dict
that can be serialized into natural language for LLM consumption.
"""

from typing import Any

import pandas as pd

from trail.data.schema import (
    AGE_LABELS, SEX_LABELS, EMPLOYMENT_LABELS, INCOME_LABELS,
    PERIOD_LABELS, CAR_LABELS, MODE_LABELS, PURPOSE_LABELS,
)


def record_to_context(row: pd.Series | dict) -> dict[str, Any]:
    """
    Convert a single harmonized trip record to a structured context dict.

    This dict is used by llm/templates.py to fill Jinja2 prompts.
    """
    if isinstance(row, pd.Series):
        row = row.to_dict()

    def label(mapping: dict, key: Any, default: str = "Unknown") -> str:
        try:
            return mapping.get(int(key), default)
        except (TypeError, ValueError):
            return default

    journey_min = row.get("journey_time")
    journey_str = f"{int(journey_min)} minutes" if pd.notna(journey_min) and journey_min > 0 else "unknown"

    context = {
        "case_id": row.get("case_id", ""),
        "survey_year": row.get("survey_year", ""),
        "person": {
            "age_group": label(AGE_LABELS, row.get("age_group")),
            "sex": label(SEX_LABELS, row.get("sex")),
            "employment_status": label(EMPLOYMENT_LABELS, row.get("employment_status")),
            "income_group": label(INCOME_LABELS, row.get("income_group")),
        },
        "household": {
            "hh_size": int(row.get("hh_size", -1)) if pd.notna(row.get("hh_size")) else "unknown",
            "car_availability": label(CAR_LABELS, row.get("car_availability")),
        },
        "trip": {
            "trip_purpose": label(PURPOSE_LABELS, row.get("trip_purpose")),
            "departure_period": label(PERIOD_LABELS, row.get("departure_period")),
            "journey_time": journey_str,
            "origin_zone": str(row.get("origin_zone", "")),
            "destination_zone": str(row.get("destination_zone", "")),
        },
    }
    return context


def example_to_demonstration(row: pd.Series | dict) -> dict[str, Any]:
    """
    Convert a labeled example to a demonstration dict (context + label).
    Used for few-shot examples in LLM prompts.
    """
    context = record_to_context(row)
    if isinstance(row, pd.Series):
        row = row.to_dict()

    mode_val = row.get("main_mode", -1)
    mode_label = MODE_LABELS.get(int(mode_val), "Unknown") if pd.notna(mode_val) else "Unknown"

    return {
        **context,
        "label": {
            "main_mode": int(mode_val) if pd.notna(mode_val) else -1,
            "main_mode_label": mode_label,
        },
    }


def batch_to_contexts(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame of records to a list of context dicts."""
    return [record_to_context(row) for _, row in df.iterrows()]


def batch_to_demonstrations(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame of labeled records to a list of demonstration dicts."""
    return [example_to_demonstration(row) for _, row in df.iterrows()]
