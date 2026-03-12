"""
LLM response parser: extract structured predictions from JSON output.

Enforces:
  - predicted_mode: int in valid mode set
  - confidence: float in [0, 1]
  - reasoning_brief: str
"""

import json
import re
from typing import Any, Optional

from trail.data.schema import MODE_LABELS
from trail.utils.logging import get_logger

logger = get_logger(__name__)

VALID_MODES = set(MODE_LABELS.keys())


def parse_mode_prediction(raw_response: str) -> dict[str, Any]:
    """
    Parse a raw LLM response string into a structured mode prediction.

    Expected JSON:
      {"predicted_mode": <int>, "confidence": <float>, "reasoning_brief": "<str>"}

    Returns:
      dict with keys: predicted_mode (int), confidence (float), reasoning_brief (str)
      On parse failure, returns {"predicted_mode": -1, "confidence": 0.0, "reasoning_brief": "parse_error"}
    """
    if not raw_response or not raw_response.strip():
        return _error_result("empty_response")

    # Try to parse JSON
    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError:
        # Try to extract JSON substring
        data = _extract_json(raw_response)
        if data is None:
            return _error_result("json_decode_error")

    # Extract predicted_mode
    raw_mode = data.get("predicted_mode", data.get("mode", data.get("main_mode", -1)))
    try:
        mode = int(raw_mode)
    except (TypeError, ValueError):
        # Try to extract from string like "1 (Public Transport)"
        mode = _extract_mode_from_string(str(raw_mode))

    if mode not in VALID_MODES:
        logger.warning(f"Invalid mode predicted: {mode}. Raw: {raw_mode}")
        mode = _fallback_mode(data)

    # Extract confidence
    raw_conf = data.get("confidence", data.get("probability", 0.5))
    try:
        confidence = float(raw_conf)
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5

    # Extract reasoning
    reasoning = str(data.get("reasoning_brief", data.get("reasoning", data.get("reason", ""))))

    return {
        "predicted_mode": mode,
        "confidence": confidence,
        "reasoning_brief": reasoning[:500],  # truncate long reasoning
    }


def _extract_json(text: str) -> Optional[dict]:
    """Try to extract a JSON object from a text that may contain extra content."""
    # Find first { ... } block
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _extract_mode_from_string(s: str) -> int:
    """Extract mode integer from strings like '1 (Public Transport)' or 'mode 3'."""
    m = re.search(r"\b([1-9])\b", s)
    if m:
        return int(m.group(1))
    return -1


def _fallback_mode(data: dict) -> int:
    """If mode is invalid, try common fallback keys."""
    for key in ["mode_code", "transport_mode", "predicted"]:
        if key in data:
            try:
                return int(data[key])
            except (TypeError, ValueError):
                pass
    return 1  # Default to PT as most common mode


def _error_result(reason: str) -> dict[str, Any]:
    return {"predicted_mode": -1, "confidence": 0.0, "reasoning_brief": reason}
