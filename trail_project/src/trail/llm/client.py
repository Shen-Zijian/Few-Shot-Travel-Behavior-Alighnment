"""
OpenAI API client wrapper for TRAIL.

Handles:
  - Chat completions with JSON mode
  - Retry logic with exponential backoff
  - Token logging
  - Model name from config
"""

import os
import time
from typing import Any, Optional

from trail.utils.config import load_model_config
from trail.utils.logging import get_logger

logger = get_logger(__name__)


class LLMClient:
    """
    OpenAI chat completion client.
    Reads OPENAI_API_KEY from environment.
    Model and parameters from configs/model/llm_base.yaml.
    """

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_model_config("llm_base")

        self.model = config.get("model", "gpt-4o")
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 512)
        self.timeout = config.get("timeout", 60)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 5)
        self.output_format = config.get("output_format", "json_object")
        self.log_tokens = config.get("log_tokens", True)

        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._call_count = 0

        self._client = None  # Lazy init

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise EnvironmentError(
                        "OPENAI_API_KEY not set. Create a .env file or export the variable."
                    )
                self._client = OpenAI(base_url="https://api.poe.com/v1", api_key=api_key, timeout=self.timeout)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client

    def chat(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a chat completion request and return the response content.

        Args:
            messages: list of {"role": ..., "content": ...} dicts
            model, temperature, max_tokens: overrides for this call

        Returns:
            Response content string (should be JSON when output_format=json_object)
        """
        client = self._get_client()
        model = model or self.model
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens

        for attempt in range(self.max_retries):
            try:
                kwargs: dict[str, Any] = dict(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if self.output_format == "json_object":
                    kwargs["response_format"] = {"type": "json_object"}

                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content

                if self.log_tokens and response.usage:
                    self._total_prompt_tokens += response.usage.prompt_tokens
                    self._total_completion_tokens += response.usage.completion_tokens
                    self._call_count += 1

                return content or ""

            except Exception as e:
                logger.warning(f"LLM call attempt {attempt+1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    raise

    @property
    def token_usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
            "call_count": self._call_count,
        }

    def log_usage(self) -> None:
        usage = self.token_usage
        logger.info(
            f"LLM usage: {usage['call_count']} calls, "
            f"{usage['prompt_tokens']:,} prompt tokens, "
            f"{usage['completion_tokens']:,} completion tokens"
        )
