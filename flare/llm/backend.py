# BOUNDARY: flare/llm/backend.py → imports from stdlib and openai only; no flare.* imports
from __future__ import annotations

import os
from abc import ABC, abstractmethod

from openai import OpenAI


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, system: str, user: str) -> dict:
        # returns: {"text": str, "logprobs": [{"token": str, "logprob": float}]}
        raise NotImplementedError

    @property
    @abstractmethod
    def backend_id(self) -> str:
        raise NotImplementedError


class OpenAIBackend(LLMBackend):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self._client = OpenAI(api_key=api_key, max_retries=3, timeout=30.0)
        self._model = model

    @property
    def backend_id(self) -> str:
        return f"openai:{self._model}"

    def generate(self, system: str, user: str) -> dict:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            logprobs=True,
            top_logprobs=1,
        )
        text = response.choices[0].message.content or ""
        raw_logprobs = response.choices[0].logprobs
        if raw_logprobs is None or raw_logprobs.content is None:
            logprobs: list[dict] = []
        else:
            logprobs = [
                {"token": entry.token, "logprob": entry.logprob}
                for entry in raw_logprobs.content
            ]
        return {"text": text, "logprobs": logprobs}
