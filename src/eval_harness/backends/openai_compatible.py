import os
import time

from openai import OpenAI

from eval_harness.backends.base import ModelBackend
from eval_harness.schemas.responses import ModelResponse


class OpenAICompatibleBackend(ModelBackend):

    @property
    def model_name(self) -> str:
        return self._model_name

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str = os.environ.get("LM_API_KEY", ""),
    ):
        self._model_name = model_name

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def complete(
        self,
        messages,
        temperature=0.0,
        max_tokens=512,
    ):
        start = time.time()

        response = self.client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        latency_ms = (time.time() - start) * 1000

        return ModelResponse(
            text=response.choices[0].message.content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            latency_ms=latency_ms,
            raw=response.model_dump(),
        )
