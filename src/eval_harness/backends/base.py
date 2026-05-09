from abc import ABC, abstractmethod

from eval_harness.schemas.responses import ModelResponse


class ModelBackend(ABC):
    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> ModelResponse:
        pass
