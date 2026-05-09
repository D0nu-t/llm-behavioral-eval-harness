from abc import ABC, abstractmethod

from eval_harness.schemas.probes import ProbeItem
from eval_harness.schemas.responses import ModelResponse
from eval_harness.schemas.scoring import ScoredResult


class Scorer(ABC):
    @abstractmethod
    def score(
        self,
        item: ProbeItem,
        response: ModelResponse,
    ) -> ScoredResult:
        pass
