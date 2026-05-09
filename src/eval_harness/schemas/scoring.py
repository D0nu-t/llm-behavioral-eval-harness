from dataclasses import dataclass
from typing import Any


@dataclass
class ScoredResult:
    item_id: str
    score: float
    passed: bool
    reasoning: str
    metadata: dict[str, Any]
