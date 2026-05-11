from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ModelResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    raw: dict[str, Any]
    # Per-layer final-token tensors. None for OpenAICompatibleBackend.
    hidden_states: list[Any] | None = field(default=None)
    # Single-layer vector for NLA verbalization.
    # Populated when HFTransformerBackend is initialised with nla_layer set.
    # None otherwise — NLAVerbalizer will skip gracefully.
    nla_activation: Any | None = field(default=None)
