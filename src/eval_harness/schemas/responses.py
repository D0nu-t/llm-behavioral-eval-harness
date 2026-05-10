from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    raw: dict[str, Any]
    # Per-layer hidden state vectors (final token), populated by HFTransformerBackend.
    # None when using OpenAICompatibleBackend (no activation access via API).
    hidden_states: list[Any] | None = field(default=None)
