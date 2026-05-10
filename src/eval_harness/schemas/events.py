from pydantic import BaseModel
from typing import Optional


class LayerMetrics(BaseModel):
    cosine_drift: list[float]
    norm_ratio: list[float]
    effective_rank: list[float]


class EvalResultEvent(BaseModel):
    item_id: str
    score: float
    passed: bool
    reasoning: str
    response_text: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    layer_metrics: Optional[LayerMetrics]


class RunMetadata(BaseModel):
    model_name: str
    probe_type: str


class SSEEvent(BaseModel):
    run_metadata: RunMetadata
    result: EvalResultEvent
