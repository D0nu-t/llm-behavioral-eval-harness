"""
orchestrator/orchestrator.py

EvalOrchestrator.run() is a generator yielding one result dict per probe item.
Both the CLI and FastAPI SSE endpoint consume the same loop.

Per-item execution order:
  1. Baseline pass (if item.baseline_messages is set)
  2. Pressured pass — used for scoring
  3. Score pressured response
  4. Compute full layerwise metrics (cosine_drift, norm_ratio, effective_rank)
  5. Log to MLflow. Yield flat result dict.
"""

import mlflow

from eval_harness.interpretability.drift_metrics import full_layerwise_metrics
from eval_harness.interpretability.live_capture import clear_activations


class EvalOrchestrator:
    def __init__(self, backend, probe, scorer, logger):
        self.backend = backend
        self.probe = probe
        self.scorer = scorer
        self.logger = logger

    def run(self):
        with mlflow.start_run():
            for item in self.probe:

                # --- baseline pass ---
                baseline_states = None
                if item.baseline_messages is not None:
                    clear_activations()
                    baseline_resp = self.backend.complete(item.baseline_messages)
                    baseline_states = baseline_resp.hidden_states

                # --- pressured pass ---
                clear_activations()
                response = self.backend.complete(item.messages)
                pressured_states = response.hidden_states

                # --- scoring ---
                scored = self.scorer.score(item, response)
                self.logger.log_result(item, response, scored)

                # --- full metrics ---
                metrics = None
                if (
                    baseline_states is not None
                    and pressured_states is not None
                    and len(baseline_states) == len(pressured_states)
                ):
                    metrics = full_layerwise_metrics(baseline_states, pressured_states)
                    self.logger.log_metrics(item, metrics)

                def _round(vals):
                    return [round(v, 6) for v in vals] if vals else None

                yield {
                    "item_id": scored.item_id,
                    "model_name": self.backend.model_name,
                    "probe_type": item.probe_type,
                    "score": scored.score,
                    "passed": scored.passed,
                    "reasoning": scored.reasoning,
                    "response_text": response.text,
                    "latency_ms": round(response.latency_ms, 2),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "drift_per_layer":          _round(metrics["cosine_drift"])    if metrics else None,
                    "norm_ratio_per_layer":     _round(metrics["norm_ratio"])      if metrics else None,
                    "effective_rank_per_layer": _round(metrics["effective_rank"])  if metrics else None,
                }
