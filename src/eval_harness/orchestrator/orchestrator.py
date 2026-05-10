"""
orchestrator/orchestrator.py

EvalOrchestrator.run() is a generator yielding one result dict per probe item.
Both the CLI and FastAPI SSE endpoint consume the same loop — the CLI iterates
and prints, the server iterates and streams SSE events.

Per-item execution order:
  1. Baseline pass (if item.baseline_messages is set) — captures neutral activations.
  2. Pressured pass — captures adversarial activations, used for scoring.
  3. Score pressured response.
  4. Compute layerwise cosine drift between baseline and pressured states (if both present).
  5. Log to MLflow. Yield result dict.
"""

import mlflow

from eval_harness.interpretability.drift_metrics import layerwise_drift
from eval_harness.interpretability.live_capture import clear_activations


class EvalOrchestrator:
    def __init__(self, backend, probe, scorer, logger):
        self.backend = backend
        self.probe = probe
        self.scorer = scorer
        self.logger = logger

    def run(self):
        """
        Generator. Yields one dict per probe item. Caller must fully consume
        the generator to ensure the MLflow run context closes correctly.
        """
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

                # --- drift ---
                drift_per_layer = None
                if (
                    baseline_states is not None
                    and pressured_states is not None
                    and len(baseline_states) == len(pressured_states)
                ):
                    drift_per_layer = layerwise_drift(baseline_states, pressured_states)
                    self.logger.log_drift(item, drift_per_layer)

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
    "drift_per_layer": (
        [round(d, 6) for d in drift_per_layer]
        if drift_per_layer is not None else None
    ),
}
