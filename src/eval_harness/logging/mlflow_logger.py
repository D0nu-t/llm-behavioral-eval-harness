import mlflow


class MLflowLogger:
    def __init__(self, experiment_name: str):
        mlflow.set_experiment(experiment_name)

    def log_result(self, item, response, scored):
        mlflow.log_metric(f"{item.item_id}_score", scored.score)
        mlflow.log_text(response.text, f"responses/{item.item_id}.txt")

    def log_drift(self, item, drift_per_layer: list[float]):
        """Log per-layer cosine drift and mean drift for a probe item."""
        for i, drift in enumerate(drift_per_layer):
            mlflow.log_metric(f"{item.item_id}_layer{i}_cosine_drift", drift)
        mean_drift = sum(drift_per_layer) / len(drift_per_layer)
        mlflow.log_metric(f"{item.item_id}_mean_cosine_drift", mean_drift)
