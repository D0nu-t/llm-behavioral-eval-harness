import mlflow


class MLflowLogger:
    def __init__(self, experiment_name: str):
        mlflow.set_experiment(experiment_name)

    def log_result(self, item, response, scored):
        mlflow.log_metric(f"{item.item_id}_score", scored.score)
        mlflow.log_text(response.text, f"responses/{item.item_id}.txt")

    def log_metrics(self, item, metrics: dict[str, list[float]]):
        """
        Log all layerwise metrics from full_layerwise_metrics().
        Keys: cosine_drift, norm_ratio, effective_rank.
        Logs per-layer scalars and a mean summary for each type.
        """
        for metric_name, values in metrics.items():
            for i, v in enumerate(values):
                mlflow.log_metric(f"{item.item_id}_{metric_name}_L{i}", v)
            mlflow.log_metric(
                f"{item.item_id}_mean_{metric_name}",
                sum(values) / len(values),
            )
