import mlflow


class MLflowLogger:
    def __init__(self, experiment_name: str):
        mlflow.set_experiment(experiment_name)

    def log_result(
        self,
        item,
        response,
        scored,
    ):
        mlflow.log_metric(
            f"{item.item_id}_score",
            scored.score,
        )

        mlflow.log_text(
            response.text,
            f"responses/{item.item_id}.txt",
        )
