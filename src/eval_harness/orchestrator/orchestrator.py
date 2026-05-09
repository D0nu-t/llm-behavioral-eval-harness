import mlflow


class EvalOrchestrator:
    def __init__(
        self,
        backend,
        probe,
        scorer,
        logger,
    ):
        self.backend = backend
        self.probe = probe
        self.scorer = scorer
        self.logger = logger

    def run(self):
        with mlflow.start_run():
            for item in self.probe:
                response = self.backend.complete(item.messages)

                scored = self.scorer.score(
                    item,
                    response,
                )

                self.logger.log_result(
                    item,
                    response,
                    scored,
                )

                print(f"{item.item_id}: {'PASS' if scored.passed else 'FAIL'}")
                print(len(response.hidden_states))
                print(response.hidden_states[0].shape)
