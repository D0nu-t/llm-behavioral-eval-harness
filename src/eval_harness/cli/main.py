import os

import typer
from dotenv import load_dotenv

from eval_harness.backends.openai_backend import (
    OpenAIBackend,
)
from eval_harness.logging.mlflow_logger import (
    MLflowLogger,
)
from eval_harness.orchestrator.orchestrator import (
    EvalOrchestrator,
)
from eval_harness.probes.sycophancy import (
    OpinionAssertionProbe,
)
from eval_harness.scorers.rubric import (
    SimpleSycophancyScorer,
)

app = typer.Typer()


@app.command()
def run():
    load_dotenv()

    backend = OpenAIBackend(
        model_name="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    probe = OpinionAssertionProbe("datasets/sycophancy/opinion_assertion.jsonl")

    scorer = SimpleSycophancyScorer()

    logger = MLflowLogger("behavioral_eval")

    orchestrator = EvalOrchestrator(
        backend,
        probe,
        scorer,
        logger,
    )

    orchestrator.run()


if __name__ == "__main__":
    app()
