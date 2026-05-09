import os
from pathlib import Path
import typer
from dotenv import load_dotenv

#
from eval_harness.backends.openai_compatible import (
    OpenAICompatibleBackend,
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

app = typer.Typer(no_args_is_help=True)


@app.command()
def run():
    load_dotenv()

    backend = OpenAICompatibleBackend(
        model_name=os.getenv("MODEL_NAME"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("BASE_URL"),
    )

    ROOT = Path(__file__).resolve().parents[3]

    dataset_path = ROOT / "datasets" / "sycophancy" / "opinion_assertion.jsonl"

    probe = OpinionAssertionProbe(str(dataset_path))
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
