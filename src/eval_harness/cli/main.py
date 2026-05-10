import os
from pathlib import Path

import typer
from dotenv import load_dotenv

from eval_harness.backends.hf_transformer import HFTransformerBackend
from eval_harness.logging.mlflow_logger import MLflowLogger
from eval_harness.orchestrator.orchestrator import EvalOrchestrator
from eval_harness.probes.sycophancy import OpinionAssertionProbe
from eval_harness.scorers.rubric import SimpleSycophancyScorer

app = typer.Typer(no_args_is_help=True)


@app.command()
def run():
    load_dotenv()

    backend = HFTransformerBackend(model_name=os.getenv("MODEL_NAME"))
    ROOT = Path(__file__).resolve().parents[3]
    dataset_path = ROOT / "datasets" / "sycophancy" / "opinion_assertion.jsonl"

    orchestrator = EvalOrchestrator(
        backend=backend,
        probe=OpinionAssertionProbe(str(dataset_path)),
        scorer=SimpleSycophancyScorer(),
        logger=MLflowLogger("behavioral_eval"),
    )

    for result in orchestrator.run():
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{result['item_id']}: {status}  latency={result['latency_ms']}ms")
        if result["drift_per_layer"] is not None:
            mean = sum(result["drift_per_layer"]) / len(result["drift_per_layer"])
            print(f"  mean cosine drift: {mean:.4f}")
            for i, d in enumerate(result["drift_per_layer"]):
                print(f"  layer {i}: {d:.6f}")


if __name__ == "__main__":
    app()
