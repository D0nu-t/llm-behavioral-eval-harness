import json

from eval_harness.probes.base import ProbeDataset
from eval_harness.schemas.probes import ProbeItem


class OpinionAssertionProbe(ProbeDataset):
    """
    Reads opinion_assertion.jsonl and yields one ProbeItem per row.

    Supported JSONL formats (both work in the same file):

    Pressured-only (original):
        {"item_id": "syc_001", "biased_prefix": "...", "question": "..."}

    Paired drift (extended):
        {"item_id": "syc_001", "baseline": "...", "biased_prefix": "...", "question": "..."}

    When "baseline" is present, baseline_messages is populated and the
    orchestrator runs both turns to compute representational drift.
    """

    def __init__(self, path: str):
        self.path = path

    def __iter__(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                pressured_content = f"{row['biased_prefix']}\n\n{row['question']}"
                messages = [{"role": "user", "content": pressured_content}]

                baseline_messages = None
                if "baseline" in row:
                    baseline_messages = [{"role": "user", "content": row["baseline"]}]

                yield ProbeItem(
                    item_id=row["item_id"],
                    probe_type="sycophancy",
                    messages=messages,
                    metadata=row,
                    baseline_messages=baseline_messages,
                )
