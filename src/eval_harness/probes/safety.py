import json
from pathlib import Path

from eval_harness.schemas.probes import ProbeItem


class SafetyProbe:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)

    def __iter__(self):
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)

                yield ProbeItem(
                    item_id=row["item_id"],
                    probe_type="safety",
                    messages=[
                        {
                            "role": "user",
                            "content": row["prompt"],
                        }
                    ],
                    metadata=row,
                )
