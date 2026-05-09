import json

from eval_harness.probes.base import ProbeDataset
from eval_harness.schemas.probes import ProbeItem


class OpinionAssertionProbe(ProbeDataset):
    def __init__(self, path: str):
        self.path = path

    def __iter__(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)

                yield ProbeItem(
                    item_id=row["item_id"],
                    probe_type="sycophancy",
                    messages=[
                        {
                            "role": "user",
                            "content": (f"{row['biased_prefix']}\n\n{row['question']}"),
                        }
                    ],
                    metadata=row,
                )
