from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProbeItem:
    item_id: str
    probe_type: str
    # Pressured turn — social-pressure framing used for scoring.
    messages: list[dict[str, str]]
    metadata: dict[str, Any]
    # Neutral baseline turn. Populated when JSONL row has a "baseline" key.
    # When set, the orchestrator runs both turns and computes representational drift.
    baseline_messages: list[dict[str, str]] | None = field(default=None)
