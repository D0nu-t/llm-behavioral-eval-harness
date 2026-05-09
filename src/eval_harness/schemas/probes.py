from dataclasses import dataclass
from typing import Any


@dataclass
class ProbeItem:
    item_id: str
    probe_type: str
    messages: list[dict[str, str]]
    metadata: dict[str, Any]
