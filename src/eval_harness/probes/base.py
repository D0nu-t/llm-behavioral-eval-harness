from abc import ABC, abstractmethod
from collections.abc import Iterator

from eval_harness.schemas.probes import ProbeItem


class ProbeDataset(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[ProbeItem]:
        pass
