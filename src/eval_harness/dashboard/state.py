import torch
from collections import defaultdict

# BUG WAS HERE: torch was used in type annotation but never imported.
LIVE_ACTIVATIONS: defaultdict[str, torch.Tensor] = defaultdict(list)

LIVE_TEXT: list[str] = []

CURRENT_PROMPT: str = ""
