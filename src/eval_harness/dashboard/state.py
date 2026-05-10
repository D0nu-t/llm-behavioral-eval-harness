from collections import defaultdict

LIVE_ACTIVATIONS: defaultdict[str, torch.Tensor] = defaultdict(list)

LIVE_TEXT: list[str] = []

CURRENT_PROMPT: str = ""
