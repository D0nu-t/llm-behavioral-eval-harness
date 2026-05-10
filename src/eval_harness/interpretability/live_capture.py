"""
interpretability/live_capture.py

Forward hooks + atomic activation persistence.

Path is anchored to the project root via __file__ resolution so both the
CLI process and the Streamlit dashboard process always agree on the location,
regardless of which directory each was launched from.
"""

import json
import os
from pathlib import Path

import torch

from eval_harness.dashboard.state import LIVE_ACTIVATIONS

# parents: [0]=interpretability [1]=eval_harness [2]=src [3]=project_root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DUMP = _PROJECT_ROOT / "activation_dump.json"


def save_activations(path: Path | None = None) -> None:
    """Atomic write via .tmp rename. Call once post-generate(), not per hook."""
    target = Path(path) if path else _DEFAULT_DUMP
    serializable = {k: v.tolist() for k, v in LIVE_ACTIVATIONS.items()}
    tmp = str(target) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(serializable, f)
        f.flush()
    os.replace(tmp, str(target))


def clear_activations() -> None:
    """Reset in-memory store between probe items."""
    LIVE_ACTIVATIONS.clear()


def make_hook(name: str):
    """
    Forward hook storing final-token hidden state as [1, hidden_dim].
    Does NOT write to disk — caller must call save_activations() after pass.
    """
    def hook(module, inputs, outputs):
        tensor = outputs[0] if isinstance(outputs, tuple) else outputs
        if not isinstance(tensor, torch.Tensor):
            return
        LIVE_ACTIVATIONS[name] = tensor[:, -1, :].detach().cpu()
    return hook
