"""
interpretability/live_capture.py

Registers forward hooks on transformer layers and writes activations to
activation_dump.json ONCE per forward pass — not per layer.

BUG THAT WAS HERE: save_activations() was called inside every hook
invocation. With N layers this caused N partial writes per forward pass.
Any concurrent reader (the Streamlit dashboard) could open the file between
writes and see an incomplete dict. The file also appeared to "wipe" because
each write starts from the current LIVE_ACTIVATIONS state which grows
incrementally during the pass.

FIX: hooks only mutate LIVE_ACTIVATIONS in memory. The orchestrator calls
save_activations() once after each complete forward pass.
"""

import json
import torch

from eval_harness.dashboard.state import LIVE_ACTIVATIONS


def save_activations(path: str = "activation_dump.json") -> None:
    """Write LIVE_ACTIVATIONS to disk atomically via a .tmp swap."""
    serializable = {k: v.tolist() for k, v in LIVE_ACTIVATIONS.items()}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(serializable, f)
        f.flush()
    # Atomic rename — reader never sees a partial file
    import os
    os.replace(tmp, path)


def make_hook(name: str):
    """
    Returns a forward hook that captures the final-token hidden state.
    Does NOT write to disk — call save_activations() after the full
    forward pass completes.
    """
    def hook(module, inputs, outputs):
        tensor = outputs[0] if isinstance(outputs, tuple) else outputs
        if not isinstance(tensor, torch.Tensor):
            return
        # tensor shape: [batch, seq_len, hidden_dim]
        # keep final token, detach from graph
        LIVE_ACTIVATIONS[name] = tensor[:, -1, :].detach().cpu()
    return hook
