import json

import plotly.graph_objects as go
import streamlit as st
import torch
from streamlit_autorefresh import st_autorefresh
from pathlib import Path
_DUMP_PATH = Path(__file__).resolve().parents[3] / "activation_dump.json"

try:
    with open(_DUMP_PATH, "r") as f:
        activations = json.load(f)
except FileNotFoundError:
    activations = None
except json.JSONDecodeError:
    st.warning("Activation file mid-write, retrying...")
    st.stop()

st.set_page_config(layout="wide")
st_autorefresh(interval=1000, key="refresh")
st.title("LLM Behavioral Interpretability Dashboard")
st.subheader("Live Activation Stream")


def activation_norm_plot(norms):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=norms, mode="lines+markers"))
    fig.update_layout(
        title="Layer Activation Norms",
        xaxis_title="Layer",
        yaxis_title="Norm",
    )
    return fig


# BUG WAS HERE: activations=None then `if activations:` is always False.
# Read the file first, then branch on the result.
activations = None
try:
    with open(_DUMP_PATH, "r") as f:
        activations = json.load(f)
except FileNotFoundError:
    pass
except json.JSONDecodeError:
    st.warning("Activation file mid-write, retrying...")
    st.stop()

if activations:
    latest_states = []
    for layer_name in sorted(activations.keys()):
        tensor_data = activations[layer_name]
        # Shape coming in: [1, hidden_dim] (from tensor[:, -1, :].tolist())
        tensor = torch.tensor(tensor_data[0])  # strip the batch dim -> [hidden_dim]
        latest_states.append(tensor)

    norms = [s.norm().item() for s in latest_states]
    fig = activation_norm_plot(norms)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Layer Norm Values")
    for i, norm in enumerate(norms):
        st.write(f"Layer {i}: {norm:.6f}")
else:
    st.info("Waiting for activations — run the eval pipeline first.")
