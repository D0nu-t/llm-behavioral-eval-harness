import json
import os
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
import streamlit as st
import torch

st.set_page_config(layout="wide")
st_autorefresh(
    interval=1000,
    key="refresh",
)
st.title("LLM Behavioral Interpretability Dashboard")

st.subheader("Live Activation Stream")


def activation_norm_plot(norms):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=norms,
            mode="lines+markers",
        )
    )

    fig.update_layout(
        title="Layer Activation Norms",
        xaxis_title="Layer",
        yaxis_title="Norm",
    )

    return fig


activations = None

if activations:
    try:
        with open(
            "activation_dump.json",
            "r",
        ) as f:
            activations = json.load(f)

    except json.JSONDecodeError:
        st.warning("Activation file updating...")

        st.stop()

    latest_states = []

    for layer_name in sorted(activations.keys()):
        tensor_data = activations[layer_name]

        tensor = torch.tensor(tensor_data)

        final_token = tensor[0]

        latest_states.append(final_token)

    norms = [s.norm().item() for s in latest_states]

    fig = activation_norm_plot(norms)

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader("Layer Norm Values")

    for i, norm in enumerate(norms):
        st.write(f"Layer {i}: {norm:.6f}")

else:
    st.info("Waiting for activations...")
