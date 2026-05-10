import plotly.graph_objects as go


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


def drift_plot(drifts):
    fig = go.Figure()

    fig.add_trace(go.Bar(y=drifts))

    fig.update_layout(
        title="Layerwise Semantic Drift",
        xaxis_title="Layer",
        yaxis_title="Cosine Drift",
    )

    return fig
