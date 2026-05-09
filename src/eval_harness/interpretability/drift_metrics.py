import torch.nn.functional as F


def cosine_drift(v1, v2):
    return (
        1
        - F.cosine_similarity(
            v1.unsqueeze(0),
            v2.unsqueeze(0),
        ).item()
    )


def layerwise_drift(
    baseline_states,
    pressured_states,
):
    drifts = []

    for b, p in zip(
        baseline_states,
        pressured_states,
    ):
        drifts.append(cosine_drift(b, p))

    return drifts
