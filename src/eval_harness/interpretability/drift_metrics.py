"""
interpretability/drift_metrics.py

Layerwise metrics comparing baseline and pressured activation vectors.

All functions operate on 1-D tensors (one vector per layer, already extracted
as final-token states by HFTransformerBackend).

Metrics
-------
cosine_drift    : 1 - cos(v, w). Range [0, 2]. 0 = identical direction.
norm_ratio      : ||w|| / ||v||. >1 = pressured vector has larger magnitude.
effective_rank  : exp(H) where H is Shannon entropy of the SVD singular value
                  spectrum of [v; w]. Collapses toward 1.0 when the two vectors
                  are nearly parallel (low-rank pair); approaches 2.0 when they
                  are orthogonal (full-rank pair). Useful as a continuous
                  companion to cosine_drift — drift can be large while rank
                  stays low if the pressure merely scales the vector.
layerwise_*     : apply the scalar metric across matched layer pairs.
"""

import numpy as np
import torch
import torch.nn.functional as F


def cosine_drift(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """
    1 - cosine_similarity(v1, v2).
    Range [0, 2]. 0 = identical direction, 1 = orthogonal, 2 = opposite.
    """
    return 1 - F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


def norm_ratio(v_baseline: torch.Tensor, v_pressured: torch.Tensor) -> float:
    """
    ||v_pressured|| / ||v_baseline||.
    >1 means social pressure inflated the residual stream magnitude at this layer.
    <1 means it suppressed it.
    Returns inf if baseline norm is zero (degenerate layer).
    """
    denom = v_baseline.norm().item()
    if denom == 0.0:
        return float("inf")
    return v_pressured.norm().item() / denom


def effective_rank(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """
    Effective rank of the 2×d matrix [v1; v2] via SVD entropy.

    r_eff = exp(-sum_i p_i * log(p_i))  where  p_i = s_i / sum(s)

    For two vectors the SVD has at most 2 non-zero singular values:
      - s1 captures the shared direction (mean)
      - s2 captures the difference direction

    Interpretation:
      r_eff → 1.0  : vectors nearly parallel — pressure collapsed them
      r_eff → 2.0  : vectors orthogonal — pressure opened a new direction
    """
    v1_np = v1.float().numpy()
    v2_np = v2.float().numpy()
    mat = np.vstack([v1_np, v2_np])               # shape [2, d]
    _, s, _ = np.linalg.svd(mat, full_matrices=False)
    s = s[s > 1e-12]                              # drop numerical zeros
    if len(s) == 0:
        return 1.0
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))


def layerwise_drift(
    baseline_states: list[torch.Tensor],
    pressured_states: list[torch.Tensor],
) -> list[float]:
    return [cosine_drift(b, p) for b, p in zip(baseline_states, pressured_states)]


def layerwise_norm_ratio(
    baseline_states: list[torch.Tensor],
    pressured_states: list[torch.Tensor],
) -> list[float]:
    return [norm_ratio(b, p) for b, p in zip(baseline_states, pressured_states)]


def layerwise_effective_rank(
    baseline_states: list[torch.Tensor],
    pressured_states: list[torch.Tensor],
) -> list[float]:
    return [effective_rank(b, p) for b, p in zip(baseline_states, pressured_states)]


def full_layerwise_metrics(
    baseline_states: list[torch.Tensor],
    pressured_states: list[torch.Tensor],
) -> dict[str, list[float]]:
    """
    Convenience wrapper returning all three metric vectors in one call.
    Used by the orchestrator to avoid iterating the layer pairs three times.
    """
    cos, nratio, erank = [], [], []
    for b, p in zip(baseline_states, pressured_states):
        cos.append(cosine_drift(b, p))
        nratio.append(norm_ratio(b, p))
        erank.append(effective_rank(b, p))
    return {
        "cosine_drift": cos,
        "norm_ratio": nratio,
        "effective_rank": erank,
    }
