"""Scaling of uncertainty values into 1-10 confidence scores.

`unc_to_confidence` is the DeBUnc formula (Yoffe et al.). It knows nothing
about category names or output format: the coarse-grained drivers hand it one
entropy per agent, the fine-grained ones hand it every (agent, category)
entropy at once."""

import numpy as np


def unc_to_confidence(uncertainties: np.ndarray) -> np.ndarray:
    """Cross-agent normalise N entropies into N integers in [1, 10].

        c_i = 1 / u_i
        s_i = c_i / sum(c_j) * (5N - 1) + 1/N      # mean(s_i) = 5
        s_i = round(clip(s_i, 1, 10))

    The original repo hard-coded 14 as the multiplier, which is 5N - 1 for N=3
    only; for N=2 that gave a mean of 7.5. This uses the N-aware form."""
    n = uncertainties.shape[0]
    confidences = 1 / uncertainties
    confidences = confidences * (5 * n - 1) / np.sum(confidences) + 1 / n
    confidences = np.clip(confidences, 1, 10)
    confidences = np.round(confidences).astype(int)
    return confidences
