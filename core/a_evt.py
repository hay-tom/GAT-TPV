from typing import Dict
import numpy as np


def fit_evt_for_view(
    scores: np.ndarray,
    q: float = 0.995,
    beta: float = 1.0,
    gamma: float = 3.0,
) -> Dict[str, float]:

    scores = np.asarray(scores, dtype=np.float64)

    if scores.ndim != 1:
        raise ValueError(f"scores must be 1-D, got {scores.shape}")

    if scores.size == 0:
        raise ValueError("scores must not be empty")

    if not (0.0 < q < 1.0):
        raise ValueError(f"q must be in (0, 1), got {q}")

    T = float(np.quantile(scores, q))
    med = float(np.median(scores))
    MAD = float(np.median(np.abs(scores - med)) + 1e-8)

    return {
        "T": T,
        "MAD": MAD,
        "q": float(q),
        "beta": float(beta),
        "gamma": float(gamma),
    }


def compute_aevt_threshold(
    conf: np.ndarray,
    evt_params: Dict[str, float],
) -> np.ndarray:
    """
    Compute confidence-conditioned adaptive thresholds.
    """

    c = np.clip(np.asarray(conf, dtype=np.float64), 0.0, 1.0)

    T = float(evt_params["T"])
    MAD = float(evt_params["MAD"])
    beta = float(evt_params.get("beta", 1.0))
    gamma = float(evt_params.get("gamma", 3.0))

    lower = T - gamma * MAD
    upper = T + beta * MAD

    T_prime = T + (beta * (1.0 - c) - gamma * c) * MAD
    T_prime = np.clip(T_prime, lower, upper)

    return T_prime.astype(np.float32)
