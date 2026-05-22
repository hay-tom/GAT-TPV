import numpy as np


KNOWN = 0
ALERT = 1
ANOMALY = 2


def tpcv_decision(
    scores: np.ndarray,
    thresholds: np.ndarray,
    delta: float = 0.0,
    k: int = 2,
) -> np.ndarray:

    scores = np.asarray(scores, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)

    if scores.ndim != 2:
        raise ValueError(f"scores must be [N, V], got shape {scores.shape}")

    if thresholds.ndim == 1:
        if thresholds.shape[0] != scores.shape[1]:
            raise ValueError(
                f"thresholds shape {thresholds.shape} does not match "
                f"number of views {scores.shape[1]}"
            )
        thresholds = np.broadcast_to(thresholds[None, :], scores.shape)

    if thresholds.shape != scores.shape:
        raise ValueError(
            f"thresholds must have shape [V] or [N, V], got {thresholds.shape}"
        )

    if not (1 <= k <= scores.shape[1]):
        raise ValueError(f"k must be in [1, {scores.shape[1]}], got {k}")

    diff = scores - thresholds

    count_abnormal = (diff >= 0.0).sum(axis=1)
    count_strong = (diff >= delta).sum(axis=1)

    decision = np.full(scores.shape[0], KNOWN, dtype=np.int64)

    alert_mask = (count_abnormal > 0) & (count_strong < k)
    anomaly_mask = count_strong >= k

    decision[alert_mask] = ALERT
    decision[anomaly_mask] = ANOMALY

    return decision


def anomaly_score_from_diff(diff: np.ndarray) -> np.ndarray:
    """
    Continuous anomaly score used for AUROC/AUPR ranking.

    This follows the paper's idea of using a margin-derived
    score, e.g., max_v Delta_v(x).
    """
    diff = np.asarray(diff, dtype=np.float64)

    if diff.ndim != 2:
        raise ValueError(f"diff must be [N, V], got shape {diff.shape}")

    return diff.max(axis=1).astype(np.float32)


def decision_to_name(decision: np.ndarray) -> np.ndarray:
    mapping = {
        KNOWN: "KNOWN",
        ALERT: "ALERT",
        ANOMALY: "ANOMALY",
    }
    decision = np.asarray(decision, dtype=np.int64)
    return np.array([mapping[int(d)] for d in decision])
