from typing import Dict, Optional, Tuple
import numpy as np


def _robust_zscore(scores: np.ndarray, med: float, mad: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    denom = 1.4826 * (float(mad) + 1e-8)
    return (scores - float(med)) / denom


def fit_gaml_params(
    scores_dict: Dict[str, np.ndarray],
    normal_mask: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:

    params = {}

    for view, scores in scores_dict.items():
        scores = np.asarray(scores, dtype=np.float64)

        if normal_mask is not None:
            mask = np.asarray(normal_mask, dtype=bool)
            if mask.shape[0] != scores.shape[0]:
                raise ValueError(
                    f"normal_mask length {mask.shape[0]} does not match "
                    f"scores length {scores.shape[0]} for view {view}"
                )
            scores_fit = scores[mask]
        else:
            scores_fit = scores

        if scores_fit.size == 0:
            raise ValueError(f"No samples available to fit GAML for view {view}")

        med = np.median(scores_fit)
        mad = np.median(np.abs(scores_fit - med)) + 1e-8

        params[view] = {
            "med": float(med),
            "mad": float(mad),
        }

    return params


def transform_gaml_scores(
    scores_dict: Dict[str, np.ndarray],
    params: Dict[str, Dict[str, float]],
    return_conf: bool = False,
):
    """
    Apply robust score alignment and optionally compute confidence.
    """

    s_calib = {}
    conf = {}

    for view, scores in scores_dict.items():
        if view not in params:
            raise KeyError(f"Missing GAML params for view {view}")

        scores = np.asarray(scores, dtype=np.float64)
        med = params[view]["med"]
        mad = params[view]["mad"]

        z = _robust_zscore(scores, med, mad).astype(np.float32)
        s_calib[view] = z

        absz = np.minimum(np.abs(z).astype(np.float32), np.float32(1e6))
        c = absz / (1.0 + absz)
        conf[view] = c.astype(np.float32)

    if return_conf:
        return s_calib, conf

    return s_calib


def calibrate_scores_with_gaml(
    scores_dict: Dict[str, np.ndarray],
    labels: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Dict[str, float]]] = None,
    return_conf: bool = False,
    normal_mask: Optional[np.ndarray] = None,
):

    if params is None:
        return fit_gaml_params(scores_dict, normal_mask=normal_mask)

    return transform_gaml_scores(
        scores_dict=scores_dict,
        params=params,
        return_conf=return_conf,
    )
