"""Numeric metric helpers for dataset statistics."""

from sklearn.metrics import roc_auc_score
import numpy as np


def compute_auroc(
    y_true: list[bool] | list[int] | np.ndarray,
    y_scores: list[float] | np.ndarray,
) -> float:
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC).

    Args:
        y_true: True binary labels.
        y_scores: Target scores — probability estimates of the positive class
            or confidence values.

    Returns:
        The AUROC score.
    """
    return float(roc_auc_score(y_true, y_scores))
