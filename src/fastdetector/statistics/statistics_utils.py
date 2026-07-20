"""Numeric metric helpers for dataset statistics.

Plotting functions are re-exported from :mod:`fastdetector.statistics.plotting`
for backward compatibility.
"""

from sklearn.metrics import roc_auc_score
import numpy as np

# Re-export plotting functions for backward compatibility.
from fastdetector.statistics.plotting import (  # noqa: F401
    get_histogram,
    get_sweeping_classifier_plot,
    get_confusion_matrix,
    get_scatterplot,
)


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
