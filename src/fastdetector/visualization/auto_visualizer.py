import numpy as np
from datasets import Dataset
from typing import List, Optional, Callable

from fastdetector.visualization.metrics import compute_threshold_sweep, compute_classifier_metrics
from fastdetector.visualization.plotting import get_sweep_plot, format_confusion_matrix

MaskFn = Callable[[Dataset], np.ndarray]

def _extract(ds: Dataset, column: str, mask_fn: Optional[MaskFn]) -> np.ndarray:
    """Extract column values from a dataset, optionally applying a mask function.

    Args:
        ds: The input Dataset.
        column: Column name to extract.
        mask_fn: Optional callable returning a boolean mask array for filtering rows.

    Returns:
        Numpy float array of extracted column values.
    """
    # asarray, not array: callers never mutate the result, so a column that is
    # already a float array (see analysis.ColumnCache) is used without copying.
    arr = np.asarray(ds[column], dtype=float)
    if mask_fn is not None:
        mask = np.asarray(mask_fn(ds), dtype=bool)
        arr = arr[mask]
    return arr

class StatWrapper:
    """Wrapper computing univariate summary statistics for a dataset column.

    Non-finite entries (``None``, NaN, ``inf`` - a metric that errored out or
    was never computed for that row) are counted in ``invalid`` and excluded
    from every other statistic, so one failed row cannot turn a whole column's
    mean into NaN.
    """

    def __init__(self, ds: Dataset, column: str, name: str, mask_fn: Optional[MaskFn] = None) -> None:
        """Initialize StatWrapper.

        Args:
            ds: The input Dataset.
            column: Name of the dataset column.
            name: Descriptive display name for this statistic.
            mask_fn: Optional boolean masking function.
        """
        self.name = name
        self.column = column
        self.arr = _extract(ds, column, mask_fn)
        self.finite = self.arr[np.isfinite(self.arr)]

        self.count = int(self.arr.size)
        self.invalid = int(self.arr.size - self.finite.size)

        has_values = self.finite.size > 0
        self.mean = float(np.mean(self.finite)) if has_values else float('nan')
        self.median = float(np.median(self.finite)) if has_values else float('nan')
        self.std = float(np.std(self.finite)) if has_values else float('nan')
        self.min = float(np.min(self.finite)) if has_values else float('nan')
        self.max = float(np.max(self.finite)) if has_values else float('nan')
        self.values = {
            "count": self.count,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "invalid": self.invalid,
        }

class ThresholdWrapper:
    """Wrapper that sweeps thresholds and extracts optimal decision thresholds."""

    def __init__(self, arrays: List[np.ndarray], column_classes: List[bool], threshold_type: str, flip_class: bool = False, name: str = "", column_names: List[str] = None) -> None:
        """Initialize ThresholdWrapper.

        Args:
            arrays: List of score arrays per class.
            column_classes: List of class labels (True = positive class).
            threshold_type: Type of threshold to extract ("accuracy", "f1", "fpr_1pct", etc.).
            flip_class: If True, scores <= threshold indicate positive class.
            name: Display name for the threshold.
            column_names: Optional custom names for input datasets.
        """
        self.name = name
        self.threshold_type = threshold_type
        self.flip_class = flip_class
        self.column_names = column_names or [f"Class {i}" for i in range(len(arrays))]
        
        threshold_dict, optimal_acc, sweep_data = compute_threshold_sweep(arrays, column_classes, flip_class)
        
        self.threshold_dict = threshold_dict
        self.sweep_data = sweep_data
        self.optimal_acc = optimal_acc
        self.threshold_value = threshold_dict.get(threshold_type)
        self.values = {"threshold_value": self.threshold_value, "optimal_acc": self.optimal_acc}
        
    def render_sweep_plot(self) -> bytes:
        """Render a threshold sweep line plot.

        Returns:
            PNG image bytes of the threshold sweep plot.
        """
        thresholds, per_dataset_accs, agg_accs = self.sweep_data
        return get_sweep_plot(thresholds, per_dataset_accs, agg_accs, self.column_names, self.threshold_dict, f"Threshold Sweep: {self.name}")

class StaticThresholdWrapper:
    """Wrapper holding a static, manually specified threshold value."""

    def __init__(self, value: float, flip_class: bool = False, name: str = "") -> None:
        """Initialize StaticThresholdWrapper.

        Args:
            value: Static threshold float value.
            flip_class: If True, scores <= threshold indicate positive class.
            name: Display name for the threshold.
        """
        self.name = name
        self.threshold_value = value
        self.flip_class = flip_class
        self.values = {"threshold_value": self.threshold_value}

class ClassifierWrapper:
    """Wrapper evaluating a classifier across dataset arrays at a fixed threshold."""

    def __init__(self, arrays: List[np.ndarray], column_classes: List[bool], threshold_wrapper, name: str = "") -> None:
        """Initialize ClassifierWrapper.

        Args:
            arrays: List of score arrays per class.
            column_classes: List of class labels (True = positive class).
            threshold_wrapper: ThresholdWrapper or StaticThresholdWrapper instance.
            name: Display name for the evaluation.
        """
        self.name = name
        
        threshold_val = threshold_wrapper.threshold_value
        flip_class = threshold_wrapper.flip_class
        
        metrics = compute_classifier_metrics(arrays, column_classes, threshold_val, flip_class)
        self.metrics = metrics
        self.metrics["confusion_matrix"] = format_confusion_matrix(
            metrics["TP"], metrics["FP"], metrics["TN"], metrics["FN"], f"Confusion Matrix: {name}"
        )
        self.values = self.metrics