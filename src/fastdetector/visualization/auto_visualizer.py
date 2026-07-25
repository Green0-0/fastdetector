import numpy as np
from datasets import Dataset
from typing import List, Optional, Tuple, Dict, Callable

from fastdetector.visualization.metrics import compute_threshold_sweep, compute_classifier_metrics, FPR_TARGETS
from fastdetector.visualization.plotting import get_histogram, get_sweep_plot, format_confusion_matrix, get_scatterplot

MaskFn = Callable[[Dataset], np.ndarray]

def _extract(ds: Dataset, column: str, mask_fn: Optional[MaskFn]) -> np.ndarray:
    arr = np.array(ds[column], dtype=float)
    if mask_fn is not None:
        mask = np.asarray(mask_fn(ds), dtype=bool)
        arr = arr[mask]
    return arr

class StatWrapper:
    def __init__(self, ds: Dataset, column: str, name: str, mask_fn: Optional[MaskFn] = None):
        self.name = name
        self.column = column
        self.arr = _extract(ds, column, mask_fn)
        self.mean = float(np.mean(self.arr)) if len(self.arr) > 0 else float('nan')
        self.std = float(np.std(self.arr)) if len(self.arr) > 0 else float('nan')
        self.min = float(np.min(self.arr)) if len(self.arr) > 0 else float('nan')
        self.max = float(np.max(self.arr)) if len(self.arr) > 0 else float('nan')
        self.values = {"mean": self.mean, "std": self.std, "min": self.min, "max": self.max}

class ThresholdWrapper:
    def __init__(self, arrays: List[np.ndarray], column_classes: List[bool], threshold_type: str, flip_class: bool = False, name: str = "", column_names: List[str] = None):
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
        thresholds, per_dataset_accs, agg_accs = self.sweep_data
        return get_sweep_plot(thresholds, per_dataset_accs, agg_accs, self.column_names, self.threshold_dict, f"Threshold Sweep: {self.name}")

class StaticThresholdWrapper:
    def __init__(self, value: float, flip_class: bool = False, name: str = ""):
        self.name = name
        self.threshold_value = value
        self.flip_class = flip_class
        self.values = {"threshold_value": self.threshold_value}

class ClassifierWrapper:
    def __init__(self, arrays: List[np.ndarray], column_classes: List[bool], threshold_wrapper, name: str = ""):
        self.name = name
        
        threshold_val = threshold_wrapper.threshold_value
        flip_class = threshold_wrapper.flip_class
        
        metrics = compute_classifier_metrics(arrays, column_classes, threshold_val, flip_class)
        self.metrics = metrics
        self.metrics["confusion_matrix"] = format_confusion_matrix(
            metrics["TP"], metrics["FP"], metrics["TN"], metrics["FN"], f"Confusion Matrix: {name}"
        )
        self.values = self.metrics