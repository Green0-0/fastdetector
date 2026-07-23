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

def generate_histogram(wrappers: List[StatWrapper], title: str, bins: int = 50, figsize: Tuple[int, int] = (8, 5)) -> bytes:
    data_lists = [w.arr for w in wrappers]
    labels = [w.name for w in wrappers]
    return get_histogram(data_lists, labels, title, bins=bins, figsize=figsize)

def generate_scatterplot(x_wrapper: StatWrapper, y_wrappers: List[StatWrapper], title: str, xlabel: str = "X", ylabel: str = "Y", point_alpha: float = 0.5, rolling_mean_window: int = 0, figsize: Tuple[int, int] = (8, 5)) -> bytes:
    x_data = x_wrapper.arr
    y_data_lists = [w.arr for w in y_wrappers]
    labels = [w.name for w in y_wrappers]
    return get_scatterplot(x_data, y_data_lists, labels, title, xlabel=xlabel, ylabel=ylabel, point_alpha=point_alpha, rolling_mean_window=rolling_mean_window, figsize=figsize)

def generate_pearson_heatmap(wrappers: List[StatWrapper], title: str) -> bytes:
    import matplotlib.pyplot as plt
    import io
    
    arrays = [w.arr for w in wrappers]
    names = [w.name for w in wrappers]
    n = len(wrappers)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if len(arrays[i]) == 0 or len(arrays[j]) == 0:
                matrix[i, j] = float("nan")
            else:
                matrix[i, j] = float(np.corrcoef(arrays[i], arrays[j])[0, 1])
                
    fig, ax = plt.subplots(figsize=(max(6, n), max(6, n)))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(n), labels=names)
    ax.set_yticks(np.arange(n), labels=names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            text_color = "w" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color)
            
    ax.set_title(title)
    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return buf.getvalue()

def compute_row_emojis(rows: List[dict], emoji_config: Optional[dict]) -> Dict[str, str]:
    if not emoji_config:
        return {r["name"]: "" for r in rows}
        
    widx = emoji_config["wrapper_idx"]
    stat = emoji_config["stat"]
    higher_is_better = emoji_config.get("higher_is_better", True)
    skip_names = set(emoji_config.get("skip_names", ()))
    
    rank_values = []
    for r in rows:
        cells = r.get("cells", [])
        if widx >= len(cells):
            rank_values.append(float("nan"))
            continue
        val = cells[widx].values.get(stat)
        if val is None:
            rank_values.append(float("nan"))
        else:
            rank_values.append(val)
            
    valid = [(i, v) for i, v in enumerate(rank_values) if v is not None and v == v and rows[i]["name"] not in skip_names]
    
    best: set[int] = set()
    worst: set[int] = set()
    
    if valid:
        valid_sorted = sorted(valid, key=lambda x: x[1])
        n = len(valid_sorted)
        if emoji_config["mode"] == "single":
            if higher_is_better:
                best = {valid_sorted[-1][0]}
                worst = {valid_sorted[0][0]}
            else:
                best = {valid_sorted[0][0]}
                worst = {valid_sorted[-1][0]}
        else:
            n_top = max(1, int(n * emoji_config["pct"]))
            if higher_is_better:
                worst = {idx for idx, _ in valid_sorted[:n_top]}
                best = {idx for idx, _ in valid_sorted[-n_top:]}
            else:
                best = {idx for idx, _ in valid_sorted[:n_top]}
                worst = {idx for idx, _ in valid_sorted[-n_top:]}
                
    return {r["name"]: ("✔️ " if i in best else ("❗ " if i in worst else "")) for i, r in enumerate(rows)}

def generate_table(rows: List[dict], columns: List[dict], emoji_config: Optional[dict] = None, row_header: str = "Name") -> Tuple[str, Dict[str, str]]:
    emojis = compute_row_emojis(rows, emoji_config)
    row_names = [emojis[r["name"]] + r["name"] for r in rows]
    
    header = f"| {row_header} | " + " | ".join(c["header"] for c in columns) + " |\n"
    sep = "|---|" + "|".join(["---" for _ in columns]) + "|\n"
    
    lines = []
    for i, row in enumerate(rows):
        cells = []
        for col in columns:
            widx = col["wrapper_idx"]
            stat = col["stat"]
            if widx >= len(row["cells"]):
                cells.append("-")
                continue
            wrapper = row["cells"][widx]
            val = wrapper.values.get(stat)
            if val is None:
                cells.append("-")
                continue
                
            stat_2 = col.get("stat_2")
            if stat_2 is not None:
                val2 = wrapper.values.get(stat_2)
                if val2 is not None:
                    fmt = col.get("format", "{value:.4f} ± {value_2:.4f}")
                    cells.append(fmt.format(value=val, value_2=val2))
                else:
                    cells.append(f"{val:.4f}")
            else:
                fmt = col.get("format", "{value:.4f}")
                cells.append(fmt.format(value=val))
        lines.append(f"| {row_names[i]} | " + " | ".join(cells) + " |")
        
    return header + sep + "\n".join(lines) + "\n", emojis
