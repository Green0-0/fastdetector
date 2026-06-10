import io
import matplotlib.pyplot as plt
import numpy as np

def get_histogram(data_lists: list[list[float]], labels: list[str], title: str, bins: int = 50, figsize: tuple[int, int] = (8, 5)) -> bytes:
    """Generates a single histogram with multiple datasets overlayed with their corresponding label.

    Returns the histogram as a bytes object.

    Args:
        data_lists (list[list[float]]): List of data lists.
        labels (list[str]): List of labels corresponding to the data lists.
        title (str): Title of the histogram.
        bins (int, optional): Number of bins. Defaults to 50.
        figsize (tuple[int, int], optional): Size of the histogram. Defaults to (8, 5).
    
    Returns:
        bytes: Histogram as a bytes object.
    """
    plt.figure(figsize=figsize)
    for data, label in zip(data_lists, labels):
        if data is not None:
            plt.hist(data, bins=bins, alpha=0.5, density=True, label=label)
    plt.title(title)
    if labels and any(labels):
        plt.legend()
    plt.grid(True)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf.read()

def get_sweeping_classifier_plot(data_lists: list[list[float]], correct_labels: list[bool], flip_inequality: bool, generate_aggregate_line: bool, labels: list[str], title: str, figsize: tuple[int, int] = (8, 5)) -> tuple[bytes, float, float]:
    """Generates a single sweeping classifier plot with multiple datasets overlayed with their corresponding label. The classifier works by sweeping along all possible thresholds within the data lists and calculating the accuracy of classification where values greater than the threshold are classified as one class and values less than or equal to the threshold are classified as the other class.

    Returns the sweeping classifier plot as a bytes object, the threshold where the optimal average accuracy was found, and the optimal average accuracy itself.

    Args:
        data_lists (list[list[float]]): List of data lists.
        correct_labels (list[bool]): List of correct labels corresponding to the data lists.
        flip_inequality (bool): Whether to flip the inequality.
        generate_aggregate_line (bool): Whether to generate an aggregate line.
        labels (list[str]): List of labels corresponding to the data lists.
        title (str): Title of the sweeping classifier plot.
        figsize (tuple[int, int], optional): Size of the sweeping classifier plot. Defaults to (8, 5).
    
    Returns:
        tuple[bytes, float, float]: Sweeping classifier plot as a bytes object, the optimal threshold, and the optimal accuracy.
    """
    all_data = []
    for d in data_lists:
        all_data.extend(d)
        
    if not all_data:
        return b"", 0.0, 0.0
        
    min_val, max_val = np.min(all_data), np.max(all_data)
    thresholds = np.linspace(min_val, max_val, 100)
    
    plt.figure(figsize=figsize)
    total_len = sum(len(d) for d in data_lists)
    agg_accs = []
    
    for i, data in enumerate(data_lists):
        arr = np.array(data)
        if len(arr) == 0:
            continue
            
        is_pos = correct_labels[i]
        accs = []
        
        for t in thresholds:
            if not flip_inequality:
                correct = np.sum(arr > t) if is_pos else np.sum(arr <= t)
            else:
                correct = np.sum(arr <= t) if is_pos else np.sum(arr > t)
            accs.append(correct / len(arr))
            
        plt.plot(thresholds, accs, label=labels[i])
        
    optimal_threshold = 0.0
    optimal_accuracy = 0.0
    if total_len > 0:
        for t in thresholds:
            total_correct = 0
            for i, data in enumerate(data_lists):
                arr = np.array(data)
                is_pos = correct_labels[i]
                if not flip_inequality:
                    correct = np.sum(arr > t) if is_pos else np.sum(arr <= t)
                else:
                    correct = np.sum(arr <= t) if is_pos else np.sum(arr > t)
                total_correct += correct
            agg_accs.append(total_correct / total_len)
            
        optimal_idx = int(np.argmax(agg_accs))
        optimal_threshold = float(thresholds[optimal_idx])
        optimal_accuracy = float(agg_accs[optimal_idx])
        
        if generate_aggregate_line:
            plt.plot(thresholds, agg_accs, label='Aggregate Accuracy', color='black', linestyle='--')
        
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title(title)
    if any(labels) or generate_aggregate_line:
        plt.legend()
    plt.grid(True)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf.read(), optimal_threshold, optimal_accuracy

def get_confusion_matrix(data_lists: list[list[float]], correct_labels: list[bool], flip_inequality: bool, target_threshold: float, title: str) -> str:
    """Generates a markdown confusion matrix table at a target threshold.
    
    Args:
        data_lists (list[list[float]]): List of data lists.
        correct_labels (list[bool]): List of correct labels corresponding to the data lists.
        flip_inequality (bool): Whether to flip the inequality.
        target_threshold (float): The threshold to classify values.
        title (str): Title of the confusion matrix.
    
    Returns:
        str: Markdown formatted confusion matrix.
    """
    actual = []
    predicted = []
    
    for i, data in enumerate(data_lists):
        arr = np.array(data)
        if len(arr) == 0:
            continue
        is_pos = correct_labels[i]
        
        actual.extend([is_pos] * len(arr))
        
        if not flip_inequality:
            preds = arr > target_threshold
        else:
            preds = arr <= target_threshold
            
        predicted.extend(preds.tolist())
        
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    TP = np.sum((actual == True) & (predicted == True))
    FP = np.sum((actual == False) & (predicted == True))
    TN = np.sum((actual == False) & (predicted == False))
    FN = np.sum((actual == True) & (predicted == False))
    
    actual_pos = TP + FN
    pred_pos = TP + FP
    
    total = TP + FP + TN + FN
    
    precision = TP / pred_pos if pred_pos > 0 else 0
    recall = TP / actual_pos if actual_pos > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    tp_prop = TP / total if total > 0 else 0
    fp_prop = FP / total if total > 0 else 0
    fn_prop = FN / total if total > 0 else 0
    tn_prop = TN / total if total > 0 else 0
    
    md = f"### {title} (F1 Score: {f1:.4f})\n"
    md += "| | Predicted Positive | Predicted Negative |\n"
    md += "|---|---|---|\n"
    md += f"| **Actual Positive** | {TP} ({tp_prop:.2%}) | {FN} ({fn_prop:.2%}) |\n"
    md += f"| **Actual Negative** | {FP} ({fp_prop:.2%}) | {TN} ({tn_prop:.2%}) |\n"
    
    return md

def get_scatterplot(x_data: list[float] | list[list[float]], y_data_lists: list[list[float]], labels: list[str], title: str, xlabel: str = "X", ylabel: str = "Y", figsize: tuple[int, int] = (8, 5)) -> bytes:
    """Generates a scatterplot of multiple y datasets against a single x dataset.

    Args:
        x_data (list[float] | list[list[float]]): The x-axis data, or a list of x-axis data lists corresponding to y_data_lists.
        y_data_lists (list[list[float]]): List of y-axis data lists.
        labels (list[str]): List of labels for each y dataset.
        title (str): Title of the plot.
        xlabel (str, optional): Label for the x-axis. Defaults to "X".
        ylabel (str, optional): Label for the y-axis. Defaults to "Y".
        figsize (tuple[int, int], optional): Size of the figure. Defaults to (8, 5).

    Returns:
        bytes: The generated scatterplot image as bytes.
    """
    plt.figure(figsize=figsize)
    
    if len(x_data) > 0 and isinstance(x_data[0], (list, np.ndarray)):
        x_lists = x_data
    else:
        x_lists = [x_data] * len(y_data_lists)
        
    for x_vals, y_vals, label in zip(x_lists, y_data_lists, labels):
        if x_vals is not None and y_vals is not None and len(x_vals) == len(y_vals):
            plt.scatter(x_vals, y_vals, alpha=0.5, label=label, marker='o', s=15)
            
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if labels and any(labels):
        plt.legend()
    plt.grid(True)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf.read()