"""CLI entry point: generate the evaluation README, its charts, and the summary JSON.

The README is assembled from a fixed section list (see :func:`_build_readme`),
in this order:

1. the run configuration the script was invoked with, then a data-derived
   summary of what the report contains,
2. a contents list (headed "Contents" - see _build_readme),
3. univariate stats and correlations over every statistic of interest,
4. distance and classifier histograms over the whole dataset,
5. a classifier comparison table and the threshold sweeps behind it,
6. one report per classifier, broken down by prompt subset and by bin,
7. a hardcoded full report (currently: the first classifier, per generator
   model/genconfig).
"""

from typing import Optional
from fastdetector.visualization.plotting import generate_table
from fastdetector.visualization.plotting import generate_pearson_heatmap
from fastdetector.visualization.plotting import get_histogram
import argparse
import json
import re
import numpy as np
from datasets import Dataset

from fastdetector.frontend.toml_config import AnalysisConfig
from fastdetector.frontend.toml_loader import load_config_pair
from fastdetector.utils import load_dataset_all_shards, upload_readme, apply_filter_conditions
from fastdetector.visualization.auto_visualizer import (
    StatWrapper,
    ClassifierWrapper,
    ThresholdWrapper,
    StaticThresholdWrapper,
    _extract
)

#: Univariate table: one row per statistic of interest.
UNIVARIATE_COLUMNS = [
    {"header": "N", "wrapper_idx": 0, "stat": "count", "format": "{value:,.0f}"},
    {"header": "Mean", "wrapper_idx": 0, "stat": "mean"},
    {"header": "Median", "wrapper_idx": 0, "stat": "median"},
    {"header": "Std Dev", "wrapper_idx": 0, "stat": "std"},
    {"header": "Min", "wrapper_idx": 0, "stat": "min"},
    {"header": "Max", "wrapper_idx": 0, "stat": "max"},
    {"header": "Invalid/Error", "wrapper_idx": 0, "stat": "invalid", "format": "{value:,.0f}"},
]

#: Per-subset classifier table (prompt subsets, bins, generator configs).
SUBSET_COLUMNS = [
    {"header": "N", "wrapper_idx": 0, "stat": "n", "format": "{value:,.0f}"},
    {"header": "AUROC", "wrapper_idx": 0, "stat": "auroc"},
    {"header": "TPR", "wrapper_idx": 0, "stat": "tpr"},
    {"header": "FPR", "wrapper_idx": 0, "stat": "fpr"},
    {"header": "Accuracy", "wrapper_idx": 0, "stat": "acc"},
    {"header": "F1", "wrapper_idx": 0, "stat": "f1"},
]

#: Classifier comparison table. Cell 0 is the overall ClassifierWrapper and
#: cell 1 the threshold wrapper it was pinned with.
COMPARISON_COLUMNS = [
    {"header": "Threshold", "wrapper_idx": 1, "stat": "threshold_value"},
    {"header": "AUROC", "wrapper_idx": 0, "stat": "auroc"},
    {"header": "TPR @ Threshold", "wrapper_idx": 0, "stat": "tpr"},
    {"header": "FPR @ Threshold", "wrapper_idx": 0, "stat": "fpr"},
    {"header": "Accuracy", "wrapper_idx": 0, "stat": "acc"},
    {"header": "F1", "wrapper_idx": 0, "stat": "f1"},
]

class NumpyEncoder(json.JSONEncoder):
    """JSONEncoder subclass for serializing NumPy scalar and array types."""

    def default(self, obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def _parse_genparams(val: any) -> Optional[dict]:
    """Parse JSON string or dictionary of generation parameters.

    Args:
        val: JSON string, dict, or None.

    Returns:
        Parsed parameters dictionary or None.
    """
    if isinstance(val, dict): return val
    if not val: return None
    return json.loads(val)

def extract_prompt_types(result_ds: Dataset, prompt_col: str) -> tuple[np.ndarray, bool]:
    """Extract prompt type metadata labels from dataset prompt metadata column.

    Args:
        result_ds: Input evaluation Dataset.
        prompt_col: Name of prompt metadata column.

    Returns:
        Tuple of (prompt_types_array, has_prompt_metadata_boolean).
    """
    if not (prompt_col and prompt_col in result_ds.column_names):
        return np.array(["Unknown"] * len(result_ds)), False
    pts = []
    for p in result_ds[prompt_col]:
        pt = "Unknown"
        if p and isinstance(p.get("metadata"), dict):
            pt = str(p["metadata"].get("PROMPT_TYPE", "Unknown"))
        pts.append(pt)
    return np.array(pts), True

def extract_model_genconfig(result_ds: Dataset, model_col: str) -> tuple[np.ndarray, bool]:
    """Extract combined model identifier and generation configuration labels.

    Args:
        result_ds: Input evaluation Dataset.
        model_col: Name of generator model column.

    Returns:
        Tuple of (genconfig_labels_array, has_model_genconfig_boolean).

    Raises:
        ValueError: If model_col or generation_params column is missing.
    """
    has_model_col = model_col in result_ds.column_names
    has_genparams = "generation_params" in result_ds.column_names
    if not has_model_col and not has_genparams:
        return np.array(["Unknown"] * len(result_ds)), False
    if not has_model_col or not has_genparams:
        raise ValueError("Missing columns for model/genconfig extraction.")
    parsed = []
    for m, g in zip(result_ds[model_col], result_ds["generation_params"]):
        m_str = str(m).split('/')[-1] if m else "Unknown"
        d = _parse_genparams(g)
        temp = d.get("temperature", "Unknown") if d is not None else "Unknown"
        parsed.append(f"{m_str} (Temp: {temp})")
    return np.array(parsed), True

def compute_quantile_bins(values: np.ndarray, num_bins: int, column: str) -> tuple[np.ndarray, list[str]]:
    """Assign each row to an equal-count (quantile) bin of *values*.

    Bins are quantile-based rather than equally spaced because distance
    metrics are heavily skewed: fixed-width bins put almost every row in one
    bucket and leave the rest empty.

    Rows whose value is non-finite get the empty label and therefore belong to
    no bin subset. Duplicate quantile edges (a column with few distinct
    values) collapse into a single bin rather than producing empty ones.

    Args:
        values: Numeric values to bin, one per row.
        num_bins: Requested number of bins.
        column: Column name, used to label the bins.

    Returns:
        Tuple of (per_row_bin_labels, ordered_unique_bin_labels). Both are
        empty when the column cannot be binned.
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if num_bins < 2 or finite.size == 0:
        return np.array([""] * arr.size), []

    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, num_bins + 1)))
    if edges.size < 2:
        return np.array([""] * arr.size), []

    labels = [f"{column} {edges[i]:.4g} to {edges[i + 1]:.4g}" for i in range(edges.size - 1)]
    idx = np.clip(np.searchsorted(edges, arr, side="right") - 1, 0, edges.size - 2)
    row_labels = np.array([
        labels[i] if np.isfinite(v) else "" for i, v in zip(idx, arr)
    ])
    return row_labels, labels

class ColumnCache:
    """A read-only Dataset view that materialises each column exactly once.

    Caching column extractions avoids rebuilding columns repeatedly when
    generating multi-classifier subset reports.

    Only the read paths the report uses are implemented: column access,
    ``column_names`` and ``len``.
    """

    def __init__(self, ds: Dataset) -> None:
        """Wrap *ds*, caching nothing until a column is asked for.

        Args:
            ds: The dataset to read through.
        """
        self._ds = ds
        self._columns: dict[str, np.ndarray] = {}

    @property
    def column_names(self) -> list[str]:
        """The wrapped dataset's column names."""
        return self._ds.column_names

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, column: str) -> np.ndarray:
        if column not in self._columns:
            self._columns[column] = np.asarray(self._ds[column])
        return self._columns[column]

def _column_equals_mask(column: str, value: str):
    """Build a masking function that filters rows where dataset[column] == value.

    Args:
        column: Column name to match.
        value: Target string value.

    Returns:
        Mask function returning boolean numpy array.
    """
    cache: dict[int, np.ndarray] = {}

    def mask_fn(ds) -> np.ndarray:
        key = id(ds)
        if key not in cache:
            cache[key] = np.asarray(ds[column]) == value
        return cache[key]
    return mask_fn

def _safe_name(name: str) -> str:
    """Sanitize string for use in file names or identifiers.

    Args:
        name: Raw string.

    Returns:
        Sanitized uppercase alphanumeric string with underscores.
    """
    raw = re.sub(r"[^a-zA-Z0-9]", "_", name)
    return re.sub(r"_+", "_", raw).strip("_").upper()

def _anchor(title: str) -> str:
    """Return the markdown heading anchor a renderer derives from *title*.

    Args:
        title: Heading text (without the leading ``#``s).

    Returns:
        The anchor slug, e.g. ``"classifier-report-editlens-score"``.
    """
    slug = re.sub(r"[^\w\s-]", "", title.lower())
    return re.sub(r"[\s_]+", "-", slug.strip())

def _build_table_of_contents(body: str) -> list[str]:
    """Build the nested contents list from the headings *body* actually has.

    Reading the headings back out of the rendered markdown (rather than
    listing them a second time by hand) is what keeps every entry a working
    link: a section that is renamed, added or skipped changes its own entry.
    Repeated headings get the ``-1``, ``-2`` ... anchor suffixes markdown
    renderers assign them, in document order.

    Args:
        body: The rendered markdown body, below the introduction.

    Returns:
        List of markdown list items: ``##`` headings numbered at the top
        level, ``###`` headings indented beneath them.
    """
    seen: dict[str, int] = {}
    entries: list[str] = []
    section_number = 0

    for line in body.split("\n"):
        match = re.match(r"^(#{2,3})\s+(.+?)\s*$", line)
        if match is None:
            continue
        level, title = len(match.group(1)), match.group(2)

        anchor = _anchor(title)
        occurrence = seen.get(anchor, 0)
        seen[anchor] = occurrence + 1
        if occurrence:
            anchor = f"{anchor}-{occurrence}"

        if level == 2:
            section_number += 1
            entries.append(f"{section_number}. [{title}](#{anchor})")
        else:
            entries.append(f"    - [{title}](#{anchor})")

    return entries

def _fmt(value, spec: str = ".4f") -> str:
    """Format a metric for prose, rendering missing/NaN values as ``n/a``.

    Args:
        value: Numeric value or None.
        spec: Format spec applied to valid values.

    Returns:
        Formatted string.
    """
    if value is None or (isinstance(value, float) and value != value):
        return "n/a"
    return format(value, spec)

class Subset:
    """Represents a named data subset defined by a row-masking function."""

    def __init__(self, name: str, mask_fn):
        """Initialize Subset.

        Args:
            name: Subset display name.
            mask_fn: Row masking function.
        """
        self.name = name
        self.mask_fn = mask_fn
        self.safe = _safe_name(name)

def _extract_classifier_data(ds: Dataset, eval_config, clf_config, mask_fn) -> tuple[list, list, list]:
    """Extract score arrays, class labels, and column names for classifier evaluation.

    Args:
        ds: Input Dataset.
        eval_config: AnalysisConfig object.
        clf_config: ClassifierConfig object.
        mask_fn: Subset masking function or None.

    Returns:
        Tuple of (arrays_list, classes_boolean_list, column_names_list).

        Rows whose score is missing or non-finite are dropped: the metrics
        pair each array with a class label positionally rather than row-wise,
        so a scorer that failed on a handful of rows must not be allowed to
        turn the whole classifier's threshold sweep and AUROC into NaN. The
        univariate table reports how many rows this affects per column.
    """
    extracted_human = []
    extracted_ai = []
    col_names_human = []
    col_names_ai = []

    def scored(arr: np.ndarray) -> np.ndarray:
        """Drop the entries a classifier produced no usable score for."""
        return arr[np.isfinite(arr)]

    for i, base_col in enumerate(eval_config.base_columns):
        col_name = f"{base_col}{clf_config.suffix}"
        arr = _extract(ds, col_name, mask_fn)

        if eval_config.fixed_classes is not None:
            is_ai = eval_config.fixed_classes[i]
            if is_ai:
                extracted_ai.append(scored(arr))
                col_names_ai.append(col_name)
            else:
                extracted_human.append(scored(arr))
                col_names_human.append(col_name)
        elif eval_config.auto_class_column is not None:
            labels = np.array(ds[eval_config.auto_class_column])
            if mask_fn is not None:
                labels = labels[np.asarray(mask_fn(ds), dtype=bool)]

            ai_mask = (labels == eval_config.ai_label)
            human_mask = ~ai_mask

            extracted_ai.append(scored(arr[ai_mask]))
            col_names_ai.append(f"{col_name} (AI)")

            extracted_human.append(scored(arr[human_mask]))
            col_names_human.append(f"{col_name} (Human)")
        else:
            raise ValueError("Must provide fixed_classes or auto_class_column")

    arrays = extracted_human + extracted_ai
    classes = [False] * len(extracted_human) + [True] * len(extracted_ai)
    names = col_names_human + col_names_ai

    return arrays, classes, names

def _class_series(arrays: list, classes: list, names: list) -> list:
    """Label extracted classifier arrays as Human/AI series for a histogram.

    Args:
        arrays: Score arrays, as returned by :func:`_extract_classifier_data`.
        classes: Aligned class flags (True = AI).
        names: Aligned source column names.

    Returns:
        List of ``(values, label)`` tuples ready for ``get_histogram``.
    """
    series = []
    for arr, is_ai, name in zip(arrays, classes, names):
        cls = "AI" if is_ai else "Human"
        label = name if name.endswith(f"({cls})") else f"{cls} ({name})"
        series.append((arr, label))
    return series

def _threshold_settings(eval_config, clf) -> tuple:
    """Resolve (manual_threshold, threshold_type) for a classifier.

    Each classifier uses either the score or the bin threshold settings,
    selected by its threshold_kind. A manual threshold, where set, applies to
    every classifier of that kind; the validation split is only skipped
    entirely when *all* configured classifiers have a manual threshold.

    Args:
        eval_config: AnalysisConfig instance.
        clf: ClassifierConfig instance.

    Returns:
        Tuple of (manual_threshold or None, threshold_type string).
    """
    if clf.threshold_kind == "bin":
        return eval_config.manual_threshold_bin, eval_config.threshold_type_bin
    return eval_config.manual_threshold_score, eval_config.threshold_type_score

def _threshold_description(eval_config, clf) -> str:
    """Describe, in prose, how a classifier's decision threshold was chosen.

    Args:
        eval_config: AnalysisConfig instance.
        clf: ClassifierConfig instance.

    Returns:
        Human-readable description string.
    """
    manual, threshold_type = _threshold_settings(eval_config, clf)
    if manual is not None:
        return f"pinned manually at {manual:g}"
    return f"swept for `{threshold_type}` on the validation split"

def select_available(ds: Dataset, eval_config) -> tuple[list, list, list, list]:
    """Split the configured metrics and classifiers into present and missing.

    A config that names every statistic the pipeline can compute is the useful
    default, but any given dataset has only the ones whose stats stage was
    actually run (and whose flags were enabled). Rather than crashing on the
    first absent column, the report evaluates what is there and states what it
    skipped, so a missing stage is visible instead of silently unreported.

    Args:
        ds: The evaluation Dataset.
        eval_config: AnalysisConfig instance.

    Returns:
        Tuple of (distance_metrics, missing_distance_metrics, classifiers,
        skipped_classifiers), where skipped_classifiers maps a classifier name
        to the columns it needed and the dataset does not have.
    """
    metrics, missing_metrics = [], []
    for m in eval_config.distance_metrics:
        (metrics if m in ds.column_names else missing_metrics).append(m)

    classifiers: list = []
    skipped: dict[str, list[str]] = {}
    for clf in eval_config.classifiers:
        absent = [
            f"{base}{clf.suffix}"
            for base in eval_config.base_columns
            if f"{base}{clf.suffix}" not in ds.column_names
        ]
        if absent:
            skipped[clf.name] = absent
        else:
            classifiers.append(clf)

    return metrics, missing_metrics, classifiers, skipped

def _rank_subsets(subset_metrics: dict, subsets: list, stat: str = "auroc") -> list:
    """Rank subsets by a metric, best first, dropping subsets without a value.

    Args:
        subset_metrics: Mapping of Subset -> ClassifierWrapper.
        subsets: Subsets to rank.
        stat: Metric key to rank on.

    Returns:
        List of (Subset, value) tuples sorted from best to worst.
    """
    scored = []
    for sub in subsets:
        wrapper = subset_metrics.get(sub)
        if wrapper is None:
            continue
        value = wrapper.values.get(stat)
        if value is None or value != value:
            continue
        scored.append((sub, float(value)))
    return sorted(scored, key=lambda item: item[1], reverse=True)

def _build_run_configuration(eval_config, run_info: dict) -> list[str]:
    """Render the configuration block describing how the script was run.

    Mirrors the parameter dumps gen.py and filter.py write, so a dataset card
    always states the settings that produced it.

    Args:
        eval_config: AnalysisConfig instance.
        run_info: Run metadata collected by :func:`main` and
            :func:`_build_readme` (dataset name, row counts, config paths).

    Returns:
        List of markdown lines.
    """
    classes = eval_config.fixed_classes
    if classes is not None:
        column_roles = ", ".join(
            f"`{col}` ({'AI' if is_ai else 'Human'})"
            for col, is_ai in zip(eval_config.base_columns, classes)
        )
    else:
        column_roles = (
            ", ".join(f"`{col}`" for col in eval_config.base_columns)
            + f" (classed by `{eval_config.auto_class_column}`, AI label "
            f"`{eval_config.ai_label}`)"
        )

    if eval_config.filter_conditions:
        conditions = f" {eval_config.filter_type} ".join(
            f"`{c.column} {c.operator} {c.value}`" for c in eval_config.filter_conditions
        )
    else:
        conditions = "None"

    if run_info["bin_column"]:
        bins = f"{run_info['bin_count']} quantile bins of `{run_info['bin_column']}`"
    else:
        bins = "None"

    lines = [
        f"- Dataset: `{run_info['dataset']}`",
        f"- Globals Config: `{run_info['globals_config']}`",
        f"- Analysis Config: `{run_info['analysis_config']}`",
        f"- Rows Loaded: {run_info['rows_loaded']:,}",
        f"- Filter Conditions: {conditions}",
        f"- Rows Analyzed (after filtering): {run_info['rows_analyzed']:,}",
        f"- Evaluation / Validation Rows: {run_info['rows_test']:,} / {run_info['rows_val']:,}"
        f" (validation_size = {eval_config.validation_size})",
        f"- Base Columns: {column_roles}",
        f"- Distance Metrics: {', '.join(f'`{m}`' for m in run_info['distance_metrics']) or 'None'}",
        f"- Distance Metrics Skipped (not in this dataset):"
        f" {', '.join(f'`{m}`' for m in run_info['missing_metrics']) or 'None'}",
        f"- Threshold Types: score = `{eval_config.threshold_type_score}`,"
        f" bin = `{eval_config.threshold_type_bin}`",
        f"- Manual Thresholds: score = {eval_config.manual_threshold_score},"
        f" bin = {eval_config.manual_threshold_bin}",
        f"- Bins: {bins}",
        f"- Prompt Subsets: {len(run_info['unique_prompts'])}"
        f" ({', '.join(run_info['unique_prompts']) or 'None'})",
        f"- Generator Configs: {len(run_info['unique_models'])}"
        f" ({', '.join(run_info['unique_models']) or 'None'})",
        "- Classifiers:",
    ]
    for clf in eval_config.classifiers:
        missing = run_info["skipped_classifiers"].get(clf.name)
        if missing:
            # Naming the absent columns is the whole point: it says which
            # statistics stage has not been run for this dataset.
            lines.append(
                f"    - ~~**{clf.name}**~~ - SKIPPED, the dataset has no"
                f" `{'`, `'.join(missing)}` column(s)"
            )
            continue
        lines.append(
            f"    - **{clf.name}** - columns `*{clf.suffix}`, direction"
            f" `{clf.direction}`, threshold kind `{clf.threshold_kind}`"
            f" ({_threshold_description(eval_config, clf)})"
        )
    if not eval_config.classifiers:
        lines.append("    - None configured")
    return lines

def _build_intro(clf_data: dict, overall: Subset, prompt_subsets: list,
                 bin_subsets: list, model_subsets: list, dist_wrappers: list,
                 run_info: dict) -> str:
    """Write the data-derived paragraph describing what this report contains.

    Args:
        clf_data: Per-classifier results (see :func:`_build_readme`).
        overall: The "Overall" Subset.
        prompt_subsets: Prompt-type subsets.
        bin_subsets: Quantile-bin subsets.
        model_subsets: Generator model/genconfig subsets.
        dist_wrappers: StatWrappers over the distance metrics.
        run_info: Run metadata.

    Returns:
        A markdown paragraph.
    """
    n_clf = len(clf_data)
    sentences = [
        f"This readme computes the detection report for `{run_info['dataset']}`:"
        f" {run_info['rows_analyzed']:,} human/AI text pairs"
        f" from {len(run_info['unique_models'])} generator configuration(s)"
        f" and {len(run_info['unique_prompts'])} prompt type(s), summarised"
        f" univariately, correlated against each other, and used to score"
        f" {n_clf} classifier(s) as AI-text detectors."
    ]

    overall_metrics = {
        name: data["subsets"][overall].values for name, data in clf_data.items()
    }
    ranked = sorted(
        (
            (metrics.get("auroc"), name)
            for name, metrics in overall_metrics.items()
            if metrics.get("auroc") is not None and metrics.get("auroc") == metrics.get("auroc")
        ),
        reverse=True,
    )
    if len(ranked) >= 2:
        (best_auroc, best), (worst_auroc, worst) = ranked[0], ranked[-1]
        sentences.append(
            f"Over the whole evaluation split **{best}** separates the two"
            f" classes best (AUROC {_fmt(best_auroc)}, catching"
            f" {_fmt(overall_metrics[best].get('tpr'), '.2%')} of AI rows at"
            f" {_fmt(overall_metrics[best].get('fpr'), '.2%')} false positives),"
            f" while **{worst}** is weakest (AUROC {_fmt(worst_auroc)})."
        )
    elif ranked:
        auroc, name = ranked[0]
        sentences.append(
            f"Over the whole evaluation split **{name}** scores AUROC"
            f" {_fmt(auroc)}, catching {_fmt(overall_metrics[name].get('tpr'), '.2%')}"
            f" of AI rows at {_fmt(overall_metrics[name].get('fpr'), '.2%')} false"
            f" positives at its pinned threshold."
        )

    if dist_wrappers:
        sentences.append(
            f"{len(dist_wrappers)} pairwise distance metric(s)"
            f" ({', '.join(f'`{w.name}`' for w in dist_wrappers)}) measure how far"
            f" each AI response moved from its human original; they are profiled"
            f" here and correlated against every classifier score."
        )

    if clf_data:
        first_name = next(iter(clf_data))
        first_subsets = clf_data[first_name]["subsets"]
        breakdown = (
            f"Every classifier is then broken down over"
            f" {len(prompt_subsets)} prompt subset(s)"
        )
        if bin_subsets:
            breakdown += f" and {len(bin_subsets)} `{run_info['bin_column']}` bin(s)"
        breakdown += "."
        sentences.append(breakdown)

        prompt_ranking = _rank_subsets(first_subsets, prompt_subsets)
        if len(prompt_ranking) >= 2:
            (easy_sub, easy_auroc), (hard_sub, hard_auroc) = prompt_ranking[0], prompt_ranking[-1]
            sentences.append(
                f"For **{first_name}**, the easiest subset is"
                f" {easy_sub.name.removeprefix('Prompt: ')} (AUROC {_fmt(easy_auroc)})"
                f" and the hardest {hard_sub.name.removeprefix('Prompt: ')}"
                f" (AUROC {_fmt(hard_auroc)})."
            )
        bin_ranking = _rank_subsets(first_subsets, bin_subsets)
        if len(bin_ranking) >= 2:
            sentences.append(
                f"Its detection is strongest on the"
                f" `{run_info['bin_column']}` bin"
                f" {bin_ranking[0][0].name.removeprefix('Bin: ')}"
                f" (AUROC {_fmt(bin_ranking[0][1])}) and weakest on"
                f" {bin_ranking[-1][0].name.removeprefix('Bin: ')}"
                f" (AUROC {_fmt(bin_ranking[-1][1])}), which is where the"
                f" dataset's remaining headroom is."
            )
        if model_subsets:
            sentences.append(
                f"The final section reports **{first_name}** across the"
                f" {len(model_subsets)} generator configuration(s) that wrote"
                f" the AI side of the corpus."
            )

    return " ".join(sentences)

def _build_readme(result_ds: Dataset, eval_config, run_info: dict, unique_prompts: list,
                  unique_mg_strs: list, bin_labels: list) -> tuple[str, dict[str, bytes], dict]:
    """Build the evaluation README, its charts, and the machine-readable summary.

    Args:
        result_ds: Evaluation Dataset (already filtered, with the
            ``prompt_type`` / ``model_genconfig`` / ``stat_bin`` label columns
            added by :func:`main`).
        eval_config: AnalysisConfig instance.
        run_info: Run metadata (dataset name, config paths, row counts).
        unique_prompts: Unique prompt type strings (empty if unavailable).
        unique_mg_strs: Unique model/genconfig strings (empty if unavailable).
        bin_labels: Ordered quantile bin labels (empty if unavailable).

    Returns:
        Tuple of (readme_markdown_string, dict_of_filename_to_bytes,
        summary_stats_dict). The summary stats dict is the machine-readable
        JSON spec of all classifier results (see :func:`_build_summary_stats`).
    """
    distance_metrics, missing_metrics, classifiers, skipped_classifiers = select_available(
        result_ds, eval_config
    )
    for metric in missing_metrics:
        print(f"Notice: distance metric '{metric}' is not in the dataset; skipping it.")
    for name, columns in skipped_classifiers.items():
        print(f"Notice: classifier '{name}' needs missing columns {columns}; skipping it.")

    skip_val = all(_threshold_settings(eval_config, c)[0] is not None for c in classifiers)

    val_split = None if skip_val else eval_config.validation_size
    test_ds = result_ds
    val_ds = result_ds
    if val_split is not None and val_split > 0:
        splits = result_ds.train_test_split(test_size=val_split, seed=42)
        test_ds = splits["train"]
        val_ds = splits["test"]

    test_ds = ColumnCache(test_ds)
    val_ds = ColumnCache(val_ds)

    overall = Subset("Overall", None)
    prompt_subsets = [
        Subset(f"Prompt: {p}", _column_equals_mask("prompt_type", p)) for p in unique_prompts
    ]
    bin_subsets = [
        Subset(f"Bin: {b}", _column_equals_mask("stat_bin", b)) for b in bin_labels
    ]
    model_subsets = [
        Subset(f"Model: {m}", _column_equals_mask("model_genconfig", m)) for m in unique_mg_strs
    ]

    charts: dict[str, bytes] = {}

    # ---------------------------------------------------------------- stats
    # "Statistics of interest" = every column the report does arithmetic on:
    # the distance metrics plus each classifier's score column per base column.
    dist_wrappers = [StatWrapper(test_ds, m, name=m) for m in distance_metrics]

    clf_stat_wrappers = []
    for clf in classifiers:
        for i, base_col in enumerate(eval_config.base_columns):
            col = f"{base_col}{clf.suffix}"
            if eval_config.fixed_classes is not None:
                role = "AI" if eval_config.fixed_classes[i] else "Human"
            else:
                role = base_col
            clf_stat_wrappers.append(StatWrapper(test_ds, col, name=f"{clf.name} ({role})"))

    all_variables = dist_wrappers + clf_stat_wrappers

    # ---------------------------------------------------------- classifiers
    clf_data: dict[str, dict] = {}
    for index, clf in enumerate(classifiers):
        flip = (clf.direction == "lower_is_ai")
        manual_threshold, threshold_type = _threshold_settings(eval_config, clf)

        if manual_threshold is not None:
            tw = StaticThresholdWrapper(manual_threshold, flip_class=flip, name=clf.name)
        else:
            val_arrays, val_classes, val_names = _extract_classifier_data(val_ds, eval_config, clf, None)
            tw = ThresholdWrapper(
                val_arrays, val_classes, threshold_type, flip_class=flip,
                name=clf.name, column_names=val_names,
            )
            charts[f"SWEEP_{_safe_name(clf.name)}.png"] = tw.render_sweep_plot()

        # Only the first classifier gets the per-generator breakdown; that
        # section is the hardcoded "full report" slot.
        clf_subsets = [overall] + prompt_subsets + bin_subsets
        if index == 0:
            clf_subsets = clf_subsets + model_subsets

        subset_metrics = {}
        for sub in clf_subsets:
            arrays, classes, _ = _extract_classifier_data(test_ds, eval_config, clf, sub.mask_fn)
            subset_metrics[sub] = ClassifierWrapper(
                arrays, classes, tw, name=f"{clf.name} ({sub.name})"
            )

        clf_data[clf.name] = {"tw": tw, "subsets": subset_metrics, "config": clf}

    def classifier_histogram(clf, sub: Subset, filename: str, title: str) -> str:
        """Render one Human/AI overlaid histogram of a classifier's scores.

        Args:
            clf: ClassifierConfig instance.
            sub: Subset to restrict the rows to.
            filename: Chart filename to register in *charts*.
            title: Figure title.

        Returns:
            The markdown image embed for the rendered chart.
        """
        arrays, classes, names = _extract_classifier_data(test_ds, eval_config, clf, sub.mask_fn)
        charts[filename] = get_histogram(_class_series(arrays, classes, names), title=title)
        return f"![{title}]({filename})"

    def distance_histograms_by_subset(subsets: list, prefix: str, group: str) -> list[str]:
        """Render one histogram per distance metric, overlaying the subsets.

        Args:
            subsets: Subsets to overlay.
            prefix: Chart filename prefix.
            group: Name of the subset family, used in titles.

        Returns:
            List of markdown image embeds (empty when there is nothing to draw).
        """
        embeds = []
        for wrapper in dist_wrappers:
            filename = f"{prefix}_{_safe_name(wrapper.name)}.png"
            # Distances do not depend on the classifier, so the same chart is
            # embedded in every classifier's section; render it only once.
            if filename not in charts:
                series = [
                    (_extract(test_ds, wrapper.column, sub.mask_fn), sub.name)
                    for sub in subsets
                ]
                charts[filename] = get_histogram(series, title=f"{wrapper.name} by {group}")
            embeds.append(f"![{wrapper.name} by {group}]({filename})")
        return embeds

    # ------------------------------------------------------------- sections
    sections: list[tuple[str, list[str]]] = []

    univariate: list[str] = []
    if all_variables:
        rows = [{"name": w.name, "cells": [w]} for w in all_variables]
        table, _ = generate_table(rows, UNIVARIATE_COLUMNS, row_header="Statistic")
        univariate.append(
            "Every statistic the report does arithmetic on, over the"
            f" {len(test_ds):,}-row evaluation split. `Invalid/Error` counts rows"
            " whose value is missing or non-finite; those rows are excluded from"
            " the other columns."
        )
        univariate.append("")
        univariate.append(table.rstrip())
    else:
        univariate.append("No statistics of interest were found in this dataset.")
    sections.append(("Univariate Analysis", univariate))

    correlation: list[str] = []
    if all_variables:
        charts["CORRELATIONS.png"] = generate_pearson_heatmap(
            [(w.arr, w.name) for w in all_variables], title="Correlations"
        )
        correlation.append(
            "Pearson correlation between every statistic of interest, computed"
            " over the rows where both statistics are present."
        )
        correlation.append("")
        correlation.append("![CORRELATIONS](CORRELATIONS.png)")
    else:
        correlation.append("No statistics of interest were found in this dataset.")
    sections.append(("Correlation Heatmap", correlation))

    distances: list[str] = []
    if dist_wrappers:
        distances.append(
            "Distribution of each pairwise distance between a human original and"
            " its AI rewrite, over the whole evaluation split."
        )
        distances.append("")
        for wrapper in dist_wrappers:
            filename = f"DIST_HIST_{_safe_name(wrapper.name)}.png"
            charts[filename] = get_histogram(
                [(wrapper.arr, wrapper.name)], title=f"Distance: {wrapper.name}"
            )
            distances.append(f"![Distance: {wrapper.name}]({filename})")
    else:
        distances.append("No distance metrics were configured or found.")
    sections.append(("Histogram, Distances", distances))

    classification: list[str] = []
    if classifiers:
        classification.append(
            "Human and AI score distributions for each classifier, overlaid, over"
            " the whole evaluation split."
        )
        classification.append("")
        for clf in classifiers:
            classification.append(
                classifier_histogram(
                    clf, overall, f"CLF_HIST_{_safe_name(clf.name)}.png",
                    f"Classifier: {clf.name}",
                )
            )
    else:
        classification.append("No classifiers were configured.")
    sections.append(("Histogram, Classification", classification))

    comparison: list[str] = []
    if classifiers:
        rows = [
            {
                "name": clf.name,
                "cells": [clf_data[clf.name]["subsets"][overall], clf_data[clf.name]["tw"]],
            }
            for clf in classifiers
        ]
        table, _ = generate_table(
            rows, COMPARISON_COLUMNS, row_header="Classifier",
            emoji_config={"mode": "single", "wrapper_idx": 0, "stat": "auroc", "higher_is_better": True},
        )
        comparison.append(
            "AUROC is threshold-free; TPR, FPR, accuracy and F1 are measured at"
            " each classifier's own pinned threshold (shown in the first column)."
            " Rows a classifier produced no usable score for are excluded from"
            " its metrics and counted in the univariate table's `Invalid/Error`"
            " column."
        )
        comparison.append("")
        comparison.append(table.rstrip())
        comparison.append("")
        comparison.append("✔️ marks the best AUROC, ❗ the worst.")
    else:
        comparison.append("No classifiers were configured.")
    sections.append(("Classifiers Comparison Table", comparison))

    thresholds: list[str] = []
    swept = [c for c in classifiers if _threshold_settings(eval_config, c)[0] is None]
    if swept:
        thresholds.append(
            f"Accuracy against threshold on the {len(val_ds):,}-row validation"
            " split, with each candidate threshold type marked. The type named"
            " in the run configuration is the one pinned for the tables above."
        )
        thresholds.append("")
        for clf in swept:
            safe_clf = _safe_name(clf.name)
            thresholds.append(f"**{clf.name}** ({_threshold_description(eval_config, clf)})\n")
            thresholds.append(f"![SWEEP_{safe_clf}](SWEEP_{safe_clf}.png)\n")
    else:
        thresholds.append(
            "Every classifier has a manual threshold, so no validation sweep was run."
        )
    sections.append(("Classifier Thresholds", thresholds))

    for clf in classifiers:
        subset_metrics = clf_data[clf.name]["subsets"]
        body: list[str] = [
            f"Threshold {_threshold_description(eval_config, clf)}"
            f" = {_fmt(clf_data[clf.name]['tw'].threshold_value)}; scores read from"
            f" `*{clf.suffix}` ({clf.direction})."
        ]

        body.append(f"\n### {clf.name}: By Prompt Subset")
        prompt_rows = [{"name": overall.name, "cells": [subset_metrics[overall]]}]
        prompt_rows += [
            {"name": sub.name.removeprefix("Prompt: "), "cells": [subset_metrics[sub]]}
            for sub in prompt_subsets
        ]
        table, _ = generate_table(
            prompt_rows, SUBSET_COLUMNS, row_header="Prompt Subset",
            emoji_config={"mode": "single", "wrapper_idx": 0, "stat": "auroc", "skip_names": {"Overall"}},
        )
        body.append(table.rstrip())

        body.append(f"\n### {clf.name}: By Bin")
        if bin_subsets:
            bin_rows = [{"name": overall.name, "cells": [subset_metrics[overall]]}]
            bin_rows += [
                {"name": sub.name.removeprefix("Bin: "), "cells": [subset_metrics[sub]]}
                for sub in bin_subsets
            ]
            table, _ = generate_table(
                bin_rows, SUBSET_COLUMNS, row_header=f"Bin ({run_info['bin_column']})",
                emoji_config={"mode": "single", "wrapper_idx": 0, "stat": "auroc", "skip_names": {"Overall"}},
            )
            body.append(table.rstrip())
        else:
            body.append(
                "No bin column was available, so this breakdown was skipped."
                " Set `bin_column` in the analysis config to enable it."
            )

        body.append(f"\n### {clf.name}: Score Histograms per Prompt Subset")
        if prompt_subsets:
            for sub in prompt_subsets:
                body.append(
                    classifier_histogram(
                        clf, sub,
                        f"CLF_HIST_{_safe_name(clf.name)}_{sub.safe}.png",
                        f"{clf.name}: {sub.name}",
                    )
                )
        else:
            body.append("No prompt metadata was found, so there are no prompt subsets.")

        if dist_wrappers and prompt_subsets:
            body.append(f"\n### {clf.name}: Distance Histograms per Prompt Subset")
            body.extend(distance_histograms_by_subset(prompt_subsets, "DIST_BY_PROMPT", "Prompt Subset"))

        sections.append((f"Classifier Report: {clf.name}", body))

    manual: list[str] = []
    if classifiers:
        first = classifiers[0]
        subset_metrics = clf_data[first.name]["subsets"]
        manual.append(
            f"This section is hardcoded: it reports the *first* configured"
            f" classifier (**{first.name}**) over the generator model/sampling"
            f" configurations that produced the AI side of each pair."
        )
        if model_subsets:
            rows = [{"name": overall.name, "cells": [subset_metrics[overall]]}]
            rows += [
                {"name": sub.name.removeprefix("Model: "), "cells": [subset_metrics[sub]]}
                for sub in model_subsets
            ]
            table, _ = generate_table(
                rows, SUBSET_COLUMNS, row_header="Generator Config",
                emoji_config={"mode": "single", "wrapper_idx": 0, "stat": "auroc", "skip_names": {"Overall"}},
            )
            manual.append("")
            manual.append(table.rstrip())

            manual.append("\n### Score Histograms per Generator Config")
            for sub in model_subsets:
                manual.append(
                    classifier_histogram(
                        first, sub,
                        f"CLF_HIST_{_safe_name(first.name)}_{sub.safe}.png",
                        f"{first.name}: {sub.name}",
                    )
                )

            if dist_wrappers:
                manual.append("\n### Distance Histograms per Generator Config")
                manual.extend(
                    distance_histograms_by_subset(model_subsets, "DIST_BY_MODEL", "Generator Config")
                )
        else:
            manual.append(
                "\nNo generator model/genconfig metadata was found in this dataset,"
                " so there is nothing to break down here."
            )
    else:
        manual.append("No classifiers were configured.")
    sections.append(("Manually Specified Full Report", manual))

    # -------------------------------------------------------------- assembly
    run_info = {
        **run_info,
        "rows_test": len(test_ds),
        "rows_val": 0 if skip_val else len(val_ds),
        "distance_metrics": [w.name for w in dist_wrappers],
        "missing_metrics": missing_metrics,
        "skipped_classifiers": skipped_classifiers,
        "unique_prompts": unique_prompts,
        "unique_models": unique_mg_strs,
        "bin_count": len(bin_labels),
    }

    body_lines: list[str] = []
    for title, body in sections:
        body_lines.append(f"## {title}")
        body_lines.extend(body)
        body_lines.append("")
    body_md = "\n".join(body_lines)

    lines = ["# Auto-Generated FastDetector Dataset", ""]
    lines.extend(_build_run_configuration(eval_config, run_info))
    lines.append("")
    lines.append(
        _build_intro(
            clf_data, overall, prompt_subsets, bin_subsets,
            model_subsets, dist_wrappers, run_info,
        )
    )
    lines.append("")
    # NOT "Table of Contents": the Hugging Face card renderer silently strips a
    # section whose heading is exactly that (it is boilerplate in the Hub's own
    # dataset-card template), heading and list both, so the readme rendered on
    # the Hub had no contents at all. Verified against the rendered pages of
    # squad and glue, which lose theirs the same way, while cards heading the
    # same list "Contents" keep it. Renaming this back will delete it again.
    lines.append("## Contents")
    lines.extend(_build_table_of_contents(body_md))
    lines.append("")
    lines.append(body_md)

    summary_stats = _build_summary_stats(clf_data, overall, prompt_subsets, bin_subsets, model_subsets)
    return "\n".join(lines), charts, summary_stats


def _build_summary_stats(clf_data: dict, overall: Subset, prompt_subsets: list,
                         bin_subsets: list, model_subsets: list) -> dict:
    """Assemble the machine-readable JSON spec of all classifier results.

    The spec is keyed by classifier name so it works for any configured
    classifier set (unlike the old fixed "score"/"bin" EditLens schema)::

        {"overall": {clf: metrics},
         "prompts": {prompt: {clf: metrics}},
         "bins": {bin: {clf: metrics}},
         "models": {model: {clf: metrics}},
         "thresholds": {clf: threshold_values}}

    where ``metrics`` is the scalar dict from
    :func:`fastdetector.visualization.metrics.compute_classifier_metrics`
    (the markdown confusion matrix is dropped). Subset groups a classifier was
    not evaluated over are simply absent, which is why the per-generator
    entries only carry the first classifier.

    Args:
        clf_data: Per-classifier dict with 'tw' (threshold wrapper) and
            'subsets' (Subset -> ClassifierWrapper) entries.
        overall: The "Overall" Subset.
        prompt_subsets: The per-prompt Subset objects.
        bin_subsets: The per-bin Subset objects.
        model_subsets: The per-generator-config Subset objects.

    Returns:
        The summary stats dict (JSON-serializable with NumpyEncoder).
    """
    def scalar_metrics(cw) -> dict:
        return {k: v for k, v in cw.metrics.items() if k != "confusion_matrix"}

    groups = [
        ("prompts", prompt_subsets, "Prompt: "),
        ("bins", bin_subsets, "Bin: "),
        ("models", model_subsets, "Model: "),
    ]

    summary: dict = {"overall": {}, "prompts": {}, "bins": {}, "models": {}, "thresholds": {}}
    for clf_name, data in clf_data.items():
        summary["overall"][clf_name] = scalar_metrics(data['subsets'][overall])
        summary["thresholds"][clf_name] = dict(data['tw'].values)
        for key, subsets, prefix in groups:
            for sub in subsets:
                wrapper = data['subsets'].get(sub)
                if wrapper is None:
                    continue
                summary[key].setdefault(sub.name.removeprefix(prefix), {})[clf_name] = (
                    scalar_metrics(wrapper)
                )
    return summary

def main() -> None:
    """Run analysis and evaluation reporting pipeline from configuration.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals-config", type=str, default="config/globals.toml")
    parser.add_argument("--analysis-config", type=str, default="config/analysis.toml")
    args = parser.parse_args()

    globals_config, eval_config = load_config_pair(args.globals_config, args.analysis_config, AnalysisConfig)

    target_dataset = globals_config.resolve_output_dataset(globals_config.stat_suffix)
    print(f"Loading all shards for dataset {target_dataset}...")
    result_ds = load_dataset_all_shards(target_dataset, split="train")
    rows_loaded = len(result_ds)

    if eval_config.filter_conditions:
        print("Applying filters...")
        result_ds = apply_filter_conditions(result_ds, eval_config.filter_conditions, eval_config.filter_type)

    pts, has_prompts = extract_prompt_types(result_ds, eval_config.prompt_metadata_column)
    mg_strs, has_mg = extract_model_genconfig(result_ds, eval_config.model_metadata_column)

    if has_prompts: result_ds = result_ds.add_column("prompt_type", pts)
    if has_mg: result_ds = result_ds.add_column("model_genconfig", mg_strs)

    unique_prompts = sorted(set(pts)) if has_prompts else []
    unique_mgs = sorted(set(mg_strs)) if has_mg else []

    # Bins default to the first available distance metric: "how far did the
    # rewrite move" is the axis detection difficulty actually varies along.
    bin_column = eval_config.bin_column
    if bin_column is None:
        bin_column = next((m for m in eval_config.distance_metrics if m in result_ds.column_names), None)

    bin_labels: list[str] = []
    if bin_column is not None and bin_column in result_ds.column_names:
        print(f"Binning rows into {eval_config.num_bins} quantile bins of '{bin_column}'...")
        row_bins, bin_labels = compute_quantile_bins(
            np.array(result_ds[bin_column], dtype=float), eval_config.num_bins, bin_column
        )
        result_ds = result_ds.add_column("stat_bin", row_bins)
    elif bin_column is not None:
        print(f"Notice: bin column '{bin_column}' is not in the dataset; skipping the bin breakdown.")
        bin_column = None

    run_info = {
        "dataset": target_dataset,
        "globals_config": args.globals_config,
        "analysis_config": args.analysis_config,
        "rows_loaded": rows_loaded,
        "rows_analyzed": len(result_ds),
        "bin_column": bin_column,
    }

    print("Generating visualizer README and charts...")
    readme_md, charts, summary_stats = _build_readme(
        result_ds, eval_config, run_info, unique_prompts, unique_mgs, bin_labels
    )

    # Machine-readable spec of the results, consumed by compare_summary.py.
    charts["summary_stats.json"] = json.dumps(summary_stats, indent=2, cls=NumpyEncoder).encode("utf-8")

    print("Uploading README, charts, and summary_stats.json to Hub...")
    upload_readme(target_dataset, files=charts, readme_content=readme_md)
    print("Done!")

if __name__ == "__main__":
    main()
