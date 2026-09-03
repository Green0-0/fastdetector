from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Iterator, Optional
import argparse
import json
import re

import numpy as np
from datasets import Dataset

from fastdetector.frontend.toml_config import AnalysisConfig, ClassifierConfig
from fastdetector.frontend.toml_loader import load_config_pair
from fastdetector.utils import apply_filter_conditions, load_dataset_all_shards, upload_readme
from fastdetector.visualization import metrics, plotting


# --------------------------------------------------------------------------
# Dataset preparation
# --------------------------------------------------------------------------

def extract_prompt_types(ds: Dataset, column: str) -> tuple[np.ndarray, bool]:
    """Label each row with the prompt type that produced it.

    Args:
        ds: The dataset.
        column: Name of the prompt metadata column.

    Returns:
        Tuple of (per-row labels, whether the metadata was there at all).
    """
    if not (column and column in ds.column_names):
        return np.array(["Unknown"] * len(ds)), False
    return np.array([str(p["metadata"].get("PROMPT_TYPE", "Unknown"))
                     if p and isinstance(p.get("metadata"), dict) else "Unknown"
                     for p in ds[column]]), True


def extract_model_genconfig(ds: Dataset, column: str) -> tuple[np.ndarray, bool]:
    """Label each row with the generator model and temperature that wrote it.

    Args:
        ds: The dataset.
        column: Name of the generator model column.

    Returns:
        Tuple of (per-row labels, whether the metadata was there at all).

    Raises:
        ValueError: If only one of the two source columns is present, which
            means the dataset was assembled wrongly and guessing would mislabel
            every row.
    """
    wanted = (column, "generation_params")
    found = [name for name in wanted if name in ds.column_names]
    if not found:
        return np.array(["Unknown"] * len(ds)), False
    if len(found) < len(wanted):
        raise ValueError("Missing columns for model/genconfig extraction.")

    labels = []
    for model, raw in zip(ds[column], ds["generation_params"]):
        params = raw if isinstance(raw, dict) else (json.loads(raw) if raw else None)
        temperature = params.get("temperature", "Unknown") if params is not None else "Unknown"
        labels.append(f"{str(model).split('/')[-1] if model else 'Unknown'} (Temp: {temperature})")
    return np.array(labels), True


def select_available(ds: Dataset, cfg: AnalysisConfig) -> tuple[list, list, list, dict]:
    """Split the configured metrics and classifiers into present and missing.

    A config that names every statistic the pipeline can compute is the useful
    default, but any given dataset has only the ones whose stats stage was
    actually run. Rather than crashing on the first absent column, the report
    evaluates what is there and states what it skipped, so a stage that has not
    been run is visible instead of silently unreported.

    Args:
        ds: The dataset.
        cfg: Analysis configuration.

    Returns:
        Tuple of (distance metrics, missing distance metrics, classifiers,
        {skipped classifier name: the columns it needed and did not find}).
    """
    present = set(ds.column_names)
    classifiers, skipped = [], {}
    for clf in cfg.classifiers:
        absent = [c for c in (f"{b}{clf.suffix}" for b in cfg.base_columns) if c not in present]
        if absent:
            skipped[clf.name] = absent
        else:
            classifiers.append(clf)
    return ([m for m in cfg.distance_metrics if m in present],
            [m for m in cfg.distance_metrics if m not in present], classifiers, skipped)


# --------------------------------------------------------------------------
# Subsets and scores
# --------------------------------------------------------------------------

@dataclass(eq=False)
class Subset:
    """A named slice of the evaluation rows.

    Attributes:
        group: Subset family ("Prompt" or "Model"), empty for Overall.
        label: The subset's own name within its family.
        mask: Per-row boolean mask, or None to select every row.
    """

    group: str
    label: str
    mask: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        """Qualified display name, e.g. ``"Prompt: rewrite"``."""
        return f"{self.group}: {self.label}" if self.group else self.label

    @property
    def safe(self) -> str:
        """Display name, sanitised for use in a chart filename."""
        return safe_name(self.name)


@dataclass
class SubsetGroups:
    """Every subset family the report breaks its classifiers down over.

    Attributes:
        overall: The whole evaluation split.
        prompts: One subset per prompt type.
        models: One subset per generator model/sampling configuration.
    """

    overall: Subset = field(default_factory=lambda: Subset("", "Overall"))
    prompts: list[Subset] = field(default_factory=list)
    models: list[Subset] = field(default_factory=list)


@dataclass
class Scores:
    """One classifier's scores over a split, flattened across the base columns.

    Every score is one entry, tagged with the row and the base column it came
    from, which makes both subsetting and per-column plotting plain indexing.

    Attributes:
        values: Score of each (row, base column) pair that has a usable score.
        is_ai: Whether each score came from the AI side of its pair.
        rows: Row each score came from, for restricting to a subset.
        source: Index into *sources* of the column each score came from.
        sources: (is_ai, column name) per source column, human columns first.
    """

    values: np.ndarray
    is_ai: np.ndarray
    rows: np.ndarray
    source: np.ndarray
    sources: list[tuple]

    def subset(self, mask: Optional[np.ndarray]) -> "Scores":
        """Restrict to the rows *mask* selects.

        Args:
            mask: Per-row boolean mask, or None to keep everything.

        Returns:
            The restricted Scores (self, when there is no mask).
        """
        if mask is None:
            return self
        keep = mask[self.rows]
        return Scores(self.values[keep], self.is_ai[keep], self.rows[keep], self.source[keep], self.sources)

    def by_source(self) -> Iterator[tuple]:
        """Split back into the columns the scores were read from.

        A source column holds one class throughout, so *is_ai* comes back as the
        column's declared class rather than a per-score array.

        Yields:
            (values, is_ai, column name) per source column, human columns first.
        """
        for index, (is_ai, name) in enumerate(self.sources):
            yield self.values[self.source == index], is_ai, name

    def series(self) -> list[tuple]:
        """Label each source column Human/AI for an overlaid histogram.

        Returns:
            (values, label) pairs, one per source column.
        """
        return [(values, name if name.endswith(f"({side})") else f"{side} ({name})")
                for values, is_ai, name in self.by_source() for side in ["AI" if is_ai else "Human"]]


def read_scores(read: Callable, cfg: AnalysisConfig, suffix: str,
                auto_ai: Optional[np.ndarray] = None) -> Scores:
    """Collect one classifier's scores from every base column.

    Scores that are missing or non-finite are dropped here rather than allowed
    to NaN out the classifier's whole sweep and AUROC; the univariate table
    counts them under ``Invalid/Error``.

    Args:
        read: Column reader over the split.
        cfg: Analysis configuration.
        suffix: The classifier's column suffix.
        auto_ai: Per-row AI flags when classes come from a column instead of
            from ``fixed_classes``; each base column then contributes both a
            human and an AI series.

    Returns:
        The flattened Scores.
    """
    parts = []
    for index, base in enumerate(cfg.base_columns):
        column, values = f"{base}{suffix}", read(f"{base}{suffix}", float)
        usable = np.isfinite(values)
        if auto_ai is None:
            parts.append((bool(cfg.fixed_classes[index]), column, values, usable))
        else:
            parts += [(is_ai, f"{column} ({side})", values, usable & (auto_ai == is_ai))
                      for is_ai, side in ((False, "Human"), (True, "AI"))]

    parts.sort(key=lambda part: part[0])  # Human columns first, as the legends expect.
    kept = [np.flatnonzero(usable) for *_, usable in parts]
    return Scores(
        values=np.concatenate([values[rows] for (_, _, values, _), rows in zip(parts, kept)]),
        is_ai=np.concatenate([np.full(rows.size, is_ai, bool) for (is_ai, *_), rows in zip(parts, kept)]),
        rows=np.concatenate(kept),
        source=np.concatenate([np.full(rows.size, index) for index, rows in enumerate(kept)]),
        sources=[(is_ai, column) for is_ai, column, _, _ in parts])


# --------------------------------------------------------------------------
# Classifier evaluation
# --------------------------------------------------------------------------

@dataclass
class Run:
    """One configured classifier, as evaluated over every subset.

    Attributes:
        config: The ClassifierConfig this run came from.
        scores: Its scores over the evaluation split, for the histograms.
        threshold: Decision threshold its metrics were measured at.
        values: Threshold columns merged into the comparison table.
        subsets: Subset -> metric dict.
        sweep_chart: The validation sweep plot, or None for a manual threshold.
    """

    config: ClassifierConfig
    scores: Scores
    threshold: float
    values: dict
    subsets: dict
    sweep_chart: Optional[bytes] = None


def threshold_description(clf: ClassifierConfig) -> str:
    """Describe, in prose, how a classifier's decision threshold was chosen.

    Args:
        clf: Classifier configuration.

    Returns:
        Human-readable description.
    """
    if clf.manual_threshold is not None:
        return f"pinned manually at {clf.manual_threshold:g}"
    return f"swept for `{clf.threshold_type}` on the validation split"


def evaluate(clf: ClassifierConfig, test: Scores, val: Optional[Scores],
             subsets: list[Subset]) -> Run:
    """Pin a classifier's threshold, then score it over every subset.

    Args:
        clf: Classifier configuration.
        test: The classifier's scores over the evaluation split.
        val: Its scores over the validation split, swept for the threshold.
            Unused, and so allowed to be None, when the threshold is pinned.
        subsets: Subsets to break the evaluation split down over.

    Returns:
        The completed Run.
    """
    flip, manual = clf.direction == "lower_is_ai", clf.manual_threshold

    if manual is not None:
        threshold, values, chart = manual, {"threshold": manual}, None
    else:
        points = metrics.operating_points(val.values, val.is_ai, flip)
        threshold = points[clf.threshold_type].threshold
        values = {"threshold": threshold, "optimal_accuracy": points["accuracy"].accuracy}
        # The chart wants an even axis every source column can share, which is
        # not the exact grid the threshold above was pinned on.
        thresholds, accuracy = metrics.sweep(val.values, val.is_ai, flip)
        chart = plotting.sweep_plot(
            thresholds,
            [(metrics.sweep(scores, np.full(scores.size, is_ai), flip, thresholds)[1], name)
             for scores, is_ai, name in val.by_source() if scores.size],
            accuracy, {name: point.threshold for name, point in points.items()},
            f"Threshold Sweep: {clf.name}")

    scored = {sub: metrics.classifier_metrics(part.values, part.is_ai, threshold, flip)
              for sub in subsets for part in [test.subset(sub.mask)]}
    return Run(clf, test, threshold, values, scored, chart)


# --------------------------------------------------------------------------
# Markdown helpers
# --------------------------------------------------------------------------

def safe_name(name: str) -> str:
    """Sanitise a display name for use in a chart filename.

    Args:
        name: Raw display name.

    Returns:
        Uppercase alphanumeric string with single underscores.
    """
    return re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9]", "_", name)).strip("_").upper()


def fmt(value, spec: str = ".4f") -> str:
    """Format a metric for prose, rendering a missing or NaN value as ``n/a``.

    Args:
        value: Numeric value or None.
        spec: Format spec applied to valid values.

    Returns:
        Formatted string.
    """
    return "n/a" if value is None or value != value else format(value, spec)


def subset_table(run: Run, overall: Subset, subsets: list[Subset], row_header: str) -> str:
    """Render one classifier's metrics across a family of subsets.

    Args:
        run: The evaluated classifier.
        overall: Overall subset, listed first and never marked best or worst.
        subsets: Subsets to tabulate, in order.
        row_header: Header label for the first column.

    Returns:
        Markdown table string.
    """
    rows = [{"name": overall.name, "values": run.subsets[overall]}]
    rows += [{"name": sub.label, "values": run.subsets[sub]} for sub in subsets]
    return plotting.table(rows, ["n", "auroc", "tpr", "fpr", "accuracy", "f1"], row_header,
                          mark_key="auroc", skip_marks={overall.name})


def build_contents(body: str, base_level: int = 2) -> list[str]:
    """Build the nested contents list from the headings *body* actually has.

    Reading the headings back out of the rendered markdown (rather than listing
    them a second time by hand) is what keeps every entry a working link: a
    section that is renamed, added or skipped changes its own entry. Repeated
    headings get the ``-1``, ``-2`` ... anchor suffixes markdown renderers
    assign them, in document order.

    Args:
        body: The rendered markdown body.
        base_level: Heading level to number. Its immediate children are
            indented beneath it.

    Returns:
        Markdown list items: headings at *base_level* numbered at the top
        level, with their immediate child headings indented beneath them.
    """
    seen: dict[str, int] = {}
    entries, number = [], 0
    headings = rf"^({'#' * base_level}|{'#' * (base_level + 1)})[ \t]+(.+?)[ \t]*$"
    for match in re.finditer(headings, body, re.M):
        title = match[2]
        slug = re.sub(r"[\s_]+", "-", re.sub(r"[^\w\s-]", "", title.lower()).strip())
        seen[slug] = seen.get(slug, 0) + 1
        anchor = slug if seen[slug] == 1 else f"{slug}-{seen[slug] - 1}"
        number += len(match[1]) == base_level
        entries.append(f"{number}. [{title}](#{anchor})" if len(match[1]) == base_level
                       else f"    - [{title}](#{anchor})")
    return entries


def sorted_metric_rows(rows: list[dict], key: str = "auroc") -> list[dict]:
    """Sort table rows by a metric, highest first and missing values last."""
    def order(row: dict) -> tuple:
        value = row["values"].get(key)
        valid = value is not None and value == value
        return (not valid, -float(value) if valid else 0.0, row["name"])

    return sorted(rows, key=order)


def averaged_subset_rows(runs: dict[str, Run], subsets: list[Subset]) -> list[dict]:
    """Average classifier metrics for each corpus subset."""
    columns = ("auroc", "tpr", "fpr", "accuracy", "f1")
    rows = []
    for subset in subsets:
        values = {}
        for column in columns:
            observed = np.asarray([run.subsets[subset][column] for run in runs.values()], dtype=float)
            finite = observed[np.isfinite(observed)]
            values[f"average_{column}"] = float(np.mean(finite)) if finite.size else float("nan")
        rows.append({"name": subset.name, "values": values})
    return sorted_metric_rows(rows, key="average_auroc")


def performance_table(run: Run, groups: SubsetGroups) -> str:
    """Render one classifier over the overall, prompt, and generator subsets."""
    rows = [{"name": groups.overall.name, "values": run.subsets[groups.overall]}]
    rows += [{"name": sub.name, "values": run.subsets[sub]}
             for sub in [*groups.prompts, *groups.models]]
    return plotting.table(rows, ["n", "auroc", "tpr", "fpr", "accuracy", "f1"], "Subset",
                          mark_key="auroc", skip_marks={groups.overall.name})


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------

def main() -> None:
    """Load the statistics dataset, build its report, and upload it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals-config", type=str, default="config/globals.toml")
    parser.add_argument("--analysis-config", type=str, default="config/analysis.toml")
    args = parser.parse_args()
    globals_config, cfg = load_config_pair(args.globals_config, args.analysis_config, AnalysisConfig)
    if cfg.fixed_classes is None and cfg.auto_class_column is None:
        raise ValueError("Must provide fixed_classes or auto_class_column")
    known = set(metrics.THRESHOLD_TYPES)
    named = {c.threshold_type for c in cfg.classifiers}
    if not named <= known:
        raise ValueError(f"Unknown threshold type(s) {sorted(named - known)}; pick from {sorted(known)}")

    dataset = globals_config.resolve_dataset(globals_config.stat_dataset)
    print(f"Loading all shards for dataset {dataset}...")
    ds = load_dataset_all_shards(dataset, split="train")
    if cfg.filter_conditions:
        print("Applying filters...")
        ds = apply_filter_conditions(ds, cfg.filter_conditions, cfg.filter_type)
    rows_analyzed = len(ds)

    prompts, has_prompts = extract_prompt_types(ds, cfg.prompt_metadata_column)
    models, has_models = extract_model_genconfig(ds, cfg.model_metadata_column)
    if has_prompts: ds = ds.add_column("prompt_type", prompts)
    if has_models: ds = ds.add_column("model_genconfig", models)
    unique_prompts = sorted(set(prompts)) if has_prompts else []
    unique_models = sorted(set(models)) if has_models else []

    print("Generating visualizer README and charts...")
    distance_metrics, missing_metrics, classifiers, skipped = select_available(ds, cfg)
    for metric in missing_metrics:
        print(f"Notice: distance metric '{metric}' is not in the dataset; skipping it.")
    for name, columns in skipped.items():
        print(f"Notice: classifier '{name}' needs missing columns {columns}; skipping it.")

    # The validation split exists only to sweep thresholds on, so it is skipped
    # entirely when every classifier already has a manual threshold.
    swept = [clf for clf in classifiers if clf.manual_threshold is None]
    test_ds, val_ds = ds, ds
    if swept and cfg.validation_size > 0:
        split = ds.train_test_split(test_size=cfg.validation_size, seed=42)
        test_ds, val_ds = split["train"], split["test"]

    # Each column is materialised as an array at most once per split.
    test = lru_cache(maxsize=None)(lambda name, dtype=None: np.asarray(test_ds[name], dtype=dtype))
    val = lru_cache(maxsize=None)(lambda name, dtype=None: np.asarray(val_ds[name], dtype=dtype))

    groups = SubsetGroups(
        prompts=[Subset("Prompt", p, test("prompt_type") == p) for p in unique_prompts],
        models=[Subset("Model", m, test("model_genconfig") == m) for m in unique_models])

    auto = None if cfg.fixed_classes is not None else cfg.auto_class_column
    test_ai = None if auto is None else test(auto) == cfg.ai_label
    val_ai = None if auto is None or not swept else val(auto) == cfg.ai_label
    runs = {}
    for clf in classifiers:
        subsets = [groups.overall, *groups.prompts, *groups.models]
        # A pinned classifier never looks at the validation split, so it does
        # not pay to flatten it - which with every threshold pinned is the whole
        # dataset a second time.
        val_scores = read_scores(val, cfg, clf.suffix, val_ai) if clf in swept else None
        runs[clf.name] = evaluate(clf, read_scores(test, cfg, clf.suffix, test_ai),
                                  val_scores, subsets)

    distances = [(name, test(name, float)) for name in distance_metrics]
    statistics = distances + [
        (f"{clf.name} ({base if cfg.fixed_classes is None else 'AI' if cfg.fixed_classes[i] else 'Human'})",
         test(f"{base}{clf.suffix}", float))
        for clf in classifiers for i, base in enumerate(cfg.base_columns)]

    # ----------------------------------------------------------------- charts
    # Every chart is rendered once, here, and embedded by filename below.
    charts: dict[str, bytes] = {}
    if statistics:
        charts["CORRELATIONS.png"] = plotting.heatmap(
            metrics.correlations([values for _, values in statistics]),
            [name for name, _ in statistics], "Correlations")
    for name, values in distances:
        charts[f"DIST_HIST_{safe_name(name)}.png"] = plotting.histogram([(values, name)], f"Distance: {name}")
    for clf in swept:
        charts[f"SWEEP_{safe_name(clf.name)}.png"] = runs[clf.name].sweep_chart
    for clf in classifiers:
        scores = runs[clf.name].scores
        charts[f"CLF_HIST_{safe_name(clf.name)}.png"] = plotting.histogram(scores.series(),
                                                                          f"Classifier: {clf.name}")
        for sub in [*groups.prompts, *groups.models]:
            charts[f"CLF_HIST_{safe_name(clf.name)}_{sub.safe}.png"] = plotting.histogram(
                scores.subset(sub.mask).series(), f"{clf.name}: {sub.name}")
    # Distances are properties of the corpus, so each family is rendered once
    # and shared by the overview and appendix.
    for family, prefix, group in ((groups.prompts, "DIST_BY_PROMPT", "Prompt Subset"),
                                  (groups.models, "DIST_BY_MODEL", "Generator Config")):
        for name, values in (distances if family else []):
            charts[f"{prefix}_{safe_name(name)}.png"] = plotting.histogram(
                [(values[sub.mask], sub.name) for sub in family], f"{name} by {group}")

    # ---------------------------------------------------------- configuration
    if cfg.fixed_classes is not None:
        roles = ", ".join(f"`{col}` ({'AI' if is_ai else 'Human'})"
                          for col, is_ai in zip(cfg.base_columns, cfg.fixed_classes))
    else:
        roles = (", ".join(f"`{col}`" for col in cfg.base_columns)
                 + f" (classed by `{cfg.auto_class_column}`, AI label `{cfg.ai_label}`)")

    conditions = f" {cfg.filter_type} ".join(
        f"`{c.column} {c.operator} {c.value}`" for c in cfg.filter_conditions) or "None"
    classifier_names = ", ".join(
        f"{clf.name} [skipped]" if clf.name in skipped else clf.name for clf in cfg.classifiers) or "None"

    provenance = f"""- Dataset: `{dataset}`
- Globals Config: `{args.globals_config}`
- Analysis Config: `{args.analysis_config}`
- Rows: {rows_analyzed:,}"""

    evaluation_config = f"""- Prompt Subsets: {len(unique_prompts)} ({', '.join(unique_prompts) or 'None'})
- Generator Configs: {len(unique_models)} ({', '.join(unique_models) or 'None'})
- Classifiers: {len(cfg.classifiers)} ({classifier_names})
- Filter Conditions: {conditions}
- Evaluation / Validation Rows: {len(test_ds):,} / {len(val_ds) if swept else 0:,} (validation_size = {cfg.validation_size})
- Base Columns: {roles}"""

    # ----------------------------------------------------- evaluation summary
    overall = {name: run.subsets[groups.overall] for name, run in runs.items()}
    classifier_rows = sorted_metric_rows([
        {"name": name, "values": {**run.subsets[groups.overall], **run.values}}
        for name, run in runs.items()])
    subset_rows = averaged_subset_rows(runs, [*groups.prompts, *groups.models]) if runs else []

    summary = []
    ranked_classifiers = [row for row in classifier_rows
                          if row["values"]["auroc"] == row["values"]["auroc"]]
    if ranked_classifiers:
        best = ranked_classifiers[0]
        summary.append(f"The best classifier was **{best['name']}** with an AUROC of "
                       f"{fmt(best['values']['auroc'])}.")

    def hardest(family: list[Subset]) -> Optional[dict]:
        names = {sub.name for sub in family}
        candidates = [row for row in subset_rows
                      if row["name"] in names
                      and row["values"]["average_tpr"] == row["values"]["average_tpr"]]
        return min(candidates, key=lambda row: (row["values"]["average_tpr"], row["name"])) if candidates else None

    hard_prompt, hard_model = hardest(groups.prompts), hardest(groups.models)
    if hard_prompt and hard_model:
        summary.append(f"The hardest prompt subset was **{hard_prompt['name'].split(': ', 1)[-1]}** "
                       f"with a TPR of {fmt(hard_prompt['values']['average_tpr'])}, and the hardest generator "
                       f"config was **{hard_model['name'].split(': ', 1)[-1]}** with a TPR of "
                       f"{fmt(hard_model['values']['average_tpr'])}.")
    elif hard_prompt:
        summary.append(f"The hardest prompt subset was **{hard_prompt['name'].split(': ', 1)[-1]}** "
                       f"with a TPR of {fmt(hard_prompt['values']['average_tpr'])}.")
    elif hard_model:
        summary.append(f"The hardest generator config was **{hard_model['name'].split(': ', 1)[-1]}** "
                       f"with a TPR of {fmt(hard_model['values']['average_tpr'])}.")

    comparison = "No classifiers were configured or found."
    if classifier_rows:
        comparison = plotting.table(
            classifier_rows, ["threshold", "auroc", "tpr", "fpr", "accuracy", "f1"],
            row_header="Classifier", mark_key="auroc")

    subset_comparison = "No prompt or generator subsets were available to compare."
    if subset_rows:
        subset_comparison = plotting.table(
            subset_rows, ["average_auroc", "average_tpr", "average_fpr", "average_accuracy",
                          "average_f1"], row_header="Subset", mark_key="average_auroc")

    # ------------------------------------------------------------- the report
    univariate = correlation = "No statistics of interest were found in this dataset."
    if statistics:
        univariate_table = plotting.table(
            [{"name": name, "values": metrics.describe(values)} for name, values in statistics],
            ["n", "mean", "median", "std", "min", "max", "invalid"], row_header="Statistic")
        univariate = f"""Every statistic the report does arithmetic on, over the {len(test_ds):,}-row evaluation split. `Invalid` counts rows whose value is missing or non-finite; those rows are excluded from the other columns.

{univariate_table}"""
        correlation = """Pearson correlation between every statistic of interest, computed over the rows where both statistics are present.

![CORRELATIONS](CORRELATIONS.png)"""

    def distance_embeds(prefix: str, label: str, family: list[Subset]) -> str:
        if not distances:
            return "No distance metrics were configured or found."
        if not family and prefix != "DIST_HIST":
            return f"No {label.lower()} metadata was found, so there are no subsets."
        return "\n".join(f"![{name} by {label}]({prefix}_{safe_name(name)}.png)"
                         if prefix != "DIST_HIST" else
                         f"![Distance: {name}](DIST_HIST_{safe_name(name)}.png)"
                         for name, _ in distances)

    interest = []
    available_distances = {name.lower(): name for name, _ in distances}
    for wanted, family, prefix, label in (
            ("jaccard_1", groups.prompts, "DIST_BY_PROMPT", "Prompt Subset"),
            ("cosdist", groups.prompts, "DIST_BY_PROMPT", "Prompt Subset"),
            ("jaccard_1", groups.models, "DIST_BY_MODEL", "Generator Config"),
            ("cosdist", groups.models, "DIST_BY_MODEL", "Generator Config")):
        if wanted in available_distances and family:
            name = available_distances[wanted]
            interest.append(f"![{name.upper()} by {label}]({prefix}_{safe_name(name)}.png)")
    statistics_of_interest = "\n".join(interest) or "No JACCARD_1 or COSDIST subset statistics were available."

    reports = ""
    for clf in classifiers:
        run, safe = runs[clf.name], safe_name(clf.name)
        prompt_hists = ("\n".join(f"![{clf.name}: {sub.name}](CLF_HIST_{safe}_{sub.safe}.png)"
                                  for sub in groups.prompts)
                        or "No prompt metadata was found, so there are no prompt subsets.")
        model_hists = ("\n".join(f"![{clf.name}: {sub.name}](CLF_HIST_{safe}_{sub.safe}.png)"
                                 for sub in groups.models)
                       or "No generator model/genconfig metadata was found, so there are no generator subsets.")
        if clf.manual_threshold is None:
            thresholding = (f"- Direction: `{clf.direction}`\n"
                            f"- Swept for `{clf.threshold_type}` with a found threshold of "
                            f"{fmt(run.threshold)}.\n\n"
                            f"![Threshold Sweep: {clf.name}](SWEEP_{safe}.png)")
        else:
            thresholding = (f"- Direction: `{clf.direction}`\n"
                            f"- Sweeping skipped and fixed at {fmt(run.threshold)}.")
        reports += f"""### Classifier: {clf.name}

#### Performance:

{performance_table(run, groups)}

#### Thresholding:

{thresholding}

#### Classification Histograms:

![Classifier: {clf.name}](CLF_HIST_{safe}.png)

**Per Prompt Subset**

{prompt_hists}

**Per Generator Config Subset**

{model_hists}

"""

    appendix_body = f"""### Univariate Analysis

{univariate}

### Correlation Heatmap

{correlation}

### Distance Histograms

{distance_embeds("DIST_HIST", "Distance", [groups.overall])}

### Distance Histograms per Prompt Subset

{distance_embeds("DIST_BY_PROMPT", "Prompt Subset", groups.prompts)}

### Distance Histograms per Generator Config Subset

{distance_embeds("DIST_BY_MODEL", "Generator Config", groups.models)}

{reports if reports else "No classifiers were configured or found."}
"""
    contents = "\n".join(f"> {line}" if line else ">" for line in
                         ["Table of contents", "", *build_contents(appendix_body, base_level=3)])
    readme = f"""# Auto-Generated FastDetector Dataset

{provenance}

## Evaluation Results

{evaluation_config}

{' '.join(summary)}

{comparison}

Classifier metrics averaged within each prompt and generator subset:

{subset_comparison}

✔️ marks the best AUROC, ❗ the worst.

## Statistics of Interest

{statistics_of_interest}

## Appendix

{contents}

{appendix_body}"""

    print("Uploading README and charts to Hub...")
    upload_readme(dataset, files=charts, readme_content=readme)
    print("Done!")


if __name__ == "__main__":
    main()
