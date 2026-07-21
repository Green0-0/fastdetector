"""AutoVisualizer: compile-then-evaluate README/chart builder.

The user spends the first phase declaring *what* to analyze (binding columns,
declaring IDs, wiring up plots). Nothing runs during this phase. The second
phase is a single :meth:`apply` call that traces the entire description,
deduplicates array extractions, computes everything, and produces the final
output.

Design principles
-----------------
1. **Lazy evaluation** — no computation happens until ``apply()`` is called.
   This allows deduplication of shared array extractions and catches all
   configuration errors (unresolved IDs, missing thresholds, wrong columns)
   at a single well-defined moment.

2. **Template string substitution** — the README is a plain string with
   ``{{ID}}`` placeholders. ``apply()`` replaces every placeholder with a
   scalar value (formatted), a markdown image embed (for plots), or a
   markdown chunk (for tables / pearson sections). Unresolved IDs fail
   loudly.

3. **Library stays generic** — no knowledge of stat vs eval, no hardcoded
   column names, no metric registry. All such knowledge lives in the
   scripts that configure the visualizer.

4. **Wrappers are handles, not values** — ``bind_*`` returns a wrapper object
   on which the user calls ``specify_stats`` to map internal stat names to
   template IDs. The wrapper does not carry computed data until ``apply()``
   runs.
"""

import re
import itertools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from datasets import Dataset

from fastdetector.visualization.metrics import (
    compute_threshold_sweep,
    compute_classifier_metrics,
    FPR_TARGETS,
)
from fastdetector.visualization.plotting import (
    get_histogram,
    get_sweep_plot,
    format_confusion_matrix,
    get_scatterplot,
)


# A mask function receives a dataset split (same format as the full dataset)
# and returns a boolean numpy array of length ``len(split)``.
MaskFn = Callable[[Dataset], np.ndarray]


_TEMPLATE_PATTERN = re.compile(r"\{\{([^}]+)\}\}")


class _Wrapper:
    """Base class for all wrapper types.

    A wrapper is a *declaration* of what to analyze. It does not hold computed
    data until :meth:`_resolve` is called during ``apply()``.
    """

    def __init__(self, viz: "AutoVisualizer", name: str):
        self._viz = viz
        self._name = name
        self._stat_ids: Dict[str, str] = {}  # stat_name -> template_id
        self._values: Dict[str, Any] = {}  # stat_name -> computed value

    @property
    def name(self) -> str:
        return self._name

    def specify_stats(self, **kwargs: Optional[str]) -> "_Wrapper":
        """Map internal stat names to template IDs.

        Every keyword argument's value is either ``None`` (skip) or a template
        ID string. Each specified stat will be computed during ``apply()`` and
        its value registered under the given ID.

        Returns ``self`` for chaining.
        """
        for stat_name, tid in kwargs.items():
            if tid is None:
                continue
            self._stat_ids[stat_name] = tid
            self._viz._register_id(tid, self, self._stat_id_type(stat_name))
        return self

    def _stat_id_type(self, stat_name: str) -> str:
        """Return the template ID type for a stat: "scalar", "plot", or "markdown".

        Subclasses override this to declare which stats produce plots or
        markdown chunks vs scalars.
        """
        return "scalar"

    def _resolve(self, viz: "AutoVisualizer") -> None:
        raise NotImplementedError

    def _get_value_array(self, viz: "AutoVisualizer") -> np.ndarray:
        """Return the raw numeric array for plots (histograms, scatterplots).

        For StatWrapper this is the column itself. For ClassifierStatWrapper
        this is the positive-class scores. Threshold wrappers raise.
        """
        raise TypeError(
            f"{type(self).__name__} cannot be used directly in plots. "
            f"Bind the score column as a StatWrapper instead."
        )


class StatWrapper(_Wrapper):
    """A numeric column bound for univariate analysis (mean, std, min, max)."""

    def __init__(
        self,
        viz: "AutoVisualizer",
        name: str,
        column: str,
        mask_fn: MaskFn,
        split: str = "test",
    ):
        super().__init__(viz, name)
        self._column = column
        self._mask_fn = mask_fn
        self._split = split

    def _resolve(self, viz: "AutoVisualizer") -> None:
        arr = viz._extract(self._column, self._mask_fn, self._split)
        if "mean" in self._stat_ids:
            self._values["mean"] = float(np.mean(arr))
        if "std" in self._stat_ids:
            self._values["std"] = float(np.std(arr))
        if "min" in self._stat_ids:
            self._values["min"] = float(np.min(arr))
        if "max" in self._stat_ids:
            self._values["max"] = float(np.max(arr))

    def _get_value_array(self, viz: "AutoVisualizer") -> np.ndarray:
        return viz._extract(self._column, self._mask_fn, self._split)


class ClassifierStatWrapper(_Wrapper):
    """Score column(s) bound as a classifier, evaluated at a fixed threshold.

    The threshold is looked up at ``apply()`` time via the ``threshold_id``
    — this is the template ID that a threshold wrapper registered through
    ``specify_stats(threshold_value=...)``. Flip polarity and other
    threshold-side properties are inherited from the threshold binding, so
    the user never specifies them twice.
    """

    def __init__(
        self,
        viz: "AutoVisualizer",
        name: str,
        column_names: List[str],
        column_classes: List[bool],
        mask_fn: MaskFn,
        threshold_id: str,
        autoclass_column: Optional[str] = None,
        split: str = "test",
    ):
        super().__init__(viz, name)
        self._column_names = column_names
        self._column_classes = column_classes
        self._mask_fn = mask_fn
        self._threshold_id = threshold_id
        self._autoclass_column = autoclass_column
        self._split = split

    def _resolve(self, viz: "AutoVisualizer") -> None:
        threshold_val, flip_class = viz._get_threshold(self._threshold_id)

        arrays, classes = self._extract_arrays(viz)
        metrics = compute_classifier_metrics(arrays, classes, threshold_val, flip_class)

        for stat_name in self._stat_ids:
            if stat_name == "confusion_matrix":
                self._values["confusion_matrix"] = format_confusion_matrix(
                    metrics["TP"], metrics["FP"], metrics["TN"], metrics["FN"],
                    f"Confusion Matrix: {self._name}",
                )
            elif stat_name in metrics:
                self._values[stat_name] = metrics[stat_name]
            else:
                raise ValueError(
                    f"ClassifierStatWrapper does not support stat '{stat_name}'. "
                    f"Available: acc, f1, auroc, tpr, fnr, fpr, tnr, precision, "
                    f"recall, confusion_matrix."
                )

    def _stat_id_type(self, stat_name: str) -> str:
        if stat_name == "confusion_matrix":
            return "markdown"
        return "scalar"

    def _extract_arrays(
        self, viz: "AutoVisualizer"
    ) -> Tuple[List[np.ndarray], List[bool]]:
        if self._autoclass_column is not None:
            scores = viz._extract(self._column_names[0], self._mask_fn, self._split)
            labels = viz._extract(self._autoclass_column, self._mask_fn, self._split)
            labels = np.asarray(labels, dtype=bool)
            return [scores[~labels], scores[labels]], [False, True]
        else:
            arrays = [
                viz._extract(c, self._mask_fn, self._split) for c in self._column_names
            ]
            return arrays, list(self._column_classes)

    def _get_value_array(self, viz: "AutoVisualizer") -> np.ndarray:
        """Return positive-class scores for plots."""
        if self._autoclass_column is not None:
            scores = viz._extract(self._column_names[0], self._mask_fn, self._split)
            labels = viz._extract(self._autoclass_column, self._mask_fn, self._split)
            labels = np.asarray(labels, dtype=bool)
            return scores[labels]
        else:
            try:
                pos_idx = self._column_classes.index(True)
            except ValueError:
                raise ValueError(
                    f"ClassifierStatWrapper '{self._name}' has no positive class "
                    f"(column_classes={self._column_classes}). Cannot extract "
                    f"value array for plots."
                )
            return viz._extract(self._column_names[pos_idx], self._mask_fn, self._split)


class ClassifierThresholdStatWrapper(_Wrapper):
    """A classifier threshold determined by sweeping over a split.

    The sweep runs on the validation split (by default). The resulting
    threshold — selected by ``threshold_type`` — is registered under the
    template ID from ``specify_stats(threshold_value=...)`` so that
    :class:`ClassifierStatWrapper` instances can look it up.
    """

    def __init__(
        self,
        viz: "AutoVisualizer",
        name: str,
        column_names: List[str],
        column_classes: List[bool],
        mask_fn: MaskFn,
        threshold_type: str,
        autoclass_column: Optional[str] = None,
        flip_class: bool = False,
        split: str = "val",
    ):
        super().__init__(viz, name)
        self._column_names = column_names
        self._column_classes = column_classes
        self._mask_fn = mask_fn
        self._threshold_type = threshold_type
        self._autoclass_column = autoclass_column
        self._flip_class = flip_class
        self._split = split
        self._sweep_data: Optional[tuple] = None
        self._threshold_dict: dict = {}

    def _resolve(self, viz: "AutoVisualizer") -> None:
        ds = viz._get_split(self._split)
        if ds is None:
            raise ValueError(
                f"Cannot sweep threshold on split '{self._split}': this split "
                f"is not available. Pass val_split to AutoVisualizer or use "
                f"bind_static_threshold."
            )

        arrays, classes = self._extract_arrays(viz)
        threshold_dict, optimal_acc, sweep_data = compute_threshold_sweep(
            arrays, classes, self._flip_class
        )
        self._threshold_dict = threshold_dict
        self._sweep_data = sweep_data

        selected = threshold_dict.get(self._threshold_type)
        if selected is None:
            raise ValueError(
                f"Unknown threshold_type '{self._threshold_type}'. "
                f"Available: {list(threshold_dict.keys())}"
            )

        # Register the threshold value + flip_class under each threshold_value ID
        # so ClassifierStatWrapper can look it up.
        for stat_name, tid in self._stat_ids.items():
            if stat_name == "threshold_value":
                viz._register_threshold(tid, selected, self._flip_class)
                self._values["threshold_value"] = selected
            elif stat_name == "sweep_plot":
                self._values["sweep_plot"] = None  # rendered later
            elif stat_name == "optimal_acc":
                self._values["optimal_acc"] = optimal_acc
            else:
                raise ValueError(
                    f"ClassifierThresholdStatWrapper does not support stat "
                    f"'{stat_name}'. Available: threshold_value, sweep_plot, "
                    f"optimal_acc."
                )

    def _stat_id_type(self, stat_name: str) -> str:
        if stat_name == "sweep_plot":
            return "plot"
        return "scalar"

    def _extract_arrays(
        self, viz: "AutoVisualizer"
    ) -> Tuple[List[np.ndarray], List[bool]]:
        if self._autoclass_column is not None:
            scores = viz._extract(self._column_names[0], self._mask_fn, self._split)
            labels = viz._extract(self._autoclass_column, self._mask_fn, self._split)
            labels = np.asarray(labels, dtype=bool)
            return [scores[~labels], scores[labels]], [False, True]
        else:
            arrays = [
                viz._extract(c, self._mask_fn, self._split) for c in self._column_names
            ]
            return arrays, list(self._column_classes)

    def _render_sweep_plot(self) -> bytes:
        if self._sweep_data is None:
            raise RuntimeError("Sweep data not available; _resolve was not called.")
        thresholds, per_dataset_accs, agg_accs = self._sweep_data
        labels = [c for c in self._column_names] if self._autoclass_column is None else ["Negative", "Positive"]
        return get_sweep_plot(
            thresholds,
            per_dataset_accs,
            agg_accs,
            labels,
            self._threshold_dict,
            f"Threshold Sweep: {self._name}",
        )


class StaticThresholdWrapper(_Wrapper):
    """A manual threshold (no sweep, no plot).

    Produces only ``threshold_value``. Any other stat requested via
    ``specify_stats`` fails loudly — this is the reason StaticThreshold gets
    its own type (so that requesting ``sweep_plot`` on a static threshold
    fails at resolve time rather than silently producing empty output).
    """

    def __init__(
        self,
        viz: "AutoVisualizer",
        name: str,
        value: float,
        flip_class: bool = False,
    ):
        super().__init__(viz, name)
        self._value = value
        self._flip_class = flip_class

    def _resolve(self, viz: "AutoVisualizer") -> None:
        for stat_name, tid in self._stat_ids.items():
            if stat_name == "threshold_value":
                viz._register_threshold(tid, self._value, self._flip_class)
                self._values["threshold_value"] = self._value
            else:
                raise ValueError(
                    f"StaticThresholdWrapper does not support stat '{stat_name}'. "
                    f"Only 'threshold_value' is available."
                )


class AutoVisualizer:
    """Compile-then-evaluate README/chart builder.

    Usage
    -----
    1. Construct: ``viz = AutoVisualizer(ds, val_split=0.1)``
    2. Bind thresholds: ``tw = viz.bind_classifier_threshold(...); tw.specify_stats(threshold_value="T")``
    3. Bind stats and classifier stats: ``sw = viz.bind_stat(...); sw.specify_stats(mean="M")``
    4. Specify plots: ``viz.specify_histogram("H", [sw1, sw2])``
    5. Apply: ``readme, charts, values = viz.apply(template)``
    """

    def __init__(
        self,
        ds: Dataset,
        val_split: Optional[float] = None,
        seed: int = 42,
    ):
        """Initialize the visualizer.

        Args:
            ds: The HuggingFace dataset.
            val_split: If set, proportion of the dataset to use as the
                validation split (for threshold sweeping). The remainder is
                the test split. If ``None``, no split is performed and
                ``bind_classifier_threshold`` will fail.
            seed: Random seed for the val/test split (deterministic regeneration).
        """
        self._ds_full = ds
        self._ds_val: Optional[Dataset] = None
        self._ds_test: Dataset = ds

        if val_split is not None:
            if not (0 < val_split < 1):
                raise ValueError(f"val_split must be in (0, 1), got {val_split}.")
            n = len(ds)
            val_size = max(1, int(n * val_split))
            rng = np.random.RandomState(seed)
            indices = np.arange(n)
            rng.shuffle(indices)
            val_idx = indices[:val_size]
            test_idx = indices[val_size:]
            self._ds_val = ds.select(val_idx.tolist())
            self._ds_test = ds.select(test_idx.tolist())

        self._wrappers: List[_Wrapper] = []
        self._threshold_wrappers: List[ClassifierThresholdStatWrapper] = []
        self._static_wrappers: List[StaticThresholdWrapper] = []

        # Plot/table/pearson specs: id -> spec dict
        self._plot_specs: Dict[str, dict] = {}
        self._table_specs: Dict[str, dict] = {}
        self._pearson_specs: Dict[str, dict] = {}

        # ID registry: template_id -> (wrapper, type)  where type is "scalar"/"plot"/"markdown"
        self._id_registry: Dict[str, Tuple[_Wrapper, str]] = {}

        # Threshold registry: threshold_id -> (value, flip_class)
        self._threshold_registry: Dict[str, Tuple[float, bool]] = {}

        # Extraction cache: (column, id(mask_fn), split) -> np.ndarray
        self._extract_cache: Dict[Tuple[str, int, str], np.ndarray] = {}

    # ------------------------------------------------------------------
    # Split access
    # ------------------------------------------------------------------

    def _get_split(self, split: str) -> Optional[Dataset]:
        if split == "val":
            return self._ds_val
        elif split == "test":
            return self._ds_test
        else:
            raise ValueError(f"Unknown split '{split}'. Use 'val' or 'test'.")

    # ------------------------------------------------------------------
    # Array extraction (with caching)
    # ------------------------------------------------------------------

    def _extract(self, column: str, mask_fn: MaskFn, split: str) -> np.ndarray:
        """Extract a numeric column from a split, filtered by a mask function.

        Results are cached per (column, mask_fn, split) so that multiple
        wrappers sharing the same column + mask + split don't re-extract.
        """
        cache_key = (column, id(mask_fn), split)
        if cache_key in self._extract_cache:
            return self._extract_cache[cache_key]

        ds = self._get_split(split)
        if ds is None:
            raise ValueError(
                f"Split '{split}' is not available. "
                f"Pass val_split to AutoVisualizer if you need a validation split."
            )

        if column not in ds.column_names:
            raise ValueError(
                f"Column '{column}' not found in dataset. "
                f"Available columns: {ds.column_names}"
            )

        mask = mask_fn(ds)
        mask = np.asarray(mask, dtype=bool)
        if len(mask) != len(ds):
            raise ValueError(
                f"Mask function returned {len(mask)} elements but split "
                f"'{split}' has {len(ds)} rows."
            )

        col_data = ds[column]
        try:
            arr = np.array(col_data, dtype=float)[mask]
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Column '{column}' could not be converted to float for "
                f"split '{split}'. AutoVisualizer only handles numeric "
                f"columns. Original error: {e}"
            ) from e

        self._extract_cache[cache_key] = arr
        return arr

    # ------------------------------------------------------------------
    # ID / threshold registration
    # ------------------------------------------------------------------

    def _register_id(self, tid: str, wrapper: _Wrapper, id_type: str) -> None:
        if tid in self._id_registry:
            existing_wrapper, _ = self._id_registry[tid]
            if existing_wrapper is not wrapper:
                raise ValueError(
                    f"Template ID '{tid}' is already registered by another "
                    f"wrapper. Each ID must be unique."
                )
        self._id_registry[tid] = (wrapper, id_type)

    def _register_threshold(self, tid: str, value: float, flip_class: bool) -> None:
        self._threshold_registry[tid] = (value, flip_class)

    def _get_threshold(self, threshold_id: str) -> Tuple[float, bool]:
        if threshold_id not in self._threshold_registry:
            raise ValueError(
                f"Threshold ID '{threshold_id}' was not registered by any "
                f"threshold wrapper. Make sure a ClassifierThresholdStatWrapper "
                f"or StaticThresholdWrapper has called "
                f"specify_stats(threshold_value='{threshold_id}')."
            )
        return self._threshold_registry[threshold_id]

    # ------------------------------------------------------------------
    # Bind functions
    # ------------------------------------------------------------------

    def bind_stat(
        self,
        column: str,
        mask_fn: MaskFn,
        name: Optional[str] = None,
        split: str = "test",
    ) -> StatWrapper:
        """Bind a numeric column for univariate analysis.

        Args:
            column: Column name in the dataset.
            mask_fn: Function(ds_split) -> bool array. Applied to the
                chosen split to select rows.
            name: Display name (for tables, pearson output, etc.). Defaults
                to the column name.
            split: Which split to use ("test" or "val"). Defaults to "test".

        Returns:
            A :class:`StatWrapper`.
        """
        w = StatWrapper(self, name or column, column, mask_fn, split)
        self._wrappers.append(w)
        return w

    def bind_classifier_stat(
        self,
        column_names: Union[str, List[str]],
        column_classes: Union[bool, List[bool]],
        mask_fn: MaskFn,
        threshold_id: str,
        autoclass_column: Optional[str] = None,
        name: Optional[str] = None,
        split: str = "test",
    ) -> ClassifierStatWrapper:
        """Bind score column(s) as a classifier, evaluated at a fixed threshold.

        Args:
            column_names: Either a list of score column names (one per class)
                or a single score column name (when ``autoclass_column`` is set).
            column_classes: Either a list of booleans (True = positive class,
                one per column) or a single boolean (when a single column is
                passed without ``autoclass_column``).
            mask_fn: Function(ds_split) -> bool array.
            threshold_id: Template ID of the threshold to use (must be
                registered by a threshold wrapper's
                ``specify_stats(threshold_value=...)``).
            autoclass_column: If set, a boolean column whose per-row value
                determines the class. Mutually exclusive with passing multiple
                column_names.
            name: Display name. Defaults to joined column names.
            split: Which split to use. Defaults to "test".

        Returns:
            A :class:`ClassifierStatWrapper`.
        """
        if isinstance(column_names, str):
            column_names = [column_names]
        if isinstance(column_classes, bool):
            column_classes = [column_classes]

        if autoclass_column is not None:
            if len(column_names) != 1:
                raise ValueError(
                    f"When autoclass_column is set, exactly one score column "
                    f"must be provided, got {column_names}."
                )
            default_name = column_names[0]
        else:
            if len(column_names) != len(column_classes):
                raise ValueError(
                    f"column_names ({column_names}) and column_classes "
                    f"({column_classes}) must have the same length."
                )
            default_name = " vs ".join(column_names)

        w = ClassifierStatWrapper(
            self, name or default_name, column_names, column_classes,
            mask_fn, threshold_id, autoclass_column, split,
        )
        self._wrappers.append(w)
        return w

    def bind_classifier_threshold(
        self,
        column_names: Union[str, List[str]],
        column_classes: Union[bool, List[bool]],
        mask_fn: MaskFn,
        threshold_type: str,
        autoclass_column: Optional[str] = None,
        flip_class: bool = False,
        name: Optional[str] = None,
        split: str = "val",
    ) -> ClassifierThresholdStatWrapper:
        """Bind a classifier threshold determined by sweeping.

        The sweep runs on the specified split (val by default). The selected
        threshold (by ``threshold_type``) is registered under the template ID
        from ``specify_stats(threshold_value=...)``.

        Args:
            column_names: Score column name(s).
            column_classes: Class label(s) (True = positive).
            mask_fn: Function(ds_split) -> bool array.
            threshold_type: Which threshold to select from the sweep
                (``"accuracy"``, ``"f1"``, ``"fpr_1pct"``, etc.).
            autoclass_column: Optional boolean column for autoclass mode.
            flip_class: If True, values <= threshold are positive.
            name: Display name (used for sweep plot title).
            split: Which split to sweep on. Defaults to "val".

        Returns:
            A :class:`ClassifierThresholdStatWrapper`.
        """
        if isinstance(column_names, str):
            column_names = [column_names]
        if isinstance(column_classes, bool):
            column_classes = [column_classes]

        if autoclass_column is not None:
            if len(column_names) != 1:
                raise ValueError(
                    f"When autoclass_column is set, exactly one score column "
                    f"must be provided, got {column_names}."
                )
            default_name = column_names[0]
        else:
            if len(column_names) != len(column_classes):
                raise ValueError(
                    f"column_names ({column_names}) and column_classes "
                    f"({column_classes}) must have the same length."
                )
            default_name = " vs ".join(column_names)

        valid_types = ["accuracy", "f1"] + list(FPR_TARGETS.keys())
        if threshold_type not in valid_types:
            raise ValueError(
                f"Unknown threshold_type '{threshold_type}'. "
                f"Valid: {valid_types}"
            )

        w = ClassifierThresholdStatWrapper(
            self, name or default_name, column_names, column_classes,
            mask_fn, threshold_type, autoclass_column, flip_class, split,
        )
        self._threshold_wrappers.append(w)
        self._wrappers.append(w)
        return w

    def bind_static_threshold(
        self,
        value: float,
        flip_class: bool = False,
        name: Optional[str] = None,
    ) -> StaticThresholdWrapper:
        """Bind a manual threshold (no sweep, no plot).

        Args:
            value: The threshold value.
            flip_class: If True, values <= threshold are positive.
            name: Internal name (for debugging; not displayed).

        Returns:
            A :class:`StaticThresholdWrapper`.
        """
        w = StaticThresholdWrapper(self, name or f"static_{value}", value, flip_class)
        self._static_wrappers.append(w)
        self._wrappers.append(w)
        return w

    # ------------------------------------------------------------------
    # Plot / table / pearson spec functions
    # ------------------------------------------------------------------

    def specify_histogram(
        self,
        id: str,
        wrappers: List[_Wrapper],
        bins: int = 50,
        figsize: Tuple[int, int] = (8, 5),
        title: Optional[str] = None,
    ) -> None:
        """Register a histogram under a template ID.

        Each wrapper contributes one dataset overlaid on the same histogram.
        The wrapper's ``_get_value_array`` is called to extract the data.

        Args:
            id: Template ID. The output filename will be ``{id}.png``.
            wrappers: List of wrappers (StatWrapper or ClassifierStatWrapper).
            bins: Number of histogram bins.
            figsize: Figure size.
            title: Plot title. Defaults to the ID.
        """
        if not wrappers:
            raise ValueError(f"specify_histogram('{id}'): wrappers list is empty.")
        self._register_id(id, wrappers[0], "plot")
        self._plot_specs[id] = {
            "type": "histogram",
            "wrappers": wrappers,
            "bins": bins,
            "figsize": figsize,
            "title": title or id,
        }

    def specify_scatterplot(
        self,
        id: str,
        x_wrapper: _Wrapper,
        y_wrappers: List[_Wrapper],
        xlabel: str = "X",
        ylabel: str = "Y",
        point_alpha: float = 0.5,
        rolling_mean_window: int = 0,
        figsize: Tuple[int, int] = (8, 5),
        title: Optional[str] = None,
    ) -> None:
        """Register a scatterplot under a template ID.

        Args:
            id: Template ID. Output filename: ``{id}.png``.
            x_wrapper: Wrapper providing the x-axis data.
            y_wrappers: List of wrappers providing y-axis series.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            point_alpha: Scatter point transparency.
            rolling_mean_window: Window for rolling mean overlay (0 = none).
            figsize: Figure size.
            title: Plot title. Defaults to the ID.
        """
        if not y_wrappers:
            raise ValueError(f"specify_scatterplot('{id}'): y_wrappers list is empty.")
        self._register_id(id, x_wrapper, "plot")
        self._plot_specs[id] = {
            "type": "scatterplot",
            "x_wrapper": x_wrapper,
            "y_wrappers": y_wrappers,
            "xlabel": xlabel,
            "ylabel": ylabel,
            "point_alpha": point_alpha,
            "rolling_mean_window": rolling_mean_window,
            "figsize": figsize,
            "title": title or id,
        }

    def specify_pearson(
        self,
        id: str,
        wrappers: List[_Wrapper],
        title: Optional[str] = None,
    ) -> None:
        """Register a Pearson correlation section under a template ID.

        Computes all pairwise Pearson correlations among the wrappers' value
        arrays and formats them as a markdown list.

        Args:
            id: Template ID. The template ``{{id}}`` is replaced with the
                markdown list.
            wrappers: List of wrappers (>= 2 required).
            title: Section heading. Defaults to the ID.
        """
        if len(wrappers) < 2:
            raise ValueError(
                f"specify_pearson('{id}'): need at least 2 wrappers, "
                f"got {len(wrappers)}."
            )
        self._register_id(id, wrappers[0], "markdown")
        self._pearson_specs[id] = {
            "wrappers": wrappers,
            "title": title or id,
        }

    def specify_table(
        self,
        id: str,
        rows: List[dict],
        columns: List[dict],
        emoji_config: Optional[dict] = None,
    ) -> None:
        """Register a markdown table under a template ID.

        Each row provides a list of wrappers (``cells``), and each column
        specifies which wrapper (by index) and which stat to pull.

        Args:
            id: Template ID. ``{{id}}`` is replaced with the markdown table.
            rows: List of ``{"name": str, "cells": [wrapper, ...]}``.
            columns: List of ``{"header": str, "wrapper_idx": int,
                "stat": str, "stat_2": str|None, "format": str}``.
                ``format`` defaults to ``"{value:.4f}"``; when ``stat_2`` is
                set it defaults to ``"{value:.4f} ± {value_2:.4f}"``.
            emoji_config: Optional ``{"mode": "single"|"pct", "pct": float,
                "wrapper_idx": int, "stat": str}``. Marks best (✔️) and
                worst (❗) rows by the specified stat.
        """
        if not rows:
            raise ValueError(f"specify_table('{id}'): rows list is empty.")
        if not columns:
            raise ValueError(f"specify_table('{id}'): columns list is empty.")
        self._register_id(id, rows[0]["cells"][0], "markdown")
        self._table_specs[id] = {
            "rows": rows,
            "columns": columns,
            "emoji_config": emoji_config,
        }

    # ------------------------------------------------------------------
    # Apply: the single evaluation entry point
    # ------------------------------------------------------------------

    def apply(
        self, readme_template: str
    ) -> Tuple[str, Dict[str, bytes], Dict[str, Any]]:
        """Evaluate all bindings and produce the final README + charts.

        Args:
            readme_template: A string with ``{{ID}}`` placeholders. IDs may
                include a format spec: ``{{ID:.4f}}``, ``{{ID:.2%}}``.

        Returns:
            A tuple of ``(readme, charts, values_dict)``:

            - ``readme``: The substituted markdown string.
            - ``charts``: ``{filename: png_bytes}`` for all registered plots.
            - ``values_dict``: ``{id: value}`` for all registered IDs (scalars
              as floats, plots as filenames, markdown as strings).

        Raises:
            ValueError: If any ``{{ID}}`` in the template is unresolved.
        """
        # Phase 1: Resolve static thresholds (just constants, but register them).
        for w in self._static_wrappers:
            w._resolve(self)

        # Phase 2: Resolve swept thresholds (run sweeps on val).
        for w in self._threshold_wrappers:
            w._resolve(self)

        # Phase 3: Resolve remaining wrappers (classifier stats + plain stats).
        for w in self._wrappers:
            if isinstance(w, (ClassifierThresholdStatWrapper, StaticThresholdWrapper)):
                continue
            w._resolve(self)

        # Phase 4: Render plots and collect chart bytes.
        charts: Dict[str, bytes] = {}
        values_dict: Dict[str, Any] = {}

        for tid, spec in self._plot_specs.items():
            png = self._render_plot(spec)
            filename = f"{tid}.png"
            charts[filename] = png
            values_dict[tid] = filename

        # Phase 4b: Render sweep plots from threshold wrappers.
        for w in self._threshold_wrappers:
            if "sweep_plot" in w._stat_ids:
                tid = w._stat_ids["sweep_plot"]
                png = w._render_sweep_plot()
                filename = f"{tid}.png"
                charts[filename] = png
                values_dict[tid] = filename

        # Phase 5: Render pearson sections.
        for tid, spec in self._pearson_specs.items():
            md = self._render_pearson(spec)
            values_dict[tid] = md

        # Phase 6: Render tables.
        for tid, spec in self._table_specs.items():
            md = self._render_table(spec)
            values_dict[tid] = md

        # Phase 7: Collect scalar values from all wrappers.
        # Skip stats that were already handled as plots/markdown above.
        for w in self._wrappers:
            for stat_name, tid in w._stat_ids.items():
                if tid in values_dict:
                    continue  # already set (plot filename or markdown)
                if stat_name in w._values and w._values[stat_name] is not None:
                    values_dict[tid] = w._values[stat_name]

        # Phase 8: Template substitution.
        readme = self._substitute(readme_template, values_dict)

        return readme, charts, values_dict

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _render_plot(self, spec: dict) -> bytes:
        if spec["type"] == "histogram":
            data_lists = [w._get_value_array(self) for w in spec["wrappers"]]
            labels = [w.name for w in spec["wrappers"]]
            return get_histogram(
                data_lists, labels, spec["title"],
                bins=spec["bins"], figsize=spec["figsize"],
            )
        elif spec["type"] == "scatterplot":
            x_data = spec["x_wrapper"]._get_value_array(self)
            y_data_lists = [w._get_value_array(self) for w in spec["y_wrappers"]]
            labels = [w.name for w in spec["y_wrappers"]]
            return get_scatterplot(
                x_data, y_data_lists, labels, spec["title"],
                xlabel=spec["xlabel"], ylabel=spec["ylabel"],
                point_alpha=spec["point_alpha"],
                rolling_mean_window=spec["rolling_mean_window"],
                figsize=spec["figsize"],
            )
        else:
            raise ValueError(f"Unknown plot type: {spec['type']}")

    def _render_pearson(self, spec: dict) -> str:
        wrappers = spec["wrappers"]
        arrays = [w._get_value_array(self) for w in wrappers]
        names = [w.name for w in wrappers]

        md = f"### {spec['title']}\n\n"
        for (i, j) in itertools.combinations(range(len(wrappers)), 2):
            if len(arrays[i]) == 0 or len(arrays[j]) == 0:
                corr = float("nan")
            elif len(arrays[i]) != len(arrays[j]):
                raise ValueError(
                    f"Pearson correlation between '{names[i]}' and '{names[j]}' "
                    f"failed: arrays have different lengths ({len(arrays[i])} "
                    f"vs {len(arrays[j])}). Make sure the wrappers use the same "
                    f"mask and split."
                )
            else:
                corr = float(np.corrcoef(arrays[i], arrays[j])[0, 1])
            md += f"- **{names[i]} vs {names[j]}**: {corr:.4f}\n"
        return md

    def _render_table(self, spec: dict) -> str:
        rows = spec["rows"]
        columns = spec["columns"]
        emoji_config = spec.get("emoji_config")

        # --- Apply emoji markers to row names ---
        row_names = [r["name"] for r in rows]
        if emoji_config:
            widx = emoji_config["wrapper_idx"]
            stat = emoji_config["stat"]
            rank_values = []
            for r in rows:
                w = r["cells"][widx]
                rank_values.append(w._values.get(stat, float("nan")))

            valid = [(i, v) for i, v in enumerate(rank_values) if v == v]  # not NaN
            if valid:
                valid_sorted = sorted(valid, key=lambda x: x[1])
                n = len(valid_sorted)
                if emoji_config["mode"] == "single":
                    best = {valid_sorted[-1][0]}
                    worst = {valid_sorted[0][0]}
                else:  # "pct"
                    n_top = max(1, int(n * emoji_config["pct"]))
                    worst = {idx for idx, _ in valid_sorted[:n_top]}
                    best = {idx for idx, _ in valid_sorted[-n_top:]}
                row_names = [
                    ("✔️ " if i in best else ("❗ " if i in worst else "")) + name
                    for i, name in enumerate(row_names)
                ]

        # --- Build table ---
        header = "| Subset | " + " | ".join(c["header"] for c in columns) + " |\n"
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
                val = wrapper._values.get(stat)
                if val is None:
                    cells.append("-")
                    continue

                stat_2 = col.get("stat_2")
                if stat_2 is not None:
                    val2 = wrapper._values.get(stat_2)
                    if val2 is not None:
                        fmt = col.get("format", "{value:.4f} ± {value_2:.4f}")
                        cells.append(fmt.format(value=val, value_2=val2))
                    else:
                        # Hardcoded fallback: the composite format string
                        # would crash with KeyError on {value_2} if val2 is
                        # missing, so use a simple single-value format.
                        cells.append(f"{val:.4f}")
                else:
                    fmt = col.get("format", "{value:.4f}")
                    cells.append(fmt.format(value=val))
            lines.append(f"| {row_names[i]} | " + " | ".join(cells) + " |")

        return header + sep + "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Template substitution
    # ------------------------------------------------------------------

    def _substitute(
        self, template: str, values_dict: Dict[str, Any]
    ) -> str:
        """Replace ``{{ID}}`` and ``{{ID:format}}`` placeholders in *template*.

        - Plot IDs → ``![ID](ID.png)``
        - Markdown IDs → the markdown string
        - Scalar IDs → formatted value (default ``{:.4f}`` for floats)
        - Unresolved IDs → raise ValueError
        """
        id_types = {tid: t for tid, (_, t) in self._id_registry.items()}

        def replacer(m: re.Match) -> str:
            full = m.group(1).strip()
            if ":" in full:
                id_str, fmt = full.split(":", 1)
                id_str = id_str.strip()
                fmt = fmt.strip()
            else:
                id_str = full
                fmt = None

            if id_str not in values_dict and id_str not in id_types:
                raise ValueError(
                    f"Unresolved template ID: '{id_str}'. "
                    f"Make sure it was registered via bind_* / specify_*."
                )

            id_type = id_types.get(id_str, "scalar")
            value = values_dict.get(id_str)

            if id_type == "plot":
                return f"![{id_str}]({id_str}.png)"
            elif id_type == "markdown":
                if value is None:
                    raise ValueError(
                        f"Template ID '{id_str}' is registered as markdown but "
                        f"was not computed. This usually means the spec was "
                        f"registered but unreachable from the template; either "
                        f"reference it or remove the specify_* call."
                    )
                return str(value)
            else:  # scalar
                if value is None:
                    raise ValueError(
                        f"Template ID '{id_str}' is registered as a scalar but "
                        f"was not computed. Check that the wrapper's _resolve "
                        f"actually populates this stat."
                    )
                if fmt:
                    return format(value, fmt)
                if isinstance(value, float):
                    return f"{value:.4f}"
                return str(value)

        return _TEMPLATE_PATTERN.sub(replacer, template)
