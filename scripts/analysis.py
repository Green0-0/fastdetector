from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Optional
import argparse
import csv
import html
import io
import json
import re

import numpy as np
from datasets import Dataset

from fastdetector.frontend.toml_config import AnalysisConfig, ClassifierConfig
from fastdetector.frontend.toml_loader import load_config_pair
from fastdetector.utils import load_dataset_all_shards, upload_readme
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


def extract_prompt_messages(ds: Dataset, column: str) -> tuple[np.ndarray, bool]:
    """Return the first template message for each row, ignoring later turns."""
    if not (column and column in ds.column_names):
        return np.array(["Unknown"] * len(ds)), False
    messages = []
    for prompt in ds[column]:
        turns = prompt.get("chat_turns", []) if isinstance(prompt, dict) else []
        messages.append(str(turns[0]).strip() if turns else "Unknown")
    return np.asarray(messages), True


def extract_prompt_turns(ds: Dataset, column: str) -> np.ndarray:
    """Count the chat turns of each row's prompt, 0 where it is unknown."""
    if not (column and column in ds.column_names):
        return np.zeros(len(ds), dtype=int)
    return np.asarray([len(prompt.get("chat_turns") or []) if isinstance(prompt, dict) else 0
                       for prompt in ds[column]], dtype=int)


def extract_dimension(ds: Dataset, column: str, aliases: tuple[str, ...]) -> tuple[np.ndarray, bool]:
    """Read a corpus dimension from a direct column, case-insensitively."""
    lookup = {name.lower(): name for name in ds.column_names}
    selected = next((lookup[name.lower()] for name in (column, *aliases)
                     if name and name.lower() in lookup), None)
    if selected is None:
        return np.array(["Unknown"] * len(ds)), False
    return np.asarray(["Unknown" if value is None else str(value) for value in ds[selected]]), True


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
    """A named slice of the full analysis corpus.

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
        overall: The complete corpus.
        prompts: One subset per prompt type.
        models: One subset per generator model/sampling configuration.
    """

    overall: Subset = field(default_factory=lambda: Subset("", "Overall"))
    prompts: list[Subset] = field(default_factory=list)
    models: list[Subset] = field(default_factory=list)


@dataclass
class Scores:
    """One classifier's corpus scores, flattened across the base columns.

    Every usable score is tagged with its row and class for fast subsetting.

    Attributes:
        values: Score of each (row, base column) pair that has a usable score.
        is_ai: Whether each score came from the AI side of its pair.
        rows: Row each score came from, for restricting to a subset.
    """

    values: np.ndarray
    is_ai: np.ndarray
    rows: np.ndarray

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
        return Scores(self.values[keep], self.is_ai[keep], self.rows[keep])


def read_scores(read: Callable, cfg: AnalysisConfig, suffix: str,
                auto_ai: Optional[np.ndarray] = None) -> Scores:
    """Collect one classifier's scores from every base column.

    Scores that are missing or non-finite are dropped here rather than allowed
    to NaN out the classifier's whole sweep and AUROC; the univariate table
    counts them under ``Invalid/Error``.

    Args:
        read: Cached dataset-column reader.
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
        values = read(f"{base}{suffix}", float)
        usable = np.isfinite(values)
        if auto_ai is None:
            parts.append((bool(cfg.fixed_classes[index]), values, usable))
        else:
            parts += [(is_ai, values, usable & (auto_ai == is_ai)) for is_ai in (False, True)]

    parts.sort(key=lambda part: part[0])  # Human scores first.
    kept = [np.flatnonzero(usable) for _, _, usable in parts]
    return Scores(
        values=np.concatenate([values[rows] for (_, values, _), rows in zip(parts, kept)]),
        is_ai=np.concatenate([np.full(rows.size, is_ai, bool) for (is_ai, _, _), rows in zip(parts, kept)]),
        rows=np.concatenate(kept))


# --------------------------------------------------------------------------
# Classifier evaluation
# --------------------------------------------------------------------------

@dataclass
class Run:
    """A classifier's full-corpus scores, operating points, and plots."""

    config: ClassifierConfig
    scores: Scores
    subsets: dict
    sweep_chart: bytes
    points: dict

    @property
    def flip(self) -> bool:
        """Whether a lower score means AI."""
        return self.config.direction == "lower_is_ai"


def evaluate(clf: ClassifierConfig, scores: Scores, subsets: list[Subset]) -> Run:
    """Fit both report operating points on the complete corpus.

    Args:
        clf: Classifier configuration.
        scores: The classifier's scores over the complete analysis corpus.
        subsets: Corpus subsets to compare at the same operating points.

    Returns:
        The completed Run.
    """
    flip = clf.direction == "lower_is_ai"
    points = metrics.operating_points(scores.values, scores.is_ai, flip)
    markers = {
        "1% FPR": points["fpr_1pct"].threshold,
        "0.1% FPR": points["fpr_0_1pct"].threshold,
    }
    thresholds, tpr, fpr = metrics.sweep_rates(scores.values, scores.is_ai, flip)
    chart = plotting.detector_sweep_plot(
        thresholds, tpr, fpr, markers, f"Threshold sweep: {clf.name}")
    scored = {sub: metrics.report_metrics(part.values, part.is_ai, flip, points)
              for sub in subsets for part in [scores.subset(sub.mask)]}
    return Run(clf, scores, scored, chart, points)


# --------------------------------------------------------------------------
# Report helpers
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


def _row_class_means(scores: Scores, row_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Collapse any repeated score sources to one human and AI mean per row."""
    outputs = []
    for wanted in (False, True):
        total, count = np.zeros(row_count), np.zeros(row_count, dtype=int)
        selected = scores.is_ai == wanted
        np.add.at(total, scores.rows[selected], scores.values[selected])
        np.add.at(count, scores.rows[selected], 1)
        outputs.append(np.divide(total, count, out=np.full(row_count, np.nan), where=count > 0))
    return outputs[0], outputs[1]


def _mean(values: np.ndarray) -> float:
    """Mean of the finite entries, NaN (without a warning) when there are none."""
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


# --------------------------------------------------------------------------
# Corpus metadata
# --------------------------------------------------------------------------

@dataclass
class Corpus:
    """Per-row metadata the report breaks the corpus down by.

    Attributes:
        prompts: Prompt category of each row.
        messages: First prompt message of each row.
        turns: Number of chat turns in each row's prompt (0 when unknown).
        models: Generator config of each row.
        topics: Topic of each row.
        formats: Format of each row.
        has_messages: Whether prompt messages were available.
        has_topics: Whether a topic column was found.
        has_formats: Whether a format column was found.
    """

    prompts: np.ndarray
    messages: np.ndarray
    turns: np.ndarray
    models: np.ndarray
    topics: np.ndarray
    formats: np.ndarray
    has_messages: bool = False
    has_topics: bool = False
    has_formats: bool = False


def prompt_boilerplate(messages: np.ndarray) -> frozenset[str]:
    """Find the lines most distinct prompt messages share, such as output-format trailers.

    Args:
        messages: First prompt message of each row.

    Returns:
        Lines found in at least 60% of the distinct messages, when there are
        at least three to compare.
    """
    unique = {message for message in messages if message != "Unknown"}
    if len(unique) < 3:
        return frozenset()
    counts = Counter(line for message in unique for line in {
        line.strip() for line in _strip_document(message).splitlines() if line.strip()})
    return frozenset(line for line, count in counts.items() if count >= 0.6 * len(unique))


def _strip_document(message: str) -> str:
    return re.sub(r"<document>.*?</document>", "", message, flags=re.S)


def prompt_instruction(message: str, boilerplate: frozenset[str] = frozenset()) -> str:
    """Reduce a stored first message to the instruction a reader cares about.

    Args:
        message: The prompt's first chat turn.
        boilerplate: Lines shared by most prompts, dropped unless nothing else is left.

    Returns:
        The instruction on one line, with the document placeholder named.
    """
    lines = [line.strip() for line in _strip_document(message).splitlines() if line.strip()]
    kept = [line for line in lines if line not in boilerplate] or lines
    return re.sub(r"\s+", " ", " ".join(kept)).replace("{{DOC}}", "[source text]").strip()


def _dataset_title(dataset: str) -> str:
    name = dataset.split("/")[-1]
    name = re.sub(r"-(?:val|validation|test)$", "", name, flags=re.I)
    return " ".join(part.upper() if part.lower() in {"cc"} else part.title()
                    for part in name.split("-"))


# --------------------------------------------------------------------------
# HTML building blocks
# --------------------------------------------------------------------------
#
# Hugging Face renders the card without scripts or <style> blocks, so every
# element carries its own inline style and all interactivity is <details>.

INK, MUTED, FAINT = "#1b2733", "#5c6672", "#8a929b"
LINE, HAIR, PAPER, TINT = "#dcd3c3", "#efe9dd", "#fffdf9", "#fbf8f1"
ACCENT = "#e07b39"
MONO = "'IBM Plex Mono',ui-monospace,monospace"
SANS = "'Source Sans Pro',system-ui,sans-serif"
SERIF = "Georgia,'Iowan Old Style','Palatino Linotype',serif"
TH = (f"font-family:{SANS};font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;"
      f"color:{FAINT};padding:8px 10px;border:0;border-bottom:1.5px solid {INK};background:{PAPER};"
      "white-space:nowrap")
TD = f"padding:9px 10px;border:0;border-bottom:1px solid {HAIR}"
NUM = f"{TD};text-align:right;font-family:{MONO};font-size:12.5px;white-space:nowrap"
CHEVRON = ('<span class="motion-safe:transition-transform group-open:rotate-180" '
           f'style="display:inline-block;font-size:15px;color:{FAINT};line-height:1">⌄</span>')
# Tailwind's group-open matches *any* open ancestor, so a chevron inside an
# already-open <details> would flip before its own is opened; nested
# summaries use this static marker instead.
NESTED_MARKER = f'<span style="font-size:12px;color:{ACCENT};font-weight:600;white-space:nowrap">show ▾</span>'

esc = html.escape


def _colour(index: int) -> str:
    return plotting.PALETTE[index % len(plotting.PALETTE)]


def _swatch(colour: str) -> str:
    return (f'<span style="display:inline-block;width:9px;height:9px;border-radius:50%;'
            f'background:{colour};margin-right:6px"></span>')


def _code(text: str, chip: bool = True) -> str:
    background = f"background:{HAIR};padding:2px 6px;border-radius:5px" if chip else "background:transparent;padding:0"
    return (f'<code style="font-family:{MONO};font-size:12px;{background};color:{INK};'
            f'word-break:break-word">{esc(text)}</code>')


def _number(value) -> str:
    return format(int(value), ",d") if isinstance(value, (int, np.integer)) else fmt(value)


def _table(headers: list[tuple[str, str]], rows: list[list[str]], min_width: int = 660) -> str:
    """Render a small reference-styled table.

    Args:
        headers: (label, "left" | "right") per column.
        rows: Pre-rendered ``<td>`` cells per row.
        min_width: Width below which the table scrolls rather than squeezes.

    Returns:
        HTML string.
    """
    head = "".join(f'<th style="text-align:{align};{TH}">{esc(label)}</th>' for label, align in headers)
    body = "".join("<tr>" + "".join(row) + "</tr>" for row in rows)
    return (f'<div style="overflow-x:auto;margin:0;border:1px solid {LINE};border-radius:12px;'
            f'background:{PAPER}"><table style="width:100%;min-width:{min_width}px;border-collapse:collapse;'
            f'font-size:13.5px;margin:0;border:0;display:table"><thead><tr>{head}</tr></thead>'
            f'<tbody>{body}</tbody></table></div>')


def _td(content: str, numeric: bool = False) -> str:
    return f'<td style="{NUM if numeric else TD}">{content}</td>'


def _image_card(dataset: str, filename: str, alt: str, caption: str = "") -> str:
    href = f'https://huggingface.co/datasets/{dataset}/resolve/main/{filename}'
    return (f'<div style="margin:12px 0 0;padding:10px;background:{PAPER};border:1px solid {LINE};'
            f'border-radius:12px"><img src="{esc(filename)}" alt="{esc(alt)}" '
            'style="display:block;width:100%;max-width:100%;height:auto;margin:0 auto;border:0;'
            f'border-radius:6px;background:{PAPER}"/><p style="margin:8px 2px 0;font-size:12.5px;'
            f'color:{FAINT};line-height:1.5;display:flex;justify-content:space-between;gap:12px;'
            f'flex-wrap:wrap"><span>{esc(caption)}</span><a href="{esc(href)}" '
            f'style="color:{ACCENT};text-decoration:none;white-space:nowrap;font-weight:600">'
            'Open full size ↗</a></p></div>')


def _section(number: str, anchor: str, title: str, description: str) -> str:
    return ('<div style="display:flex;align-items:flex-end;margin:54px 0 0"><div style="flex:0 0 44px;'
            f'height:3px;background:{ACCENT}"></div><div style="flex:1;height:1px;background:rgba(27,39,51,.45)"'
            '></div></div><div style="display:flex;align-items:baseline;gap:14px;margin:20px 0 8px">'
            f'<span style="font-family:{MONO};font-size:13px;font-weight:700;color:{ACCENT}">'
            f'{number}</span><h2 id="{anchor}" style="margin:0;padding:0;border:0;font-family:{SERIF};'
            f'font-size:clamp(24px,3.2vw,30px);font-weight:400;letter-spacing:-.01em;line-height:1.2;'
            f'color:{INK}">{esc(title)}</h2></div>'
            f'<p style="margin:0 0 22px;max-width:760px;color:{MUTED};font-size:15px;line-height:1.65">'
            f'{esc(description)}</p>')


def _details(title: str, subtitle: str, body: str, identifier: str = "") -> str:
    id_attr = f' id="{esc(identifier)}"' if identifier else ""
    return (f'<details class="group"{id_attr} style="margin:10px 0;border:1px solid {LINE};'
            f'border-radius:12px;background:{PAPER};overflow:hidden"><summary class="hover:bg-gray-50" '
            'style="cursor:pointer;list-style:none;display:flex;align-items:center;gap:12px;flex-wrap:wrap;'
            f'padding:14px 18px;color:{INK}"><span style="font-weight:650">{esc(title)}</span>'
            f'<span style="font-size:12.5px;color:{FAINT}">{esc(subtitle)}</span>'
            f'<span style="margin-left:auto">{CHEVRON}</span></summary>'
            f'<div style="padding:4px 18px 20px;border-top:1px solid {HAIR}">{body}</div></details>')


def _option(group: str, title: str, subtitle: str, body: str, identifier: str = "",
            lead: str = "", trailing: str = "", is_open: bool = False, nested: bool = False) -> str:
    """One choice in a picker: a row of an exclusive accordion.

    Every option sharing *group* is a ``<details name>`` of the same name, so
    opening one closes whichever was open before and the picker behaves like
    a single-choice selector whose chosen option shows its content below.
    """
    id_attr = f' id="{esc(identifier)}"' if identifier else ""
    return (f'<details name="{esc(group)}" class="group"{id_attr}{" open" if is_open else ""} '
            f'style="margin:0;border-bottom:1px solid {HAIR};background:{PAPER}">'
            '<summary class="hover:bg-gray-50" style="cursor:pointer;list-style:none;display:flex;'
            f'align-items:center;gap:12px;flex-wrap:wrap;padding:11px 14px;color:{INK}">{lead}'
            f'<span style="flex:1 1 220px;font-weight:650">{esc(title)}'
            f'<span style="font-weight:400;font-size:12.5px;color:{FAINT};margin-left:10px">{esc(subtitle)}</span>'
            f'</span>{trailing}{NESTED_MARKER if nested else CHEVRON}</summary>'
            f'<div style="padding:6px 14px 18px;background:{TINT};border-top:1px dashed {LINE}">{body}</div>'
            '</details>')


def _picker(label: str, options: str) -> str:
    return (f'<div style="font-family:{MONO};font-size:11px;font-weight:600;letter-spacing:.14em;'
            f'text-transform:uppercase;color:{FAINT};margin:14px 0 8px">{esc(label)}</div>'
            f'<div style="border:1px solid {LINE};border-radius:12px;background:{PAPER};overflow:hidden">'
            f'{options}</div>')


def _subheading(title: str, top: int = 26) -> str:
    return (f'<h3 style="margin:{top}px 0 10px;padding:0;border:0;font-family:{SANS};font-size:12px;'
            f'font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:{FAINT}">{esc(title)}</h3>')


def _paragraph(text: str, margin: str = "12px 0") -> str:
    """A body paragraph; *text* is trusted HTML."""
    return f'<p style="margin:{margin};color:{MUTED};font-size:14px;line-height:1.65">{text}</p>'


def _muted(text: str) -> str:
    return f'<p style="color:{FAINT};margin:12px 0">{esc(text)}</p>'


# --------------------------------------------------------------------------
# Report sections
# --------------------------------------------------------------------------

def _subset_difficulty(runs: dict[str, Run], subsets: list[Subset]) -> Optional[tuple[Subset, float]]:
    """Find the subset whose best available detector has the lowest TPR."""
    ranked = []
    for subset in subsets:
        values = [run.subsets[subset]["tpr_at_fpr_1pct"] for run in runs.values()]
        finite = [value for value in values if value == value]
        if finite:
            ranked.append((max(finite), subset.label, subset))
    return (min(ranked)[2], min(ranked)[0]) if ranked else None


def _metric_strip(values: dict) -> str:
    """The leaderboard's four headline numbers, right-aligned in one row."""
    return "".join(
        f'<span style="min-width:86px;text-align:right;font-family:{MONO};font-size:12px">'
        f'<small style="display:block;color:{FAINT};font-family:{SANS}">{esc(label)}</small>'
        f'{_number(values[key])}</span>' for key, label in REPORT_COLUMNS)


REPORT_COLUMNS = [("n", "N"), ("auroc", "AUROC"),
                  ("tpr_at_fpr_1pct", "TPR @ 1% FPR"), ("tpr_at_fpr_0_1pct", "TPR @ 0.1% FPR")]


def _leaderboard(dataset: str, rows: list[dict], runs: dict[str, Run], charts: dict,
                 min_distance: str) -> str:
    entries = []
    for rank, row in enumerate(rows, 1):
        name, run = row["name"], runs[row["name"]]
        safe = safe_name(name)
        thresholds = (f'<div style="display:flex;gap:18px;flex-wrap:wrap;margin:12px 0;color:{MUTED};'
                      f'font-size:13px"><span><b>1% FPR threshold</b> '
                      f'{_code(fmt(run.points["fpr_1pct"].threshold))}</span>'
                      f'<span><b>0.1% FPR threshold</b> '
                      f'{_code(fmt(run.points["fpr_0_1pct"].threshold))}</span></div>')
        body = thresholds + _image_card(dataset, f"SWEEP_{safe}.png", f"Threshold sweep: {name}",
                                        "TPR and FPR at every threshold · both FPR operating points marked")
        if f"MINDIST_{safe}.png" in charts:
            body += _image_card(
                dataset, f"MINDIST_{safe}.png", f"{name}: TPR against minimum {min_distance}",
                f"AI rows with {min_distance} below the cutoff are dropped · human rows and "
                "thresholds are unchanged")
        body += _subheading("Score distributions", 24) + "".join([
            _image_card(dataset, f"CLF_HUMAN_{safe}.png", f"{name}: overall human",
                        "Overall · human only"),
            _image_card(dataset, f"CLF_AI_{safe}.png", f"{name}: overall AI",
                        "Overall · AI only"),
            _image_card(dataset, f"CLF_PROMPTS_{safe}.png", f"{name}: prompt subsets",
                        "AI texts · one envelope per prompt subset"),
            _image_card(dataset, f"CLF_MODELS_{safe}.png", f"{name}: generator subsets",
                        "AI texts · one envelope per generator config"),
        ])
        entries.append(
            f'<details class="group" id="classifier-{safe.lower().replace("_", "-")}" '
            f'style="border-bottom:1px solid {HAIR};margin:0"><summary class="hover:bg-gray-50" '
            'style="cursor:pointer;list-style:none;display:flex;align-items:center;gap:12px;flex-wrap:wrap;'
            f'padding:11px 12px;color:{INK}"><span style="width:26px;text-align:right;color:{FAINT};'
            f'font-family:{MONO}">{rank:02d}</span><span style="flex:1 1 220px;'
            f'font-weight:650">{esc(name)}</span>{_metric_strip(row["values"])}{CHEVRON}'
            f'</summary><div style="padding:4px 14px 20px;background:{TINT};border-top:1px dashed {LINE}">'
            f'{body}</div></details>')
    if not entries:
        return "<p>No score-based classifiers were available.</p>"
    return (f'<div style="border:1px solid {LINE};border-radius:12px;background:{PAPER};overflow:hidden">'
            + "".join(entries) + "</div>")


#: Distance measures summarised per prompt in the specific-prompt leaderboard.
PROMPT_DISTANCES = ("cosdist", "jaccard_1")

#: Hardest prompts shown inline per detector; the full ranking is a CSV beside
#: the card, since hundreds of prompts times every detector would bloat it.
PROMPT_TABLE_LIMIT = 12

#: Hugging Face rejects dataset cards past roughly 1 MB (HTTP 413 from its YAML
#: validator), so a larger card fails here rather than mid-upload.
MAX_README_BYTES = 950 * 1024


def prompt_ranking_file(name: str) -> str:
    """Repository path of a detector's full specific-prompt ranking."""
    return f"prompt_rankings/{safe_name(name)}.csv"


def prompt_ranking_csv(rows: list[dict], distance_names: list[str]) -> bytes:
    """Serialise a full specific-prompt ranking, hardest first."""
    keys = ["text", "category", "turns", "rows", "auroc", "tpr_at_fpr_1pct", "tpr_at_fpr_0_1pct",
            "human", "ai", "movement", *distance_names]
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["rank", "prompt", "category", "turns", "rows", "auroc", "tpr_at_1pct_fpr",
                     "tpr_at_0_1pct_fpr", "mean_human_score", "mean_ai_score",
                     "mean_abs_score_movement", *(f"mean_{name}" for name in distance_names)])
    for rank, row in enumerate(rows, 1):
        writer.writerow([rank, *(row.get(key) for key in keys)])
    return buffer.getvalue().encode("utf-8")


def _specific_prompt_rows(run: Run, corpus: Corpus, human: np.ndarray, ai: np.ndarray,
                          distances: list[tuple], boilerplate: frozenset[str]) -> list[dict]:
    """Score every distinct first prompt message, hardest first."""
    categories = sorted(set(corpus.prompts))
    movement = np.abs(ai - human)
    rows = []
    for message in sorted(set(corpus.messages)):
        mask = corpus.messages == message
        part = run.scores.subset(mask)
        labels, counts = np.unique(corpus.prompts[mask], return_counts=True)
        category = str(labels[np.argmax(counts)])
        turns = set(corpus.turns[mask].tolist()) - {0}
        rows.append({
            "text": prompt_instruction(message, boilerplate),
            "category": category, "colour": _colour(categories.index(category)),
            "turns": ("single & two-turn" if len(turns) > 1 else
                      "two-turn" if turns and min(turns) > 1 else "single-turn" if turns else ""),
            "rows": int(mask.sum()),
            **metrics.report_metrics(part.values, part.is_ai, run.flip, run.points),
            "human": _mean(human[mask]), "ai": _mean(ai[mask]), "movement": _mean(movement[mask]),
            **{name: _mean(values[mask]) for name, values in distances if name in PROMPT_DISTANCES},
        })
    return sorted(rows, key=lambda row: (
        np.inf if row["tpr_at_fpr_1pct"] != row["tpr_at_fpr_1pct"] else row["tpr_at_fpr_1pct"],
        row["tpr_at_fpr_0_1pct"], row["text"]))


def _prompt_leaderboard(rows: list[dict], distance_names: list[str]) -> str:
    """Render the ranked prompt table.

    Rows are many (every prompt, for every detector), so cells carry no inline
    style of their own; the table's defaults and ``rules`` do the styling.
    """
    headers = [("Prompt (first message)", "left"), ("Rows", "right"), ("AUROC", "right"),
               ("TPR @ 1% FPR", "right"), ("TPR @ 0.1% FPR", "right"),
               ("Human score", "right"), ("AI score", "right"), ("|AI − human|", "right"),
               *((name, "right") for name in distance_names)]
    head = "".join(f'<th style="text-align:{align};{TH}">{esc(label)}</th>' for label, align in headers)
    body = []
    for row in rows:
        tag = " · ".join(filter(None, [row["category"], row["turns"]]))
        tpr = row["tpr_at_fpr_1pct"]
        bar = ("" if tpr != tpr else
               f'<div style="height:3px;margin-top:5px;background:{HAIR};border-radius:2px">'
               f'<div style="width:{100 * tpr:.1f}%;height:3px;background:{ACCENT};border-radius:2px">'
               '</div></div>')
        body.append(
            f'<tr><td align="left" style="min-width:300px;max-width:520px;white-space:normal;'
            f'line-height:1.45;font-variant-numeric:normal">{esc(row["text"])}'
            f'<div style="margin-top:4px;font-family:{MONO};font-size:11px;color:{FAINT}">'
            f'{_swatch(row["colour"])}{esc(tag)}</div></td>'
            + "".join(f'<td style="text-align:right">{value}</td>' for value in [
                f'{row["rows"]:,}', fmt(row["auroc"]), fmt(tpr) + bar, fmt(row["tpr_at_fpr_0_1pct"]),
                fmt(row["human"]), fmt(row["ai"]), fmt(row["movement"]),
                *(fmt(row.get(name)) for name in distance_names)]) + "</tr>")
    return (f'<div style="overflow:auto;max-height:860px;border:1px solid {LINE};border-radius:12px;'
            f'background:{PAPER}"><table cellpadding="9" cellspacing="0" rules="rows" '
            f'style="width:100%;min-width:1080px;margin:0;border:0;border-color:{HAIR};border-collapse:collapse;'
            f'font-size:12.5px;text-align:right;vertical-align:top;font-variant-numeric:tabular-nums;'
            f'color:{INK}"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def _analytics(dataset: str, rows: list[dict], runs: dict[str, Run], charts: dict,
               rankings: dict[str, list[dict]], distances: list[tuple]) -> str:
    """Two-step picker: choose a detector, then a split, and the split's charts show below."""
    distance_names = [name for name, _ in distances if name in PROMPT_DISTANCES]
    options = []
    for rank, row in enumerate(rows, 1):
        name, run = row["name"], runs[row["name"]]
        safe = safe_name(name)
        slug = safe.lower().replace("_", "-")

        def grid(kind: str, caption: str, prefix: str) -> str:
            filename = f"{prefix}_{kind}_{safe}.png"
            return _image_card(dataset, filename, f"{name}: {caption}", caption) if filename in charts else ""

        topic_format = "".join([
            grid("HUMAN", "Human texts · mean score per topic and format", "ANALYTICS_TOPIC_FORMAT"),
            grid("AI", "AI texts · mean score, same colour scale as the human grid", "ANALYTICS_TOPIC_FORMAT"),
            grid("DISTANCE", "Mean unsigned |AI − human| score per pair", "ANALYTICS_TOPIC_FORMAT"),
        ]) or _muted("No topic and format columns were found.")
        generator_prompt = "".join([
            grid("AI", "AI texts · mean score per generator config and prompt category",
                 "ANALYTICS_GENERATOR_PROMPT"),
            grid("DISTANCE", "Mean unsigned |AI − human| score per pair", "ANALYTICS_GENERATOR_PROMPT"),
        ]) or _muted("No generator or prompt metadata was found.")
        prompt_rows = rankings.get(name, [])
        shown = prompt_rows[:PROMPT_TABLE_LIMIT]
        filename = prompt_ranking_file(name)
        full = (f' Only the {len(shown)} hardest of {len(prompt_rows):,} are shown; the '
                f'<a href="https://huggingface.co/datasets/{esc(dataset)}/blob/main/{esc(filename)}" '
                f'style="color:{ACCENT};font-weight:600">full ranking ↗</a> is a CSV beside this card.'
                if len(prompt_rows) > len(shown) and filename in charts else "")
        specific = (_paragraph(
            "One row per distinct first message (a second turn, where there is one, is ignored), "
            "hardest first: ranked by TPR at the detector's corpus-wide 1% FPR threshold. Score "
            "columns are the detector's mean over the prompt's rows, <b>|AI − human|</b> is the mean "
            "per-pair score movement, and distance columns are the mean distance between each "
            "source and its rewrite." + full, "12px 0") + _prompt_leaderboard(shown, distance_names)
                    if prompt_rows else _muted("No prompt messages were available."))

        group = f"analytics-split-{slug}"
        splits = "".join([
            _option(group, "Topic × format", "human · AI · |AI − human| grids", topic_format,
                    f"analytics-{slug}-topic-format", is_open=True, nested=True),
            _option(group, "Generator × prompt category", "AI · |AI − human| grids", generator_prompt,
                    f"analytics-{slug}-generator-prompt", nested=True),
            _option(group, "Specific prompt", f"{len(prompt_rows):,} prompts · hardest first",
                    specific, f"analytics-{slug}-specific-prompt", nested=True),
        ])
        values = row["values"]
        options.append(_option(
            "analytics-model", name, "", _picker(f"Select a split · {name}", splits),
            f"analytics-{slug}",
            lead=(f'<span style="width:26px;text-align:right;color:{FAINT};font-family:{MONO}">'
                  f'{rank:02d}</span>'),
            trailing=(f'<span style="font-family:{MONO};font-size:12px;color:{MUTED}">'
                      f'AUROC {fmt(values["auroc"])} · TPR@1% {fmt(values["tpr_at_fpr_1pct"])}</span>')))
    if not options:
        return "<p>No models were available to analyse.</p>"
    return _picker("Select a model", "".join(options))


def _distances(dataset: str, distances: list[tuple], groups: SubsetGroups) -> str:
    blocks = []
    for name, _ in distances:
        safe = safe_name(name)
        cards = _image_card(dataset, f"DIST_HIST_{safe}.png", f"{name}: overall", "Overall")
        if groups.prompts:
            cards += _image_card(dataset, f"DIST_BY_PROMPT_{safe}.png", f"{name}: prompt subsets",
                                 "One envelope per prompt subset")
        if groups.models:
            cards += _image_card(dataset, f"DIST_BY_MODEL_{safe}.png", f"{name}: generator configs",
                                 "One envelope per generator config")
        blocks.append(_details(name, "overall · prompt subsets · generator configs", cards))
    return "".join(blocks) or "<p>No distance measures were available.</p>"


def _generation_params(raw) -> dict:
    """Flatten one row's sampling parameters, nested request extras included."""
    params = raw if isinstance(raw, dict) else (json.loads(raw) if raw else {})
    flat = {}
    for key, value in (params or {}).items():
        flat.update(_generation_params(value) if isinstance(value, dict) else {key: value})
    return flat


def _generator_table(ds: Dataset, cfg: AnalysisConfig, groups: SubsetGroups) -> str:
    """Generator configs with their model link, temperature, sampling, and row counts."""
    column = cfg.model_metadata_column
    rows = []
    for index, subset in enumerate(groups.models):
        first = int(np.flatnonzero(subset.mask)[0])
        model = ds[first][column] if column in ds.column_names else None
        params = (_generation_params(ds[first]["generation_params"])
                  if "generation_params" in ds.column_names else {})
        temperature = params.pop("temperature", None)
        sampling = " · ".join(f"{key} {value}" for key, value in params.items()) or "—"
        name = (f'<a href="https://huggingface.co/{esc(str(model))}" style="color:{INK};'
                f'text-decoration:underline;text-decoration-color:{LINE};text-underline-offset:3px">'
                f'{esc(str(model))}</a>' if model and "/" in str(model) else esc(str(model or subset.label)))
        rows.append([
            f'<td style="{TD};font-size:13px">{_swatch(_colour(index))}{name}</td>',
            _td(f"<b>{esc(str(temperature))}</b>" if temperature is not None else "—", numeric=True),
            f'<td style="{TD};font-size:12px;color:{MUTED};line-height:1.5">{esc(sampling)}</td>',
            _td(f"{int(subset.mask.sum()):,}", numeric=True),
            _td(f"{subset.mask.mean():.1%}", numeric=True)])
    total = int(sum(subset.mask.sum() for subset in groups.models))
    rows.append([_td("<b>Total</b>"), _td(""), _td(""), _td(f"<b>{total:,}</b>", numeric=True),
                 _td("<b>100%</b>", numeric=True)])
    return _table([("Generator model", "left"), ("Temp", "right"), ("Sampling", "left"),
                   ("Rows", "right"), ("Share", "right")], rows)


def _prompt_cards(corpus: Corpus, groups: SubsetGroups) -> str:
    """One card per prompt category, listing its instructions and their counts."""
    boilerplate = prompt_boilerplate(corpus.messages)
    cards = []
    for index, subset in enumerate(groups.prompts):
        colour = _colour(index)
        turns = corpus.turns[subset.mask]
        single, multi = int(np.sum(turns == 1)), int(np.sum(turns > 1))
        messages, counts = np.unique(corpus.messages[subset.mask], return_counts=True)
        known = [(message, count) for message, count in zip(messages, counts) if message != "Unknown"]
        shape = (f"{single:,} single · {multi:,} two-turn" if single and multi else
                 "two-turn" if multi else "single-turn" if single else "")
        meta = " · ".join(filter(None, [shape, f"{len(known)} instructions" if known else ""]))
        listing = "".join(
            f'<li style="margin:0 0 6px"><span style="font-family:{MONO};font-size:11px;color:{FAINT}">'
            f'{count:,}×</span> {esc(prompt_instruction(message, boilerplate))}</li>'
            for message, count in sorted(known, key=lambda item: (-item[1], item[0])))
        instructions = (
            f'<details style="margin:auto 0 0;border-top:1px dashed {LINE};padding-top:6px">'
            f'<summary style="cursor:pointer;list-style:none;font-size:12.5px;color:{ACCENT};font-weight:600;'
            f'display:flex;align-items:center;gap:6px">Show the {len(known)} instructions ▾</summary><ul style="margin:8px 0 4px;padding-left:16px;font-size:12.5px;'
            f'line-height:1.5;color:{MUTED}">{listing}</ul></details>') if known else ""
        cards.append(
            f'<div style="background:{PAPER};border:1px solid {LINE};border-top:3px solid {colour};'
            'border-radius:12px;padding:16px 16px 12px;display:flex;flex-direction:column">'
            '<div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px;flex-wrap:wrap">'
            f'<code style="font-family:{MONO};font-size:13px;font-weight:700;background:transparent;padding:0;'
            f'color:{INK}">{_swatch(colour)}{esc(subset.label)}</code><span style="font-family:{MONO};'
            f'font-size:12px;color:{MUTED}">{int(subset.mask.sum()):,} rows</span></div>'
            f'<div style="font-size:11.5px;color:{FAINT};margin:2px 0 10px;font-family:{MONO}">{esc(meta)}</div>'
            f'{instructions}</div>')
    return ('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px">'
            + "".join(cards) + "</div>")


def _frequency_table(labels: np.ndarray, heading: str) -> str:
    values, counts = np.unique(labels.astype(str), return_counts=True)
    total = int(counts.sum())
    return _table([(heading, "left"), ("Rows", "right"), ("Share", "right")], [
        [_td(esc(value)), _td(f"{count:,}", numeric=True), _td(f"{count / total:.1%}", numeric=True)]
        for value, count in sorted(zip(values, counts), key=lambda item: (-item[1], item[0]))], 420)


def _key_values(pairs: list[tuple[str, str]]) -> str:
    """A bordered list of labelled facts; values are trusted HTML."""
    return (f'<div style="border:1px solid {LINE};border-radius:12px;background:{PAPER};overflow:hidden">'
            + "".join(
                f'<div style="{"border-top:1px solid " + HAIR + ";" if index else ""}display:flex;'
                'flex-wrap:wrap;gap:4px 18px;padding:11px 16px"><div style="flex:0 0 110px;'
                f'font-family:{MONO};font-size:11px;letter-spacing:.1em;text-transform:uppercase;'
                f'color:{FAINT};padding-top:2px">{esc(label)}</div><div style="flex:1 1 300px;'
                f'font-size:14px;color:{MUTED}">{value}</div></div>'
                for index, (label, value) in enumerate(pairs)) + "</div>")


def _class_names(cfg: AnalysisConfig) -> dict[str, str]:
    if cfg.fixed_classes is None:
        return {}
    return {base: "AI" if is_ai else "Human" for base, is_ai in zip(cfg.base_columns, cfg.fixed_classes)}


def _composition(ds: Dataset, cfg: AnalysisConfig, groups: SubsetGroups, corpus: Corpus,
                 runs: dict[str, Run], skipped: dict, analysis_config: str) -> str:
    classes = _class_names(cfg)
    pairing = " and ".join(f"{_code(base)} ({classes.get(base, 'class from ' + str(cfg.auto_class_column))})"
                           for base in cfg.base_columns)
    body = _paragraph(f"Every row pairs {pairing}: a human-written source and the text a generator "
                      "model produced from it. Inclusion filtering happens while the dataset is built, "
                      "so this report evaluates the published corpus exactly as it is.", "12px 0 0")
    if groups.models:
        body += _subheading("Generator configs", 20) + _generator_table(ds, cfg, groups)
    if groups.prompts:
        body += _subheading("Prompt subsets", 30) + _paragraph(
            f"Instructions are grouped by their first message. Counts cover all {len(ds):,} rows.",
            "0 0 12px") + _prompt_cards(corpus, groups)
    if corpus.has_topics:
        body += _subheading("Topics", 30) + _frequency_table(corpus.topics, "Topic")
    if corpus.has_formats:
        body += _subheading("Formats", 30) + _frequency_table(corpus.formats, "Format")
    flipped = [name for name, run in runs.items() if run.flip]
    body += _subheading("Evaluation protocol", 30) + _key_values([
        ("Classes", ", ".join(f"{_code(base)} is {label}" for base, label in classes.items())
         + ". Each detector scores both texts of every row." if classes else
         f"Read per row from {_code(str(cfg.auto_class_column))}."),
        ("Thresholds", f"All {len(ds):,} rows are used; there is no sweep/evaluation split. Each "
         "detector's thresholds are placed exactly on its human scores so that 1% and 0.1% of human "
         "texts are called AI."
         + (f" {len(flipped)} detectors call lower scores AI (for example, perplexity)." if flipped else "")),
        ("Metrics", "N (texts scored), AUROC, and TPR at the 1% and 0.1% FPR thresholds. Subsets are "
         "scored at the corpus-wide thresholds rather than refitted."),
        ("Filters", "None. No cosine, soft-ngram, Jaccard, or other analysis-time filtering is applied."),
        ("Skipped", esc(", ".join(f"{name} ({'; '.join(reason)})" for name, reason in skipped.items()))
         or "None"),
        ("Config", _code(analysis_config)),
    ])
    return body


#: Descriptions of the columns every generated corpus shares.
KNOWN_COLUMNS = {
    "response_0": "Generator output for the first turn",
    "response_1": "Generator output for the second turn (empty for single-turn prompts)",
    "generation_params": "JSON string of sampling parameters",
}


def _field(name: str, description: str) -> str:
    return (f'<div style="padding:8px 0;border-top:1px solid {HAIR}">'
            f'<code style="font-family:{MONO};font-size:12px;background:transparent;padding:0;color:{INK};'
            f'word-break:break-word;font-weight:600">{esc(name)}</code><div style="font-size:12.5px;'
            f'color:{FAINT};line-height:1.45;margin-top:2px">{esc(description)}</div></div>')


def _schema(dataset: str, ds: Dataset, cfg: AnalysisConfig, runs: dict[str, Run],
            distances: list[tuple], corpus: Corpus) -> str:
    classes = _class_names(cfg)
    present = set(ds.column_names)
    text = [(base, f"{'Human-written source' if classes.get(base) == 'Human' else 'Generated text'}"
             f" (class: {classes[base]})" if base in classes else "Text scored by the detectors")
            for base in cfg.base_columns]
    text += [(name, KNOWN_COLUMNS[name]) for name in ("response_0", "response_1") if name in present]
    generation = [(name, description) for name, description in (
        (cfg.prompt_metadata_column, "struct: chat_turns (list of strings), use_multiturn, examples, "
                                     "metadata.PROMPT_TYPE"),
        (cfg.model_metadata_column, "Hugging Face id of the generator model"),
        ("generation_params", KNOWN_COLUMNS["generation_params"])) if name in present]
    metadata = [(name, f"{label} of the human source") for name, label, found in (
        (cfg.topic_column, "Topic", corpus.has_topics), (cfg.format_column, "Format", corpus.has_formats))
        if found and name in present]
    alternation = "{" + "|".join(cfg.base_columns) + "}"
    detectors = [(run.config.suffix, name) for name, run in runs.items()]
    groups = [
        ("Text", "#3b6ea8", text),
        ("Generation", "#7a5aa6", generation),
        ("Metadata", "#2a8c82", metadata),
        ("Distances", MUTED, [(", ".join(name for name, _ in distances),
                               f"{len(distances)} float columns measuring how far the rewrite moved "
                               "from the original")] if distances else []),
    ]
    described = ({name for _, _, fields in groups[:3] for name, _ in fields}
                 | {name for name, _ in distances}
                 | {f"{base}{run.config.suffix}" for run in runs.values() for base in cfg.base_columns})
    others = [name for name in ds.column_names if name not in described]
    if others:
        groups.append(("Other columns", FAINT, [(", ".join(others), "Not used by this report")]))
    def card(title: str, colour: str, content: str, wide: bool = False) -> str:
        return (f'<div style="{"grid-column:1/-1;" if wide else ""}background:{PAPER};border:1px solid {LINE};'
                f'border-left:3px solid {colour};border-radius:12px;padding:14px 16px 8px">'
                f'<p style="margin:0 0 6px;font-family:{MONO};font-size:11px;font-weight:600;'
                f'letter-spacing:.14em;text-transform:uppercase;color:{MUTED}">{esc(title)}</p>{content}</div>')

    cards = "".join(card(title, colour, "".join(_field(name, description) for name, description in fields))
                    for title, colour, fields in groups if fields)
    if detectors:
        cards += card("Detector scores", ACCENT, (
            f'<div style="font-size:12.5px;color:{FAINT};margin:0 0 4px">One float column per text: '
            f'{_code(alternation + "<suffix>", chip=False)}</div>'
            '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));column-gap:24px">'
            + "".join(_field(suffix, name) for suffix, name in detectors) + "</div>"), wide=True)
    loading = (
        'from datasets import concatenate_datasets, get_dataset_config_names, load_dataset\n\n'
        f'repo = "{dataset}"\n'
        '# every generator config is its own dataset config\n'
        'ds = concatenate_datasets([\n'
        '    load_dataset(repo, config, split="train")\n'
        '    for config in get_dataset_config_names(repo)\n'
        '])')
    return (_paragraph(f"{len(ds.column_names):,} columns per row. Detector columns come in pairs, one "
                       "per text of the row.", "12px 0")
            + '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px">'
            + cards + "</div>" + _subheading("Load it", 30)
            # A blank line either side lets the fence render as highlighted markdown.
            + f"\n\n```python\n{loading}\n```\n\n")


def _univariate(statistics: list[tuple], ds: Dataset) -> tuple[str, str]:
    """The univariate table and its one-line summary."""
    described = [(name, metrics.describe(values)) for name, values in statistics]
    invalid = sum(summary["invalid"] for _, summary in described)
    rows = [[f'<td style="{TD};white-space:nowrap;font-size:13px">{_code(name, chip=False)}</td>',
             *(_td(fmt(summary[key]), numeric=True) for key in ("mean", "median", "std", "min", "max")),
             _td(f"{summary['invalid']:,}", numeric=True)] for name, summary in described]
    table = _table([("Statistic", "left"), ("Mean", "right"), ("Median", "right"), ("Std", "right"),
                    ("Min", "right"), ("Max", "right"), ("Invalid", "right")], rows, 700)
    note = (f"all {len(ds):,} rows valid" if not invalid else f"{invalid:,} invalid values")
    return table, f"{len(described)} statistics · {note}"


def _build_html_report(dataset: str, analysis_config: str, cfg: AnalysisConfig, ds: Dataset,
                       groups: SubsetGroups, runs: dict[str, Run], corpus: Corpus,
                       distances: list[tuple], statistics: list[tuple], charts: dict,
                       rankings: dict, skipped: dict) -> str:
    """Assemble the self-contained, interactive Hugging Face HTML card."""
    title = _dataset_title(dataset)
    overall_rows = sorted(
        [{"name": name, "values": run.subsets[groups.overall]} for name, run in runs.items()],
        key=lambda row: (-row["values"]["auroc"] if row["values"]["auroc"] == row["values"]["auroc"] else np.inf,
                         row["name"]))
    best = max(overall_rows, key=lambda row: row["values"]["tpr_at_fpr_1pct"], default=None)
    hard_prompt, hard_model = _subset_difficulty(runs, groups.prompts), _subset_difficulty(runs, groups.models)

    def hero_callout(label: str, name: str, value: float, note: str, colour: str) -> str:
        return (f'<div style="flex:1 1 190px;min-width:0;border-left:2px solid {colour};padding:2px 0 2px 12px">'
                f'<div style="font-family:{MONO};font-size:10px;letter-spacing:.12em;'
                f'text-transform:uppercase;color:#9aa6b2">{esc(label)}</div><div style="font-size:15px;'
                f'font-weight:600;color:#f6f1e7;margin:3px 0 2px;line-height:1.3">{esc(name)}</div>'
                f'<span style="font-family:{SERIF};font-size:22px;color:#f6f1e7">{fmt(value)}</span> '
                f'<span style="font-family:{MONO};font-size:10.5px;color:#9aa6b2">'
                f'{esc(note)}</span></div>')

    callouts = ""
    if best:
        callouts += hero_callout("Best detector", best["name"], best["values"]["tpr_at_fpr_1pct"],
                                 "TPR @ 1% FPR", "#2a8c82")
    if hard_prompt:
        callouts += hero_callout("Hardest prompt subset", hard_prompt[0].label, hard_prompt[1],
                                 "max detector TPR @ 1% FPR", "#c8475a")
    if hard_model:
        callouts += hero_callout("Hardest generator config", hard_model[0].label, hard_model[1],
                                 "max detector TPR @ 1% FPR", "#d4a02a")

    hero_stats = [(len(ds), "rows"), (len(groups.models), "generator configs"),
                  (len(groups.prompts), "prompt subsets"), (len(runs), "detectors")]
    hero_stats_html = "".join(
        f'<div style="flex:1 1 125px;padding:16px 12px 18px 0;border-top:1px solid rgba(255,255,255,.12)">'
        f'<div style="font-family:{SERIF};font-size:25px;color:#f6f1e7">{value:,}</div>'
        f'<div style="font-family:{MONO};font-size:10px;text-transform:uppercase;'
        f'color:#9aa6b2;margin-top:5px">{esc(label)}</div></div>' for value, label in hero_stats)

    univariate, univariate_note = _univariate(statistics, ds)
    appendix = _details(
        "How the dataset was built",
        " · ".join(filter(None, [f"{len(groups.models):,} generator configs" if groups.models else "",
                                 f"{len(groups.prompts):,} prompt subsets" if groups.prompts else "",
                                 "evaluation protocol"])),
        _composition(ds, cfg, groups, corpus, runs, skipped, analysis_config), "composition")
    appendix += _details("Schema & usage",
                         f"{len(ds.column_names):,} columns · loading snippet",
                         _schema(dataset, ds, cfg, runs, distances, corpus), "schema-usage")
    if statistics:
        appendix += _details(
            "Univariate statistics", univariate_note,
            _paragraph("Every statistic the report does arithmetic on, over the complete corpus. "
                       "Missing or non-finite values are counted as invalid.", "12px 0") + univariate,
            "univariate-statistics")
        appendix += _details("Correlation heatmap", "pairwise Pearson r",
                             _image_card(dataset, "CORRELATIONS.png", "Statistic correlations",
                                         "Each pair is correlated over the rows where both are finite."),
                             "correlation-heatmap")

    min_distance = cfg.min_distance_metric
    nav = [("01", "leaderboard", "Leaderboard"), ("02", "analytics", "Model analytics"),
           ("03", "distances", "Distances"), ("04", "appendix", "Appendix")]
    nav_html = "".join(
        f'<a href="#{anchor}" class="hover:translate-x-px motion-safe:transition-transform" '
        f'style="display:inline-block;color:{INK};text-decoration:none;font-size:13px;font-weight:500">'
        f'<span style="font-family:{MONO};color:{ACCENT};font-size:11.5px;margin-right:7px">'
        f'{number}</span>{label}</a>' for number, anchor, label in nav)
    return f'''<div id="fastdetector-card" style="background-color:#f5f3ec;background-image:linear-gradient(#e9e6dc 1px,transparent 1px),linear-gradient(90deg,#e9e6dc 1px,transparent 1px);background-size:28px 28px;background-position:-1px -1px;color:{INK};font-family:{SANS};font-size:15px;line-height:1.6;border:1px solid {LINE};border-radius:18px;overflow:hidden;margin:4px 0 0">
<div style="background:#15202b;color:#f3f1ea;padding:clamp(24px,4.5vw,48px) clamp(20px,4vw,44px) 0;background-image:radial-gradient(circle at 100% 0%,rgba(224,123,57,.20),transparent 42%),radial-gradient(circle at 0% 100%,rgba(59,110,168,.30),transparent 48%),linear-gradient(rgba(255,255,255,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px);background-size:auto,auto,26px 26px,26px 26px">
<div style="font-family:{MONO};font-size:11px;font-weight:600;letter-spacing:.16em;text-transform:uppercase;color:#9aa6b2;padding-bottom:16px;margin-bottom:34px;border-bottom:1px solid rgba(255,255,255,.12)">Auto-generated FastDetector dataset</div>
<h1 style="margin:0;padding:0;border:0;font-family:{SERIF};font-size:clamp(42px,7vw,74px);font-weight:400;letter-spacing:-.025em;line-height:1;color:#f6f1e7">{esc(title)}</h1>
<div style="display:flex;flex-wrap:wrap;gap:16px 24px;margin:26px 0 24px">{callouts}</div>
<div style="display:flex;flex-wrap:wrap;margin-top:22px">{hero_stats_html}</div>
<div style="height:4px;margin:0 calc(-1 * clamp(20px,4vw,44px));background:linear-gradient(90deg,#3b6ea8,#7a5aa6 50%,{ACCENT})"></div></div>
<div style="display:flex;flex-wrap:wrap;gap:6px 22px;padding:14px clamp(20px,4vw,44px);background:{PAPER};border-bottom:1px solid {LINE}">{nav_html}</div>
<div style="padding:4px clamp(20px,4vw,44px) 36px">
{_section("01", "leaderboard", "Detector leaderboard", f"Score-based detectors ranked by overall AUROC. Thresholds are placed exactly on every human score, so TPR is reported at a true 1% and 0.1% false positive rate. Open a row for its threshold sweep, its TPR as AI rows with low {min_distance} are dropped, and its score distributions.")}
{_leaderboard(dataset, overall_rows, runs, charts, min_distance)}
{_section("02", "analytics", "Model-Specific Analytics", "Select a model, then a split; the split's charts appear below it. Topic × format compares human scores, AI scores and their unsigned gap; generator × prompt category shows only the AI side, since neither changes the human text; specific prompt ranks every first message by difficulty.")}
{_analytics(dataset, overall_rows, runs, charts, rankings, distances)}
{_section("03", "distances", "How far do the rewrites move?", "Every distance measure between each source and its rewrite: overall, then with every prompt subset and every generator config overlaid on one chart each.")}
{_distances(dataset, distances, groups)}
{_section("04", "appendix", "Appendix", "Reference material: how the dataset was built and evaluated, its schema and loading code, univariate statistics, and cross-statistic correlations.")}
{appendix}
<div style="margin:44px 0 0;padding-top:18px;border-top:1px solid rgba(27,39,51,.35);display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px;font-size:12px;color:{FAINT}"><span>Generated by the FastDetector analysis pipeline from the complete published corpus.</span><span style="font-family:{MONO}">{esc(analysis_config)}</span></div>
</div></div>'''


def main() -> None:
    """Generate and upload the interactive full-data evaluation card."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--globals-config", default="config/globals.toml")
    parser.add_argument("--analysis-config", default="config/analysis.toml")
    parser.add_argument("--dataset", help="Exact Hugging Face statistics dataset to analyse")
    args = parser.parse_args()
    globals_config, cfg = load_config_pair(args.globals_config, args.analysis_config, AnalysisConfig)
    if cfg.fixed_classes is None and cfg.auto_class_column is None:
        raise ValueError("Must provide fixed_classes or auto_class_column")

    dataset = args.dataset or f"{globals_config.resolve_dataset(globals_config.stat_dataset)}-val"
    print(f"Loading all shards for statistics dataset {dataset}...")
    ds = load_dataset_all_shards(dataset, split="train")

    prompts, has_prompts = extract_prompt_types(ds, cfg.prompt_metadata_column)
    messages, has_messages = extract_prompt_messages(ds, cfg.prompt_metadata_column)
    models, has_models = extract_model_genconfig(ds, cfg.model_metadata_column)
    topics, has_topics = extract_dimension(ds, cfg.topic_column,
                                           ("topic", "text_topic", "source_topic"))
    formats, has_formats = extract_dimension(ds, cfg.format_column,
                                             ("format", "text_format", "source_format"))
    corpus = Corpus(prompts, messages, extract_prompt_turns(ds, cfg.prompt_metadata_column), models,
                    topics, formats, has_messages, has_topics, has_formats)

    distance_metrics, missing_metrics, available, skipped = select_available(ds, cfg)
    classifiers = [clf for clf in available
                   if "bucket" not in clf.name.lower() and "bucket" not in clf.suffix.lower()]
    for clf in available:
        if clf not in classifiers:
            skipped[clf.name] = ["bucket output intentionally excluded"]
    for metric in missing_metrics:
        print(f"Notice: distance metric '{metric}' is not in the dataset; skipping it.")
    for name, reason in skipped.items():
        print(f"Notice: classifier '{name}' skipped ({', '.join(reason)}).")

    read = lru_cache(maxsize=None)(lambda name, dtype=None: np.asarray(ds[name], dtype=dtype))
    groups = SubsetGroups(
        prompts=[Subset("Prompt", value, prompts == value)
                 for value in (sorted(set(prompts)) if has_prompts else [])],
        models=[Subset("Model", value, models == value)
                for value in (sorted(set(models)) if has_models else [])])
    auto = None if cfg.fixed_classes is not None else cfg.auto_class_column
    auto_ai = None if auto is None else read(auto) == cfg.ai_label
    subsets = [groups.overall, *groups.prompts, *groups.models]
    runs = {clf.name: evaluate(clf, read_scores(read, cfg, clf.suffix, auto_ai), subsets)
            for clf in classifiers}

    distances = [(name, read(name, float)) for name in distance_metrics]
    statistics = distances + [
        (f"{clf.name} ({base if cfg.fixed_classes is None else 'AI' if cfg.fixed_classes[i] else 'Human'})",
         read(f"{base}{clf.suffix}", float))
        for clf in classifiers for i, base in enumerate(cfg.base_columns)]
    min_distance = (read(cfg.min_distance_metric, float)
                    if cfg.min_distance_metric in ds.column_names else None)

    print("Rendering charts and interactive README...")
    charts: dict[str, bytes] = {}
    if statistics:
        charts["CORRELATIONS.png"] = plotting.heatmap(
            metrics.correlations([values for _, values in statistics]),
            [name for name, _ in statistics], "Correlations")
    for name, values in distances:
        safe = safe_name(name)
        charts[f"DIST_HIST_{safe}.png"] = plotting.histogram([(values, "Overall")], f"Distance: {name}")
        if groups.prompts:
            charts[f"DIST_BY_PROMPT_{safe}.png"] = plotting.histogram(
                [(values[sub.mask], sub.label) for sub in groups.prompts], f"{name} by prompt subset")
        if groups.models:
            charts[f"DIST_BY_MODEL_{safe}.png"] = plotting.histogram(
                [(values[sub.mask], sub.label) for sub in groups.models], f"{name} by generator config")

    rankings = {}
    boilerplate = prompt_boilerplate(messages)
    distance_names = [name for name, _ in distances if name in PROMPT_DISTANCES]
    for name, run in runs.items():
        safe = safe_name(name)
        thresholds = {"1% FPR": run.points["fpr_1pct"].threshold,
                      "0.1% FPR": run.points["fpr_0_1pct"].threshold}
        scores = run.scores
        charts[f"SWEEP_{safe}.png"] = run.sweep_chart
        if min_distance is not None:
            ai = scores.is_ai
            cutoffs, curves, kept = metrics.tpr_by_min_distance(
                scores.values[ai], min_distance[scores.rows[ai]],
                {f"TPR @ {label}": value for label, value in thresholds.items()}, run.flip)
            if cutoffs.size:
                charts[f"MINDIST_{safe}.png"] = plotting.min_distance_plot(
                    cutoffs, curves, kept, f"{name}: TPR against minimum {cfg.min_distance_metric}",
                    cfg.min_distance_metric)
        charts[f"CLF_HUMAN_{safe}.png"] = plotting.histogram(
            [(scores.values[~scores.is_ai], "Overall, human")], f"{name}: overall human", markers=thresholds)
        charts[f"CLF_AI_{safe}.png"] = plotting.histogram(
            [(scores.values[scores.is_ai], "Overall, AI")], f"{name}: overall AI", markers=thresholds)
        charts[f"CLF_PROMPTS_{safe}.png"] = plotting.histogram(
            [(part.values[part.is_ai], sub.label) for sub in groups.prompts
             for part in [scores.subset(sub.mask)]], f"{name}: AI by prompt subset", markers=thresholds)
        charts[f"CLF_MODELS_{safe}.png"] = plotting.histogram(
            [(part.values[part.is_ai], sub.label) for sub in groups.models
             for part in [scores.subset(sub.mask)]], f"{name}: AI by generator config", markers=thresholds)
        human, ai = _row_class_means(scores, len(ds))
        if has_messages:
            rankings[name] = _specific_prompt_rows(run, corpus, human, ai, distances, boilerplate)
            charts[prompt_ranking_file(name)] = prompt_ranking_csv(rankings[name], distance_names)
        if has_topics and has_formats:
            panels = plotting.score_grid_panels(
                human, ai, topics, formats, "Topic", "Format", f"{name}: topic × format")
            charts.update({f"ANALYTICS_TOPIC_FORMAT_{kind}_{safe}.png": image
                           for kind, image in panels.items()})
        if groups.prompts and groups.models:
            panels = plotting.score_grid_panels(
                human, ai, models, prompts, "Generator config", "Prompt category",
                f"{name}: generator × prompt category", include_human=False)
            charts.update({f"ANALYTICS_GENERATOR_PROMPT_{kind}_{safe}.png": image
                           for kind, image in panels.items()})

    readme = _build_html_report(dataset, args.analysis_config, cfg, ds, groups, runs, corpus,
                                distances, statistics, charts, rankings, skipped)
    size = len(readme.encode("utf-8"))
    print(f"README size: {size / 1024:.1f} KiB")
    if size > MAX_README_BYTES:
        raise RuntimeError(f"README is {size / 1024:.0f} KiB, over the {MAX_README_BYTES // 1024} KiB "
                           "Hugging Face accepts; lower PROMPT_TABLE_LIMIT or trim the card.")
    print("Uploading README and charts to Hub...")
    upload_readme(dataset, filename="README.md", files=charts, readme_content=readme)
    print("Done!")


if __name__ == "__main__":
    main()
