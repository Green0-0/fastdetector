"""AutoVisualizer: compile-then-evaluate README/chart builder.

See :mod:`fastdetector.visualization.auto_visualizer` for the full design.
"""

from fastdetector.visualization.auto_visualizer import (
    AutoVisualizer,
    StatWrapper,
    ClassifierStatWrapper,
    ClassifierThresholdStatWrapper,
    StaticThresholdWrapper,
)

__all__ = [
    "AutoVisualizer",
    "StatWrapper",
    "ClassifierStatWrapper",
    "ClassifierThresholdStatWrapper",
    "StaticThresholdWrapper",
]
