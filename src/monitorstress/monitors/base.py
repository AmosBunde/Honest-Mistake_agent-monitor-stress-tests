"""The monitor interface from docs/DESIGN.md.

A monitor may read three things from a trajectory when it scores: task_prompt, steps and
events. `fit` additionally reads `label` (it is supervised training); `score` never does.
"""
from __future__ import annotations

import numpy as np

from ..schema import Trajectory


class Monitor:
    """Base class. Higher scores mean more suspicious."""

    name: str = "monitor"
    view: str = "commands"
    trainable: bool = False

    def fit(self, trajs: list[Trajectory]) -> None:
        """Train on labeled trajectories. A no-op for monitors that are not trained."""
        return None

    def score(self, trajs: list[Trajectory]) -> np.ndarray:
        """One score per trajectory, in order."""
        raise NotImplementedError


def labels_of(trajs: list[Trajectory]) -> np.ndarray:
    """Training labels. Only `fit` methods may call this."""
    return np.array([int(t.label) for t in trajs], dtype=int)
