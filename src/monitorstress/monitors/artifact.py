"""A baseline that sees only the structure of a trajectory: how many steps, which command names.

It reads traj.steps and nothing else, and from each command only its first token (the
command name, no arguments). It is not a serious monitor. It is reported beside the others
so that every table shows how much of a number the shape of the trajectory alone gives,
and the same classifier is the E0 artifact check.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MaxAbsScaler

from ..schema import Trajectory
from .base import Monitor, labels_of


def command_name_features(traj: Trajectory) -> dict[str, float]:
    """Number of steps and how often each command name (first token) is used. No arguments."""
    counts: dict[str, float] = defaultdict(float)
    for step in traj.steps:
        tokens = step.command.split()
        counts["cmd:" + (tokens[0] if tokens else "")] += 1.0
    features = {name: math.log1p(count) for name, count in counts.items()}
    features["n_steps"] = float(len(traj.steps))
    return features


class ArtifactMonitor(Monitor):
    name = "artifact"
    view = "commands"
    trainable = True

    def __init__(self, C: float = 1.0, seed: int = 0):
        self.C = C
        self.seed = seed
        self.model: Pipeline | None = None

    def fit(self, trajs: list[Trajectory]) -> None:
        self.model = make_pipeline(
            DictVectorizer(), MaxAbsScaler(),
            LogisticRegression(C=self.C, max_iter=2000, class_weight="balanced", random_state=self.seed),
        )
        self.model.fit([command_name_features(t) for t in trajs], labels_of(trajs))

    def score(self, trajs: list[Trajectory]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ArtifactMonitor.score called before fit")
        if not trajs:
            return np.zeros(0)
        return self.model.decision_function([command_name_features(t) for t in trajs]).astype(float)
