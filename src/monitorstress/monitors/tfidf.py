"""Character n-gram TF-IDF and logistic regression over a text view of the trajectory."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

from ..schema import Trajectory
from ..views import TEXT_VIEWS, render
from .base import Monitor, labels_of


class TfidfMonitor(Monitor):
    trainable = True

    def __init__(
        self,
        view: str = "commands",
        ngram_range: tuple[int, int] = (2, 5),
        max_features: int = 200_000,
        C: float = 10.0,
        seed: int = 0,
    ):
        if view not in TEXT_VIEWS:
            raise ValueError(f"tfidf needs a text view, got {view!r}")
        self.view = view
        self.name = "tfidf"
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.C = C
        self.seed = seed
        self.model: Pipeline | None = None

    def _texts(self, trajs: list[Trajectory]) -> list[str]:
        return [render(t, self.view) for t in trajs]

    def fit(self, trajs: list[Trajectory]) -> None:
        self.model = make_pipeline(
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=self.ngram_range,
                lowercase=True,
                sublinear_tf=True,
                min_df=2,
                max_features=self.max_features,
                dtype=np.float32,
            ),
            LogisticRegression(C=self.C, max_iter=2000, class_weight="balanced", random_state=self.seed),
        )
        self.model.fit(self._texts(trajs), labels_of(trajs))

    def score(self, trajs: list[Trajectory]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("TfidfMonitor.score called before fit")
        if not trajs:
            return np.zeros(0)
        return self.model.decision_function(self._texts(trajs)).astype(float)
