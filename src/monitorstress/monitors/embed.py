"""Sentence embeddings (MiniLM) per step, pooled over steps, and logistic regression.

Features per trajectory: mean over steps, max over steps, and the embedding of the task
prompt (3 x 384 numbers). Embeddings are cached on disk, keyed by a hash of the text, so
repeated experiments only pay for the model once.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from ..schema import Trajectory
from ..views import TEXT_VIEWS, render_steps
from .base import Monitor, labels_of

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CACHE_DIR = Path(os.environ.get("MONITORSTRESS_CACHE", ".cache/monitorstress")) / "embeddings"
_DIGEST_BYTES = 20


class EmbeddingCache:
    """Append-only file of (sha1 digest, float32 vector) records, mirrored in memory."""

    def __init__(self, path: str | Path, dim: int):
        self.path = Path(path)
        self.dim = dim
        self.record_bytes = _DIGEST_BYTES + 4 * dim
        self.vectors: dict[bytes, np.ndarray] = {}
        if self.path.exists():
            raw = self.path.read_bytes()
            usable = len(raw) - len(raw) % self.record_bytes  # ignore a torn last record
            for start in range(0, usable, self.record_bytes):
                key = raw[start:start + _DIGEST_BYTES]
                vec = np.frombuffer(raw, dtype=np.float32, count=dim, offset=start + _DIGEST_BYTES)
                self.vectors[key] = vec
            if usable != len(raw):
                with self.path.open("r+b") as f:
                    f.truncate(usable)

    @staticmethod
    def key(text: str) -> bytes:
        return hashlib.sha1(text.encode("utf-8")).digest()

    def add(self, items: dict[bytes, np.ndarray]) -> None:
        if not items:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as f:
            for key, vec in items.items():
                vec = np.asarray(vec, dtype=np.float32)
                f.write(key + vec.tobytes())
                self.vectors[key] = vec


class MiniLMEncoder:
    """all-MiniLM-L6-v2 through plain transformers: mean pooling over tokens, L2-normalized."""

    dim = 384

    def __init__(self, model_name: str = MODEL_NAME, cache_dir: str | Path | None = None,
                 max_tokens: int = 128, batch_size: int = 64):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        tag = f"{model_name.replace('/', '--')}-{max_tokens}"
        self.cache = EmbeddingCache(cache_dir / f"{tag}.bin", self.dim)
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
        self._model = AutoModel.from_pretrained(self.model_name, local_files_only=True).eval()

    def _encode_uncached(self, texts: list[str]) -> np.ndarray:
        import torch

        self._load()
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar lengths per batch
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                idx = order[start:start + self.batch_size]
                batch = self._tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                                        max_length=self.max_tokens, return_tensors="pt")
                hidden = self._model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, dim=1)
                out[idx] = pooled.numpy()
        return out

    def encode(self, texts: list[str]) -> np.ndarray:
        """One row per text. Only texts not yet in the cache go through the model."""
        keys = [EmbeddingCache.key(t) for t in texts]
        missing: dict[bytes, str] = {}
        for key, text in zip(keys, texts):
            if key not in self.cache.vectors and key not in missing:
                missing[key] = text
        if missing:
            vectors = self._encode_uncached(list(missing.values()))
            self.cache.add(dict(zip(missing.keys(), vectors)))
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.cache.vectors[k] for k in keys])


class EmbedMonitor(Monitor):
    trainable = True

    def __init__(self, view: str = "commands", encoder: MiniLMEncoder | None = None,
                 cache_dir: str | Path | None = None, C: float = 1.0, seed: int = 0):
        if view not in TEXT_VIEWS:
            raise ValueError(f"embed needs a text view, got {view!r}")
        self.view = view
        self.name = "embed"
        self.encoder = encoder if encoder is not None else MiniLMEncoder(cache_dir=cache_dir)
        self.C = C
        self.seed = seed
        self.model: Pipeline | None = None

    def features(self, trajs: list[Trajectory]) -> np.ndarray:
        """[mean over steps | max over steps | task prompt] for each trajectory."""
        texts: list[str] = []
        spans: list[tuple[int, int]] = []
        for t in trajs:
            steps = render_steps(t, self.view)
            spans.append((len(texts), len(texts) + len(steps)))
            texts.extend(steps)
        prompt_start = len(texts)
        texts.extend(t.task_prompt for t in trajs)
        vectors = self.encoder.encode(texts)
        dim = vectors.shape[1] if len(texts) else self.encoder.dim
        out = np.zeros((len(trajs), 3 * dim), dtype=np.float32)
        for i, (a, b) in enumerate(spans):
            if b > a:
                out[i, :dim] = vectors[a:b].mean(0)
                out[i, dim:2 * dim] = vectors[a:b].max(0)
            out[i, 2 * dim:] = vectors[prompt_start + i]
        return out

    def fit(self, trajs: list[Trajectory]) -> None:
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced", random_state=self.seed),
        )
        self.model.fit(self.features(trajs), labels_of(trajs))

    def score(self, trajs: list[Trajectory]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("EmbedMonitor.score called before fit")
        if not trajs:
            return np.zeros(0)
        return self.model.decision_function(self.features(trajs)).astype(float)
