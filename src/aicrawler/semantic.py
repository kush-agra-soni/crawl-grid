"""Local embedding model for semantic dedup and topic-focused scheduling (ONNX, CPU)."""
import threading
import numpy as np
from .config import ROOT

MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedder:
    def __init__(self):
        self._m, self._lock = None, threading.Lock()

    def _load(self):
        with self._lock:
            if self._m is None:
                from fastembed import TextEmbedding

                self._m = TextEmbedding(MODEL, cache_dir=str(ROOT / "models"))
        return self._m

    @staticmethod
    def _norm(m):
        m = np.asarray(m, dtype=np.float32)
        n = np.linalg.norm(m, axis=-1, keepdims=True)
        return m / np.where(n == 0, 1, n)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._norm(np.vstack(list(self._load().embed(texts)))) if texts else np.zeros((0, 384), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._norm(np.asarray(next(iter(self._load().query_embed(text)))))
