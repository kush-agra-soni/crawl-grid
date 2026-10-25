"""Exact (sha256), near-duplicate (MinHash + LSH) and optional semantic (embedding) deduplication."""
import re
import hashlib
import threading
import numpy as np

NUM_PERM = 128
BANDS = 32  # 32 bands x 4 rows: ~50% detection at Jaccard 0.42, >99% at 0.8
ROWS = NUM_PERM // BANDS
_MERSENNE = (1 << 61) - 1
_rng = np.random.default_rng(1234)
_A = _rng.integers(1, _MERSENNE, NUM_PERM, dtype=np.uint64)
_B = _rng.integers(0, _MERSENNE, NUM_PERM, dtype=np.uint64)
_WORD = re.compile(r"\w+")


def content_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()


def shingles(text: str, k: int = 5) -> set[int]:
    words = _WORD.findall(text.lower())
    if len(words) < k:
        return {hash(" ".join(words)) & 0xFFFFFFFF} if words else set()
    return {int.from_bytes(hashlib.blake2b(" ".join(words[i:i + k]).encode(), digest_size=4).digest(), "little")
            for i in range(len(words) - k + 1)}


def minhash(sh: set[int]) -> np.ndarray:
    if not sh:
        return np.full(NUM_PERM, np.iinfo(np.uint64).max, dtype=np.uint64)
    x = np.fromiter(sh, dtype=np.uint64)
    # (a*x + b) mod p over all permutations at once; uint64 wraparound is acceptable for hashing
    return ((np.outer(_A, x) + _B[:, None]) % _MERSENNE).min(axis=1)


def jaccard_est(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a == b))


class DedupIndex:
    def __init__(self, near_threshold: float = 0.8, embedder=None, semantic_threshold: float = 0.95):
        self.near_threshold = near_threshold
        self.embedder, self.semantic_threshold = embedder, semantic_threshold
        self._exact: dict[str, str] = {}
        self._buckets: dict[tuple, list[int]] = {}
        self._sigs: list[np.ndarray] = []
        self._urls: list[str] = []
        self._vecs: list[np.ndarray] = []
        self._vec_urls: list[str] = []
        self._lock = threading.Lock()

    def check_and_add(self, url: str, text: str) -> tuple[str | None, str | None, float]:
        """Return (duplicate_of_url, kind, similarity); registers the page when it is new."""
        h = content_hash(text)
        sig = minhash(shingles(text))
        with self._lock:
            if h in self._exact:
                return self._exact[h], "exact", 1.0
            cands = set()
            keys = [(b, sig[b * ROWS:(b + 1) * ROWS].tobytes()) for b in range(BANDS)]
            for k in keys:
                cands.update(self._buckets.get(k, []))
            best, best_sim = None, 0.0
            for i in cands:
                s = jaccard_est(sig, self._sigs[i])
                if s > best_sim:
                    best, best_sim = i, s
            if best is not None and best_sim >= self.near_threshold:
                return self._urls[best], "near", round(best_sim, 3)
        vec = None
        if self.embedder is not None and len(text) > 200:
            try:
                vec = np.asarray(self.embedder.embed([text[:3000]])[0], dtype=np.float32)
            except Exception:
                vec = None
        with self._lock:
            if vec is not None and self._vecs:
                sims = np.vstack(self._vecs) @ vec
                j = int(np.argmax(sims))
                if sims[j] >= self.semantic_threshold:
                    return self._vec_urls[j], "semantic", round(float(sims[j]), 3)
            idx = len(self._sigs)
            self._sigs.append(sig)
            self._urls.append(url)
            self._exact[h] = url
            for k in keys:
                self._buckets.setdefault(k, []).append(idx)
            if vec is not None:
                self._vecs.append(vec)
                self._vec_urls.append(url)
        return None, None, 0.0
