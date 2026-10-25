"""Crawl frontier: breadth-first by default, or topic-focused priority when a focus topic is set."""
import re
import heapq
import itertools
import numpy as np

_TOK = re.compile(r"[a-z0-9]+")
STOP = frozenset("the a an and or of to in for on with how what is are www http https com org html htm php index".split())


def tokens(s: str) -> set[str]:
    return {t for t in _TOK.findall(s.lower()) if t not in STOP and len(t) > 1}


class Frontier:
    def __init__(self, focus_topic: str = "", embedder=None):
        self.topic = tokens(focus_topic)
        self.embedder = embedder if focus_topic else None
        self._topic_vec = None
        if self.embedder is not None:
            try:
                self._topic_vec = np.asarray(self.embedder.embed_query(focus_topic), dtype=np.float32)
            except Exception:
                self.embedder = None
        self._heap: list = []
        self._seen: set[str] = set()
        self._counter = itertools.count()
        self.host_quality: dict[str, list[float]] = {}
        self._slugs: set[str] = set()

    def __len__(self) -> int:
        return len(self._heap)

    def seen(self, url: str) -> bool:
        return url in self._seen

    def relevance(self, url: str, anchor: str, parent_relevance: float = 0.0) -> float:
        if not self.topic:
            return 0.0
        words = tokens(anchor) | tokens(url.replace("-", " ").replace("_", " ").replace("/", " "))
        lexical = len(words & self.topic) / len(self.topic)
        sem = 0.0
        if self._topic_vec is not None and anchor.strip():
            try:
                sem = max(0.0, float(np.asarray(self.embedder.embed_query(anchor + " " + url)) @ self._topic_vec))
            except Exception:
                sem = 0.0
        return round(min(1.0, 0.6 * lexical + 0.6 * sem + 0.25 * parent_relevance), 4)

    @staticmethod
    def _slug(url: str) -> str:
        last = url.split("?")[0].rstrip("/").rsplit("/", 1)[-1]
        return last if len(last) > 3 and not last.isdigit() else ""

    def priority(self, depth: int, relevance: float, host: str, likely_copy: bool = False) -> float:
        """Higher is crawled first. Without a topic this is breadth-first (shallow first).
        Likely copies (same final path segment as an already queued URL, e.g. /print/x and /amp/x) go last."""
        if likely_copy:
            return -depth - 10
        if not self.topic:
            return -depth
        hq = self.host_quality.get(host)
        host_bonus = (sum(hq) / len(hq) - 0.5) * 0.2 if hq else 0.0
        return relevance * 2 - depth * 0.15 + host_bonus

    def push(self, url: str, depth: int, parent: str | None, anchor: str = "", parent_relevance: float = 0.0,
             host: str = "") -> bool:
        if url in self._seen:
            return False
        self._seen.add(url)
        slug = self._slug(url)
        copy = bool(slug) and slug in self._slugs
        if slug:
            self._slugs.add(slug)
        rel = self.relevance(url, anchor, parent_relevance)
        pr = self.priority(depth, rel, host, copy)
        heapq.heappush(self._heap, (-pr, next(self._counter), url, depth, parent, rel, pr))
        return True

    def pop(self):
        if not self._heap:
            return None
        _, _, url, depth, parent, rel, pr = heapq.heappop(self._heap)
        return {"url": url, "depth": depth, "parent": parent, "relevance": rel, "priority": round(pr, 4)}

    def record_quality(self, host: str, q: float) -> None:
        self.host_quality.setdefault(host, []).append(q)
