"""Page quality: interpretable features, a rule baseline and a learned logistic-regression model."""
import re
import json
import math
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).with_name("quality_model.json")
STOP = frozenset("the a an and or of to in is it for on with as by at from that this be are was were has have not "
                 "but which you your we our they their can will".split())
FEATURES = ["log_words", "text_ratio", "link_density", "avg_sentence_words", "stopword_ratio", "unique_ratio",
            "digit_ratio", "upper_ratio", "short_line_ratio", "login_form", "punct_per_word", "title_words"]
REASONS = {
    "log_words": "very little text", "text_ratio": "main content is a small part of the page",
    "link_density": "mostly links / navigation", "avg_sentence_words": "fragments rather than sentences",
    "stopword_ratio": "not natural prose", "unique_ratio": "highly repetitive", "digit_ratio": "mostly numbers",
    "upper_ratio": "mostly capitals", "short_line_ratio": "list of short lines", "login_form": "login or sign-up form",
    "punct_per_word": "little punctuation", "title_words": "missing title",
}


def features(c: dict) -> dict:
    text = c.get("text") or ""
    words = re.findall(r"\w+", text)
    n = len(words)
    lw = [w.lower() for w in words]
    sentences = max(1, len(re.findall(r"[.!?](\s|$)", text)))
    lines = [ln for ln in text.splitlines() if ln.strip()]
    full = max(1, c.get("full_text_chars") or len(text))
    return {
        "log_words": math.log1p(n),
        "text_ratio": min(1.0, len(text) / full),
        "link_density": min(1.0, (c.get("anchor_chars") or 0) / full),
        "avg_sentence_words": min(60.0, n / sentences),
        "stopword_ratio": sum(w in STOP for w in lw) / n if n else 0.0,
        "unique_ratio": len(set(lw)) / n if n else 0.0,
        "digit_ratio": sum(ch.isdigit() for ch in text) / max(1, len(text)),
        "upper_ratio": sum(ch.isupper() for ch in text) / max(1, sum(ch.isalpha() for ch in text)),
        "short_line_ratio": sum(len(ln.split()) < 5 for ln in lines) / len(lines) if lines else 1.0,
        "login_form": 1.0 if c.get("n_inputs", 0) > 0 and n < 300 else 0.0,
        "punct_per_word": len(re.findall(r"[.,;:]", text)) / n if n else 0.0,
        "title_words": min(20.0, len((c.get("title") or "").split())),
    }


def rule_score(f: dict) -> float:
    """Hand-written baseline the learned model is compared against."""
    ok = f["log_words"] >= math.log1p(80) and f["link_density"] < 0.5 and f["login_form"] == 0
    return 0.8 if ok else 0.2


class QualityModel:
    def __init__(self, weights=None, bias=0.0, mean=None, std=None, meta=None):
        self.w = np.asarray(weights, dtype=float) if weights is not None else None
        self.b, self.meta = bias, meta or {}
        self.mean = np.asarray(mean, dtype=float) if mean is not None else None
        self.std = np.asarray(std, dtype=float) if std is not None else None

    @property
    def trained(self) -> bool:
        return self.w is not None

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "QualityModel":
        if not path.exists():
            return cls()
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(d["weights"], d["bias"], d["mean"], d["std"], d.get("meta"))

    def save(self, path: Path = MODEL_PATH) -> None:
        path.write_text(json.dumps({"features": FEATURES, "weights": self.w.tolist(), "bias": self.b,
                                    "mean": self.mean.tolist(), "std": self.std.tolist(), "meta": self.meta},
                                   indent=2), encoding="utf-8")

    def _x(self, f: dict) -> np.ndarray:
        return (np.array([f[k] for k in FEATURES]) - self.mean) / self.std

    def score(self, f: dict) -> tuple[float, list[str]]:
        if not self.trained:
            s = rule_score(f)
            reasons = [] if s >= 0.5 else [r for k, r in (("log_words", REASONS["log_words"]),
                                                          ("link_density", REASONS["link_density"]),
                                                          ("login_form", REASONS["login_form"]))
                                           if (k == "log_words" and f[k] < math.log1p(80)) or
                                           (k == "link_density" and f[k] >= 0.5) or (k == "login_form" and f[k])]
            return s, reasons
        x = self._x(f)
        z = float(x @ self.w + self.b)
        p = 1 / (1 + math.exp(-max(-30, min(30, z))))
        contrib = x * self.w
        reasons = [REASONS[FEATURES[i]] for i in np.argsort(contrib)[:3] if contrib[i] < -0.5] if p < 0.5 else []
        return round(p, 4), reasons


def train(feature_rows: list[dict], labels: list[int], l2: float = 0.05, epochs: int = 3000,
          lr: float = 0.1) -> QualityModel:
    X = np.array([[r[k] for k in FEATURES] for r in feature_rows], dtype=float)
    y = np.array(labels, dtype=float)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1
    Xs = (X - mean) / std
    w, b = np.zeros(X.shape[1]), 0.0
    pos = max(1, y.sum())
    sw = np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * max(1, len(y) - pos)))  # class balancing
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-(Xs @ w + b)))
        g = (p - y) * sw
        w -= lr * (Xs.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    return QualityModel(w, b, mean, std)


def evaluate(model_fn, feature_rows: list[dict], labels: list[int], threshold: float = 0.5) -> dict:
    tp = fp = fn = tn = 0
    for f, y in zip(feature_rows, labels):
        pred = model_fn(f) >= threshold
        tp += pred and y
        fp += pred and not y
        fn += (not pred) and y
        tn += (not pred) and not y
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
            "accuracy": round((tp + tn) / max(1, len(labels)), 3), "n": len(labels)}
