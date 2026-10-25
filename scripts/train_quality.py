"""Train the learned page-quality model.

Data: (1) every page of the demo site, labelled by its ground truth, and (2) pages from small, polite crawls
of real sites, labelled good/junk by the LLM judge. Quality filtering is disabled while collecting so junk is
kept. Trains logistic regression on 70% and compares it with the rule baseline on the held-out 30%.

Usage: python scripts/train_quality.py [--seeds URL ...] [--pages-per-site 40] [--no-llm]"""
import sys
import json
import time
import httpx
import random
import asyncio
import argparse
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aicrawler import llm  # noqa: E402
from aicrawler.config import CrawlConfig, get_settings  # noqa: E402
from aicrawler.engine import CrawlJob  # noqa: E402
from aicrawler.quality import MODEL_PATH, QualityModel, evaluate, rule_score, train  # noqa: E402
from aicrawler.store import Store  # noqa: E402
from aicrawler.testsite import create_site, ground_truth  # noqa: E402

DEFAULT_SEEDS = ["https://docs.python.org/3/tutorial/index.html", "https://news.ycombinator.com/",
                 "https://books.toscrape.com/catalogue/category/books/mystery_3/index.html",
                 "https://en.wikipedia.org/wiki/Web_crawler"]


def crawl(store, cfg, transport=None, sleep=None):
    cid = store.create_crawl(cfg.model_dump(), "training")
    job = CrawlJob(cid, cfg, store, quality_model=QualityModel(), transport=transport, sleep=sleep)
    job.start()
    while job.state in ("queued", "running", "paused"):
        time.sleep(0.5)
    return store.pages(cid, limit=100000, with_text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", default=DEFAULT_SEEDS)
    ap.add_argument("--pages-per-site", type=int, default=40)
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()
    s = get_settings()
    store = Store(Path(tempfile.mkdtemp()) / "train.db")
    rows = []

    async def fast(x):
        await asyncio.sleep(min(x, 0.01))
    demo = crawl(store, CrawlConfig(seeds=["http://site.test/"], max_depth=4, max_pages=300, min_delay_s=0,
                                    max_delay_s=0, quality_filter=False, on_challenge="skip", use_cache=False),
                 httpx.ASGITransport(app=create_site(slow_s=0.05)), fast)
    gt = ground_truth()
    for p in demo:
        g = gt.get(urlsplit(p["url"]).path)
        if p.get("features") and g and g["label"] in ("good", "junk"):
            rows.append({"source": "demo", "url": p["url"], "features": p["features"], "label": int(g["label"] == "good")})
    print(f"demo site: {len(rows)} labelled pages")

    if not a.no_llm:
        for seed in a.seeds:
            pages = crawl(store, CrawlConfig(seeds=[seed], max_depth=2, max_pages=a.pages_per_site, min_delay_s=1.0,
                                             max_delay_s=2.0, concurrency=2, per_host_concurrency=2,
                                             quality_filter=False, on_challenge="skip"))
            n = 0
            for p in pages:
                if not p.get("features") or p["outcome"] not in ("stored", "duplicate"):
                    continue
                try:
                    j = llm.judge_page(s, p["url"], p["title"] or "", p["text"] or "")
                    print(f"  judged {p['url'][:80]} -> {j.get('label')}", flush=True)
                    label = str(j.get("label", "")).lower()
                except llm.LLMError as e:
                    print("  judge failed:", e)
                    continue
                if label in ("good", "junk"):
                    rows.append({"source": urlsplit(seed).netloc, "url": p["url"], "features": p["features"],
                                 "label": int(label == "good"), "reason": j.get("reason")})
                    n += 1
            print(f"{seed}: {len(pages)} pages crawled, {n} judged")

    random.Random(42).shuffle(rows)
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    cut = lambda xs: int(len(xs) * 0.7)  # noqa: E731
    tr = pos[:cut(pos)] + neg[:cut(neg)]
    te = pos[cut(pos):] + neg[cut(neg):]
    model = train([r["features"] for r in tr], [r["label"] for r in tr])
    Xte, yte = [r["features"] for r in te], [r["label"] for r in te]
    learned = evaluate(lambda f: model.score(f)[0], Xte, yte)
    rules = evaluate(rule_score, Xte, yte)
    real_te = [r for r in te if r["source"] != "demo"]
    learned_real = evaluate(lambda f: model.score(f)[0], [r["features"] for r in real_te], [r["label"] for r in real_te])
    rules_real = evaluate(rule_score, [r["features"] for r in real_te], [r["label"] for r in real_te])
    model.meta = {"trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_pages": len(tr), "test_pages": len(te),
                  "positives": len(pos), "negatives": len(neg),
                  "sources": sorted({r["source"] for r in rows}),
                  "test_learned": learned, "test_rules": rules,
                  "test_real_sites_learned": learned_real, "test_real_sites_rules": rules_real}
    model.save()
    print(json.dumps(model.meta, indent=2))
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "quality_training_data.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("saved", MODEL_PATH)


if __name__ == "__main__":
    main()
