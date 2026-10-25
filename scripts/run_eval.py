"""Crawl the demo site under several configurations and score each against its ground truth.

Usage: python scripts/run_eval.py [--real-embeddings]"""
import sys
import json
import time
import httpx
import asyncio
import argparse
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aicrawler.config import CrawlConfig  # noqa: E402
from aicrawler.engine import CrawlJob  # noqa: E402
from aicrawler.evaluation import evaluate_crawl  # noqa: E402
from aicrawler.quality import QualityModel  # noqa: E402
from aicrawler.store import Store  # noqa: E402
from aicrawler.testsite import create_site, ground_truth  # noqa: E402


async def fast(x):
    await asyncio.sleep(min(x, 0.005))


def run(name, store, embedder=None, quality=None, topic=None, **kw):
    base = dict(seeds=["http://site.test/"], max_depth=4, max_pages=300, min_delay_s=0, max_delay_s=0, concurrency=4,
                per_host_concurrency=4, on_challenge="skip", use_cache=False)
    base.update(kw)
    cfg = CrawlConfig(**base)
    cid = store.create_crawl(cfg.model_dump(), name)
    job = CrawlJob(cid, cfg, store, embedder=embedder, quality_model=quality,
                   transport=httpx.ASGITransport(app=create_site(0.02)), sleep=fast)
    t = time.time()
    job.start()
    while job.state in ("queued", "running"):
        time.sleep(0.1)
    ev = evaluate_crawl(store.pages(cid, limit=10000), ground_truth(), topic, first_n=30 if topic else None)
    ev["seconds"] = round(time.time() - t, 1)
    ev["dups_by_kind"] = job.stats()["duplicates_by_kind"]
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-embeddings", action="store_true")
    a = ap.parse_args()
    store = Store(Path(tempfile.mkdtemp()) / "eval.db")
    emb = None
    if a.real_embeddings:
        from aicrawler.semantic import FastEmbedder
        emb = FastEmbedder()
    learned = QualityModel.load()
    res = {
        "rules_quality": run("rules", store, quality=QualityModel()),
        "learned_quality": run("learned", store, quality=learned) if learned.trained else None,
        "bfs_harvest_python": run("bfs", store, topic="python", max_pages=30, concurrency=1, per_host_concurrency=1),
        "focused_python": run("focus-py", store, embedder=emb, topic="python", focus_topic="python programming",
                              max_pages=30, concurrency=1, per_host_concurrency=1),
        "bfs_harvest_cooking": run("bfs2", store, topic="cooking", max_pages=30, concurrency=1, per_host_concurrency=1),
        "focused_cooking": run("focus-cook", store, embedder=emb, topic="cooking", focus_topic="cooking recipes",
                               max_pages=30, concurrency=1, per_host_concurrency=1),
    }
    if emb:
        res["semantic_dedup"] = run("semantic", store, embedder=emb, semantic_dedup=True)
    for k, v in res.items():
        if v:
            print(f"{k:22} " + json.dumps({x: v[x] for x in v if x not in ("junk_stored_paths", "good_rejected_paths")}))
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "demo_eval.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
