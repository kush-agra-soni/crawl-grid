import sys
import time
import pytest
import asyncio
import hashlib
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "launcher")]

from aicrawler.frontier import tokens  # noqa: E402

SITE = "http://site.test/"


async def fast_sleep(s):
    await asyncio.sleep(min(s, 0.01))


class HashEmbedder:
    def _v(self, t):
        v = np.zeros(64, np.float32)
        for w in tokens(t):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed(self, texts):
        return np.vstack([self._v(t) for t in texts])

    def embed_query(self, t):
        return self._v(t)


def fast_cfg(**kw):
    base = {"seeds": [SITE], "max_depth": 4, "max_pages": 300, "min_delay_s": 0, "max_delay_s": 0,
            "concurrency": 6, "per_host_concurrency": 6, "use_cache": True}
    base.update(kw)
    return base


def wait_crawl(c, cid, timeout=60, on_wait=None):
    deadline = time.time() + timeout
    while time.time() < deadline:
        cr = c.get(f"/crawls/{cid}").json()
        if cr["status"] in ("done", "stopped", "failed"):
            return cr
        if on_wait and cr["stats"].get("waiting_for_user"):
            on_wait(cr)
            on_wait = None
        time.sleep(0.05)
    raise AssertionError(f"crawl {cid} did not finish: {c.get(f'/crawls/{cid}').json()['stats']}")


@pytest.fixture
def make_app(tmp_path):
    import httpx
    from fastapi.testclient import TestClient

    from aicrawler.api import create_app
    from aicrawler.config import Settings
    from aicrawler.testsite import create_site

    counter = {"n": 0}

    def factory(ai_fn=None, sites=None):
        counter["n"] += 1
        sites = sites if sites is not None else []

        def transport(cfg):
            site = create_site(slow_s=0.05)
            sites.append(site)
            return httpx.ASGITransport(app=site)

        app = create_app(Settings("http://llm.invalid/v1", "k", "m", tmp_path / f"d{counter['n']}"),
                         embedder=HashEmbedder(), ai_fn=ai_fn, transport_factory=transport, sleep=fast_sleep,
                         use_local_models=False, llm_enabled=False)
        c = TestClient(app)
        c.__enter__()
        c.sites = sites
        return c

    return factory
