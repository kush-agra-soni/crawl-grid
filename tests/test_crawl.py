import io
import json
import zipfile
from conftest import fast_cfg, wait_crawl


def start(c, **kw):
    r = c.post("/crawls", json={"config": fast_cfg(**kw)})
    assert r.status_code == 200, r.text
    return r.json()["crawl_id"]


def outcomes(c, cid):
    return {p["url"].replace("http://site.test", "").split("?")[0]: p for p in c.get(f"/crawls/{cid}/pages").json()}


def test_full_demo_crawl_matches_ground_truth(make_app):
    c = make_app()
    cid = start(c, on_challenge="skip")
    cr = wait_crawl(c, cid)
    assert cr["status"] == "done"
    ev = c.get(f"/crawls/{cid}/evaluate").json()
    assert ev["coverage"] == 1.0 and ev["extra_copies_stored"] == 0
    assert ev["junk_stored"].startswith("0/") and ev["good_rejected"].startswith("0/")
    assert ev["robots_respected"] and ev["challenge_detected"] and ev["faults_recovered"]
    s = cr["stats"]
    assert s["duplicates_by_kind"]["exact"] > 10 and s["duplicates_by_kind"]["near"] > 10
    p = outcomes(c, cid)
    assert p["/private/secret"]["outcome"] == "robots_blocked"
    assert "https://example.org/partner" not in [x["url"] for x in c.get(f"/crawls/{cid}/pages").json()]
    assert p["/files/guide.pdf"]["outcome"] == "skipped"
    assert p["/missing"]["outcome"] == "error"
    assert p["/login"]["outcome"] == "low_quality" and p["/login"]["quality_reasons"]


def test_challenge_pauses_host_then_resumes_with_cookies(make_app):
    c = make_app()
    cid = start(c, on_challenge="pause")

    def solve(cr):
        waiting = cr["stats"]["paused_hosts"]["site.test"]
        assert waiting >= 1 and cr["challenges"][0]["kind"] == "recaptcha"  # whole site waits politely
        r = c.post(f"/crawls/{cid}/challenges/resolve", json={"host": "site.test", "cookies": "human=yes"})
        assert r.json()["requeued"] == waiting

    cr = wait_crawl(c, cid, on_wait=solve)
    p = outcomes(c, cid)
    assert p["/protected/report"]["outcome"] in ("stored", "duplicate")  # content after the human check
    assert cr["challenges"][0]["status"] == "resolved"
    assert any("challenge resolved" in e["msg"] for e in c.get(f"/crawls/{cid}/events").json())


def test_challenge_skip_by_user(make_app):
    c = make_app()
    cid = start(c, on_challenge="pause")
    wait_crawl(c, cid, on_wait=lambda cr: c.post(f"/crawls/{cid}/challenges/resolve",
                                                  json={"host": "site.test", "skip": True}))
    assert c.get(f"/crawls/{cid}").json()["challenges"][0]["status"] == "skipped"


def test_limits_depth_pages_and_patterns(make_app):
    c = make_app()
    cid = start(c, max_depth=0)
    assert list(outcomes(c, cid) if wait_crawl(c, cid) else {}) == ["/"]
    cid = start(c, max_pages=10, use_cache=False)
    cr = wait_crawl(c, cid)
    assert cr["stats"]["fetched"] <= 10
    cid = start(c, include_patterns=["/guides/cooking/", r"^http://site\.test/$"], on_challenge="skip")
    wait_crawl(c, cid)
    stored = [u for u, p in outcomes(c, cid).items() if p["outcome"] == "stored"]
    assert stored and all(u.startswith("/guides/cooking/") for u in stored)


def test_pause_resume_and_stop(make_app):
    c = make_app()
    cid = start(c, min_delay_s=0.2, max_delay_s=0.2, concurrency=1, per_host_concurrency=1, use_cache=False)
    assert c.post(f"/crawls/{cid}/pause").json()["status"] == "paused"
    assert c.get(f"/crawls/{cid}").json()["status"] == "paused"
    assert c.post(f"/crawls/{cid}/resume").json()["status"] == "running"
    c.post(f"/crawls/{cid}/stop")
    cr = wait_crawl(c, cid)
    assert cr["status"] == "stopped" and cr["stats"]["fetched"] < 40
    assert c.post(f"/crawls/{cid}/pause").status_code == 409


def test_topic_focus_harvests_more_on_topic_pages(make_app):
    c = make_app()
    bfs = start(c, max_pages=25, use_cache=False, concurrency=1, per_host_concurrency=1, on_challenge="skip")
    wait_crawl(c, bfs)
    focused = start(c, max_pages=25, use_cache=False, concurrency=1, per_host_concurrency=1, on_challenge="skip",
                    focus_topic="cooking recipes")
    wait_crawl(c, focused)

    def harvest(cid):
        pages = c.get(f"/crawls/{cid}/pages").json()
        return sum("/cooking/" in p["url"] or "rate-limited" in p["url"] for p in pages) / len(pages)
    assert harvest(focused) > harvest(bfs) + 0.2


def test_cache_reused_across_crawls(make_app):
    c = make_app()
    first = start(c, on_challenge="skip")
    wait_crawl(c, first)
    second = start(c, on_challenge="skip")
    cr = wait_crawl(c, second)
    assert cr["stats"]["cache_hits"] > 50
    assert sum(c.sites[-1].state.site["hits"].values()) < 10
    assert c.delete("/cache").json()["cleared"] > 50


def test_ai_extraction(make_app):
    seen = []

    def fake_ai(instr, title, url, text):
        seen.append(url)
        if "Omelette" in title:
            raise RuntimeError("model timeout")
        return {"title": title, "topic": "python" if "python" in url else "cooking"}

    c = make_app(ai_fn=fake_ai)
    cid = start(c, on_challenge="skip", ai_extraction="Return the title and topic", ai_max_pages=50)
    cr = wait_crawl(c, cid)
    stored = [p for p in c.get(f"/crawls/{cid}/pages", params={"outcome": "stored"}).json()]
    assert all(p["ai_data"] for p in stored)
    ok = [p for p in stored if "data" in p["ai_data"]]
    assert ok and ok[0]["ai_data"]["data"]["title"]
    assert cr["stats"]["ai_failed"] >= 1 and cr["stats"]["ai_extracted"] == len(ok)


def test_search_exports_and_page_detail(make_app):
    c = make_app()
    cid = start(c, on_challenge="skip")
    wait_crawl(c, cid)
    hits = c.get("/search", params={"q": "sourdough starter", "crawl_id": cid}).json()
    assert hits and "sourdough" in hits[0]["url"] and "[" in hits[0]["snippet"]
    pid = hits[0]["page_id"]
    assert "patience" in c.get(f"/pages/{pid}").json()["text"]
    jl = c.get(f"/crawls/{cid}/export", params={"format": "jsonl"}).text.strip().splitlines()
    assert len(jl) == c.get(f"/crawls/{cid}").json()["stats"]["stored"] and json.loads(jl[0])["text"]
    assert c.get(f"/crawls/{cid}/export", params={"format": "csv"}).text.startswith("url,final_url")
    assert "<h1>Crawl report</h1>" in c.get(f"/crawls/{cid}/export", params={"format": "html"}).text
    z = zipfile.ZipFile(io.BytesIO(c.get(f"/crawls/{cid}/export", params={"format": "zip"}).content))
    assert set(z.namelist()) == {"manifest.json", "documents.jsonl", "config.json"}
    assert c.get(f"/crawls/{cid}/export", params={"format": "xml"}).status_code == 422


def test_templates_validation_and_delete(make_app):
    c = make_app()
    assert c.post("/templates", json={"name": "docs", "config": fast_cfg(max_depth=3)}).status_code == 200
    assert c.get("/templates").json()[0]["config"]["max_depth"] == 3
    assert c.delete("/templates/docs").status_code == 200 and c.delete("/templates/docs").status_code == 404
    assert c.post("/crawls", json={"config": {"seeds": []}}).status_code == 422
    assert c.post("/crawls", json={"config": {"seeds": ["x.com"], "max_depth": 99}}).status_code == 422
    cid = start(c, max_depth=0)
    wait_crawl(c, cid)
    assert c.delete(f"/crawls/{cid}").status_code == 200 and c.get(f"/crawls/{cid}").status_code == 404


def test_health_and_quality_model_endpoint(make_app):
    c = make_app()
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["quality_model"] in ("rules", "learned")
    assert "trained" in c.get("/quality/model").json()
