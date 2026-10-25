import re
import httpx
import pytest
import asyncio
import tempfile
from pathlib import Path
from aicrawler.store import Store
from aicrawler.fetcher import Fetcher
from aicrawler.dedup import DedupIndex
from aicrawler.frontier import Frontier
from aicrawler.config import CrawlConfig
from aicrawler.api import parse_cookie_string
from conftest import HashEmbedder, fast_sleep
from aicrawler.llm import LLMError, parse_json
from aicrawler.testsite import COOK, PY, article_html
from aicrawler.urls import allowed, normalize, registrable
from aicrawler.extract import detect_challenge, extract_content, extract_links
from aicrawler.quality import QualityModel, evaluate, features, rule_score, train


# ---------- urls ----------
def test_normalize():
    assert normalize("HTTP://Example.COM:80/a//b?utm_source=x&b=2&a=1#frag") == "http://example.com/a/b?a=1&b=2"
    assert normalize("../c", "https://x.org/a/b/") == "https://x.org/a/c"
    assert normalize("mailto:a@b.c") is None and normalize("ftp://x") is None


def test_registrable_and_allowed():
    assert registrable("docs.python.org") == "python.org" and registrable("www.bbc.co.uk") == "bbc.co.uk"
    sites = {"python.org"}
    ex = [re.compile(r"\.png$")]
    assert allowed("https://docs.python.org/3/", sites, True, [], [], ex) == (True, "")
    assert allowed("https://example.org/", sites, True, [], [], ex)[1] == "external link"
    assert allowed("https://docs.python.org/logo.png", sites, True, [], [], ex)[1] == "excluded pattern"
    assert allowed("https://a.b.com/x", sites, True, ["b.com"], [], [])[0]
    assert allowed("https://docs.python.org/2/", sites, True, [], [re.compile("/3/")], [])[1] == "not in include patterns"


def test_config_validation():
    c = CrawlConfig(seeds=["example.com"], min_delay_s=3, max_delay_s=1)
    assert c.seeds == ["https://example.com"] and c.max_delay_s == 3
    with pytest.raises(ValueError):
        CrawlConfig(seeds=[])
    with pytest.raises(ValueError):
        CrawlConfig(seeds=["x.com"], on_challenge="solve")


# ---------- extraction & challenges ----------
def test_extract_content_and_links():
    page = article_html("python", "decorators", PY["decorators"])
    c = extract_content(page, "http://site.test/guides/python/decorators")
    assert c["title"] == "Decorators" and "decorators" in c["text"].lower() and c["n_links"] > 5
    links = dict(extract_links(page, "http://site.test/guides/python/decorators"))
    assert "http://site.test/print/guides/python/decorators" in links
    assert links["http://site.test/guides/python/list-comprehensions"] == "List comprehensions"


def test_extract_links_skips_nofollow_and_schemes():
    body = '<a href="/a">A</a><a rel="nofollow" href="/b">B</a><a href="mailto:x@y.z">m</a><a href="#top">t</a>'
    assert [u for u, _ in extract_links(body, "http://h.test/")] == ["http://h.test/a"]


@pytest.mark.parametrize("status,body,kind", [
    (200, "<div class='g-recaptcha' data-sitekey='x'></div>", "recaptcha"),
    (403, "<title>Attention Required! | Cloudflare</title>", "cloudflare"),
    (200, "<script src='https://challenges.cloudflare.com/turnstile/v0/api.js'></script>", "turnstile"),
    (429, "<p>Please verify you are a human</p>", "bot_wall"),
    (200, "<p>Normal article</p>", None),
])
def test_detect_challenge(status, body, kind):
    assert detect_challenge(status, body) == kind


def test_contact_form_with_recaptcha_is_not_a_wall():
    body = "<p>" + "Long genuine article text. " * 120 + "</p><div class='g-recaptcha'></div>"
    assert detect_challenge(200, body) is None


# ---------- dedup ----------
def test_dedup_exact_near_and_distinct():
    d = DedupIndex(0.8)
    a = " ".join(f"word{i} sentence about python decorators" for i in range(60))
    assert d.check_and_add("u1", a) == (None, None, 0.0)
    assert d.check_and_add("u2", "  " + a.upper())[:2] == ("u1", "exact")
    dup, kind, sim = d.check_and_add("u3", a + " plus one extra closing sentence here.")
    assert (dup, kind) == ("u1", "near") and sim >= 0.8
    other = " ".join(f"token{i} cooking rice and lentils slowly" for i in range(60))
    assert d.check_and_add("u4", other)[0] is None


def test_semantic_dedup():
    d = DedupIndex(0.95, embedder=HashEmbedder(), semantic_threshold=0.75)
    a = "Python decorators wrap functions to add behaviour. " * 10
    b = "Behaviour is added when decorators wrap Python functions. " * 10  # reworded, few shared shingles
    d.check_and_add("u1", a)
    assert d.check_and_add("u2", b)[1] == "semantic"


# ---------- quality ----------
def test_features_and_rule_baseline():
    good = features(extract_content(article_html("cooking", "pancakes", COOK["pancakes"]), "u"))
    junk = features({"text": "Log in\nEmail\nPassword", "title": "Log in", "full_text_chars": 40,
                     "anchor_chars": 20, "n_inputs": 2})
    assert rule_score(good) > 0.5 > rule_score(junk) and junk["login_form"] == 1


def test_learned_model_trains_saves_and_explains(tmp_path):
    good = [features(extract_content(article_html("python", s, t), "u")) for s, t in PY.items()]
    junk = [features({"text": " ".join(["Home", "Tags", "Archive"] * k), "title": "", "full_text_chars": 400,
                      "anchor_chars": 380}) for k in range(1, 13)]
    X, y = good + junk, [1] * len(good) + [0] * len(junk)
    m = train(X, y)
    assert evaluate(lambda f: m.score(f)[0], X, y)["f1"] == 1.0
    p = tmp_path / "m.json"
    m.save(p)
    m2 = QualityModel.load(p)
    s, reasons = m2.score(junk[0])
    assert s < 0.5 and reasons
    assert not QualityModel.load(tmp_path / "missing.json").trained


# ---------- frontier ----------
def test_frontier_bfs_and_topic_priority():
    f = Frontier()
    f.push("http://s/deep", 3, None)
    f.push("http://s/shallow", 1, None)
    assert f.pop()["url"] == "http://s/shallow" and not f.push("http://s/deep", 1, None)
    t = Frontier("python decorators")
    t.push("http://s/cooking/rice", 1, None, "Perfect rice")
    t.push("http://s/python/decorators", 2, None, "Python decorators explained")
    assert t.pop()["url"] == "http://s/python/decorators"


def test_frontier_deprioritises_likely_copies():
    f = Frontier()
    f.push("http://s/guides/decorators", 2, None)
    f.push("http://s/print/guides/decorators", 2, None)
    f.push("http://s/guides/generators", 3, None)
    assert [f.pop()["url"] for _ in range(3)] == ["http://s/guides/decorators", "http://s/guides/generators",
                                                   "http://s/print/guides/decorators"]


def test_frontier_semantic_relevance():
    t = Frontier("python decorators", embedder=HashEmbedder())
    assert t.relevance("http://s/x", "decorators in python") > t.relevance("http://s/y", "baking bread")


# ---------- misc ----------
def test_parse_cookie_string_and_json():
    assert parse_cookie_string("a=1; b = two ;bad") == {"a": "1", "b": "two"}
    assert parse_json('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert parse_json('<think>x</think>{"label": "good"}') == {"label": "good"}
    with pytest.raises(LLMError):
        parse_json("nothing")


# ---------- fetcher (mock transport) ----------
def _fetcher(handler, **cfg):
    store = Store(Path(tempfile.mkdtemp()) / "f.db")
    logs = []
    conf = CrawlConfig(seeds=["http://h.test/"], min_delay_s=0, max_delay_s=0, **cfg)
    f = Fetcher(conf, store, lambda lvl, m: logs.append((lvl, m)), sleep=fast_sleep,
                transport=httpx.MockTransport(handler))
    return f, store, logs


def test_fetcher_retries_with_retry_after_and_backoff():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        if calls["n"] == 2:
            return httpx.Response(503)
        return httpx.Response(200, text="<p>ok</p>", headers={"content-type": "text/html"})

    f, _, logs = _fetcher(handler, use_cache=False)
    r = asyncio.run(f.fetch("http://h.test/a"))
    assert r.status == 200 and r.attempts == 3 and f.stats["retries"] == 2
    assert f.host("http://h.test/a").penalty > 1
    assert any("429" in m for _, m in logs)


def test_fetcher_gives_up_and_reports_errors():
    def handler(req):
        raise httpx.ConnectError("refused")

    f, _, _ = _fetcher(handler, use_cache=False, max_retries=1)
    r = asyncio.run(f.fetch("http://h.test/a"))
    assert r.status == 0 and "ConnectError" in r.error and r.attempts == 2


def test_fetcher_cache_fresh_and_conditional():
    calls = []

    def handler(req):
        calls.append(dict(req.headers))
        if req.headers.get("if-none-match") == '"v1"':
            return httpx.Response(304)
        return httpx.Response(200, text="<p>body</p>", headers={"content-type": "text/html", "etag": '"v1"'})

    f, store, _ = _fetcher(handler, cache_ttl_hours=1)
    assert not asyncio.run(f.fetch("http://h.test/a")).from_cache
    assert asyncio.run(f.fetch("http://h.test/a")).from_cache and len(calls) == 1
    f.cfg = f.cfg.model_copy(update={"cache_ttl_hours": 0})
    r = asyncio.run(f.fetch("http://h.test/a"))
    assert r.from_cache and r.body == "<p>body</p>" and f.stats["revalidated"] == 1


def test_robots_and_crawl_delay():
    def handler(req):
        if req.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /no/\nCrawl-delay: 3\n")
        return httpx.Response(200, text="ok")

    f, _, logs = _fetcher(handler)
    assert asyncio.run(f.allowed_by_robots("http://h.test/yes")) is True
    assert asyncio.run(f.allowed_by_robots("http://h.test/no/x")) is False
    assert f.host("http://h.test/").crawl_delay == 3 and any("crawl-delay" in m for _, m in logs)


def test_headers_referer_and_non_html():
    seen = {}

    def handler(req):
        seen.update(req.headers)
        return httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"})

    f, _, _ = _fetcher(handler, use_cache=False, user_agent="MyBot/1.0")
    r = asyncio.run(f.fetch("http://h.test/x.pdf", referer="http://h.test/"))
    assert seen["user-agent"] == "MyBot/1.0" and seen["referer"] == "http://h.test/" and "accept-language" in seen
    assert r.content_type == "application/pdf" and r.body == ""
