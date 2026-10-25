"""A deterministic demo website with ground truth, used for tests, evaluation and the offline demo.

Content: 24 articles in two topics (python, cooking), exact duplicates (/print/...), near duplicates (/amp/...),
junk (tag pages, archives, login/signup, cookie notice), a robots.txt-disallowed area, a reCAPTCHA wall,
and fault pages (503-then-200, 429 with Retry-After, slow, redirect, 404)."""
import random
import asyncio
from html import escape
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse

PY = {
    "list-comprehensions": "List comprehensions",
    "virtual-environments": "Virtual environments",
    "async-await": "Async and await",
    "type-hints": "Type hints",
    "decorators": "Decorators",
    "context-managers": "Context managers",
    "dataclasses": "Dataclasses",
    "error-handling": "Error handling",
    "testing-with-pytest": "Testing with pytest",
    "packaging": "Packaging a library",
    "generators": "Generators",
    "logging": "Logging",
}
COOK = {
    "sourdough-starter": "Sourdough starter",
    "perfect-rice": "Perfect rice",
    "knife-skills": "Knife skills",
    "tomato-sauce": "Tomato sauce",
    "roast-vegetables": "Roast vegetables",
    "dal-tadka": "Dal tadka",
    "pancakes": "Pancakes",
    "stir-fry": "Stir fry",
    "soup-stock": "Soup stock",
    "pizza-dough": "Pizza dough",
    "omelette": "Omelette",
    "chai": "Masala chai",
}
PY_SENT = [
    "In Python, {t} help you write code that is shorter and easier to read.",
    "This guide explains how {t} work, when to use them and which mistakes to avoid.",
    "The interpreter evaluates the expression lazily, which keeps memory usage low even for large inputs.",
    "A common pattern is to combine {t} with the standard library so that each function has one clear purpose.",
    "Remember that readability counts: if a construct becomes hard to follow, split it into named steps.",
    "Unit tests make refactoring safe, because they show immediately when behaviour changes.",
    "Most production code bases enforce a consistent style with a formatter and a linter in continuous integration.",
    "Performance rarely depends on clever syntax; measure with a profiler before optimising anything.",
    "The example below shows a small function, its expected output and the edge case that usually breaks it.",
    "Type checkers such as mypy catch many errors before the program runs, especially in larger teams.",
]
COOK_SENT = [
    "Getting {t} right at home mostly comes down to patience and good ingredients.",
    "Start with fresh produce, measure carefully the first time, and taste as you go.",
    "Heat control matters more than any special equipment: medium heat gives you time to react.",
    "Many cooks rush this step, but letting the flavours develop slowly makes a noticeable difference.",
    "Salt in small amounts throughout cooking rather than all at once at the end.",
    "If the mixture looks too thick, add a splash of warm water and stir until it loosens.",
    "Leftovers keep for three days in the fridge and often taste better the next day.",
    "A heavy pan distributes heat evenly and prevents the bottom from burning.",
    "Serve immediately with something fresh and acidic to balance the richness.",
    "Once you are comfortable with the basic method, experiment with spices and herbs you enjoy.",
]
NAV = '<nav><a href="/">Home</a> | <a href="/guides/python/">Python</a> | <a href="/guides/cooking/">Cooking</a> | ' \
      '<a href="/tags/beginner">Beginner</a> | <a href="/archive/2026/1">Archive</a> | <a href="/login">Log in</a></nav>'
FOOT = '<footer><a href="/legal/cookies">Cookies</a> | <a href="/signup">Sign up</a> | ' \
       '<a href="https://example.org/partner">Partner site</a> | <img src="/img/logo.png"></footer>'


def _page(title: str, body: str, extra_head: str = "") -> str:
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{escape(title)}</title>{extra_head}"
            f"</head><body>{NAV}<main>{body}</main>{FOOT}</body></html>")


def article_text(topic: str, slug: str, title: str) -> list[str]:
    rnd = random.Random(slug)
    pool = PY_SENT if topic == "python" else COOK_SENT
    paras = []
    for _ in range(4):
        sents = rnd.sample(pool, 4)
        paras.append(" ".join(s.format(t=title.lower()) for s in sents))
    paras.insert(1, f"This article about {title.lower()} is part of our {topic} series, written for readers who "
                    f"want practical advice rather than theory. Article reference {slug.upper()}-{len(slug) * 7}.")
    return paras


def article_html(topic: str, slug: str, title: str, variant: str = "") -> str:
    paras = article_text(topic, slug, title)
    if variant == "amp":
        paras[-1] = paras[-1] + " Updated for the mobile edition."
    items = COOK if topic == "cooking" else PY
    related = [s for s in items if s != slug][:3]
    rel = " ".join(f'<a href="/guides/{topic}/{s}">{items[s]}</a>' for s in related)
    body = (f"<article><h1>{escape(title)}</h1>" + "".join(f"<p>{escape(p)}</p>" for p in paras) +
            f"<p>Related: {rel}</p><p><a href='/print/guides/{topic}/{slug}'>Print</a> "
            f"<a href='/amp/guides/{topic}/{slug}'>Mobile</a> <a href='/tags/{topic}'>#{topic}</a></p></article>")
    return _page(title + (" (print)" if variant == "print" else ""), body)


def ground_truth() -> dict:
    """label: good/junk/robots/challenge; cluster: pages with the same content share one id."""
    gt = {}
    for topic, items in (("python", PY), ("cooking", COOK)):
        gt[f"/guides/{topic}/"] = {"label": "junk", "topic": topic, "dup_of": None}
        for slug in items:
            canon = f"/guides/{topic}/{slug}"
            gt[canon] = {"label": "good", "topic": topic, "dup_of": None, "cluster": canon}
            gt[f"/print/guides/{topic}/{slug}"] = {"label": "good", "topic": topic, "dup_of": canon, "kind": "exact",
                                                   "cluster": canon}
            gt[f"/amp/guides/{topic}/{slug}"] = {"label": "good", "topic": topic, "dup_of": canon, "kind": "near",
                                                 "cluster": canon}
        gt[f"/tags/{topic}"] = {"label": "junk", "topic": None, "dup_of": None}
    for p in ["/", "/tags/beginner", "/archive/2026/1", "/archive/2026/2", "/login", "/signup", "/legal/cookies",
              "/slow"]:
        gt[p] = {"label": "junk", "topic": None, "dup_of": None}
    gt["/private/secret"] = {"label": "robots", "topic": None, "dup_of": None}
    gt["/protected/report"] = {"label": "challenge", "topic": None, "dup_of": None}
    gt["/flaky/1"] = {"label": "good", "topic": "python", "dup_of": "/guides/python/generators", "kind": "exact",
                      "cluster": "/guides/python/generators", "fault": "503 once"}
    gt["/rate-limited"] = {"label": "good", "topic": "cooking", "dup_of": "/guides/cooking/chai", "kind": "exact",
                           "cluster": "/guides/cooking/chai", "fault": "429 once"}
    gt["/old-link"] = {"label": "good", "topic": "cooking", "dup_of": "/guides/cooking/omelette", "kind": "exact",
                       "cluster": "/guides/cooking/omelette", "fault": "301 redirect"}
    return gt


def create_site(slow_s: float = 1.5) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    state = {"flaky": 0, "rl": 0, "solved": False, "hits": {}}

    @app.middleware("http")
    async def count(request: Request, call_next):
        state["hits"][request.url.path] = state["hits"].get(request.url.path, 0) + 1
        return await call_next(request)

    @app.get("/robots.txt", response_class=PlainTextResponse)
    def robots():
        return "User-agent: *\nDisallow: /private/\nCrawl-delay: 0\n"

    @app.get("/", response_class=HTMLResponse)
    def home():
        links = "".join(f'<li><a href="/guides/python/{s}?utm_source=home">{t}</a></li>' for s, t in list(PY.items())[:4])
        body = ("<h1>Field Notes</h1><ul>" + links + "</ul><p><a href='/guides/python/'>All Python guides</a> "
                "<a href='/guides/cooking/'>All cooking guides</a> <a href='/private/secret'>Staff</a> "
                "<a href='/protected/report'>Annual report</a> <a href='/flaky/1'>Flaky</a> "
                "<a href='/rate-limited'>Busy</a> <a href='/slow'>Slow</a> <a href='/old-link'>Old link</a> "
                "<a href='/missing'>Missing</a> <a href='/files/guide.pdf'>PDF</a></p>")
        return _page("Field Notes - Home", body)

    @app.get("/guides/{topic}/", response_class=HTMLResponse)
    def section(topic: str):
        items = PY if topic == "python" else COOK
        links = "".join(f'<li><a href="/guides/{topic}/{s}">{t}</a></li>' for s, t in items.items())
        return _page(f"{topic.title()} guides", f"<h1>{topic.title()} guides</h1><ul>{links}</ul>")

    @app.get("/guides/{topic}/{slug}", response_class=HTMLResponse)
    def article(topic: str, slug: str):
        items = PY if topic == "python" else COOK
        if slug not in items:
            return HTMLResponse(_page("Not found", "<h1>404</h1>"), status_code=404)
        return article_html(topic, slug, items[slug])

    @app.get("/print/guides/{topic}/{slug}", response_class=HTMLResponse)
    def printed(topic: str, slug: str):
        items = PY if topic == "python" else COOK
        return article_html(topic, slug, items[slug], "print")

    @app.get("/amp/guides/{topic}/{slug}", response_class=HTMLResponse)
    def amp(topic: str, slug: str):
        items = PY if topic == "python" else COOK
        return article_html(topic, slug, items[slug], "amp")

    @app.get("/tags/{tag}", response_class=HTMLResponse)
    def tag(tag: str):
        items = list(PY.items()) + list(COOK.items())
        links = " ".join(f'<a href="/archive/2026/{i % 2 + 1}">{t}</a>' for i, (_, t) in enumerate(items))
        return _page(f"Tag: {tag}", f"<h1>#{tag}</h1><p>{links}</p>")

    @app.get("/archive/2026/{n}", response_class=HTMLResponse)
    def archive(n: int):
        links = " ".join(f'<a href="/guides/cooking/{s}">{t}</a>' for s, t in COOK.items())
        return _page(f"Archive page {n}", f"<h1>Archive</h1><p>{links}</p><a href='/archive/2026/{n % 2 + 1}'>Next</a>")

    @app.get("/login", response_class=HTMLResponse)
    def login():
        return _page("Log in", "<h1>Log in</h1><form><input type='email' name='e'><input type='password' name='p'>"
                               "<button>Log in</button></form><a href='/signup'>Create account</a>")

    @app.get("/signup", response_class=HTMLResponse)
    def signup():
        return _page("Sign up", "<h1>Sign up</h1><form><input type='email' name='e'><input type='password' "
                                "name='p'><button>Create account</button></form>")

    @app.get("/legal/cookies", response_class=HTMLResponse)
    def cookies():
        return _page("Cookie notice", "<p>We use cookies.</p>")

    @app.get("/private/secret", response_class=HTMLResponse)
    def secret():
        return _page("Staff only", "<p>Internal staff handbook. This page is disallowed by robots.txt.</p>")

    @app.get("/protected/report", response_class=HTMLResponse)
    def protected(request: Request):
        if request.cookies.get("human") == "yes" or state["solved"]:
            return article_html("python", "logging", "Annual report")
        return HTMLResponse("<html><head><title>Verify you are human</title>"
                            "<script src='https://www.google.com/recaptcha/api.js'></script></head><body>"
                            "<div class='g-recaptcha' data-sitekey='demo'></div></body></html>", status_code=200)

    @app.get("/flaky/1", response_class=HTMLResponse)
    def flaky():
        state["flaky"] += 1
        if state["flaky"] == 1:
            return HTMLResponse("busy", status_code=503)
        return article_html("python", "generators", PY["generators"])

    @app.get("/rate-limited", response_class=HTMLResponse)
    def limited():
        state["rl"] += 1
        if state["rl"] == 1:
            return HTMLResponse("slow down", status_code=429, headers={"Retry-After": "1"})
        return article_html("cooking", "chai", COOK["chai"])

    @app.get("/slow", response_class=HTMLResponse)
    async def slow():
        await asyncio.sleep(slow_s)
        return _page("Slow page", "<p>Sorry for the wait.</p>")

    @app.get("/old-link")
    def old():
        return RedirectResponse("/guides/cooking/omelette", status_code=301)

    @app.get("/files/guide.pdf")
    def pdf():
        return PlainTextResponse("%PDF-1.4 fake", media_type="application/pdf")

    @app.get("/__groundtruth.json")
    def gt():
        return JSONResponse(ground_truth())

    @app.get("/__hits")
    def hits():
        return state["hits"]

    app.state.site = state
    return app
