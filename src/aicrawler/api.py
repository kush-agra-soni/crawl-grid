import io
import csv
import html
import json
import time
import httpx
import socket
import zipfile
import threading
from . import llm
from .store import Store
from .urls import host_of
from .engine import CrawlJob
from .quality import QualityModel
from pydantic import BaseModel, Field
from .evaluation import evaluate_crawl
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from .testsite import create_site, ground_truth
from fastapi.responses import HTMLResponse, Response
from .config import CrawlConfig, Settings, get_settings

TERMINAL = ("done", "stopped", "failed")


class StartRequest(BaseModel):
    config: CrawlConfig
    name: str | None = Field(default=None, max_length=100)


class ResolveRequest(BaseModel):
    host: str
    cookies: str = ""
    skip: bool = False


class TemplateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    config: CrawlConfig


def parse_cookie_string(s: str) -> dict[str, str]:
    out = {}
    for part in s.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            if k.strip():
                out[k.strip()] = v.strip()
    return out


def create_app(settings: Settings | None = None, *, embedder=None, ai_fn=None, transport_factory=None,
               sleep=None, use_local_models: bool = True, llm_enabled: bool = True) -> FastAPI:
    settings = settings or get_settings()
    store = Store(settings.db_path)
    store.mark_interrupted()
    jobs: dict[str, CrawlJob] = {}
    demo = {"url": None, "server": None}
    lazy = {"embedder": embedder}
    if llm_enabled and ai_fn is None:
        ai_fn = lambda instr, title, url, text: llm.ai_extract(settings, instr, title, url, text)  # noqa: E731

    def get_embedder():
        if lazy["embedder"] is None and use_local_models:
            from .semantic import FastEmbedder
            lazy["embedder"] = FastEmbedder()
        return lazy["embedder"]

    @asynccontextmanager
    async def lifespan(_app):
        yield
        for j in jobs.values():
            j.stop()
        if demo["server"]:
            demo["server"].should_exit = True

    app = FastAPI(title="AI Web Crawler", version="1.0.0", lifespan=lifespan)
    app.state.store, app.state.jobs = store, jobs

    def crawl_view(c: dict) -> dict:
        j = jobs.get(c["id"])
        if j and j.state not in TERMINAL:
            c = {**c, "status": j.state, "stats": j.stats()}
        c["challenges"] = store.challenges(c["id"])
        return c

    def get_or_404(cid: str) -> dict:
        c = store.get_crawl(cid)
        if not c:
            raise HTTPException(404, "crawl not found")
        return c

    # ---------- health ----------
    @app.get("/health")
    def health(check_llm: bool = False) -> dict:
        model = QualityModel.load()
        out = {"status": "ok", "active_crawls": sum(j.state in ("running", "paused") for j in jobs.values()),
               "quality_model": "learned" if model.trained else "rules", "cache_entries": store.cache_size(),
               "demo_site": demo["url"]}
        if check_llm:
            out["llm"] = llm.ping(settings) if llm_enabled else False
        return out

    # ---------- crawls ----------
    @app.post("/crawls")
    def start(req: StartRequest) -> dict:
        cfg = req.config
        if any(j.state in ("running", "paused") for j in jobs.values()) and len(jobs) >= 4:
            raise HTTPException(429, "Too many active crawls")
        cid = store.create_crawl(cfg.model_dump(), req.name or ", ".join(host_of(s) for s in cfg.seeds)[:100])
        emb = get_embedder() if (cfg.semantic_dedup or cfg.focus_topic) else None
        transport = transport_factory(cfg) if transport_factory else None
        job = CrawlJob(cid, cfg, store, settings, emb, ai_fn if cfg.ai_extraction else None, transport=transport,
                       sleep=sleep)
        jobs[cid] = job
        job.start()
        return {"crawl_id": cid}

    @app.get("/crawls")
    def list_crawls() -> list[dict]:
        return [crawl_view(c) for c in store.list_crawls()]

    @app.get("/crawls/{cid}")
    def crawl(cid: str) -> dict:
        return crawl_view(get_or_404(cid))

    def control(cid: str, action: str) -> dict:
        get_or_404(cid)
        j = jobs.get(cid)
        if not j or j.state in TERMINAL:
            raise HTTPException(409, "crawl is not active")
        getattr(j, action)()
        return {"crawl_id": cid, "status": j.state}

    @app.post("/crawls/{cid}/pause")
    def pause(cid: str) -> dict:
        return control(cid, "pause")

    @app.post("/crawls/{cid}/resume")
    def resume(cid: str) -> dict:
        return control(cid, "resume")

    @app.post("/crawls/{cid}/stop")
    def stop(cid: str) -> dict:
        return control(cid, "stop")

    @app.delete("/crawls/{cid}")
    def delete(cid: str) -> dict:
        get_or_404(cid)
        j = jobs.pop(cid, None)
        if j:
            j.stop()
        store.delete_crawl(cid)
        return {"deleted": cid}

    @app.get("/crawls/{cid}/events")
    def events(cid: str, after: int = 0) -> list[dict]:
        return store.events(cid, after)

    @app.get("/crawls/{cid}/pages")
    def pages(cid: str, outcome: str | None = None, q: str | None = None, limit: int = 500, offset: int = 0) -> list[dict]:
        get_or_404(cid)
        return store.pages(cid, outcome, q, min(limit, 5000), offset)

    @app.get("/pages/{pid}")
    def page(pid: int) -> dict:
        p = store.page(pid)
        if not p:
            raise HTTPException(404, "page not found")
        return p

    @app.get("/search")
    def search(q: str, crawl_id: str | None = None, limit: int = 20) -> list[dict]:
        return store.search(q, crawl_id, min(limit, 100))

    # ---------- challenges ----------
    @app.post("/crawls/{cid}/challenges/resolve")
    def resolve(cid: str, req: ResolveRequest) -> dict:
        get_or_404(cid)
        j = jobs.get(cid)
        if not j or j.state in TERMINAL:
            store.resolve_challenges(cid, req.host, "skipped" if req.skip else "resolved")
            return {"requeued": 0, "note": "crawl not active"}
        n = j.resolve_host(req.host, parse_cookie_string(req.cookies), req.skip)
        return {"requeued": 0 if req.skip else n}

    @app.post("/crawls/{cid}/challenges/browser")
    def solve_in_browser(cid: str, req: ResolveRequest) -> dict:
        """Open a real browser window so the user can pass the check themselves; cookies are then reused."""
        get_or_404(cid)
        url = next((c["url"] for c in store.challenges(cid) if c["host"] == req.host and c["status"] == "open"), None)
        if not url:
            raise HTTPException(404, "no open challenge for this host")
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            raise HTTPException(501, "Browser solving needs Playwright: .venv\\Scripts\\pip install playwright && "
                                     ".venv\\Scripts\\playwright install chromium. Or paste cookies instead.") from e
        from .extract import detect_challenge
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            ctx = browser.new_context()
            pg = ctx.new_page()
            pg.goto(url)
            deadline = time.time() + 300
            while time.time() < deadline:
                try:
                    if not detect_challenge(200, pg.content()):
                        break
                except Exception:
                    pass
                time.sleep(1.5)
            cookies = {c["name"]: c["value"] for c in ctx.cookies() if req.host.split(":")[0].endswith(c["domain"].lstrip("."))}
            browser.close()
        j = jobs.get(cid)
        n = j.resolve_host(req.host, cookies) if j and j.state not in TERMINAL else 0
        return {"requeued": n, "cookies": len(cookies)}

    # ---------- export ----------
    @app.get("/crawls/{cid}/export")
    def export(cid: str, format: str = "jsonl", outcome: str = "stored") -> Response:
        c = get_or_404(cid)
        rows = store.pages(cid, outcome or None, limit=100000, with_text=True)
        name = f"crawl_{cid}"
        cols = ["url", "final_url", "depth", "status_code", "outcome", "title", "words", "lang", "quality",
                "dup_of", "dup_kind", "fetched_at"]
        if format == "jsonl":
            body = "\n".join(json.dumps({**{k: r.get(k) for k in cols}, "text": r.get("text"),
                                         "ai_data": r.get("ai_data")}) for r in rows) + "\n"
            return Response(body, media_type="application/x-ndjson",
                            headers={"Content-Disposition": f'attachment; filename="{name}.jsonl"'})
        if format == "csv":
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(cols + ["ai_data"])
            for r in rows:
                w.writerow([r.get(k) for k in cols] + [json.dumps(r["ai_data"]) if r.get("ai_data") else ""])
            return Response(buf.getvalue(), media_type="text/csv",
                            headers={"Content-Disposition": f'attachment; filename="{name}.csv"'})
        if format == "html":
            s = crawl_view(c)["stats"]
            summary = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
                              for k, v in s.items() if not isinstance(v, (list, dict)))
            trs = "".join(f"<tr><td><a href='{html.escape(r['url'])}'>{html.escape(r['title'] or r['url'])}</a></td>"
                          f"<td>{r['outcome']}</td><td>{r['quality']}</td><td>{r['words']}</td></tr>" for r in rows)
            doc = (f"<!doctype html><meta charset='utf-8'><title>Crawl report {cid}</title><style>body{{font-family:"
                   f"sans-serif;margin:2rem}}td,th{{padding:4px 8px;border-bottom:1px solid #ddd;text-align:left}}"
                   f"</style><h1>Crawl report</h1><p>Seeds: {html.escape(', '.join(c['config']['seeds']))}</p>"
                   f"<table>{summary}</table><h2>Pages ({len(rows)})</h2><table><tr><th>Page</th><th>Outcome</th>"
                   f"<th>Quality</th><th>Words</th></tr>{trs}</table>")
            return HTMLResponse(doc, headers={"Content-Disposition": f'attachment; filename="{name}.html"'})
        if format == "zip":
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("manifest.json", json.dumps({"format": "aicrawler-corpus", "version": 1, "crawl_id": cid,
                                                        "seeds": c["config"]["seeds"], "documents": len(rows),
                                                        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2))
                z.writestr("documents.jsonl", "\n".join(json.dumps({"url": r["url"], "title": r["title"],
                                                                      "text": r["text"], "quality": r["quality"]})
                                                          for r in rows) + "\n")
                z.writestr("config.json", json.dumps(c["config"], indent=2))
            return Response(buf.getvalue(), media_type="application/zip",
                            headers={"Content-Disposition": f'attachment; filename="{name}.zip"'})
        raise HTTPException(422, "format must be jsonl, csv, html or zip")

    # ---------- templates & cache ----------
    @app.get("/templates")
    def templates() -> list[dict]:
        return store.templates()

    @app.post("/templates")
    def save_template(req: TemplateRequest) -> dict:
        store.save_template(req.name, req.config.model_dump())
        return {"saved": req.name}

    @app.delete("/templates/{name}")
    def delete_template(name: str) -> dict:
        if not store.delete_template(name):
            raise HTTPException(404, "template not found")
        return {"deleted": name}

    @app.delete("/cache")
    def clear_cache() -> dict:
        return {"cleared": store.cache_clear()}

    @app.get("/quality/model")
    def quality_model() -> dict:
        m = QualityModel.load()
        return {"trained": m.trained, "meta": m.meta}

    # ---------- demo site & evaluation ----------
    @app.post("/demo/start")
    def demo_start() -> dict:
        if demo["url"]:
            return {"url": demo["url"]}
        import uvicorn

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(create_site(), host="127.0.0.1", port=port, log_level="warning"))
        threading.Thread(target=server.run, daemon=True).start()
        url = f"http://127.0.0.1:{port}/"
        for _ in range(100):
            try:
                httpx.get(url + "robots.txt", timeout=1)
                break
            except httpx.HTTPError:
                time.sleep(0.05)
        demo.update(url=url, server=server)
        return {"url": url}

    @app.get("/crawls/{cid}/evaluate")
    def evaluate(cid: str) -> dict:
        c = get_or_404(cid)
        topic = None
        ft = c["config"].get("focus_topic", "").lower()
        if "python" in ft:
            topic = "python"
        elif "cook" in ft or "recipe" in ft:
            topic = "cooking"
        return evaluate_crawl(store.pages(cid, limit=100000), ground_truth(), topic, first_n=30 if topic else None)

    return app
