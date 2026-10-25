"""Crawl job: async worker pool over a frontier, with pause/resume/stop and challenge handling."""
import re
import time
import queue
import asyncio
import threading
from .fetcher import Fetcher
from collections import deque
from .frontier import Frontier
from .config import CrawlConfig
from .dedup import DedupIndex, content_hash
from .quality import QualityModel, features
from .urls import allowed, host_of, normalize, registrable
from .extract import detect_challenge, extract_content, extract_links


class CrawlJob:
    def __init__(self, crawl_id: str, cfg: CrawlConfig, store, settings=None, embedder=None, ai_fn=None,
                 quality_model: QualityModel | None = None, transport=None, sleep=None):
        self.id, self.cfg, self.store = crawl_id, cfg, store
        self.settings, self.embedder, self.ai_fn = settings, embedder, ai_fn
        self.quality = quality_model or QualityModel.load()
        self.transport, self._sleep = transport, sleep
        self.state = "queued"
        self._pause = threading.Event()
        self._pause.set()
        self._stop = threading.Event()
        self.paused_hosts: dict[str, list] = {}
        self._resume_q: queue.Queue = queue.Queue()
        self.recent = deque(maxlen=8)
        self.counts = {k: 0 for k in ("fetched", "stored", "duplicate", "low_quality", "error", "robots_blocked",
                                      "challenge", "skipped", "ai_extracted", "ai_failed")}
        self.dups = {"exact": 0, "near": 0, "semantic": 0}
        self.started = None
        self.thread: threading.Thread | None = None
        self._ai_q: queue.Queue = queue.Queue()

    # ---------- control ----------
    def start(self) -> None:
        self.thread = threading.Thread(target=self._thread_main, daemon=True, name=f"crawl-{self.id}")
        self.thread.start()

    def pause(self) -> None:
        if self.state in ("running", "queued"):
            self._pause.clear()
            self.state = "paused"
            self._log("INFO", "Paused by user")

    def resume(self) -> None:
        if self.state == "paused":
            self._pause.set()
            self.state = "running"
            self._log("INFO", "Resumed")

    def stop(self) -> None:
        self._stop.set()
        self._pause.set()
        if self.state in ("running", "paused", "queued"):
            self.state = "stopping"
            self._log("INFO", "Stopping: finishing in-flight requests")

    def resolve_host(self, host: str, cookies: dict | None = None, skip: bool = False) -> int:
        self._resume_q.put((host, cookies or {}, skip))
        return len(self.paused_hosts.get(host, []))

    # ---------- helpers ----------
    def _log(self, level: str, msg: str) -> None:
        self.store.event(self.id, level, msg)

    def stats(self) -> dict:
        el = (time.time() - self.started) if self.started else 0
        return {**self.counts, "duplicates_by_kind": self.dups, "frontier": len(self.frontier) if hasattr(self, "frontier") else 0,
                "paused_hosts": {h: len(v) for h, v in self.paused_hosts.items()},
                "waiting_for_user": bool(self.paused_hosts) and self.state == "running" and len(self.frontier) == 0
                if hasattr(self, "frontier") else False,
                "elapsed_s": round(el, 1), "pages_per_min": round(self.counts["fetched"] / el * 60, 1) if el > 1 else 0.0,
                "recent": list(self.recent), **(self.fetcher.stats if hasattr(self, "fetcher") else {})}

    def _thread_main(self) -> None:
        ai_thread = None
        if self.cfg.ai_extraction and self.ai_fn:
            ai_thread = threading.Thread(target=self._ai_worker, daemon=True)
            ai_thread.start()
        try:
            asyncio.run(self._run())
        except Exception as e:
            self.state = "failed"
            self._log("ERROR", f"Crawl failed: {e}")
            self.store.update_crawl(self.id, status="failed", error=str(e), finished=time.time(), stats=self.stats())
        finally:
            self._ai_q.put(None)
            if ai_thread:
                ai_thread.join(timeout=600)
                self.store.update_crawl(self.id, stats=self.stats())

    def _ai_worker(self) -> None:
        while True:
            item = self._ai_q.get()
            if item is None:
                return
            pid, url, title, text = item
            try:
                data = self.ai_fn(self.cfg.ai_extraction, title, url, text)
                self.store.set_ai_data(pid, {"data": data})
                self.counts["ai_extracted"] += 1
            except Exception as e:
                self.store.set_ai_data(pid, {"error": str(e)[:300]})
                self.counts["ai_failed"] += 1
                self._log("WARNING", f"AI extraction failed for {url}: {e}")

    # ---------- main loop ----------
    async def _run(self) -> None:
        cfg = self.cfg
        self.started = time.time()
        self.state = "running" if self._pause.is_set() else "paused"
        self.store.update_crawl(self.id, status=self.state, started=self.started)
        kw = {"sleep": self._sleep} if self._sleep else {}
        self.fetcher = Fetcher(cfg, self.store, self._log, transport=self.transport, **kw)
        self.frontier = Frontier(cfg.focus_topic, self.embedder)
        self.dedup = DedupIndex(cfg.near_dup_threshold, self.embedder if cfg.semantic_dedup else None)
        self.seed_sites = {registrable(host_of(s)) for s in cfg.seeds}
        self.include = [re.compile(p, re.I) for p in cfg.include_patterns]
        self.exclude = [re.compile(p, re.I) for p in cfg.exclude_patterns]
        for s in cfg.seeds:
            u = normalize(s)
            if u:
                self.frontier.push(u, 0, None, "", 0.0, host_of(u))
        mode = f"topic-focused ('{cfg.focus_topic}')" if cfg.focus_topic else "breadth-first"
        self._log("INFO", f"Crawl started: {len(cfg.seeds)} seed(s), depth {cfg.max_depth}, max {cfg.max_pages} pages, "
                          f"{mode}, delay {cfg.min_delay_s:g}-{cfg.max_delay_s:g}s, robots.txt "
                          f"{'respected' if cfg.respect_robots else 'ignored'}")
        self.in_flight = 0
        last_flush = 0.0
        workers = [asyncio.create_task(self._worker(i)) for i in range(cfg.concurrency)]
        try:
            while True:
                self._drain_resumes()
                if time.time() - last_flush > 1:
                    self.store.update_crawl(self.id, stats=self.stats())
                    last_flush = time.time()
                if all(w.done() for w in workers):
                    break
                await asyncio.sleep(0.2)
        finally:
            for w in workers:
                w.cancel()
            await self.fetcher.close()
        open_hosts = {h: len(v) for h, v in self.paused_hosts.items() if v}
        final = "stopped" if self._stop.is_set() else "done"
        self.state = final
        msg = (f"Crawl {final}: {self.counts['fetched']} fetched, {self.counts['stored']} stored, "
               f"{self.counts['duplicate']} duplicates, {self.counts['low_quality']} low quality, "
               f"{self.counts['error']} errors")
        if open_hosts:
            msg += f"; unresolved challenges on {', '.join(open_hosts)}"
        self._log("INFO", msg)
        self.store.update_crawl(self.id, status=final, finished=time.time(), stats=self.stats())

    def _drain_resumes(self) -> None:
        while not self._resume_q.empty():
            host, cookies, skip = self._resume_q.get()
            items = self.paused_hosts.pop(host, [])
            self.store.resolve_challenges(self.id, host, "skipped" if skip else "resolved")
            if skip:
                self._log("INFO", f"{host}: skipped by user; {len(items)} queued page(s) dropped")
                continue
            if cookies and items:
                self.fetcher.set_cookies(items[0]["url"], cookies)
            for it in items:
                self.frontier._seen.discard(it["url"])
                self.frontier.push(it["url"], it["depth"], it["parent"], "", it.get("relevance", 0), host)
            self._log("INFO", f"{host}: challenge resolved; re-queued {len(items)} page(s)"
                              + (f" with {len(cookies)} cookie(s)" if cookies else ""))

    async def _worker(self, n: int) -> None:
        idle = 0.0
        while not self._stop.is_set():
            while not self._pause.is_set():
                await asyncio.sleep(0.2)
            if self._stop.is_set():
                return
            if self.counts["fetched"] + self.in_flight >= self.cfg.max_pages:
                if self.in_flight == 0:
                    return
                await asyncio.sleep(0.1)
                continue
            item = self.frontier.pop()
            if item is None:
                if self.in_flight == 0 and idle > 0.6 and not self._has_waiting_hosts():
                    return
                idle += 0.1
                await asyncio.sleep(0.1)
                continue
            idle = 0.0
            host = host_of(item["url"])
            if host in self.paused_hosts:
                self.paused_hosts[host].append(item)
                continue
            self.in_flight += 1
            try:
                await self._process(item, host)
            except Exception as e:
                self.counts["error"] += 1
                self._log("ERROR", f"{item['url']}: {e}")
            finally:
                self.in_flight -= 1

    def _has_waiting_hosts(self) -> bool:
        """Paused hosts with queued pages keep the crawl alive until the user resolves, skips or stops."""
        return self.cfg.on_challenge == "pause" and (any(self.paused_hosts.values()) or not self._resume_q.empty())

    async def _process(self, item: dict, host: str) -> None:
        cfg, url, depth = self.cfg, item["url"], item["depth"]
        base = {"url": url, "depth": depth, "parent": item["parent"], "priority": item["priority"]}
        if not await self.fetcher.allowed_by_robots(url):
            self.counts["robots_blocked"] += 1
            self.store.add_page(self.id, {**base, "outcome": "robots_blocked"})
            self._log("INFO", f"robots.txt disallows {url}")
            return
        self.recent.appendleft(url)
        res = await self.fetcher.fetch(url, item["parent"])
        self.counts["fetched"] += 1
        base.update(final_url=res.final_url, status_code=res.status, elapsed_ms=res.elapsed_ms,
                    from_cache=int(res.from_cache))
        if res.error and not res.status:
            self.counts["error"] += 1
            self.store.add_page(self.id, {**base, "outcome": "error", "error": res.error})
            self._log("ERROR", f"{url}: {res.error}")
            return
        kind = detect_challenge(res.status, res.body)
        if kind:
            self.store.cache_delete(url)  # never serve a challenge page from cache after it is solved
            self.counts["challenge"] += 1
            self.store.add_page(self.id, {**base, "outcome": "challenge", "error": kind})
            self.store.add_challenge(self.id, host, url, kind)
            if cfg.on_challenge == "pause":
                self.paused_hosts.setdefault(host, [])
                self.paused_hosts[host].append(item)
                self.frontier._seen.discard(url)
                self._log("WARNING", f"{kind} challenge on {url}: {host} paused until you solve it or skip it")
            else:
                self._log("WARNING", f"{kind} challenge on {url}: skipped")
            return
        if res.status >= 400:
            self.counts["error"] += 1
            self.store.add_page(self.id, {**base, "outcome": "error", "error": f"HTTP {res.status}"})
            self._log("WARNING", f"HTTP {res.status} for {url}")
            return
        if res.content_type and not res.content_type.startswith(("text/html", "application/xhtml")):
            self.counts["skipped"] += 1
            self.store.add_page(self.id, {**base, "outcome": "skipped", "error": f"not HTML ({res.content_type})"})
            return
        final = normalize(res.final_url) or url
        content = await asyncio.to_thread(extract_content, res.body, final)
        links = extract_links(res.body, final)
        added = 0
        if depth < cfg.max_depth:
            for link, anchor in links:
                ok, _ = allowed(link, self.seed_sites, cfg.same_domain_only, cfg.allowed_domains, self.include,
                                self.exclude)
                if ok and self.frontier.push(link, depth + 1, url, anchor, item["relevance"], host_of(link)):
                    added += 1
        f = features(content)
        q, reasons = self.quality.score(f)
        self.frontier.record_quality(host, q)
        text = content["text"]
        page = {**base, "title": content["title"], "text": text, "words": len(text.split()), "lang": content["lang"],
                "quality": q, "quality_reasons": reasons, "links_out": len(links), "content_hash": content_hash(text),
                "features": f}
        if cfg.quality_filter and q < cfg.quality_threshold:
            self.counts["low_quality"] += 1
            self.store.add_page(self.id, {**page, "outcome": "low_quality"})
            self._log("INFO", f"Low quality ({q:.2f}: {', '.join(reasons) or 'model score'}): {url}")
            return
        dup_of, dkind, sim = await asyncio.to_thread(self.dedup.check_and_add, url, text)
        if dup_of:
            self.counts["duplicate"] += 1
            self.dups[dkind] += 1
            self.store.add_page(self.id, {**page, "outcome": "duplicate", "dup_of": dup_of, "dup_kind": dkind})
            self._log("INFO", f"{dkind.capitalize()} duplicate (similarity {sim:.2f}) of {dup_of}: {url}")
            return
        pid = self.store.add_page(self.id, {**page, "outcome": "stored"})
        self.counts["stored"] += 1
        self._log("INFO", f"Stored [{q:.2f}] {content['title'][:70] or url} ({page['words']} words, "
                          f"+{added} links{', cached' if res.from_cache else ''})")
        if cfg.ai_extraction and self.ai_fn and self.counts["stored"] <= cfg.ai_max_pages:
            self._ai_q.put((pid, url, content["title"], text))
