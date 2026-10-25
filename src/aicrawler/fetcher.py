"""Polite async HTTP: robots.txt, per-host pacing with jitter, adaptive backoff, retries, HTTP cache."""
import time
import httpx
import random
import asyncio
from .urls import host_of
from dataclasses import dataclass
from urllib.robotparser import RobotFileParser

RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_BODY = 5 * 1024 * 1024
HTML_TYPES = ("text/html", "application/xhtml+xml")
BROWSER_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Upgrade-Insecure-Requests": "1",
    "DNT": "1",
}


@dataclass
class FetchResult:
    url: str
    status: int = 0
    final_url: str = ""
    content_type: str = ""
    body: str = ""
    from_cache: bool = False
    elapsed_ms: int = 0
    attempts: int = 0
    error: str | None = None


class HostState:
    def __init__(self, per_host: int):
        self.sem = asyncio.Semaphore(per_host)
        self.next_ok = 0.0
        self.penalty = 1.0  # multiplicative slow-down after 429/503, decays on success
        self.crawl_delay = 0.0
        self.robots: RobotFileParser | None = None
        self.robots_loaded = False


class Fetcher:
    def __init__(self, cfg, store, log, clock=time.monotonic, sleep=asyncio.sleep, transport=None):
        self.cfg, self.store, self.log = cfg, store, log
        self.clock, self.sleep = clock, sleep
        headers = {"User-Agent": cfg.user_agent, **(BROWSER_HEADERS if cfg.browser_like_headers else {})}
        self.client = httpx.AsyncClient(headers=headers, follow_redirects=True, max_redirects=10,
                                        timeout=httpx.Timeout(cfg.timeout_s, connect=min(10, cfg.timeout_s)),
                                        proxy=cfg.proxy or None, transport=transport,
                                        limits=httpx.Limits(max_connections=cfg.concurrency * 2))
        self.hosts: dict[str, HostState] = {}
        self.stats = {"requests": 0, "retries": 0, "cache_hits": 0, "revalidated": 0, "bytes": 0}

    async def close(self) -> None:
        await self.client.aclose()

    def host(self, url: str) -> HostState:
        h = host_of(url)
        if h not in self.hosts:
            self.hosts[h] = HostState(self.cfg.per_host_concurrency)
        return self.hosts[h]

    def set_cookies(self, url: str, cookies: dict[str, str]) -> None:
        domain = host_of(url).split(":")[0]
        for k, v in cookies.items():
            self.client.cookies.set(k, v, domain=domain)

    # ---------- robots ----------
    async def allowed_by_robots(self, url: str) -> bool:
        if not self.cfg.respect_robots:
            return True
        hs = self.host(url)
        if not hs.robots_loaded:
            hs.robots_loaded = True
            scheme = url.split("://")[0]
            robots_url = f"{scheme}://{host_of(url)}/robots.txt"
            rp = RobotFileParser()
            try:
                r = await self.client.get(robots_url, headers={"Accept": "text/plain"})
                if r.status_code == 200:
                    rp.parse(r.text.splitlines())
                    hs.robots = rp
                    delay = rp.crawl_delay(self.cfg.user_agent) or rp.crawl_delay("*")
                    if delay:
                        hs.crawl_delay = min(float(delay), 60.0)
                        self.log("INFO", f"{host_of(url)}: robots.txt crawl-delay {hs.crawl_delay:g}s")
                elif r.status_code in (401, 403):
                    rp.disallow_all = True
                    hs.robots = rp
            except httpx.HTTPError:
                hs.robots = None
        return hs.robots is None or hs.robots.can_fetch(self.cfg.user_agent, url)

    # ---------- pacing ----------
    async def _wait_turn(self, hs: HostState) -> None:
        base = random.uniform(self.cfg.min_delay_s, self.cfg.max_delay_s)
        delay = max(base, hs.crawl_delay) * hs.penalty
        now = self.clock()
        wait = max(0.0, hs.next_ok - now)
        hs.next_ok = max(now, hs.next_ok) + delay
        if wait > 0:
            await self.sleep(wait)

    # ---------- fetch ----------
    async def fetch(self, url: str, referer: str | None = None) -> FetchResult:
        res = FetchResult(url=url)
        cached = self.store.cache_get(url) if self.cfg.use_cache else None
        fresh = cached and (time.time() - cached["fetched_at"]) < self.cfg.cache_ttl_hours * 3600
        if fresh:
            self.stats["cache_hits"] += 1
            return FetchResult(url, cached["status"], cached["final_url"], cached["headers"].get("content-type", ""),
                               cached["body"].decode("utf-8", "replace"), True, 0, 0)
        hs = self.host(url)
        headers = {"Referer": referer} if referer else {}
        if cached and cached.get("etag"):
            headers["If-None-Match"] = cached["etag"]
        if cached and cached.get("last_modified"):
            headers["If-Modified-Since"] = cached["last_modified"]
        async with hs.sem:
            for attempt in range(self.cfg.max_retries + 1):
                await self._wait_turn(hs)
                res.attempts = attempt + 1
                t0 = time.perf_counter()
                try:
                    async with self.client.stream("GET", url, headers=headers) as r:
                        ctype = r.headers.get("content-type", "").lower()
                        res.status, res.final_url, res.content_type = r.status_code, str(r.url), ctype
                        if r.status_code == 304 and cached:
                            self.store.cache_touch(url)
                            self.stats["revalidated"] += 1
                            res.status, res.body, res.from_cache = cached["status"], cached["body"].decode("utf-8", "replace"), True
                            res.content_type = cached["headers"].get("content-type", "")
                            res.final_url = cached["final_url"]
                            hs.penalty = max(1.0, hs.penalty * 0.8)
                            break
                        if r.status_code in RETRY_STATUS and attempt < self.cfg.max_retries:
                            wait = self._retry_after(r.headers.get("retry-after"), attempt)
                            if r.status_code in (429, 503):
                                hs.penalty = min(8.0, hs.penalty * 2)
                            self.stats["retries"] += 1
                            self.log("WARNING", f"{r.status_code} on {url}; retry {attempt + 1}/{self.cfg.max_retries} "
                                                f"in {wait:.1f}s (host pace x{hs.penalty:g})")
                            await self.sleep(wait)
                            continue
                        if any(ctype.startswith(t) for t in HTML_TYPES) or not ctype:
                            chunks, size = [], 0
                            async for chunk in r.aiter_bytes():
                                size += len(chunk)
                                if size > MAX_BODY:
                                    res.error = "page larger than 5 MB; truncated"
                                    break
                                chunks.append(chunk)
                            raw = b"".join(chunks)
                            self.stats["bytes"] += len(raw)
                            res.body = raw.decode(r.encoding or "utf-8", "replace")
                            if self.cfg.use_cache and r.status_code == 200:
                                self.store.cache_put(url, r.status_code, res.final_url,
                                                     {k: v for k, v in r.headers.items() if k in
                                                      ("content-type", "etag", "last-modified")}, raw)
                        hs.penalty = max(1.0, hs.penalty * 0.8)
                        break
                except (httpx.TimeoutException, httpx.TransportError) as e:
                    res.error = f"{e.__class__.__name__}: {e}"[:300]
                    if attempt < self.cfg.max_retries:
                        wait = self._retry_after(None, attempt)
                        self.stats["retries"] += 1
                        self.log("WARNING", f"{e.__class__.__name__} on {url}; retry {attempt + 1}/"
                                            f"{self.cfg.max_retries} in {wait:.1f}s")
                        await self.sleep(wait)
                        continue
                    break
                finally:
                    res.elapsed_ms = round((time.perf_counter() - t0) * 1000)
                    self.stats["requests"] += 1
        if res.status and res.error and not res.error.startswith("page larger"):
            res.error = None
        return res

    @staticmethod
    def _retry_after(header: str | None, attempt: int) -> float:
        if header:
            try:
                return min(120.0, max(0.0, float(header)))
            except ValueError:
                pass
        return min(60.0, 2 ** (attempt + 1) + random.uniform(0, 1))
