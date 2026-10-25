import json
import time
import uuid
import sqlite3
import threading
from pathlib import Path

SCHEMA_VERSION = 1
SCHEMA = """
CREATE TABLE IF NOT EXISTS crawls (
    id TEXT PRIMARY KEY, name TEXT, config TEXT NOT NULL, status TEXT NOT NULL, stats TEXT,
    created REAL NOT NULL, started REAL, finished REAL, error TEXT
);
CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, crawl_id TEXT NOT NULL REFERENCES crawls(id) ON DELETE CASCADE,
    url TEXT NOT NULL, final_url TEXT, depth INTEGER, parent TEXT, status_code INTEGER, outcome TEXT NOT NULL,
    title TEXT, text TEXT, words INTEGER, lang TEXT, quality REAL, quality_reasons TEXT, dup_of TEXT,
    dup_kind TEXT, content_hash TEXT, links_out INTEGER, priority REAL, elapsed_ms INTEGER,
    from_cache INTEGER DEFAULT 0, ai_data TEXT, features TEXT, error TEXT, fetched_at REAL NOT NULL,
    UNIQUE(crawl_id, url)
);
CREATE INDEX IF NOT EXISTS idx_pages_crawl ON pages(crawl_id, outcome);
CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(title, text, url UNINDEXED, page_id UNINDEXED,
    crawl_id UNINDEXED, tokenize = 'porter');
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, crawl_id TEXT NOT NULL, ts REAL NOT NULL, level TEXT NOT NULL, msg TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events ON events(crawl_id, id);
CREATE TABLE IF NOT EXISTS challenges (
    id INTEGER PRIMARY KEY AUTOINCREMENT, crawl_id TEXT NOT NULL, host TEXT NOT NULL, url TEXT NOT NULL,
    kind TEXT NOT NULL, status TEXT NOT NULL, created REAL NOT NULL, resolved REAL
);
CREATE TABLE IF NOT EXISTS http_cache (
    url TEXT PRIMARY KEY, status INTEGER, final_url TEXT, headers TEXT, body BLOB, etag TEXT,
    last_modified TEXT, fetched_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS templates (name TEXT PRIMARY KEY, config TEXT NOT NULL, created REAL NOT NULL);
"""


def _j(v):
    return None if v is None else json.dumps(v)


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = WAL")
        current = self._db.execute("PRAGMA user_version").fetchone()[0]
        has = self._db.execute("SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
        if has and current != SCHEMA_VERSION:
            raise RuntimeError(f"Database {path} uses another schema version; delete it to start fresh.")
        self._db.executescript(SCHEMA)
        self._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self._lock = threading.RLock()

    def _rows(self, sql, args=()):
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args)]

    def _one(self, sql, args=()):
        with self._lock:
            r = self._db.execute(sql, args).fetchone()
        return dict(r) if r else None

    def _exec(self, sql, args=()):
        with self._lock, self._db:
            return self._db.execute(sql, args)

    # ---------- crawls ----------
    def create_crawl(self, config: dict, name: str | None) -> str:
        cid = uuid.uuid4().hex[:10]
        self._exec("INSERT INTO crawls (id, name, config, status, created) VALUES (?,?,?,?,?)",
                   (cid, name, json.dumps(config), "queued", time.time()))
        return cid

    def update_crawl(self, cid: str, **kw) -> None:
        if "stats" in kw:
            kw["stats"] = _j(kw["stats"])
        cols = ", ".join(f"{k}=?" for k in kw)
        self._exec(f"UPDATE crawls SET {cols} WHERE id=?", (*kw.values(), cid))

    def get_crawl(self, cid: str) -> dict | None:
        c = self._one("SELECT * FROM crawls WHERE id=?", (cid,))
        if c:
            c["config"] = json.loads(c["config"])
            c["stats"] = json.loads(c["stats"]) if c["stats"] else {}
        return c

    def list_crawls(self) -> list[dict]:
        out = self._rows("SELECT * FROM crawls ORDER BY created DESC")
        for c in out:
            c["config"] = json.loads(c["config"])
            c["stats"] = json.loads(c["stats"]) if c["stats"] else {}
        return out

    def delete_crawl(self, cid: str) -> bool:
        with self._lock, self._db:
            self._db.execute("DELETE FROM pages_fts WHERE crawl_id=?", (cid,))
            self._db.execute("DELETE FROM events WHERE crawl_id=?", (cid,))
            self._db.execute("DELETE FROM challenges WHERE crawl_id=?", (cid,))
            return self._db.execute("DELETE FROM crawls WHERE id=?", (cid,)).rowcount > 0

    def mark_interrupted(self) -> None:
        self._exec("UPDATE crawls SET status='stopped', error='Interrupted by restart', finished=? "
                   "WHERE status IN ('queued','running','paused','stopping')", (time.time(),))

    # ---------- pages ----------
    def add_page(self, cid: str, p: dict) -> int:
        cols = ["crawl_id", "url", "final_url", "depth", "parent", "status_code", "outcome", "title", "text", "words",
                "lang", "quality", "quality_reasons", "dup_of", "dup_kind", "content_hash", "links_out", "priority",
                "elapsed_ms", "from_cache", "ai_data", "features", "error", "fetched_at"]
        vals = {**p, "crawl_id": cid, "fetched_at": time.time(), "quality_reasons": _j(p.get("quality_reasons")),
                "ai_data": _j(p.get("ai_data")), "features": _j(p.get("features"))}
        with self._lock, self._db:
            cur = self._db.execute(
                f"INSERT OR REPLACE INTO pages ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                [vals.get(c) for c in cols])
            pid = cur.lastrowid
            if p.get("outcome") == "stored":
                self._db.execute("INSERT INTO pages_fts (title, text, url, page_id, crawl_id) VALUES (?,?,?,?,?)",
                                 (p.get("title") or "", p.get("text") or "", p["url"], pid, cid))
        return pid

    def set_ai_data(self, page_id: int, data: dict) -> None:
        self._exec("UPDATE pages SET ai_data=? WHERE id=?", (json.dumps(data), page_id))

    def pages(self, cid: str, outcome: str | None = None, q: str | None = None, limit: int = 500,
              offset: int = 0, with_text: bool = False) -> list[dict]:
        cols = "*" if with_text else ("id, url, final_url, depth, status_code, outcome, title, words, lang, quality, "
                                      "quality_reasons, dup_of, dup_kind, links_out, priority, elapsed_ms, from_cache, "
                                      "ai_data, error, fetched_at")
        sql, args = f"SELECT {cols} FROM pages WHERE crawl_id=?", [cid]
        if outcome:
            sql += " AND outcome=?"
            args.append(outcome)
        if q:
            sql += " AND (url LIKE ? OR title LIKE ?)"
            args += [f"%{q}%", f"%{q}%"]
        out = self._rows(sql + " ORDER BY id LIMIT ? OFFSET ?", (*args, limit, offset))
        for p in out:
            p["quality_reasons"] = json.loads(p["quality_reasons"]) if p.get("quality_reasons") else []
            p["ai_data"] = json.loads(p["ai_data"]) if p.get("ai_data") else None
            if "features" in p:
                p["features"] = json.loads(p["features"]) if p["features"] else None
        return out

    def page(self, page_id: int) -> dict | None:
        p = self._one("SELECT * FROM pages WHERE id=?", (page_id,))
        if p:
            p["quality_reasons"] = json.loads(p["quality_reasons"]) if p["quality_reasons"] else []
            p["ai_data"] = json.loads(p["ai_data"]) if p["ai_data"] else None
            p["features"] = json.loads(p["features"]) if p["features"] else None
        return p

    def outcome_counts(self, cid: str) -> dict:
        return {r["outcome"]: r["n"] for r in self._rows(
            "SELECT outcome, count(*) n FROM pages WHERE crawl_id=? GROUP BY outcome", (cid,))}

    def search(self, query: str, cid: str | None = None, limit: int = 20) -> list[dict]:
        terms = " ".join(f'"{t}"' for t in query.replace('"', " ").split() if t)
        if not terms:
            return []
        sql = ("SELECT page_id, url, title, crawl_id, snippet(pages_fts, 1, '[', ']', ' ... ', 18) AS snippet, "
               "bm25(pages_fts) AS rank FROM pages_fts WHERE pages_fts MATCH ?")
        args = [terms]
        if cid:
            sql += " AND crawl_id=?"
            args.append(cid)
        return self._rows(sql + " ORDER BY rank LIMIT ?", (*args, limit))

    # ---------- events & challenges ----------
    def event(self, cid: str, level: str, msg: str) -> None:
        self._exec("INSERT INTO events (crawl_id, ts, level, msg) VALUES (?,?,?,?)", (cid, time.time(), level, msg))

    def events(self, cid: str, after: int = 0, limit: int = 500) -> list[dict]:
        return self._rows("SELECT * FROM events WHERE crawl_id=? AND id>? ORDER BY id LIMIT ?", (cid, after, limit))

    def add_challenge(self, cid: str, host: str, url: str, kind: str) -> int:
        return self._exec("INSERT INTO challenges (crawl_id, host, url, kind, status, created) VALUES (?,?,?,?,?,?)",
                          (cid, host, url, kind, "open", time.time())).lastrowid

    def challenges(self, cid: str) -> list[dict]:
        return self._rows("SELECT * FROM challenges WHERE crawl_id=? ORDER BY id", (cid,))

    def resolve_challenges(self, cid: str, host: str, status: str = "resolved") -> int:
        return self._exec("UPDATE challenges SET status=?, resolved=? WHERE crawl_id=? AND host=? AND status='open'",
                          (status, time.time(), cid, host)).rowcount

    # ---------- HTTP cache ----------
    def cache_get(self, url: str) -> dict | None:
        r = self._one("SELECT * FROM http_cache WHERE url=?", (url,))
        if r:
            r["headers"] = json.loads(r["headers"]) if r["headers"] else {}
        return r

    def cache_put(self, url: str, status: int, final_url: str, headers: dict, body: bytes) -> None:
        self._exec("INSERT OR REPLACE INTO http_cache VALUES (?,?,?,?,?,?,?,?)",
                   (url, status, final_url, json.dumps(headers), body, headers.get("etag"),
                    headers.get("last-modified"), time.time()))

    def cache_delete(self, url: str) -> None:
        self._exec("DELETE FROM http_cache WHERE url=?", (url,))

    def cache_touch(self, url: str) -> None:
        self._exec("UPDATE http_cache SET fetched_at=? WHERE url=?", (time.time(), url))

    def cache_clear(self) -> int:
        return self._exec("DELETE FROM http_cache").rowcount

    def cache_size(self) -> int:
        return self._one("SELECT count(*) n FROM http_cache")["n"]

    # ---------- templates ----------
    def save_template(self, name: str, config: dict) -> None:
        self._exec("INSERT OR REPLACE INTO templates VALUES (?,?,?)", (name, json.dumps(config), time.time()))

    def templates(self) -> list[dict]:
        out = self._rows("SELECT * FROM templates ORDER BY name")
        for t in out:
            t["config"] = json.loads(t["config"])
        return out

    def delete_template(self, name: str) -> bool:
        return self._exec("DELETE FROM templates WHERE name=?", (name,)).rowcount > 0
