import os
import html
import json
import time
import httpx
import pandas as pd
import streamlit as st

BACKEND = os.getenv("BACKEND_URL", f"http://127.0.0.1:{os.getenv('BACKEND_PORT', '8002')}")
TERMINAL = ("done", "stopped", "failed")
LEVEL_COLOR = {"INFO": "#6b7280", "WARNING": "#b45309", "ERROR": "#b91c1c"}
DEFAULT_UA = "AICrawler/1.0 (+https://github.com/; polite research crawler)"
BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"

st.set_page_config(page_title="AI Web Crawler", layout="wide")
st.markdown("<style>.log{font-family:ui-monospace,Consolas,monospace;font-size:0.78rem;white-space:pre-wrap;"
            "line-height:1.35}</style>", unsafe_allow_html=True)


def api(method, path, **kw):
    return httpx.request(method, f"{BACKEND}{path}", timeout=kw.pop("timeout", 60), **kw)


try:
    health = api("GET", "/health", params={"check_llm": True}).json()
except httpx.HTTPError:
    st.error(f"Backend not reachable at {BACKEND}")
    st.stop()

st.title("AI Web Crawler")
st.caption(f"Quality model: {health['quality_model']}  |  LLM: {'connected' if health.get('llm') else 'unreachable'}"
           f"  |  HTTP cache: {health['cache_entries']} pages  |  Active crawls: {health['active_crawls']}")
tabs = st.tabs(["Configure", "Live monitor", "Results", "Search", "Quality model"])
ss = st.session_state
ss.setdefault("cfg", {})
ss.setdefault("crawl_id", None)

# ======================= Configure =======================
with tabs[0]:
    tpls = api("GET", "/templates").json()
    t1, t2, t3 = st.columns([2, 1, 1])
    names = ["(none)"] + [t["name"] for t in tpls]
    pick = t1.selectbox("Load a saved configuration", names)
    if t2.button("Load", use_container_width=True, disabled=pick == "(none)"):
        ss.cfg = next(t["config"] for t in tpls if t["name"] == pick)
        st.rerun()
    if t3.button("Use the demo site", use_container_width=True,
                 help="A local test website with duplicates, junk pages, robots.txt rules, a CAPTCHA wall and "
                      "flaky pages, so you can try every feature offline."):
        url = api("POST", "/demo/start").json()["url"]
        ss.cfg = {**ss.cfg, "seeds": [url], "max_depth": 4, "max_pages": 150, "min_delay_s": 0.1, "max_delay_s": 0.4}
        st.rerun()
    c = ss.cfg

    with st.form("crawl"):
        st.subheader("Target")
        seeds = st.text_area("Start URLs (one per line)", "\n".join(c.get("seeds", [])), height=80,
                             placeholder="https://docs.python.org/3/tutorial/")
        a1, a2, a3 = st.columns(3)
        depth = a1.slider("Max depth", 0, 8, c.get("max_depth", 2), help="0 = only the start pages")
        max_pages = a2.number_input("Max pages", 1, 20000, c.get("max_pages", 100))
        same = a3.toggle("Stay on the start site", c.get("same_domain_only", True))
        b1, b2, b3 = st.columns(3)
        allowed = b1.text_input("Allowed domains (comma separated)", ", ".join(c.get("allowed_domains", [])))
        include = b2.text_input("Only URLs matching (regex, comma separated)", ", ".join(c.get("include_patterns", [])))
        exclude = b3.text_input("Skip URLs matching (regex, comma separated)",
                                ", ".join(c.get("exclude_patterns", [r"\.(jpg|jpeg|png|gif|svg|webp|ico|css|js|zip|gz|mp4|mp3|woff2?)(\?|$)"])))

        st.subheader("Human-like pacing and politeness")
        p1, p2, p3, p4 = st.columns(4)
        delay = p1.slider("Random delay between requests to a site (s)", 0.0, 15.0,
                          (float(c.get("min_delay_s", 1.0)), float(c.get("max_delay_s", 2.5))), 0.1)
        conc = p2.slider("Parallel requests (total)", 1, 16, c.get("concurrency", 4))
        per_host = p3.slider("Parallel requests per site", 1, 6, c.get("per_host_concurrency", 2))
        timeout = p4.slider("Timeout (s)", 5, 90, int(c.get("timeout_s", 20)))
        q1, q2, q3, q4 = st.columns(4)
        retries = q1.slider("Retries on 429/5xx/timeouts", 0, 6, c.get("max_retries", 3))
        robots = q2.toggle("Respect robots.txt", c.get("respect_robots", True))
        browser_headers = q3.toggle("Browser-like headers and cookies", c.get("browser_like_headers", True))
        ua_choice = q4.selectbox("User-Agent", ["Identify as a crawler", "Desktop browser", "Custom"],
                                 index=0 if c.get("user_agent", DEFAULT_UA) == DEFAULT_UA else
                                 (1 if c.get("user_agent") == BROWSER_UA else 2))
        u1, u2 = st.columns(2)
        ua_custom = u1.text_input("Custom User-Agent", c.get("user_agent", DEFAULT_UA) if ua_choice == "Custom" else "")
        proxy = u2.text_input("Proxy (optional)", c.get("proxy") or "", placeholder="http://user:pass@host:port")

        st.subheader("Cache")
        k1, k2 = st.columns(2)
        use_cache = k1.toggle("Reuse cached pages (with conditional requests)", c.get("use_cache", True))
        ttl = k2.slider("Cache freshness (hours)", 0, 168, int(c.get("cache_ttl_hours", 24)))

        st.subheader("Intelligence")
        i1, i2, i3 = st.columns(3)
        qf = i1.toggle("Filter low-quality pages", c.get("quality_filter", True))
        qt = i2.slider("Quality threshold", 0.0, 1.0, float(c.get("quality_threshold", 0.5)), 0.05)
        nd = i3.slider("Near-duplicate similarity", 0.5, 1.0, float(c.get("near_dup_threshold", 0.8)), 0.05)
        j1, j2 = st.columns(2)
        sem = j1.toggle("Semantic dedup (local embeddings)", c.get("semantic_dedup", False))
        topic = j2.text_input("Focus topic (crawl the most relevant links first)", c.get("focus_topic", ""))
        ai = st.text_area("AI extraction instruction (optional, uses the LLM on each stored page)",
                          c.get("ai_extraction", ""), height=68,
                          placeholder="e.g. Return {title, price, availability} for product pages, or null")
        ai_max = st.slider("Pages to run AI extraction on", 0, 200, c.get("ai_max_pages", 20))

        st.subheader("CAPTCHA and bot challenges")
        ch = st.radio("When a site shows a CAPTCHA or bot check",
                      ["Pause that site and let me solve it", "Skip the page"],
                      index=0 if c.get("on_challenge", "pause") == "pause" else 1, horizontal=True)
        st.caption("The crawler never tries to defeat CAPTCHAs automatically. Paused sites resume after you "
                   "solve the check in a browser window or paste the session cookies.")

        n1, n2, n3 = st.columns([2, 1, 1])
        tpl_name = n1.text_input("Configuration name (to save it)")
        save = n2.form_submit_button("Save configuration", use_container_width=True)
        go = n3.form_submit_button("Start crawl", type="primary", use_container_width=True)

    def split(s):
        return [x.strip() for x in s.split(",") if x.strip()]

    if go or save:
        cfg = {"seeds": [s.strip() for s in seeds.splitlines() if s.strip()], "max_depth": depth,
               "max_pages": int(max_pages), "same_domain_only": same, "allowed_domains": split(allowed),
               "include_patterns": split(include), "exclude_patterns": split(exclude), "respect_robots": robots,
               "min_delay_s": delay[0], "max_delay_s": delay[1], "concurrency": conc, "per_host_concurrency": per_host,
               "timeout_s": timeout, "max_retries": retries, "browser_like_headers": browser_headers,
               "user_agent": {"Identify as a crawler": DEFAULT_UA, "Desktop browser": BROWSER_UA}.get(ua_choice, ua_custom or DEFAULT_UA),
               "proxy": proxy or None, "use_cache": use_cache, "cache_ttl_hours": ttl, "quality_filter": qf,
               "quality_threshold": qt, "near_dup_threshold": nd, "semantic_dedup": sem, "focus_topic": topic,
               "ai_extraction": ai, "ai_max_pages": ai_max,
               "on_challenge": "pause" if ch.startswith("Pause") else "skip"}
        ss.cfg = cfg
        if not robots:
            st.warning("robots.txt is being ignored. Only do this for sites you own or have permission to crawl.")
        if save:
            r = api("POST", "/templates", json={"name": tpl_name or "untitled", "config": cfg})
            st.success("Configuration saved.") if r.status_code == 200 else st.error(r.json()["detail"])
        if go:
            r = api("POST", "/crawls", json={"config": cfg})
            if r.status_code != 200:
                st.error(r.json().get("detail"))
            else:
                ss.crawl_id = r.json()["crawl_id"]
                st.success("Crawl started. Open the Live monitor tab.")
    if st.button("Clear HTTP cache"):
        st.info(f"Removed {api('DELETE', '/cache').json()['cleared']} cached pages.")

# ======================= Live monitor =======================
crawls = api("GET", "/crawls").json()
labels = {f"{c['name']}  [{c['status']}]  {c['id']}": c["id"] for c in crawls}


def pick_crawl(key: str):
    if not crawls:
        return None
    ids = list(labels.values())
    idx = ids.index(ss.crawl_id) if ss.crawl_id in ids else 0
    choice = st.selectbox("Crawl", list(labels), index=idx, key=key)
    ss.crawl_id = labels[choice]
    return ss.crawl_id


with tabs[1]:
    cid = pick_crawl("live_pick")
    if not cid:
        st.info("Start a crawl in the Configure tab.")
    else:
        @st.fragment(run_every=1.0)
        def live():
            cr = api("GET", f"/crawls/{cid}").json()
            s, cfg = cr["stats"], cr["config"]
            st.markdown(f"**Status:** {cr['status']}")
            fetched = s.get("fetched", 0)
            st.progress(min(1.0, fetched / max(1, cfg["max_pages"])), text=f"{fetched} of up to {cfg['max_pages']} pages")
            m = st.columns(8)
            for col, (lbl, key) in zip(m, [("Fetched", "fetched"), ("Stored", "stored"), ("Duplicates", "duplicate"),
                                           ("Low quality", "low_quality"), ("Errors", "error"),
                                           ("Challenges", "challenge"), ("Pages/min", "pages_per_min"),
                                           ("Queued", "frontier")]):
                col.metric(lbl, s.get(key, 0))
            st.caption(f"Cache hits {s.get('cache_hits', 0)}  |  revalidated {s.get('revalidated', 0)}  |  retries "
                       f"{s.get('retries', 0)}  |  robots blocked {s.get('robots_blocked', 0)}  |  "
                       f"{s.get('bytes', 0) / 1e6:.1f} MB  |  AI extracted {s.get('ai_extracted', 0)}")
            if cr["status"] not in TERMINAL:
                c1, c2, c3, _ = st.columns([1, 1, 1, 5])
                if cr["status"] == "paused":
                    if c1.button("Resume", key="resume"):
                        api("POST", f"/crawls/{cid}/resume")
                elif c1.button("Pause", key="pause"):
                    api("POST", f"/crawls/{cid}/pause")
                if c2.button("Stop", key="stop"):
                    api("POST", f"/crawls/{cid}/stop")
            open_ch = [c for c in cr["challenges"] if c["status"] == "open"]
            for host in sorted({c["host"] for c in open_ch}):
                ch = next(c for c in open_ch if c["host"] == host)
                with st.container(border=True):
                    st.warning(f"{ch['kind']} check on {host}: {s.get('paused_hosts', {}).get(host, 0)} page(s) are "
                               f"waiting. Example: {ch['url']}")
                    b1, b2, b3 = st.columns([1.2, 2.5, 1])
                    if b1.button("Solve in a browser window", key=f"br_{host}"):
                        with st.spinner("Solve the check in the window that opened, then close it..."):
                            r = api("POST", f"/crawls/{cid}/challenges/browser", json={"host": host}, timeout=330)
                        st.toast(r.json().get("detail") or f"Resumed with {r.json().get('cookies')} cookies")
                    ck = b2.text_input("or paste cookies (name=value; ...)", key=f"ck_{host}", label_visibility="collapsed",
                                       placeholder="or paste cookies: name=value; name2=value2")
                    if b2.button("Resume with cookies", key=f"rc_{host}"):
                        api("POST", f"/crawls/{cid}/challenges/resolve", json={"host": host, "cookies": ck})
                    if b3.button("Skip this site", key=f"sk_{host}"):
                        api("POST", f"/crawls/{cid}/challenges/resolve", json={"host": host, "skip": True})
            if s.get("recent") and cr["status"] not in TERMINAL:
                st.markdown("**Now crawling**")
                st.code("\n".join(s["recent"][:5]), language=None)
            ev = api("GET", f"/crawls/{cid}/events").json()[-60:]
            rows = "".join(f'<div class="log" style="color:{LEVEL_COLOR.get(e["level"], "#6b7280")}">'
                           f'{time.strftime("%H:%M:%S", time.localtime(e["ts"]))}  {e["level"]:<7} '
                           f'{html.escape(e["msg"])}</div>' for e in reversed(ev))
            with st.container(height=330, border=True):
                st.markdown(rows, unsafe_allow_html=True)

        live()

# ======================= Results =======================
with tabs[2]:
    cid = pick_crawl("res_pick")
    if cid:
        cr = api("GET", f"/crawls/{cid}").json()
        f1, f2 = st.columns([1, 2])
        outcome = f1.selectbox("Outcome", ["stored", "all", "duplicate", "low_quality", "error", "challenge",
                                           "robots_blocked", "skipped"])
        q = f2.text_input("Filter by URL or title")
        pages = api("GET", f"/crawls/{cid}/pages", params={"outcome": None if outcome == "all" else outcome,
                                                            "q": q or None, "limit": 2000}).json()
        if pages:
            df = pd.DataFrame(pages)
            cols = ["url", "outcome", "title", "quality", "words", "depth", "status_code", "dup_kind", "from_cache"]
            st.dataframe(df[cols], hide_index=True, use_container_width=True, height=320,
                         column_config={"quality": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                                        "url": st.column_config.LinkColumn()})
            pick = st.selectbox("Inspect page", [p["url"] for p in pages])
            p = api("GET", f"/pages/{next(x['id'] for x in pages if x['url'] == pick)}").json()
            d1, d2 = st.columns([2, 1])
            with d1:
                st.markdown(f"**{p['title'] or p['url']}**")
                st.text(p["text"][:3000] if p.get("text") else "(no text)")
            with d2:
                st.json({k: p[k] for k in ("outcome", "quality", "quality_reasons", "dup_of", "dup_kind", "words",
                                           "lang", "status_code", "elapsed_ms", "error")})
                if p.get("ai_data"):
                    st.markdown("**AI extraction**")
                    st.json(p["ai_data"])
        else:
            st.caption("No pages match.")
        st.markdown("**Export**")
        e = st.columns(4)
        for col, fmt, mime in zip(e, ["jsonl", "csv", "html", "zip"],
                                  ["application/x-ndjson", "text/csv", "text/html", "application/zip"]):
            col.download_button(f"{fmt.upper()}", api("GET", f"/crawls/{cid}/export", params={"format": fmt}).content,
                                file_name=f"crawl_{cid}.{fmt}", mime=mime, use_container_width=True)
        if "127.0.0.1" in cr["config"]["seeds"][0] or "site.test" in cr["config"]["seeds"][0]:
            if st.button("Evaluate against the demo site's ground truth"):
                st.json(api("GET", f"/crawls/{cid}/evaluate").json())
        if cr["status"] in TERMINAL and st.button("Delete this crawl"):
            api("DELETE", f"/crawls/{cid}")
            ss.crawl_id = None
            st.rerun()

# ======================= Search =======================
with tabs[3]:
    sq = st.text_input("Full-text search across stored pages")
    scope = st.selectbox("In", ["All crawls"] + list(labels))
    if sq:
        hits = api("GET", "/search", params={"q": sq, "crawl_id": None if scope == "All crawls" else labels[scope],
                                             "limit": 30}).json()
        if not hits:
            st.caption("No results.")
        for h in hits:
            snippet = html.escape(h["snippet"]).replace("[", "<mark>").replace("]", "</mark>")
            st.markdown(f"**[{html.escape(h['title'] or h['url'])}]({h['url']})**  \n{snippet}", unsafe_allow_html=True)

# ======================= Quality model =======================
with tabs[4]:
    qm = api("GET", "/quality/model").json()
    if not qm["trained"]:
        st.info("No learned model yet; the rule baseline is used. Train one with: "
                ".venv\\Scripts\\python scripts\\train_quality.py")
    else:
        m = qm["meta"]
        st.caption(f"Trained {m.get('trained_at')} on {m.get('train_pages')} pages "
                   f"({m.get('positives')} good / {m.get('negatives')} junk) from {', '.join(m.get('sources', []))}")
        rows = []
        for name, key in [("Learned model, all test pages", "test_learned"), ("Rule baseline, all test pages", "test_rules"),
                          ("Learned model, real sites only", "test_real_sites_learned"),
                          ("Rule baseline, real sites only", "test_real_sites_rules")]:
            if m.get(key):
                rows.append({"model": name, **m[key]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Positive class = good page. Test pages were held out from training.")
