# AI Web Crawler

A polite, intelligent web crawler with a local web interface. Configure depth, pacing, caching and filters; then watch the crawl live. It handles:

- **Quality:** a learned model filters out low-quality pages.
- **Duplicates:** exact, near-duplicate and semantic copies are removed.
- **Focus:** the most on-topic links are crawled first.
- **AI extraction:** the LLM returns structured data from each page.
- **CAPTCHAs and bot checks:** the affected site is paused until you resolve it.

Everything is exportable and searchable. A one-click launcher sets it up on Windows, and a built-in demo website lets you try every feature offline.

![Live monitor](docs/screenshots/live.png)

## Contents

- [Quick start](#quick-start)
- [Features](#features)
- [Results](#results)
- [How it works](#how-it-works)
- [Responsible crawling](#responsible-crawling)
- [API](#api)
- [Development](#development)

## Quick start

**Requirements:** Windows 10/11 with Python 3.10 or newer ([python.org](https://www.python.org/downloads/), with "Add python.exe to PATH" ticked). An OpenAI-compatible LLM endpoint is only needed for AI extraction.

1. Download or clone this repository.
2. Double-click `launcher.exe`.

The first run takes 2–3 minutes. The launcher creates `.venv/`, installs `requirements.txt`, creates `.env` from `.env.example`, starts the API and UI, and opens `http://localhost:8502`. Closing the launcher window stops everything.

To try it straight away, click **Use the demo site** on the Configure tab, then **Start crawl**.

The demo site is a local website with ground truth. It contains:

- 24 articles, plus print and mobile copies of each
- tag, archive and login pages
- a robots.txt rule and a CAPTCHA wall
- flaky, rate-limited, slow and redirecting pages

Optional `.env` settings for AI extraction:

```
LLM_BASE_URL=http://localhost:8080/v1
LLM_API_KEY=your-key
LLM_MODEL=qwen3.6-35b-a3b
```

On Linux or macOS, run `./launch.sh`.

## Features

| Area | What you can configure or see |
|---|---|
| Scope | Start URLs, maximum depth, maximum pages, stay on the start site or an allow-list of domains, include and exclude URL regexes |
| Pacing | Random delay range per site, total and per-site parallelism, timeouts, retries; robots.txt and `Crawl-delay` respected by default |
| Human-like behaviour | Browser-like headers, persistent session cookies, `Referer` set to the linking page, adaptive slow-down when a site returns 429 or 503, `Retry-After` honoured, optional proxy, configurable User-Agent |
| Caching | Local HTTP cache with a freshness window; stale entries are revalidated with `ETag` / `If-Modified-Since` |
| Quality | Logistic-regression model over 12 interpretable page features, trained on demo pages and LLM-labelled real pages; every rejected page shows its reasons |
| Deduplication | Exact (normalised SHA-256), near-duplicate (MinHash with LSH) and optional semantic (`bge-small` embeddings, local ONNX) |
| Scheduling | Breadth-first by default; with a focus topic, links are ranked by anchor and URL relevance (keywords plus embeddings), site quality and depth. Likely copies such as `/print/` and `/amp/` versions are queued last |
| AI extraction | A natural-language instruction (for example "return title, price and availability") run by the LLM on each stored page; results appear in the UI and in exports |
| CAPTCHAs and bot checks | Detects reCAPTCHA, hCaptcha, Cloudflare Turnstile and bot-check pages. The site is paused (or the page skipped) and you resolve it by solving it in a browser window, pasting session cookies, or skipping the site |
| Live monitor | Progress, pages per minute, outcomes, the pages being fetched right now, pause / resume / stop, colour-coded event log |
| Results | Filterable page table, page text and AI data, full-text search (SQLite FTS5), export as JSONL, CSV, HTML report or corpus zip, saved configurations |

### Screenshots

| Configure | Results |
|---|---|
| ![Configure](docs/screenshots/configure.png) | ![Results](docs/screenshots/results.png) |

| CAPTCHA handling | Quality model |
|---|---|
| ![Challenge](docs/screenshots/challenge.png) | ![Quality](docs/screenshots/quality.png) |

## Results

**Demo site.** These results come from `scripts/run_eval.py`, scored against the site's ground truth.

| Measure | Result |
|---|---|
| Unique articles stored (coverage) | 24 / 24 |
| Duplicate pages handled (27 exact, 24 near) | 51 / 51, with no extra copies stored |
| Junk pages stored | 0 / 12 |
| Good pages wrongly rejected | 0 / 75 |
| robots.txt respected, CAPTCHA detected, 503 and 429 pages recovered after retry | yes |
| Full crawl of 90 pages (time-compressed delays) | about 2.5 s |

**Topic-focused scheduling.** Unique on-topic articles stored within the first 30 pages fetched:

| Topic | Breadth-first | Focused |
|---|---|---|
| Cooking (linked deep in the site) | 5 | 12 (all of them) |
| Python (linked from the home page) | 12 | 12 |

**Learned quality model versus the rule baseline.** The test pages were held out from training. The training data was 207 pages: 87 demo pages plus 120 pages from Python docs, Hacker News, Wikipedia and a product catalogue, labelled good or junk by the LLM. Positive class = good page.

| Test set | Model | Precision | Recall | F1 |
|---|---|---|---|---|
| Real sites (40 pages) | Rules | 0.94 | 0.63 | 0.75 |
| Real sites (40 pages) | **Learned** | **0.95** | **0.75** | **0.84** |
| All held-out pages (63) | Rules | 0.97 | 0.80 | 0.88 |
| All held-out pages (63) | **Learned** | **0.975** | **0.87** | **0.92** |

**Live crawl.** On books.toscrape.com, a public sandbox site for scraping practice, with AI extraction:

- 16 pages fetched in about 20 s at a 1–2 s polite delay
- 12 of 13 book pages kept; the category and listing pages rejected as low quality
- title, price and stock correctly extracted from all 12 stored pages

**Lesson from the first model.** It was trained without any product pages and rejected half of the book pages. Adding a product catalogue to the training data fixed this, which is why the training script crawls diverse sites.

**Limitations.** Star ratings rendered purely with CSS classes are not visible to the text-based extractor. The quality model is small and site-agnostic: retrain it with `scripts/train_quality.py` on sites like the ones you crawl.

## How it works

```
seeds -> frontier (priority queue) -> robots.txt -> per-site pacing -> fetch (cache / retry / backoff)
                 ^                                                          |
                 |                                         challenge?  -> pause site, wait for user
                 |                                                          |
          new links (scoped, scored) <- extract (trafilatura + lxml) <- HTML
                                                    |
                                     quality model -> dedup (exact / MinHash / embeddings)
                                                    |
                                  store + FTS index -> optional AI extraction (LLM worker)
```

| Module | Responsibility |
|---|---|
| `fetcher.py` | Async HTTP client, robots.txt, per-site semaphores and jittered pacing, adaptive penalty, retries, HTTP cache |
| `engine.py` | Worker pool, page pipeline, pause / resume / stop, challenge pausing and re-queueing, AI extraction worker |
| `frontier.py` | Breadth-first or topic-priority scheduling, copy detection |
| `extract.py` | Main-content extraction, link discovery, challenge detection |
| `quality.py` | Features, rule baseline, logistic regression (NumPy), evaluation |
| `dedup.py` | SHA-256, MinHash with LSH banding, semantic similarity |
| `testsite.py`, `evaluation.py` | Demo website with ground truth, and crawl scoring |
| `api.py`, `store.py` | REST API, SQLite (WAL) storage with FTS5 |

## Responsible crawling

The crawler is designed to be a good citizen:

- robots.txt is respected by default, and turning it off shows a warning.
- Delays and per-site limits are on by default.
- It slows down automatically when a server pushes back.
- The default User-Agent identifies it as a crawler.

It **does not** attempt to defeat CAPTCHAs or bot protection automatically. When a site presents one, the crawl pauses that site until a person resolves it. Crawl only sites you are permitted to crawl, and follow their terms of service.

## API

Interactive documentation is available at `http://localhost:8002/docs`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/crawls` | Start a crawl with a full configuration |
| GET | `/crawls`, `/crawls/{id}` | Status, live stats and open challenges |
| POST | `/crawls/{id}/pause`, `/resume`, `/stop` | Control a running crawl |
| GET | `/crawls/{id}/events?after=`, `/crawls/{id}/pages`, `/pages/{id}` | Event log, page list with filters, page detail |
| POST | `/crawls/{id}/challenges/resolve`, `/crawls/{id}/challenges/browser` | Resume a paused site with cookies or skip it; or solve in a browser window (needs Playwright) |
| GET | `/crawls/{id}/export?format=jsonl\|csv\|html\|zip` | Exports |
| GET | `/search?q=` | Full-text search over stored pages |
| GET, POST, DELETE | `/templates` | Saved configurations |
| POST | `/demo/start` | Start the local demo website |
| GET | `/crawls/{id}/evaluate` | Score a demo-site crawl against ground truth |

## Development

```
.venv/Scripts/python -m pytest                        # 43 tests, offline (demo site served in-process)
.venv/Scripts/python scripts/run_eval.py --real-embeddings
.venv/Scripts/python scripts/train_quality.py         # re-train the quality model (uses the LLM as labeller)
powershell -File scripts/build_launcher.ps1           # rebuild launcher.exe
```

Optional browser-based challenge solving: `.venv\Scripts\pip install playwright` then `.venv\Scripts\playwright install chromium`.
