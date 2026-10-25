import sys
import time
import asyncio
from pathlib import Path
from lxml.html import fromstring
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy


def extract_readable_text(html: str) -> str:
    """Extract readable text from HTML using lxml only."""
    if not html:
        return ""

    try:
        doc = fromstring(html)
        text = doc.text_content()

        # Clean whitespace + remove empty lines
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return "\n\n".join(lines)

    except Exception:
        # Fallback: return raw HTML
        return html.strip()


async def main():
    # Use LXML only (no SmartScrapingStrategy, no trafilatura)
    scraping_strategy = LXMLWebScrapingStrategy()

    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(max_depth=1, include_external=False),
        scraping_strategy=scraping_strategy,
        verbose=True
    )

    start_time = time.time()

    # Increase concurrency safely
    async with AsyncWebCrawler(concurrency=3) as crawler:
        results = await crawler.arun("https://www.wikipedia.org", config=config)

    elapsed = time.time() - start_time
    print(f"\nCrawled {len(results)} pages in {elapsed:.2f} seconds")

    output = []
    for r in results:
        url = getattr(r, "url", None)
        depth = r.metadata.get("depth") if r.metadata else None

        # Prefer markdown if available
        md = getattr(r, "markdown", None)
        if md:
            content = md
        else:
            html = getattr(r, "html", None)
            content = extract_readable_text(html) if html else ""

        output.append({
            "url": url,
            "depth": depth,
            "content": content
        })

    # ---- SAVE AS MARKDOWN ----
    out_path = Path("crawl_results.md")
    with out_path.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(output, start=1):
            f.write(f"# Page {idx}\n")
            f.write(f"**URL:** {item['url']}\n\n")
            f.write(f"**Depth:** {item['depth']}\n\n")
            f.write("## Content\n\n")
            f.write(item["content"] if item["content"] else "_(No content extracted)_")
            f.write("\n\n---\n\n")

    print(f"Saved {len(output)} items to {out_path.resolve()}")


if __name__ == "__main__":
    # Fix Windows async subprocess issue
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    asyncio.run(main())
