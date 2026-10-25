"""Main-content extraction (trafilatura with an lxml fallback), link discovery and challenge detection."""
import re
import json
from .urls import normalize
from lxml import html as lhtml

CHALLENGE_MARKERS = [
    ("recaptcha", re.compile(r"g-recaptcha|www\.google\.com/recaptcha|grecaptcha", re.I)),
    ("hcaptcha", re.compile(r"h-captcha|hcaptcha\.com", re.I)),
    ("turnstile", re.compile(r"challenges\.cloudflare\.com|cf-turnstile", re.I)),
    ("cloudflare", re.compile(r"cf-browser-verification|Just a moment\.\.\.|Attention Required! \| Cloudflare|"
                              r"cf_chl_opt", re.I)),
    ("bot_wall", re.compile(r"verify (that )?you are (a )?human|are you a robot|unusual traffic from your "
                            r"computer|access denied.{0,40}bot", re.I)),
]


def detect_challenge(status: int, body: str) -> str | None:
    """Return the challenge kind when the page is a CAPTCHA / bot wall instead of content."""
    head = body[:60000]
    for kind, rx in CHALLENGE_MARKERS:
        if rx.search(head):
            # a page that merely embeds a reCAPTCHA on a contact form is still content; walls are short
            if kind in ("recaptcha", "hcaptcha") and status == 200 and len(_visible_text(head)) > 1500:
                continue
            return kind
    if status in (403, 429, 503) and len(_visible_text(head)) < 400 and re.search(r"captcha|robot|bot", head, re.I):
        return "bot_wall"
    return None


def _visible_text(body: str) -> str:
    try:
        doc = lhtml.fromstring(body)
        for bad in doc.xpath("//script|//style|//noscript"):
            bad.drop_tree()
        return " ".join(doc.text_content().split())
    except Exception:
        return ""


def extract_links(body: str, base_url: str) -> list[tuple[str, str]]:
    """Return (absolute normalized url, anchor text) pairs, honouring <base href> and rel=nofollow."""
    try:
        doc = lhtml.fromstring(body)
    except Exception:
        return []
    base = doc.xpath("//base/@href")
    base_url = normalize(base[0], base_url) or base_url if base else base_url
    seen, out = set(), []
    for a in doc.xpath("//a[@href]"):
        if "nofollow" in (a.get("rel") or "").lower():
            continue
        href = a.get("href", "")
        if href.startswith(("mailto:", "javascript:", "tel:", "#")):
            continue
        u = normalize(href, base_url)
        if u and u not in seen:
            seen.add(u)
            out.append((u, " ".join(a.text_content().split())[:200]))
    return out


def extract_content(body: str, url: str) -> dict:
    """Title, main text, language and simple structure stats used by the quality model."""
    import trafilatura

    title, text, lang = None, None, None
    try:
        out = trafilatura.extract(body, url=url, output_format="json", with_metadata=True, include_comments=False,
                                  include_tables=True, favor_recall=True)
        if out:
            d = json.loads(out)
            title, text, lang = d.get("title"), d.get("text"), d.get("language")
    except Exception:
        pass
    try:
        doc = lhtml.fromstring(body)
    except Exception:
        doc = None
    if doc is not None and not title:
        t = doc.xpath("//title/text()")
        title = " ".join(t[0].split()) if t else None
    full_text = _visible_text(body)
    if not text:
        text = full_text
    anchors = " ".join(" ".join(a.text_content().split()) for a in doc.xpath("//a")) if doc is not None else ""
    return {
        "title": (title or "")[:300],
        "text": text or "",
        "lang": lang,
        "full_text_chars": len(full_text),
        "anchor_chars": len(anchors),
        "n_links": len(doc.xpath("//a[@href]")) if doc is not None else 0,
        "n_forms": len(doc.xpath("//form")) if doc is not None else 0,
        "n_inputs": len(doc.xpath("//input[@type='password' or @type='email']")) if doc is not None else 0,
    }
