import re
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

TRACKING = re.compile(r"^(utm_|fbclid$|gclid$|mc_|ref$|ref_src$|sessionid$|phpsessid$)", re.I)


def normalize(url: str, base: str | None = None) -> str | None:
    """Absolute, lowercase host, no fragment, no default port, tracking params removed, sorted query."""
    try:
        u = urljoin(base, url.strip()) if base else url.strip()
        parts = urlsplit(u)
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None
    host = parts.hostname or ""
    port = parts.port
    netloc = host.lower() + (f":{port}" if port and not (parts.scheme == "http" and port == 80 or
                                                         parts.scheme == "https" and port == 443) else "")
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    query = urlencode(sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                             if not TRACKING.match(k)))
    return urlunsplit((parts.scheme, netloc, path, query, ""))


def host_of(url: str) -> str:
    return urlsplit(url).netloc.lower()


def registrable(host: str) -> str:
    """Approximate site identity: last two labels (three for common 2-level TLDs like co.uk)."""
    host = host.split(":")[0]
    labels = host.split(".")
    if len(labels) >= 3 and labels[-2] in ("co", "com", "org", "net", "ac", "gov", "edu") and len(labels[-1]) == 2:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:]) if len(labels) >= 2 else host


def allowed(url: str, seeds_sites: set[str], same_domain: bool, allowed_domains: list[str],
            include: list[re.Pattern], exclude: list[re.Pattern]) -> tuple[bool, str]:
    h = host_of(url)
    if allowed_domains:
        if not any(h == d or h.endswith("." + d) for d in allowed_domains):
            return False, "domain not allowed"
    elif same_domain and registrable(h) not in seeds_sites:
        return False, "external link"
    if any(p.search(url) for p in exclude):
        return False, "excluded pattern"
    if include and not any(p.search(url) for p in include):
        return False, "not in include patterns"
    return True, ""
