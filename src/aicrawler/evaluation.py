"""Score a crawl of the demo site against its ground truth."""
from urllib.parse import urlsplit
from collections import defaultdict


def _path(url: str) -> str:
    return urlsplit(url).path or "/"


def evaluate_crawl(pages: list[dict], gt: dict, topic: str | None = None, first_n: int | None = None) -> dict:
    by_path = {}
    for p in pages:
        by_path.setdefault(_path(p["url"]), p)
    fetched = [p for p in pages if p["outcome"] not in ("robots_blocked",)]

    good_clusters = {g["cluster"] for g in gt.values() if g["label"] == "good" and g.get("cluster")}
    stored_per_cluster = defaultdict(int)
    seen_clusters = set()
    for path, p in by_path.items():
        g = gt.get(path)
        if g and g.get("cluster"):
            seen_clusters.add(g["cluster"])
            if p["outcome"] == "stored":
                stored_per_cluster[g["cluster"]] += 1
    covered = sum(1 for c in good_clusters if stored_per_cluster[c] >= 1)
    extra_copies = sum(max(0, n - 1) for n in stored_per_cluster.values())
    dup_pages = [path for path, g in gt.items() if g.get("dup_of") and path in by_path]
    dup_caught = sum(1 for path in dup_pages
                     if by_path[path]["outcome"] == "duplicate" or stored_per_cluster[gt[path]["cluster"]] == 1)

    junk = [path for path, g in gt.items() if g["label"] == "junk" and path in by_path]
    junk_stored = [path for path in junk if by_path[path]["outcome"] == "stored"]
    good_pages = [path for path, g in gt.items() if g["label"] == "good" and path in by_path]
    good_rejected = [path for path in good_pages if by_path[path]["outcome"] == "low_quality"]

    def outcome(path):
        return by_path.get(path, {}).get("outcome")

    out = {
        "pages_fetched": len(fetched),
        "stored": sum(1 for p in pages if p["outcome"] == "stored"),
        "coverage": round(covered / len(good_clusters), 3) if good_clusters else None,
        "clusters_seen": len(seen_clusters & good_clusters),
        "duplicate_pages_handled": f"{dup_caught}/{len(dup_pages)}",
        "extra_copies_stored": extra_copies,
        "junk_stored": f"{len(junk_stored)}/{len(junk)}",
        "junk_stored_paths": junk_stored,
        "good_rejected": f"{len(good_rejected)}/{len(good_pages)}",
        "good_rejected_paths": good_rejected,
        "robots_respected": outcome("/private/secret") in (None, "robots_blocked"),
        "challenge_detected": outcome("/protected/report") in (None, "challenge") or
        any(p["outcome"] == "challenge" for p in pages),
        "faults_recovered": all(outcome(x) in (None, "stored", "duplicate") for x in ("/flaky/1", "/rate-limited")),
    }
    if topic:
        ordered = sorted(fetched, key=lambda p: p.get("id") or 0)[: first_n or len(fetched)]
        on_topic = sum(1 for p in ordered if gt.get(_path(p["url"]), {}).get("topic") == topic)
        out["harvest_rate"] = round(on_topic / len(ordered), 3) if ordered else None
        out["unique_on_topic_stored"] = len({gt[_path(p["url"])]["cluster"] for p in ordered if p["outcome"] == "stored"
                                             and gt.get(_path(p["url"]), {}).get("topic") == topic})
        out["harvest_window"] = len(ordered)
    return out
