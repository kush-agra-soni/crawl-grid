"""Renders the Streamlit UI headlessly against an in-process backend crawling the in-process demo site."""
import time
import httpx
import pytest
import socket
import uvicorn
import threading
from aicrawler.api import create_app
from aicrawler.config import Settings
from aicrawler.testsite import create_site
from conftest import ROOT, HashEmbedder, fast_cfg, fast_sleep, wait_crawl


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    app = create_app(Settings("http://llm.invalid/v1", "k", "m", tmp_path_factory.mktemp("ui")),
                     embedder=HashEmbedder(), transport_factory=lambda cfg: httpx.ASGITransport(app=create_site(0.05)),
                     sleep=fast_sleep, use_local_models=False, llm_enabled=False)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            httpx.get(f"{url}/health")
            break
        except httpx.HTTPError:
            time.sleep(0.05)
    mp = pytest.MonkeyPatch()
    mp.setenv("BACKEND_URL", url)
    yield url
    mp.undo()
    server.should_exit = True


def test_ui_configure_live_results(backend):
    from streamlit.testing.v1 import AppTest

    c = httpx.Client(base_url=backend, timeout=30)
    cid = c.post("/crawls", json={"config": fast_cfg(on_challenge="skip")}).json()["crawl_id"]
    wait_crawl(c, cid)

    at = AppTest.from_file(str(ROOT / "ui" / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Configure", "Live monitor", "Results", "Search", "Quality model"]
    metrics = {m.label: m.value for m in at.metric}
    assert int(metrics["Stored"]) > 20 and metrics["Challenges"] == "1"

    search = next(t for t in at.text_input if t.label == "Full-text search across stored pages")
    search.set_value("sourdough starter").run()
    assert not at.exception
    assert any("sourdough" in m.value.lower() for m in at.markdown)
