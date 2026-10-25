import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
DEFAULT_UA = "AICrawler/1.0 (+https://github.com/; polite research crawler)"


@dataclass(frozen=True)
class Settings:
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    data_dir: Path
    llm_timeout: float = 180

    @property
    def db_path(self) -> Path:
        return self.data_dir / "crawler.db"


def get_settings() -> Settings:
    return Settings(
        llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:8080/v1").rstrip("/"),
        llm_api_key=os.getenv("LLM_API_KEY", "local"),
        llm_model=os.getenv("LLM_MODEL", "qwen3.6-35b-a3b"),
        data_dir=Path(os.getenv("CRAWLER_DATA", str(ROOT / "data"))),
    )


class CrawlConfig(BaseModel):
    """Every user-tunable crawl parameter, validated."""

    seeds: list[str] = Field(min_length=1, max_length=50)
    max_depth: int = Field(2, ge=0, le=10)
    max_pages: int = Field(100, ge=1, le=20000)
    same_domain_only: bool = True
    allowed_domains: list[str] = []
    include_patterns: list[str] = []
    exclude_patterns: list[str] = [r"\.(jpg|jpeg|png|gif|svg|webp|ico|css|js|zip|gz|mp4|mp3|woff2?)(\?|$)"]
    respect_robots: bool = True

    # politeness and human-like pacing
    min_delay_s: float = Field(1.0, ge=0, le=60)
    max_delay_s: float = Field(2.5, ge=0, le=120)
    concurrency: int = Field(4, ge=1, le=32)
    per_host_concurrency: int = Field(2, ge=1, le=8)
    timeout_s: float = Field(20, ge=1, le=120)
    max_retries: int = Field(3, ge=0, le=8)
    user_agent: str = DEFAULT_UA
    browser_like_headers: bool = True
    proxy: str | None = None

    # caching
    use_cache: bool = True
    cache_ttl_hours: float = Field(24, ge=0, le=24 * 30)

    # intelligence
    quality_filter: bool = True
    quality_threshold: float = Field(0.5, ge=0, le=1)
    near_dup_threshold: float = Field(0.8, ge=0.5, le=1.0)
    semantic_dedup: bool = False
    focus_topic: str = ""
    ai_extraction: str = ""
    ai_max_pages: int = Field(20, ge=0, le=500)

    # challenge (CAPTCHA / bot wall) handling: "skip" or "pause"
    on_challenge: str = Field("pause", pattern="^(skip|pause)$")

    @field_validator("seeds")
    @classmethod
    def _seeds(cls, v: list[str]) -> list[str]:
        out = []
        for s in v:
            s = s.strip()
            if not s.startswith(("http://", "https://")):
                s = "https://" + s
            out.append(s)
        return out

    @field_validator("max_delay_s")
    @classmethod
    def _delays(cls, v: float, info) -> float:
        lo = info.data.get("min_delay_s", 0)
        return max(v, lo)
