"""Runtime configuration; no evaluation labels or reference answers."""
import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def load_local_env(path=None):
    """Load only explicit provider/ES settings, without shell evaluation or value logging."""
    allowed = {"DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "DEEPSEEK_BASE_URL", "ES_URL", "ES_API_KEY"}
    path = Path(path or ROOT / ".env")
    if not path.exists():
        return
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid configuration at line {number}")
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip()
        if name not in allowed:
            raise ValueError(f"Unsupported configuration name at line {number}")
        if value.startswith(("'", '"')):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"Unclosed configuration quote at line {number}")
            value = value[1:-1]
        os.environ.setdefault(name, value)


@dataclass(frozen=True)
class Settings:
    model: str = "deepseek-flash"
    base_url: str = "https://api.deepseek.com"
    es_url: str = "http://127.0.0.1:19200"
    timeout: float = 45.0
    max_clarifications: int = 2
    max_history: int = 12
    max_retrieval_rounds: int = 1
    # B0 ablation disables only the literal-action/document-version query rules.
    surface_verification: bool = True
    # Experimental rewrite trials showed semantic regressions; opt in explicitly.
    review_answer: bool = False
    # C is opt-in; legacy B/B0 behavior stays on the existing path.
    policy: str = "B"
    max_model_requests: int = 10

    @classmethod
    def from_env(cls):
        return cls(model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
                   base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                   es_url=os.getenv("ES_URL", "http://127.0.0.1:19200"))


def load_config(path=None):
    path = Path(path or ROOT / "config/rag.v1.json").resolve()
    config = json.loads(path.read_text())
    for key in ("chunks", "sources", "vectors", "model_cache"):
        config[key] = str((ROOT / config[key]).resolve())
    return config
