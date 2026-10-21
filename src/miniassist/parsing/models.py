"""JSON-friendly structural records. Every text leaf owns its provenance."""
from dataclasses import asdict, dataclass, field
import hashlib


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_id(*parts) -> str:
    return sha256("|".join(map(str, parts)).encode())[:20]


@dataclass
class Block:
    block_id: str
    block_type: str
    page: int | None
    text: str
    section_path: list[str]
    section_ids: list[str]
    bbox: list[float] | None
    source_spans: list[dict]
    text_map: list[dict]
    metadata: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    table: dict | None = None

    def to_dict(self):
        return asdict(self)
