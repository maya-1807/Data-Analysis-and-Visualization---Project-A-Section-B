from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from utils import entry_text


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """
    Split one corpus entry into retrieval units.

    Default: single chunk per page (no chunking). Override for fixed-size or
    semantic chunking strategies.
    """
    page_id = int(record["page_id"])
    text = entry_text(record)
    return [Chunk(page_id=page_id, chunk_id=0, text=text)]