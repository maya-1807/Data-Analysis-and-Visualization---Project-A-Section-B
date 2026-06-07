"""preprocessing and chunking"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any, Dict, List

# chunking hyper-parameters
MAX_WORDS_PER_CHUNK = 200   # target chunk size
OVERLAP_SENTENCES = 2       # sentences carried into the next window
MAX_CHUNKS_PER_ENTRY = 50   # hard cap so a long entry can't explode the index
SHORT_ENTRY_CHARS = 800     # entries with title+content at or below this become one chunk


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences on ., ! or ? followed by whitespace."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p.strip()]


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """Split one corpus record into chunks.

    Short entries (title+content ≤ SHORT_ENTRY_CHARS) become a single chunk.
    Longer entries are split with a sentence aware sliding window so that every
    sentence is covered by at least one chunk. The page title is prepended to
    every chunk so the model always has document context.
    """
    page_id = int(record["page_id"])
    title = record.get("title", "")
    content = record.get("content", "")

    # Title context is prepended to every chunk
    prefix = f"{title}\n\n" if title else ""

    # if a short entry fits in one chunk; no need to split.
    if len(prefix) + len(content) <= SHORT_ENTRY_CHARS:
        return [Chunk(page_id=page_id, chunk_id=0, text=(prefix + content).strip())]

    sentences = _split_sentences(content)
    if not sentences:
        # No sentence boundaries found - fall back to a single chunk.
        return [Chunk(page_id=page_id, chunk_id=0, text=(prefix + content).strip())]

    chunks: List[Chunk] = []
    i = 0  # index of the first sentence of the current window
    while i < len(sentences) and len(chunks) < MAX_CHUNKS_PER_ENTRY:
        # Greedily pack whole sentences into this window until adding the next
        # one would exceed the word budget. The "group and" ensures an
        # oversized *first* sentence (longer than MAX_WORDS_PER_CHUNK on its own)
        # is still included rather than dropped.
        group: List[str] = []
        word_count = 0
        j = i
        while j < len(sentences):
            words_in_sent = len(sentences[j].split())
            if group and word_count + words_in_sent > MAX_WORDS_PER_CHUNK:
                break
            group.append(sentences[j])
            word_count += words_in_sent
            j += 1
            if not group:
                break

        chunk_text = prefix + " ".join(group)
        chunks.append(Chunk(page_id=page_id, chunk_id=len(chunks), text=chunk_text.strip()))

        # Slide the window forward, stepping back OVERLAP_SENTENCES from where we stopped so neighbouring chunks overlap.
        advance = max(1, len(group) - OVERLAP_SENTENCES)
        i += advance

    return chunks


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    """Chunk every entry in records and return the list of chunks."""
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks
