"""Optional preprocessing and chunking."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List

from numpy import record

from utils import EMBEDDING_MODEL_NAME, iter_entries

PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
SPECIAL_TOKEN_OVERHEAD = 2
TITLE_SEPARATOR_TOKEN_OVERHEAD = 1

GRID = [
    (
        "paragraph_pack_256",
        dict(
            method="paragraph_pack",
            token_budget=256,
            title_prepend=True,
            paragraph_overlap=0,
            fixed_overlap_ratio=0.0,
            add_title_chunk=False,
        ),
    ),
    (
        "paragraph_pack_128",
        dict(
            method="paragraph_pack",
            token_budget=128,
            title_prepend=True,
            paragraph_overlap=0,
            fixed_overlap_ratio=0.0,
            add_title_chunk=False,
        ),
    ),
    (
        "fixed_window_256_overlap_15",
        dict(
            method="fixed_window",
            token_budget=256,
            title_prepend=True,
            paragraph_overlap=0,
            fixed_overlap_ratio=0.15,
            add_title_chunk=False,
        ),
    ),
    (
        "paragraph_pack_256_overlap_1",
        dict(
            method="paragraph_pack",
            token_budget=256,
            title_prepend=True,
            paragraph_overlap=1,
            fixed_overlap_ratio=0.0,
            add_title_chunk=False,
        ),
    ),
]

_tokenizer: Any | None = None


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str


def _get_tokenizer() -> Any:
    """Load the MiniLM tokenizer lazily so importing this module stays cheap."""
    global _tokenizer
    if _tokenizer is None:
        from sentence_transformers import SentenceTransformer

        try:
            _tokenizer = SentenceTransformer(
                EMBEDDING_MODEL_NAME,
                local_files_only=True,
            ).tokenizer
        except Exception:
            _tokenizer = SentenceTransformer(EMBEDDING_MODEL_NAME).tokenizer
        try:
            _tokenizer.model_max_length = int(1e9)
        except Exception:
            pass
    return _tokenizer


def _token_ids(text: str) -> List[int]:
    if not text:
        return []
    encoded = _get_tokenizer()(
        text,
        add_special_tokens=False,
        truncation=False,
    )
    return list(encoded["input_ids"])


def _token_len(text: str) -> int:
    return len(_token_ids(text))


def _decode_token_ids(token_ids: List[int]) -> str:
    if not token_ids:
        return ""
    return _get_tokenizer().decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()


def _body_token_budget(
    title: str,
    *,
    token_budget: int,
    title_prepend: bool,
) -> int:
    if token_budget <= SPECIAL_TOKEN_OVERHEAD:
        raise ValueError("token_budget must leave room for model special tokens")

    reserved = SPECIAL_TOKEN_OVERHEAD
    if title_prepend and title:
        reserved += _token_len(title) + TITLE_SEPARATOR_TOKEN_OVERHEAD
    return max(1, token_budget - reserved)


def _format_chunk_text(title: str, body: str, *, title_prepend: bool) -> str:
    title = title.strip()
    body = body.strip()
    if title_prepend and title:
        return f"{title}\n\n{body}".strip()
    return body or title


def _split_paragraphs(content: str) -> List[str]:
    return [part.strip() for part in PARAGRAPH_SPLIT_RE.split(content) if part.strip()]


def _split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    return [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]


def _hard_split_text(text: str, body_budget: int) -> List[str]:
    token_ids = _token_ids(text)
    if not token_ids:
        return []
    return [
        _decode_token_ids(token_ids[i : i + body_budget])
        for i in range(0, len(token_ids), body_budget)
    ]


def _split_oversized_paragraph(paragraph: str, body_budget: int) -> List[str]:
    if _token_len(paragraph) <= body_budget:
        return [paragraph.strip()]

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0
    for sentence in _split_sentences(paragraph):
        sentence_tokens = _token_len(sentence)
        if sentence_tokens > body_budget:
            if current:
                chunks.append(" ".join(current).strip())
                current = []
                current_tokens = 0
            chunks.extend(_hard_split_text(sentence, body_budget))
            continue

        separator_tokens = TITLE_SEPARATOR_TOKEN_OVERHEAD if current else 0
        next_tokens = current_tokens + separator_tokens + sentence_tokens
        if current and next_tokens > body_budget:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_tokens = sentence_tokens
        else:
            current.append(sentence)
            current_tokens = next_tokens

    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _atomic_paragraph_units(paragraphs: List[str], body_budget: int) -> List[str]:
    units: List[str] = []
    for paragraph in paragraphs:
        units.extend(_split_oversized_paragraph(paragraph, body_budget))
    return units


def _paragraph_pack_chunks(
    content: str,
    *,
    body_budget: int,
    paragraph_overlap: int,
) -> List[str]:
    if paragraph_overlap < 0:
        raise ValueError("paragraph_overlap must be non-negative")

    paragraphs = _split_paragraphs(content)
    if not paragraphs:
        return []

    units = _atomic_paragraph_units(paragraphs, body_budget)
    chunks: List[str] = []
    start = 0
    while start < len(units):
        current: List[str] = []
        current_tokens = 0
        idx = start
        while idx < len(units):
            unit = units[idx]
            unit_tokens = _token_len(unit)
            separator_tokens = TITLE_SEPARATOR_TOKEN_OVERHEAD if current else 0
            next_tokens = current_tokens + separator_tokens + unit_tokens
            if current and next_tokens > body_budget:
                break
            current.append(unit)
            current_tokens = next_tokens
            idx += 1

        chunks.append("\n\n".join(current).strip())
        if idx >= len(units):
            break
        if paragraph_overlap == 0:
            start = idx
        else:
            start = max(idx - paragraph_overlap, start + 1)

    return [chunk for chunk in chunks if chunk]


def _fixed_window_chunks(
    content: str,
    *,
    body_budget: int,
    fixed_overlap_ratio: float,
) -> List[str]:
    if fixed_overlap_ratio < 0 or fixed_overlap_ratio >= 1:
        raise ValueError("fixed_overlap_ratio must be in the range [0.0, 1.0)")

    token_ids = _token_ids(content.strip())
    if not token_ids:
        return []

    overlap_tokens = round(body_budget * fixed_overlap_ratio)
    step = max(1, body_budget - overlap_tokens)
    chunks: List[str] = []
    for start in range(0, len(token_ids), step):
        window = token_ids[start : start + body_budget]
        if not window:
            break
        chunks.append(_decode_token_ids(window))
        if start + body_budget >= len(token_ids):
            break
    return [chunk for chunk in chunks if chunk]


def chunk_entry(
    record: Dict[str, Any],
    *,
    method: str = "paragraph_pack",
    token_budget: int = 256,
    title_prepend: bool = True,
    paragraph_overlap: int = 0,
    fixed_overlap_ratio: float = 0.0,
    add_title_chunk: bool = False,
) -> List[Chunk]:
    """
    Split one corpus entry into retrieval units.

    Defaults to paragraph packing with the page title prepended to each chunk.
    Pass explicit hyperparameters to reuse the same function for chunking
    experiments.
    """
    page_id = int(record["page_id"])
    title = str(record.get("title") or "").strip()
    content = str(record.get("content") or "").strip()
    body_budget = _body_token_budget(
        title,
        token_budget=token_budget,
        title_prepend=title_prepend,
    )

    if method == "paragraph_pack":
        body_chunks = _paragraph_pack_chunks(
            content,
            body_budget=body_budget,
            paragraph_overlap=paragraph_overlap,
        )
    elif method == "fixed_window":
        body_chunks = _fixed_window_chunks(
            content,
            body_budget=body_budget,
            fixed_overlap_ratio=fixed_overlap_ratio,
        )
    else:
        raise ValueError(f"Unknown chunking method: {method!r}")

    texts: List[str] = []
    if add_title_chunk and title:
        texts.append(title)
    texts.extend(
        _format_chunk_text(title, body, title_prepend=title_prepend)
        for body in body_chunks
    )
    if not texts:
        texts.append(title or content)

    return [
        Chunk(page_id=page_id, chunk_id=chunk_id, text=text)
        for chunk_id, text in enumerate(texts)
        if text
    ]


def chunk_corpus(records: List[Dict[str, Any]], **chunk_kwargs: Any) -> List[Chunk]:
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record, **chunk_kwargs))
    return chunks


if __name__ == "__main__":
    sample_record = next(iter_entries())
    for name, kwargs in GRID:
        print("*" * 40)
        print(f"Chunking method: {name}")
        sample_chunks = chunk_entry(sample_record, **kwargs)
    
        print(
            f"record page_id={sample_record['page_id']} "
            f"title={sample_record.get('title', '')!r} "
            f"chunks={len(sample_chunks)}"
        )
        print("=" * 40)
        for chunk in sample_chunks[:3]:
            print(f"page_id={chunk.page_id} chunk_id={chunk.chunk_id}")
            print(chunk.text)
            print("-" * 40)
        if len(sample_chunks) > 3:
            print(f"... {len(sample_chunks) - 3} more chunks")
