"""Offline index build and load (not timed at grading)."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pickle

import faiss
import numpy as np

from chunk import Chunk, chunk_corpus
from embed import embed_texts
from utils import ARTIFACTS_DIR, ensure_artifacts_dir, iter_entries, _tokenize

# ── file names ─────────────────────────────────────────────────────────────
FAISS_INDEX_NAME    = "index.faiss"
INDEX_META_NAME     = "index_meta.json"
CHUNK_TEXTS_NAME    = "chunk_texts.pkl"
BM25_VOCAB_NAME     = "bm25_vocab.json"
BM25_IDF_NAME       = "bm25_idf.npy"
BM25_POSTINGS_NAME  = "bm25_postings_data.npy"
BM25_OFFSETS_NAME   = "bm25_offsets.npy"
BM25_DOCLENS_NAME   = "bm25_doc_lengths.npy"

# ── BM25 hyper-parameters ──────────────────────────────────────────────────
BM25_K1   = 1.5
BM25_B    = 0.75
BM25_MIN_DF = 2          # minimum document frequency to include a term
BM25_MAX_DF_RATIO = 0.4  # exclude stop-word-like terms (> 40 % of chunks)
BM25_MAX_VOCAB = 500_000  # effectively unlimited; IDF weighting handles noise


def _build_bm25(texts: List[str]) -> Dict[str, Any]:
    """Build a BM25 inverted index over the chunks (numpy arrays + vocab).

    postings_data is int32 interleaved [chunk_id, tf, ...]; term t spans postings
    [offsets[t], offsets[t+1]). Vocab filtered by doc-frequency, capped by idf.
    """
    N = len(texts)
    max_df = int(BM25_MAX_DF_RATIO * N)

    tokenized: List[List[str]] = [_tokenize(t) for t in texts]
    doc_lengths = np.array([len(tokens) for tokens in tokenized], dtype=np.int32)
    avgdl = float(doc_lengths.mean()) if N > 0 else 1.0

    # document-frequency pass
    df: Dict[str, int] = defaultdict(int)
    for tokens in tokenized:
        for term in set(tokens):
            df[term] += 1

    # filter vocabulary
    vocab_terms = [
        term for term, freq in df.items()
        if BM25_MIN_DF <= freq <= max_df
    ]
    # rank by IDF (descending) and keep top BM25_MAX_VOCAB
    vocab_terms.sort(
        key=lambda t: math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1),
        reverse=True,
    )
    vocab_terms = vocab_terms[:BM25_MAX_VOCAB]
    vocab: Dict[str, int] = {term: idx for idx, term in enumerate(vocab_terms)}

    idf = np.array(
        [math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1) for t in vocab_terms],
        dtype=np.float32,
    )

    # build posting lists: for each term, sorted list of (chunk_id, tf)
    postings: List[List[Tuple[int, int]]] = [[] for _ in vocab_terms]
    for doc_idx, tokens in enumerate(tokenized):
        tf_map = Counter(tokens)
        for term, tf in tf_map.items():
            if term in vocab:
                postings[vocab[term]].append((doc_idx, tf))

    # flatten into two parallel int32 arrays + offsets
    flat_chunk_ids: List[int] = []
    flat_tfs: List[int] = []
    offsets = np.zeros(len(vocab_terms) + 1, dtype=np.int64)
    for tid, plist in enumerate(postings):
        offsets[tid] = len(flat_chunk_ids)
        for cid, tf in plist:
            flat_chunk_ids.append(cid)
            flat_tfs.append(tf)
    offsets[len(vocab_terms)] = len(flat_chunk_ids)

    # interleave chunk_id and tf so postings_data[2k] = chunk_id, [2k+1] = tf
    postings_data = np.empty(2 * len(flat_chunk_ids), dtype=np.int32)
    postings_data[0::2] = flat_chunk_ids
    postings_data[1::2] = flat_tfs

    return {
        "vocab": vocab,
        "idf": idf,
        "postings_data": postings_data,
        "offsets": offsets,
        "doc_lengths": doc_lengths,
        "avgdl": avgdl,
    }


def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> None:
    """
    Embed the full corpus and persist all retrieval artifacts.

    Chunks the corpus, embeds with MiniLM, and writes index.faiss (dense),
    bm25_* (sparse), chunk_texts.pkl (reranker) and index_meta.json, where
    page_ids[i] is the page of chunk row i. Reloaded at query time by load_index.
    """
    out_dir = artifacts_dir or ensure_artifacts_dir()
    records = list(iter_entries(entries_dir))
    chunks: List[Chunk] = chunk_corpus(records)

    texts   = [c.text for c in chunks]
    page_ids = [c.page_id for c in chunks]

    print(f"[build] {len(chunks):,} chunks from {len(records):,} entries — embedding …")
    vectors = embed_texts(texts)

    # ── dense FAISS index ─────────────────────────────────────────────────
    print("[build] building FAISS IndexFlatIP …")
    dim = vectors.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(vectors)
    faiss.write_index(faiss_index, str(out_dir / FAISS_INDEX_NAME))

    meta = {
        "page_ids": page_ids,
        "num_vectors": len(page_ids),
        "model": "sentence-transformers/all-MiniLM-L6-v2",
    }
    (out_dir / INDEX_META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ── chunk texts (for cross-encoder reranker) ─────────────────────────
    print("[build] saving chunk texts …")
    with open(out_dir / CHUNK_TEXTS_NAME, "wb") as f:
        pickle.dump(texts, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ── BM25 inverted index ───────────────────────────────────────────────
    print("[build] building BM25 index …")
    bm25 = _build_bm25(texts)

    (out_dir / BM25_VOCAB_NAME).write_text(
        json.dumps(bm25["vocab"], ensure_ascii=False), encoding="utf-8"
    )
    np.save(out_dir / BM25_IDF_NAME,      bm25["idf"])
    np.save(out_dir / BM25_POSTINGS_NAME, bm25["postings_data"])
    np.save(out_dir / BM25_OFFSETS_NAME,  bm25["offsets"])
    np.save(out_dir / BM25_DOCLENS_NAME,  bm25["doc_lengths"])

    # save avgdl inside meta
    meta["bm25_avgdl"]  = bm25["avgdl"]
    meta["bm25_k1"]     = BM25_K1
    meta["bm25_b"]      = BM25_B
    (out_dir / INDEX_META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[build] done — artifacts saved to {out_dir}")


def load_index(
    artifacts_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load all pre-built artifacts into one dict for query time (inverse of build_index)."""
    root = artifacts_dir or ARTIFACTS_DIR

    faiss_index = faiss.read_index(str(root / FAISS_INDEX_NAME))

    meta = json.loads((root / INDEX_META_NAME).read_text(encoding="utf-8"))
    page_ids = [int(x) for x in meta["page_ids"]]

    with open(root / CHUNK_TEXTS_NAME, "rb") as f:
        chunk_texts: List[str] = pickle.load(f)

    bm25_vocab: Dict[str, int] = json.loads(
        (root / BM25_VOCAB_NAME).read_text(encoding="utf-8")
    )
    bm25_idf          = np.load(root / BM25_IDF_NAME)
    bm25_postings     = np.load(root / BM25_POSTINGS_NAME)
    bm25_offsets      = np.load(root / BM25_OFFSETS_NAME)
    bm25_doc_lengths  = np.load(root / BM25_DOCLENS_NAME)

    return {
        "faiss_index":      faiss_index,
        "page_ids":         page_ids,
        "chunk_texts":      chunk_texts,
        "bm25_vocab":       bm25_vocab,
        "bm25_idf":         bm25_idf,
        "bm25_postings":    bm25_postings,
        "bm25_offsets":     bm25_offsets,
        "bm25_doc_lengths": bm25_doc_lengths,
        "bm25_avgdl":       float(meta.get("bm25_avgdl", 200.0)),
        "bm25_k1":          float(meta.get("bm25_k1", BM25_K1)),
        "bm25_b":           float(meta.get("bm25_b",  BM25_B)),
    }
