"""Query-time retrieval: dense (FAISS) + sparse (BM25) -> RRF -> cross-encoder rerank."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import CrossEncoder

from embed import embed_queries
from index import load_index
from utils import K_EVAL, _tokenize

DENSE_CANDIDATES   = 2000  # FAISS chunks retrieved per query
RRF_K              = 60    # constant in RRF formula
RERANK_CANDIDATES  = 60    # pages sent to cross-encoder
CE_MODEL           = "cross-encoder/ms-marco-MiniLM-L-12-v2"
CE_MAX_LENGTH      = 256
CE_BATCH_SIZE      = 128
CHUNKS_PER_PAGE    = 2

# module level cache so artifacts are loaded only once per process
_index_cache: Optional[Dict[str, Any]] = None
_reranker_cache: Optional[CrossEncoder] = None
_page_to_chunks_cache: Optional[Dict[int, List[int]]] = None
_page_ids_arr_cache: Optional[np.ndarray] = None


def _get_page_ids_arr(page_ids: List[int]) -> np.ndarray:
    """chunk_idx -> page_id as an int64 array (built once per process)."""
    global _page_ids_arr_cache
    if _page_ids_arr_cache is None:
        _page_ids_arr_cache = np.asarray(page_ids, dtype=np.int64)
    return _page_ids_arr_cache


def _get_index(artifacts_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load artifacts once and cache them for the process."""
    global _index_cache
    if _index_cache is None:
        _index_cache = load_index(artifacts_dir)
    return _index_cache


def _get_reranker() -> CrossEncoder:
    """Load the cross-encoder reranker once and cache it for the process."""
    global _reranker_cache
    if _reranker_cache is None:
        _reranker_cache = CrossEncoder(CE_MODEL, max_length=CE_MAX_LENGTH)
    return _reranker_cache


def _get_page_to_chunks(page_ids: List[int]) -> Dict[int, List[int]]:
    """Map page_id -> list of its chunk indices (built once per process)."""
    global _page_to_chunks_cache
    if _page_to_chunks_cache is None:
        mapping: Dict[int, List[int]] = {}
        for chunk_idx, pid in enumerate(page_ids):
            mapping.setdefault(pid, []).append(chunk_idx)
        _page_to_chunks_cache = mapping
    return _page_to_chunks_cache


def _top_chunks_for_page(
    pid: int,
    dense_chunk: Dict[int, float],
    sparse_chunk: Dict[int, float],
    best_chunk: int,
    page_to_chunks: Dict[int, List[int]],
    n: int = CHUNKS_PER_PAGE,
) -> List[int]:
    """Pick the n highest-scoring chunks of a page to send to the cross-encoder.

    Selection score is the chunk's dense similarity if it was a FAISS candidate,
    otherwise its BM25 score; chunks with no retrieval signal are ignored.
    """
    chunks = page_to_chunks.get(pid)
    if not chunks:
        return [best_chunk]
    scored: List[Tuple[int, float]] = []
    for c in chunks:
        s = dense_chunk.get(c)
        if s is None:
            s = sparse_chunk.get(c)
        if s is not None:
            scored.append((c, s))
    if not scored:
        return [best_chunk]
    scored.sort(key=lambda x: -x[1])
    return [c for c, _ in scored[:n]]


def _bm25_score_query(
    query_tokens: List[str],
    idx: Dict[str, Any],
) -> Dict[int, float]:
    """Return BM25 scores for all chunks that share at least one query term."""
    # ── 1. UNPACK EXTRACTED BM25 INDEX ARTIFACTS ──────────────────────────
    vocab         = idx["bm25_vocab"]
    idf           = idx["bm25_idf"]
    postings      = idx["bm25_postings"]
    offsets       = idx["bm25_offsets"]
    doc_lengths   = idx["bm25_doc_lengths"]
    avgdl         = idx["bm25_avgdl"]
    k1            = idx["bm25_k1"]
    b             = idx["bm25_b"]

    scores: Dict[int, float] = {}
    seen_terms: set = set()

    # ── 2. SCORING ENGINE LOOP ────────────────────────────────────────────
    for token in query_tokens:
        # Ignore out-of-vocabulary terms and avoid scoring the same term twice
        if token not in vocab or token in seen_terms:
            continue
        seen_terms.add(token)

        # Resolve token ID and compute index bounds for the contiguous postings array
        tid = vocab[token]
        start = int(offsets[tid])
        end   = int(offsets[tid + 1])

        # Skip if the term does not appear in any document postings
        if start == end:
            continue
        
        # ── 3. UNPACK CONTIGUOUS POSTINGS ─────────────────────────────────
        # Postings are interleaved: [chunk_id_0, tf_0, chunk_id_1, tf_1, ...]
        # We multiply pointers by 2 to adapt to this flattened layout structure.
        slice_ = postings[start * 2: end * 2]

        # Separate chunk IDs and Term Frequencies using strided array slicing [start::step]
        chunk_ids = slice_[0::2].astype(np.int64)
        tfs       = slice_[1::2].astype(np.float32)

        # ── 4. VECTORIZED BM25 CALCULATION ────────────────────────────────
        term_idf  = float(idf[tid])

        # Fetch individual text lengths for all matched chunks via fancy indexing
        dls       = doc_lengths[chunk_ids].astype(np.float32)

        # Apply document length normalization component: (1 - b) + b * (dl / avgdl)
        norm_dls  = 1.0 - b + b * dls / avgdl

        # Calculate BM25 TF scaling factor
        bm25_tf   = tfs * (k1 + 1.0) / (tfs + k1 * norm_dls)

        # Compute final chunk scores for this specific query term
        chunk_scores = term_idf * bm25_tf

        # ── 5. SCORE ACCUMULATION ─────────────────────────────────────────
        # Convert NumPy arrays back to Python native primitives to populate the score map
        for cid, sc in zip(chunk_ids.tolist(), chunk_scores.tolist()):
            scores[cid] = scores.get(cid, 0.0) + sc

    return scores


def _aggregate_to_pages(
    chunk_scores: Dict[int, float],
    page_ids_arr: np.ndarray,
) -> Dict[int, Tuple[float, int]]:
    """Max-pool chunk scores per page. also track the best chunk index.

    Vectorized: max-pooling is order-independent and exact in floating point,
    so this yields the same page scores as a scalar loop. For each page the
    best chunk is the first one (in dict-iteration order) attaining the max,
    matching the original strict-greater-than scalar logic.
    """
    n = len(chunk_scores)
    if n == 0:
        return {}
    cids = np.fromiter(chunk_scores.keys(),   dtype=np.int64,   count=n)
    scs  = np.fromiter(chunk_scores.values(), dtype=np.float64, count=n)
    pages = page_ids_arr[cids]

    uniq_pages, inv = np.unique(pages, return_inverse=True)
    inv = inv.ravel()
    page_max = np.full(uniq_pages.shape[0], -np.inf, dtype=np.float64)
    np.maximum.at(page_max, inv, scs)

    # best chunk = smallest array position (i.e. first in dict order) whose score equals its page max.
    # reverse assignment lets the earliest win.
    is_max = scs >= page_max[inv]
    cand = np.nonzero(is_max)[0]
    best_pos = np.full(uniq_pages.shape[0], n, dtype=np.int64)
    best_pos[inv[cand[::-1]]] = cand[::-1]
    best_chunk = cids[best_pos]

    return {
        int(p): (float(m), int(c))
        for p, m, c in zip(uniq_pages.tolist(), page_max.tolist(), best_chunk.tolist())
    }


def _rrf_fuse(
    dense_pages: Dict[int, Tuple[float, int]],
    sparse_pages: Dict[int, Tuple[float, int]],
    k: int = RRF_K,
) -> List[Tuple[int, float, int]]:
    """
    Combine dense and sparse page rankings with Reciprocal Rank Fusion.

    Returns list of (page_id, rrf_score, best_chunk_idx) sorted by score desc.
    """
    dense_ranked  = sorted(dense_pages.items(),  key=lambda x: -x[1][0])
    sparse_ranked = sorted(sparse_pages.items(), key=lambda x: -x[1][0])

    dense_rank  = {pid: r for r, (pid, _) in enumerate(dense_ranked,  start=1)}
    sparse_rank = {pid: r for r, (pid, _) in enumerate(sparse_ranked, start=1)}

    dense_fallback  = len(dense_ranked)  + 1
    sparse_fallback = len(sparse_ranked) + 1

    all_pids = set(dense_pages) | set(sparse_pages)
    results: List[Tuple[int, float, int]] = []
    for pid in all_pids:
        rrf = (
            1.0 / (dense_rank.get(pid,  dense_fallback)  + k)
            + 1.0 / (sparse_rank.get(pid, sparse_fallback) + k)
        )
        # prefer dense best-chunk, fall back to sparse
        if pid in dense_pages:
            best_chunk = dense_pages[pid][1]
        else:
            best_chunk = sparse_pages[pid][1]
        results.append((pid, rrf, best_chunk))

    results.sort(key=lambda x: -x[1])
    return results


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    """Return ranked page_id lists (best first) for each query.

    Per batch: embed queries -> FAISS dense + BM25 sparse -> RRF fuse to the top
    RERANK_CANDIDATES pages -> cross-encoder reranks (max-pool over top chunks/page).
    """
    # ── 1. INITIALIZATION & DATA LOADING ───────────────────────────────────
    # Retrieve the shared index artifacts and global reranker instance
    idx      = _get_index(artifacts_dir)
    reranker = _get_reranker()

    # Unpack index structures for fast lookups
    faiss_index    = idx["faiss_index"]
    page_ids       = idx["page_ids"]
    chunk_texts    = idx["chunk_texts"]
    page_to_chunks = _get_page_to_chunks(page_ids)
    page_ids_arr   = _get_page_ids_arr(page_ids)

    # ── 2. BULK DENSE EMBEDDING RETRIEVAL ──────────────────────────────────
    # Generate vector embeddings for all queries
    query_vecs = embed_queries(queries)

    # Query the FAISS index to get the top DENSE_CANDIDATES closest chunks per query
    scores_d, idxs_d = faiss_index.search(query_vecs, DENSE_CANDIDATES)

    # Global registers to flatten all cross-encoder tasks into a single GPU execution batch
    all_ce_pairs: List[Tuple[str, str]] = []

    # Tracks structural mapping: per query -> list of (page_id, [global indices in all_ce_pairs])
    all_query_cands: List[List[Tuple[int, List[int]]]] = []

    # ── 3. PER-QUERY HYBRID RETRIEVAL & RRF FUSION ─────────────────────────
    for i, query in enumerate(queries):
        # Clean up FAISS results: filter out invalid indices (-1) and build a chunk_id -> score map
        dense_chunk: Dict[int, float] = {
            int(ci): float(sc)
            for ci, sc in zip(idxs_d[i], scores_d[i])
            if ci >= 0
        }

        # Run sparse keyword search (BM25) on tokenized query text
        sparse_chunk = _bm25_score_query(_tokenize(query), idx)

        # Roll chunk-level scores up to the document/page level
        dense_pages  = _aggregate_to_pages(dense_chunk,  page_ids_arr)
        sparse_pages = _aggregate_to_pages(sparse_chunk, page_ids_arr)

        # Merge dense and sparse pages using Reciprocal Rank Fusion and prune to top candidates
        fused = _rrf_fuse(dense_pages, sparse_pages)[:RERANK_CANDIDATES]

        cands: List[Tuple[int, List[int]]] = []
        # For each candidate page, identify which specific text chunks should be evaluated by the Cross-Encoder
        for pid, _rrf, best_chunk in fused:
            chunk_idxs = _top_chunks_for_page(
                pid, dense_chunk, sparse_chunk, best_chunk, page_to_chunks
            )
            pair_idxs: List[int] = []
            # Map chunk IDs to a flattened list of global cross-encoder input pairs (query, chunk_text)
            for ci in chunk_idxs:
                pair_idxs.append(len(all_ce_pairs))
                all_ce_pairs.append((query, chunk_texts[ci]))
            cands.append((pid, pair_idxs))
        all_query_cands.append(cands)

    # ── 4. FLATTENED & LENGTH-OPTIMIZED CROSS-ENCODER INFERENCE ──────────
    if all_ce_pairs:
        # PADDING OPTIMIZATION: CrossEncoder.predict pads an entire batch to the maximum length found in that batch.
        # Sorting pairs globally by character length groups similar lengths together, preventing 
        # massive, wasted computation on trailing [PAD] tokens in deep Transformer layers.
        order = sorted(
            range(len(all_ce_pairs)),
            key=lambda k: len(all_ce_pairs[k][0]) + len(all_ce_pairs[k][1]),
        )

        # Execute batch inference on sorted text pairs
        sorted_scores = reranker.predict(
            [all_ce_pairs[k] for k in order],
            batch_size=CE_BATCH_SIZE,
            show_progress_bar=False,
        )

        # Map the sorted inference scores back to their original sequential order
        all_ce_scores = np.empty(len(all_ce_pairs), dtype=np.float32)
        all_ce_scores[np.asarray(order, dtype=np.int64)] = np.asarray(
            sorted_scores, dtype=np.float32
        )
    else:
        all_ce_scores = np.array([], dtype=np.float32)

   # ── 5. MAX-POOLING & FINAL PAGE RANKING ────────────────────────────────
    ranked: List[List[int]] = []
    for cands in all_query_cands:
        if not cands:
            ranked.append([])
            continue

        # Max-pooling strategy: The final score of a page is dictated by the score 
        # of its single highest-scoring chunk under Cross-Encoder assessment.
        scored = [
            (pid, max(float(all_ce_scores[j]) for j in pair_idxs))
            for pid, pair_idxs in cands
        ]

        # Sort candidates in descending order (highest max-pooled score first)
        scored.sort(key=lambda x: -x[1])

        # Slice out the page IDs up to the requested top_k boundary
        ranked.append([pid for pid, _ in scored[:top_k]])

    return ranked
