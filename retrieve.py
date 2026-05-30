"""Query-time retrieval (timed portion includes query embedding)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from embed import embed_queries
from index import load_index
from utils import K_EVAL


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    """
    Return ranked page_id lists (best first) for each query.

    Default: brute-force dot product on L2-normalized vectors.
    Replace with FAISS / reranking as needed.
    """
    corpus_vectors, page_ids = load_index(artifacts_dir)
    query_vectors = embed_queries(queries)
    if query_vectors.size == 0:
        return [[] for _ in queries]

    scores = query_vectors @ corpus_vectors.T
    ranked: List[List[int]] = []
    for row in scores:
        order = np.argsort(-row)
        seen: set[int] = set()
        ids: List[int] = []
        for idx in order:
            pid = page_ids[int(idx)]
            if pid in seen:
                continue
            seen.add(pid)
            ids.append(pid)
            if len(ids) >= top_k:
                break
        ranked.append(ids)
    return ranked
