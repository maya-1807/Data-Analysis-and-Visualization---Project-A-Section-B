"""
Section B entry point.

The autograder calls run(queries) once with all evaluation queries (batch of 50).
Query embedding + retrieval must complete within the time limit (GPU available).
"""
from __future__ import annotations

from typing import List

from index import build_index
from retrieve import search_batch


def run(queries: List[str]) -> List[List[int]]:
    """
    Rank corpus pages for each query.

    Parameters
    ----------
    queries : list[str]
        Batch of query strings (e.g. 50 hidden queries at grading time).

    Returns
    -------
    list[list[int]]
        One ranked list of page_id per query (most relevant first).
        Only the first 10 IDs per list are scored.
    """
    return search_batch(queries)


def build_offline_index() -> None:
    """Run once locally to create artifacts/ (not timed at grading)."""
    build_index()


if __name__ == "__main__":
    build_offline_index()
    print("Index built under artifacts/. Run: python scripts/eval_public.py")
