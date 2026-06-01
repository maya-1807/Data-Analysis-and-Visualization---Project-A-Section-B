# Corpus Analysis & Chunking Plan — Section B

## 1. Goal

The embedding model is fixed: `sentence-transformers/all-MiniLM-L6-v2`, which
**truncates input at 256 tokens** (≈190 words; ~254 usable after `[CLS]`/`[SEP]`).
The central design question is therefore whether and how to **split long pages
into chunks** so we don't silently lose most of their content before embedding.
This document records the corpus analysis, its results, and the chunking
strategies we plan to test.

## 2. What `analyze_corpus.py` does

- Reads every page JSON in `data/Wikipedia Entries/`. Reads are **parallelized**
  (thread pool, I/O-bound) and consolidated into a one-file cache
  (`corpus_cache.jsonl`) so later runs load instantly.
- Reports length distributions (percentiles) for: content chars, content words,
  title words.
- Reports paragraph structure: share of pages containing a blank-line break
  (`\n\n`), paragraphs per page, and paragraph word lengths.
- Flags **stub pages** (< 20 words, e.g. disambiguation pages).
- Applies a rough heuristic for flattened inline headings.
- With the MiniLM **tokenizer only** (not the full model), batched for speed,
  reports token-length distributions and the headline metric:
  **share of pages truncated at 256 tokens** (title + content).

The token pass loads only the tokenizer via `AutoTokenizer` and encodes in
batches, which is dramatically faster than per-document encoding.

## 3. Results (full corpus)

Corpus size: **26,974 pages**. Stub pages (< 20 words): **2,169 (8.0%)**.

### Length distributions (percentiles)

| metric | p0 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
|---|---|---|---|---|---|---|---|---|
| content chars | 1 | 1,685 | 7,628 | 19,328 | 36,821 | 50,188 | 79,107 | 197,669 |
| content words | 1 | 266 | 1,219 | 3,070 | 5,866 | 7,944 | 12,579 | 30,823 |
| title words | 1 | 1 | 2 | 3 | 4 | 4 | 6 | 14 |
| content tokens | 1 | 362 | 1,653 | 4,127 | 7,779 | 10,508 | 16,433 | 41,484 |
| title+content tokens | 4 | 367 | 1,658 | 4,131 | 7,784 | 10,511 | 16,438 | 41,488 |

### Paragraph structure

- Pages with a `\n\n` break: **21,833 (80.9%)**.
- Pages with ≥1 inline-heading match (rough heuristic): 22,864 (84.8%).

| metric | p0 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
|---|---|---|---|---|---|---|---|---|
| paragraphs / page | 1 | 3 | 13 | 33 | 62 | 84 | 131 | 331 |
| paragraph words | 1 | 84 | 96 | 105 | 112 | 115 | 120 | 6,848 |

### Key findings

- **77.9% of pages are truncated at 256 tokens** (21,018 / 26,974) when embedding
  title + content as one unit. The median page is **1,653 tokens (~6× the cap)**,
  so without chunking we embed roughly 15% of a median page and ~3% of a p90 page.
  → **Chunking is mandatory, not optional.** This is the strongest empirical
  justification for chunking and should be shown in the video.
- **Paragraphs fit the model.** Median paragraph ≈ 96 words (~125 tokens), p99
  ≈ 120 words (~160 tokens). Splitting on `\n\n` produces units that almost never
  truncate, and 80.9% of pages contain `\n\n`. → paragraphs are the natural
  semantic unit here.
- **Two complications.** Many paragraphs per page (median 13, p90 62) → a large
  number of vectors. Rare giant "paragraphs" (max 6,848 words = walls of text with
  no `\n\n`) must be hard-split. ~19% of pages have no `\n\n` at all.
- **8% stubs** (e.g. "Eiffel may refer to:") collapse to a single short chunk; the
  prepended title is essentially their only signal.
- **Inline headings are flattened** into the text ("Career.", "Research.") rather
  than markdown / wiki markup, so reliable section-based splitting is not feasible.
  Paragraphs are the practical semantic granularity.

## 4. Design implications

- Chunking is required.
- **Paragraph-packing** is the strong hypothesis; **fixed-window** is the
  comparison baseline.
- **Prepend the title to every chunk** — mid-article chunks and stubs lose their
  subject otherwise.
- Chunk → page aggregation in `retrieve.py` is already **max-pool** (dedup keeps
  each page's best-scoring chunk). With many chunks per page, max can be biased
  (a long page gets more "lottery tickets"), so we will also test top-k mean.
- **Artifact size:** roughly 340k–675k vectors depending on budget → ~0.5–1.0 GB
  in fp32. Store embeddings as **fp16** (halves size, negligible quality loss) and
  use **Git LFS**. Exact dot-product / FlatIP search at this scale is still trivial.

## 5. Chunking strategies to try

Hyperparameters split into two cost classes:

- **Embedding-time (expensive — require re-embedding all chunks):** split method,
  token budget (length), overlap, title-prepend.
- **Query-time (free — evaluated in memory over fixed embeddings):** aggregation
  (max vs top-k mean).

Lengths and overlap are *not* free to sweep (each changes the chunk texts, hence
requires a full re-embed of hundreds of thousands of chunks). Combined with the
overfitting risk of comparing many configs on only 50 public queries, we keep a
**small, principled grid** rather than an exhaustive cross.

### Proposed grid (each row = one full embedding pass)

| # | method | token budget | overlap | note | explanation |
|---|---|---|---|---|---|
| 1 | paragraph-pack | 256 | none | primary hypothesis | Split `content` on `\n\n`, then glue consecutive paragraphs together until reaching ~256 tokens. Cuts on **semantic boundaries**, with chunks near the model's cap. This is the main candidate. |
| 2 | paragraph-pack | 128 | none | length sweep | Same paragraph-packing method, but a smaller ~128-token budget (≈ one paragraph per chunk). Only the budget differs from #1, so comparing #1 vs #2 **isolates the effect of chunk length**. |
| 3 | fixed-window | 256 | ~15% | alternative method + overlap test | **Ignore paragraph boundaries**: slide a fixed ~256-token window over the text, where each chunk repeats ~15% of the previous one. The classic baseline; comparing it to #1 tests the **alternative splitting method**. (It changes both method and overlap vs #1, so it is a baseline comparison, not a clean single-variable test.) |
| 4 (optional) | paragraph-pack | 256 | 1 paragraph | overlap test on semantic method | Same as #1 but carry the last paragraph into the start of the next chunk. Only the overlap differs from #1, so comparing #1 vs #4 **isolates the effect of overlap** on the semantic method. |

- **title-prepend** is also embedding-time. Default to **on** (justified by stubs
  and mid-article chunks losing subject); confirm on/off on a single budget only,
  to avoid doubling the grid.
- **Aggregation** (max vs top-2/3 mean) is swept for free on top of every config.

### Paragraph-packing chunker spec

1. Split `content` on `\n\n` into paragraphs.
2. Greedily pack consecutive paragraphs up to a body budget (≈ budget − title
   tokens) so that title + body stays ≤ 256 tokens.
3. If a single paragraph exceeds the budget, split it into ~budget-sized windows,
   preferring sentence boundaries (regex). **Note:** `nltk` / `spacy` are not
   allowed (import restriction is numpy / sentence-transformers / faiss only), so
   sentence splitting must use a stdlib regex; hard cut as fallback.
4. Prepend the title to every chunk.
5. (Optional, to test) add a dedicated title-only chunk per page for clean
   entity-name matches.

## 6. How we will choose between configs

- Selection on the 50 public queries via our own harness that reuses
  `eval.py`'s `ndcg_at_k`, computing **per-query NDCG@10**.
- 50 queries are noisy, so compare configs with a **paired per-query** test plus
  bootstrap rather than raw mean deltas. Prefer robust, mechanistically justified
  choices. The hidden 50-query set is the real test set; avoid overfitting to the
  public set.
