#!/usr/bin/env python3
"""
Analyze the retrieval corpus to decide a chunking strategy.

Reports, over all pages:
  - content length in chars / words (percentiles)
  - paragraph structure (\n\n splits) and paragraph word-length
  - stub pages (almost no content, e.g. disambiguation pages)
  - a rough inline-heading signal (approximate)
  - OPTIONAL token length with the MiniLM tokenizer (TOKENS = True),
    incl. the share of pages truncated at 256 tokens  <-- the key number

Run from the repo root:
    python planning/analyze_corpus.py

It can also be run from planning/; paths below resolve from this file.

Configure via the Config block below (no command-line arguments).
First run builds a one-file cache so later runs load instantly.
"""
from __future__ import annotations

import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ----------------------- Config (edit these) -----------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DIR = REPO_ROOT / "data" / "Wikipedia Entries"  # corpus directory (read once, then cached)
CACHE = REPO_ROOT / "corpus_cache.jsonl"        # consolidated cache; None to disable
WORKERS = 32                     # parallel file readers (I/O-bound)
TOKENS = True                    # also compute MiniLM token lengths
TOKEN_SAMPLE = 0                 # 0 = all pages; else random sample size (fast estimate)
STUB_WORDS = 20                  # a page with fewer words than this counts as a stub
SAMPLES = 3                      # how many raw sample contents to print
BATCH = 2000                     # tokenizer batch size
# -------------------------------------------------------------------

WORD_RE = re.compile(r"\S+")
HEADING_RE = re.compile(r"(?:^|\n|\.\s)([A-Z][A-Za-z][A-Za-z ]{0,28}\.)(?=\s|\n|$)")
CAP = 256  # all-MiniLM-L6-v2 max_seq_length (incl. [CLS]/[SEP] => ~254 usable)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_one(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_from_dir(d: Path):
    paths = sorted(d.glob("*.json"))
    n = len(paths)
    if n == 0:
        return []
    records = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, rec in enumerate(ex.map(_read_one, paths), 1):
            if rec is not None:
                records.append(rec)
            if i % 2000 == 0 or i == n:
                print(f"  loading... {i}/{n}", end="\r", flush=True)
    print()
    return records


def load_corpus():
    """Load from cache if present, else read the dir (parallel) and write cache."""
    if CACHE and CACHE.exists():
        print(f"loading cache {display_path(CACHE)} ...")
        with CACHE.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    print(f"reading {display_path(DIR)} (first time) ...")
    records = load_from_dir(DIR)
    if CACHE and records:
        with CACHE.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote cache {display_path(CACHE)} ({len(records)} records) "
              f"-- later runs will load instantly")
    return records


def pctl(values, ps=(0, 25, 50, 75, 90, 95, 99, 100)):
    if not values:
        return ""
    s = sorted(values)
    n = len(s)
    out = []
    for p in ps:
        if p == 0:
            v = s[0]
        elif p == 100:
            v = s[-1]
        else:
            v = s[min(n - 1, int(round(p / 100 * (n - 1))))]
        out.append(f"p{p}={v}")
    return "  ".join(out)


def load_tokenizer():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        from sentence_transformers import SentenceTransformer
        tok = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2").tokenizer
    try:
        tok.model_max_length = int(1e9)
    except Exception:
        pass
    return tok


def token_lengths(tok, texts):
    lengths = []
    n = len(texts)
    for i in range(0, n, BATCH):
        enc = tok(texts[i:i + BATCH], add_special_tokens=False, truncation=False)
        lengths.extend(len(ids) for ids in enc["input_ids"])
        print(f"  tokenizing... {min(i + BATCH, n)}/{n}", end="\r", flush=True)
    print()
    return lengths


def main() -> None:
    entries = load_corpus()
    n = len(entries)
    if n == 0:
        print(f"No records found (DIR={DIR!r})")
        return

    char_len, word_len, title_words = [], [], []
    para_counts, para_word_len = [], []
    stub = has_blank = heading_pages = 0

    for e in entries:
        title = (e.get("title") or "").strip()
        content = (e.get("content") or "").strip()
        cw = WORD_RE.findall(content)
        title_words.append(len(WORD_RE.findall(title)))
        char_len.append(len(content))
        word_len.append(len(cw))
        if len(cw) < STUB_WORDS:
            stub += 1
        if "\n\n" in content:
            has_blank += 1
        paras = [p.strip() for p in content.split("\n\n") if p.strip()]
        para_counts.append(len(paras))
        para_word_len.extend(len(WORD_RE.findall(p)) for p in paras)
        if HEADING_RE.search(content):
            heading_pages += 1

    print(f"\npages = {n}")
    print(f"stub pages (<{STUB_WORDS} words) = {stub} ({100 * stub / n:.1f}%)\n")
    print("content chars:", pctl(char_len))
    print("content words:", pctl(word_len))
    print("title  words: ", pctl(title_words))
    print()
    print(f"pages with a blank-line break (\\n\\n) = {has_blank} ({100 * has_blank / n:.1f}%)")
    print("paragraphs/page:", pctl(para_counts))
    print("paragraph words:", pctl(para_word_len))
    print()
    print(f"pages w/ inline-heading match = {heading_pages} "
          f"({100 * heading_pages / n:.1f}%)  [rough heuristic]")

    if TOKENS:
        try:
            tok = load_tokenizer()
        except Exception as ex:
            print(f"\n[TOKENS skipped: {ex}]")
            return
        sample = entries
        if TOKEN_SAMPLE and TOKEN_SAMPLE < n:
            sample = random.Random(0).sample(entries, TOKEN_SAMPLE)
            print(f"\n[token stats on random sample of {len(sample)} pages]")
        contents = [(e.get("content") or "").strip() for e in sample]
        titles = [(e.get("title") or "").strip() for e in sample]
        print("content tokens:")
        ctoks = token_lengths(tok, contents)
        print("title tokens:")
        ttoks = token_lengths(tok, titles)
        etoks = [c if not t else c + tt + 1
                 for c, t, tt in zip(ctoks, titles, ttoks)]
        over = sum(1 for et in etoks if et + 2 > CAP)
        m = len(sample)
        print("\ncontent tokens:       ", pctl(ctoks))
        print("title+content tokens: ", pctl(etoks))
        print(f"pages truncated at {CAP} tok (title+content) = "
              f"{over} ({100 * over / m:.1f}%)   <-- KEY NUMBER")

    print("\n--- samples (first %d, raw content head) ---" % SAMPLES)
    for e in entries[:SAMPLES]:
        c = e.get("content") or ""
        print(f"[{e.get('page_id')}] {e.get('title')!r}  ({len(WORD_RE.findall(c))} words)")
        print(repr(c[:400]) + ("..." if len(c) > 400 else ""))
        print()


if __name__ == "__main__":
    main()
