# Section B — Retrieval pipeline

## Setup

```bash
cd path/to/student
pip install -r requirements.txt
```

Corpus lives at **`data/Wikipedia Entries/`** (included in the handout).

## Build index (offline, not timed — your machine only)

Run once locally to create `artifacts/`. **Submit these files** in your repo; staff do not rebuild the index at grading time.

```bash
python scripts/build_index.py
```

## Public self-test

After building, verify a fresh run loads your submitted artifacts (no rebuild):

```bash
python scripts/eval_public.py
```

## Submit

Public GitHub repo with this code, **required** `artifacts/`, and a concise README documenting artifact paths. See the assignment PDF for video and grading details.
