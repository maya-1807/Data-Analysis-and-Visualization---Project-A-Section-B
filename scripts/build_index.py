"""Build artifacts/ from the full corpus (offline, not timed)."""
from __future__ import annotations

import sys
from pathlib import Path

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

from main import build_offline_index


if __name__ == "__main__":
    build_offline_index()
