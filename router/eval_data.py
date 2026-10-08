"""Load the labelled routing eval sets from data/*.jsonl.

Each line is {"query": ..., "label": "small" | "big", "category": ...}.
eval_500 and eval_1000 used to carry these as inline Python lists; the data
now lives in data/ so it can be read by other tools and diffed on its own.
"""

from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_cases(filename: str) -> list[tuple[str, str, str]]:
    """Return [(query, label, category), ...] in file order."""
    cases = []
    with open(DATA_DIR / filename, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                cases.append((row["query"], row["label"], row["category"]))
    return cases
