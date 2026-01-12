import csv
import os
from typing import Dict, Iterable, List

DEFAULT_FIELDS = [
    "question_id",
    "subject",
    "model",
    "model_output",
    "correct_answer",
    "correct",
    "notes",
]

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_csv(path: str, fieldnames: List[str], rows: Iterable[Dict]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
