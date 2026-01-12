import csv
import pathlib
import random
from typing import Any, Dict, List, Tuple

def _read_mmlu_csv(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, cols in enumerate(reader):
            if len(cols) < 6:
                raise ValueError(f"Bad row (len={len(cols)}) at {path}:{i+1}")
            q, a, b, c, d, ans = cols[0], cols[1], cols[2], cols[3], cols[4], cols[5]
            ans = ans.strip()
            if ans not in {"A", "B", "C", "D"}:
                raise ValueError(f"Unexpected answer label '{ans}' at {path}:{i+1}")
            rows.append({"question": q.strip(), "choices": [a.strip(), b.strip(), c.strip(), d.strip()], "answer": ans})
    return rows

def list_subjects(mmlu_root: str) -> List[str]:
    test_dir = pathlib.Path(mmlu_root) / "data" / "test"
    subjects = []
    for p in sorted(test_dir.glob("*_test.csv")):
        subject = p.name[:-len("_test.csv")]
        subjects.append(subject)
    if not subjects:
        raise FileNotFoundError(f"No *_test.csv files found in {test_dir}")
    return subjects

def load_subject(mmlu_root: str, subject: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dev_path = pathlib.Path(mmlu_root) / "data" / "dev" / f"{subject}_dev.csv"
    test_path = pathlib.Path(mmlu_root) / "data" / "test" / f"{subject}_test.csv"
    if not dev_path.exists():
        raise FileNotFoundError(f"Missing dev file: {dev_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test file: {test_path}")
    return _read_mmlu_csv(str(dev_path)), _read_mmlu_csv(str(test_path))

def sample_test_rows(test: List[Dict[str, Any]], frac: float, seed: int, subject: str) -> List[Dict[str, Any]]:
    n = len(test)
    k = max(1, int(round(n * frac)))
    rng = random.Random(f"{seed}::{subject}")  # stable per subject
    idxs = list(range(n))
    rng.shuffle(idxs)
    keep = sorted(idxs[:k])

    sampled = []
    for old_i in keep:
        item = dict(test[old_i])
        item["qid"] = old_i
        sampled.append(item)
    return sampled
