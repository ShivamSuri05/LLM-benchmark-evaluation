# datasets/scienceqa/data_loader.py
"""
ScienceQA loader for local disk layout:

repo_root/
  data/
    problems.json
    test/
      <qid>/image.png
    train/
      <qid>/image.png           (may exist depending on your dump)
    val/
      <qid>/image.png           (may exist depending on your dump)

problems.json structure (as you showed):
{
  "1": {
    "question": "...",
    "choices": [...],
    "answer": 0,
    "hint": "",
    "image": "image.png",
    "task": "closed choice",
    "grade": "...",
    "subject": "...",
    "topic": "...",
    "category": "...",
    "skill": "...",
    "lecture": "...",
    "solution": "...",
    "split": "train" | "test" | "val"
  },
  ...
}

We filter by obj["split"] to get the official split.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ScienceQAExample:
    qid: str
    split: str

    question: str
    choices: List[str]
    answer_index: int

    hint: str
    lecture: str
    solution: str

    subject: str
    topic: str
    category: str
    skill: str
    grade: str
    task: str

    image_filename: Optional[str]
    image_path: Optional[str]


def load_problems_json(data_root: str) -> Dict[str, Any]:
    problems_path = os.path.join(data_root, "problems.json")
    if not os.path.exists(problems_path):
        raise FileNotFoundError(f"Missing problems.json at: {problems_path}")
    with open(problems_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_image_path(data_root: str, split: str, qid: str, image_filename: Optional[str]) -> Optional[str]:
    """
    Images are stored at: data/<split>/<qid>/<image_filename>
    Example: data/test/1/image.png
    """
    if not image_filename:
        return None
    candidate = os.path.join(data_root, split, str(qid), image_filename)
    return candidate if os.path.exists(candidate) else None


def load_split_examples(data_root: str, split: str = "test") -> List[ScienceQAExample]:
    """
    Returns examples for a given split based on the 'split' field inside problems.json.
    split should be one of: "train", "test", "val".
    """
    problems = load_problems_json(data_root)
    print(len(problems))
    out: List[ScienceQAExample] = []

    for qid, obj in problems.items():
        obj_split = (obj.get("split") or "").strip()
        if obj_split != split:
            continue

        choices = obj.get("choices") or []
        if not isinstance(choices, list) or len(choices) == 0:
            # skip malformed entry
            continue

        ans = obj.get("answer")
        if ans is None:
            continue

        image_filename = obj.get("image") or None
        image_path = _resolve_image_path(data_root, split, str(qid), image_filename)

        out.append(
            ScienceQAExample(
                qid=str(qid),
                split=obj_split,

                question=(obj.get("question") or "").strip(),
                choices=[str(c).strip() for c in choices],
                answer_index=int(ans),

                hint=(obj.get("hint") or "").strip(),
                lecture=(obj.get("lecture") or "").strip(),
                solution=(obj.get("solution") or "").strip(),

                subject=(obj.get("subject") or "").strip(),
                topic=(obj.get("topic") or "").strip(),
                category=(obj.get("category") or "").strip(),
                skill=(obj.get("skill") or "").strip(),
                grade=(obj.get("grade") or "").strip(),
                task=(obj.get("task") or "").strip(),

                image_filename=image_filename,
                image_path=image_path,
            )
        )

    # Deterministic ordering for reproducibility
    out.sort(key=lambda e: int(e.qid) if e.qid.isdigit() else e.qid)
    return out
