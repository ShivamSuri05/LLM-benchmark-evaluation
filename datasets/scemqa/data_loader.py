import os
import json
from typing import Dict, List
import re

def resolve_image_path(root: str, image_rel: str) -> str:
    base = os.path.join(root, image_rel)

    for ext in [".png", ".jpg", ".jpeg"]:
        candidate = base + ext
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"No image found for {image_rel} with png/jpg/jpeg")

def load_multiple_choice(root: str) -> Dict[str, List[Dict]]:
    json_path = os.path.join(root, "multiple_choice.json")
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data = {}

    for subject, items in raw.items():
        subject_items = []

        for idx, item in enumerate(items, start=1):
            qid = f"{subject}_{idx}"

            image_rel = item["ImagePath"]  # e.g., "Math/1"
            image_path = resolve_image_path(root, image_rel)

            raw_answer = item["Answer (final answer highlighted)"].strip()

            # Try different formats
            patterns = [            
                r"^\(?([A-E])\)?",        # (D) or D
                r"^([A-E])\b",            # D something
            ]

            gold_letter = None
            for pattern in patterns:
                m = re.match(pattern, raw_answer)
                if m:
                    gold_letter = m.group(1)
                    break

            if gold_letter is None:
                raise ValueError(f"Could not parse gold answer from: {raw_answer}")

            subject_items.append({
                "qid": qid,
                "subject": subject,
                "question": item["Question"],
                "answer": gold_letter,
                "image_path": image_path,
            })

        data[subject] = subject_items

    return data


def list_subjects(root: str):
    data = load_multiple_choice(root)
    return list(data.keys())


def sample_items(items: List[Dict], frac: float, seed: int):
    if frac >= 1.0:
        return items

    import random
    rnd = random.Random(seed)
    n = int(len(items) * frac)
    return rnd.sample(items, n)
