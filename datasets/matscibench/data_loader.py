import os
import csv
from typing import List, Dict
import random
from typing import List, Dict
import re


def sample_items(items: List[Dict], frac: float, seed: int):
    if frac >= 1.0:
        return items
    random.seed(seed)
    k = max(1, int(len(items) * frac))
    return random.sample(items, k)


def load_qa(csv_path: str, images_root: str) -> List[Dict]:
    items = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:

            # --- Handle multiple images safely ---
            image_paths = []

            raw_images = row.get("image", "")
            if raw_images:
                # Split on common separators (adjust if needed)
                parts = re.split(r",", raw_images)

                for img in parts:
                    img = img.strip()
                
                    if img:
                        full_path = os.path.join(images_root, img)
                        if os.path.isfile(full_path):
                            image_paths.append(full_path)
                        else:
                            print(f"[WARNING] Image not found: {full_path}")

            items.append({
                "qid": row["qid"],
                "question": row["question"].strip(),
                "answer": row["answer"].strip(),
                "type": row["type"].strip(),
                "difficulty": row["difficulty_level"].strip(),
                "primary_category": row["primary_category"].strip(),
                "image_paths": image_paths,  # ← list now

                # Paper categories
                "Failure": row.get("Failure Mechanisms", "").strip(),
                "Fundamental": row.get("Fundamental Mechanisms", "").strip(),
                "Materials": row.get("Materials", "").strip(),
                "Processes": row.get("Processes", "").strip(),
                "Properties": row.get("Properties", "").strip(),
                "Structures": row.get("Structures", "").strip(),
            })

    return items


def filter_by_mode(items: List[Dict], mode: str) -> List[Dict]:
    if mode == "text_only":
        return [x for x in items if not x.get("image_paths")]
    elif mode == "multimodal":
        return [x for x in items if x.get("image_paths")]
    else:
        raise ValueError("Unknown mode")

