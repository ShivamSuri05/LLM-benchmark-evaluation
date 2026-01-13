# scripts/build_scienceqa_captions_smoke.py
"""
Build (or extend) the ScienceQA caption cache for a tiny smoke subset.

Uses:
- data/problems.json
- images in data/test/<qid>/image.png
Writes:
- outputs/scienceqa_captions.json

Run:
  pip install transformers torch pillow
  python scripts/build_scienceqa_captions_smoke.py

Edit LIMIT to 2 for Phase 3 smoke, or increase later for light runs.
"""

import os
import sys
from dotenv import load_dotenv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datasets.scienceqa.data_loader import load_split_examples
from datasets.scienceqa.captioning import build_captions_for_examples, CaptionConfig


def main():
    load_dotenv()  # not strictly needed for captioning; keeps consistency
    
    data_root = os.path.join(REPO_ROOT, "datasets/scienceqa/data") 
    out_dir = os.path.join(data_root, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    cache_path = os.path.join(out_dir, "scienceqa_captions.json")

    # Phase 3 smoke: only 1–2 questions
    LIMIT = 400
    examples = load_split_examples(data_root, split="test")
    #return

    # Build iterable of (qid, image_path)
    pairs = [(ex.qid, ex.image_path) for ex in examples]

    cfg = CaptionConfig(
        model_name="nlpconnect/vit-gpt2-image-captioning",
        max_length=16,
        num_beams=4,
        batch_size=16,
        debug=True,
    )

    cache = build_captions_for_examples(pairs, cache_path=cache_path, cfg=cfg, limit=LIMIT)

    print(f"Caption cache written: {cache_path}")
    # Print the last few entries for quick verification
    shown = 0
    for qid, cap in list(cache.items())[-5:]:
        print(f"{qid}: {cap}")
        shown += 1
    if shown == 0:
        print("No captions generated (possible: no images found in the selected subset).")


if __name__ == "__main__":
    print("start")
    main()
