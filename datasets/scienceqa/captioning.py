# datasets/scienceqa/captioning.py
"""
ScienceQA image captioning (ViT + GPT-2) with on-disk caching.

IMPORTANT (Windows stability):
- DO NOT import heavy transformers vision classes at module import time.
- We use lazy imports inside the captioner to avoid hangs during import.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Iterable, Tuple, List

from PIL import Image

MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"


@dataclass
class CaptionConfig:
    model_name: str = MODEL_NAME
    device: Optional[str] = None  # "cuda" or "cpu" or None for auto
    max_length: int = 32
    num_beams: int = 4
    batch_size: int = 8
    debug: bool = False


def _lazy_import_caption_deps():
    """
    Lazy import to avoid transformers hanging at import-time on some Windows setups.
    """
    import torch
    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
    return torch, VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer


class VitGpt2Captioner:
    def __init__(self, cfg: CaptionConfig):
        self.cfg = cfg

        if cfg.debug:
            print("[captioning] importing torch/transformers...")

        torch, VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer = _lazy_import_caption_deps()
        self.torch = torch

        self.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

        if cfg.debug:
            print(f"[captioning] loading model={cfg.model_name} on device={self.device} ...")

        self.model = VisionEncoderDecoderModel.from_pretrained(cfg.model_name).to(self.device)
        self.processor = ViTImageProcessor.from_pretrained(cfg.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

        # Safe defaults
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        if cfg.debug:
            print("[captioning] model loaded.")

    def caption_image(self, image_path: str) -> str:
        img = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.to(self.device)

        with self.torch.no_grad():
            output_ids = self.model.generate(
                pixel_values,
                max_length=self.cfg.max_length,
                num_beams=self.cfg.num_beams,
            )

        caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return " ".join(caption.split())

    def caption_images(self, image_paths: List[str]) -> List[str]:
        imgs = [Image.open(p).convert("RGB") for p in image_paths]
        pixel_values = self.processor(images=imgs, return_tensors="pt").pixel_values.to(self.device)

        with self.torch.no_grad():
            output_ids = self.model.generate(
                pixel_values,
                max_length=self.cfg.max_length,
                num_beams=self.cfg.num_beams,
            )

        captions = [self.tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in output_ids]
        return [" ".join(c.split()) for c in captions]


def load_caption_cache(cache_path: str) -> Dict[str, str]:
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_caption_cache(cache_path: str, cache: Dict[str, str]) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp = cache_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp, cache_path)


def build_captions_for_examples(
    examples: Iterable[Tuple[str, Optional[str]]],
    *,
    cache_path: str,
    cfg: CaptionConfig,
    limit: Optional[int] = None,
) -> Dict[str, str]:
    """
    examples: iterable of (qid, image_path)
    limit: number of images to caption (counts only captionable images)
    """
    cache = load_caption_cache(cache_path)

    # collect pending captionable items first (so limit counts only valid images)
    pending: List[Tuple[str, str]] = []
    for qid, img_path in examples:
        if not img_path or not os.path.exists(img_path):
            continue
        if qid in cache and cache[qid]:
            continue
        pending.append((qid, img_path))
        if limit is not None and len(pending) >= limit:
            break

    if cfg.debug:
        print(f"[captioning] pending images to caption: {len(pending)}")

    if not pending:
        return cache

    captioner = VitGpt2Captioner(cfg)

    total = len(pending)

    bs = max(1, cfg.batch_size)
    for i in range(0, len(pending), bs):
        chunk = pending[i : i + bs]
        qids = [c[0] for c in chunk]
        paths = [c[1] for c in chunk]
        #print(paths)
        caps = captioner.caption_images(paths)
        for qid, cap in zip(qids, caps):
            cache[qid] = cap
        
        save_caption_cache(cache_path, cache)
        done = min(i + len(chunk), total)
        pct = (done / total) * 100
        print(f"[captioning] {done}/{total} images captioned ({pct:.1f}%)")

    return cache
