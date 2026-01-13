# datasets/scienceqa/scoring.py
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, Any, List, Tuple

from datasets.scienceqa.data_loader import ScienceQAExample


def extract_model_text(resp: Dict[str, Any]) -> str:
    choices = resp.get("choices", [])
    if not choices:
        return ""
    msg = choices[0].get("message", {}) or {}
    return (msg.get("content") or "").strip()


def _normalize(s: str) -> str:
    s = s.lower().strip()
    # remove extra whitespace
    s = " ".join(s.split())
    # remove obvious boilerplate tokens
    s = s.replace("the answer is", " ")
    s = s.replace("answer", " ")
    s = s.replace("because", " ")
    s = s.replace(":", " ")
    s = " ".join(s.split())
    return s


def _similarity(a: str, b: str) -> float:
    # Deterministic, lightweight lexical similarity.
    # Paper does not specify exact method; we will document this as a limitation.
    return SequenceMatcher(None, a, b).ratio()


def map_text_to_choice_strict(model_text: str, choices: List[str]) -> Tuple[int, Dict[int, float], str]:
    """
    STRICT paper-like mapping:
    - do NOT use letter extraction
    - map generated text to the most similar option text
    Returns (pred_index, scores_by_index, notes)
    """
    raw = model_text or ""
    norm_out = _normalize(raw)

    # If model outputs nothing, similarity will be low; still pick max deterministically.
    scores: Dict[int, float] = {}
    best_i = 0
    best_s = -1.0

    for i, ch in enumerate(choices):
        norm_ch = _normalize(ch)
        s = _similarity(norm_out, norm_ch)
        scores[i] = float(s)
        if s > best_s:
            best_s = s
            best_i = i

    # Notes for debugging/auditing
    notes = f"strict_similarity=SequenceMatcher ratio; best_score={best_s:.3f}"
    return best_i, scores, notes


def score_prediction(resp: Dict[str, Any], ex: ScienceQAExample) -> Tuple[int, bool, Dict[int, float], str, str]:
    """
    Returns:
      pred_index, correct, scores, notes, raw_text
    """
    raw_text = extract_model_text(resp)
    #print(raw_text)
    pred_idx, scores, notes = map_text_to_choice_strict(raw_text, ex.choices)
    correct = (pred_idx == ex.answer_index)
    return pred_idx, correct, scores, notes, raw_text
