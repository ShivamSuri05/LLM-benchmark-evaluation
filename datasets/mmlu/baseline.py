# datasets/mmlu/baseline.py
import re
from typing import Any, Dict, Optional, Tuple

BASELINE_PROMPT_VERSION = "baseline_regex_v1"

def subject_pretty(subject: str) -> str:
    return subject.replace("_", " ")

def build_baseline_prompt(item: Dict[str, Any], subject: str) -> str:
    q = item["question"].strip()
    A, B, C, D = [c.strip() for c in item["choices"]]
    return f"""You will be given a multiple-choice question from {subject_pretty(subject)}.
Choose the best answer based only on the information provided.

QUESTION:
{q}

OPTIONS:
A: {A}
B: {B}
C: {C}
D: {D}

Return ONLY the letter: A, B, C, or D.
"""

def extract_letter_from_text(response_text: str) -> Optional[str]:
    match = re.search(r"\b([A-D])\b", response_text.upper())
    return match.group(1) if match else None

def score_baseline_response(resp: Dict[str, Any]) -> Tuple[Optional[str], str]:
    """
    Returns (pred, notes). pred is A/B/C/D or None if not extractable.
    """
    #print(resp)
    choices = resp.get("choices", [])
    msg = choices[0].get("message", {}) if choices else {}
    text = (msg.get("content") or "").strip()
    #print(text)

    pred = extract_letter_from_text(text)
    if pred is None:
        return None, f"Regex extraction failed. Raw='{text[:120]}'"
    return pred, ""
