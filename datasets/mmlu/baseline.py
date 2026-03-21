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

def build_baseline_prompt_v2_reasoning(item: Dict[str, Any], subject: str) -> str:
    """
    Variation 2: Asks for reasoning before the final answer.
    """
    q = item["question"].strip()
    A, B, C, D = [c.strip() for c in item["choices"]]
    return f"""Answer a multiple-choice question from {subject_pretty(subject)}.

QUESTION:
{q}

OPTIONS:
A: {A}
B: {B}
C: {C}
D: {D}

Think through this step-by-step, then provide your final answer as a single letter (A, B, C, or D).
"""

def build_baseline_prompt_v3_concise(item: Dict[str, Any], subject: str) -> str:
    """
    Variation 3: Minimal, concise version with reduced instruction text.
    """
    q = item["question"].strip()
    A, B, C, D = [c.strip() for c in item["choices"]]
    return f"""Question ({subject_pretty(subject)}):
{q}

A: {A}
B: {B}
C: {C}
D: {D}

Answer: [A/B/C/D]
"""

def build_baseline_prompt_v4_detailed(item: Dict[str, Any], subject: str) -> str:
    """
    Variation 4: Emphasizes careful analysis with explicit constraint highlighting.
    """
    q = item["question"].strip()
    A, B, C, D = [c.strip() for c in item["choices"]]
    return f"""You are answering a {subject_pretty(subject)} multiple-choice question.

Question:
{q}

Select the best answer from these options:
A) {A}
B) {B}
C) {C}
D) {D}

IMPORTANT: Your response must end with ONLY the letter of your answer (A, B, C, or D). No other text after the letter.
"""

def extract_letter_from_text(response_text: str) -> Optional[str]:
    match = re.search(r"\b([A-D])\b", response_text.upper())
    return match.group(1) if match else None

def extract_letter_from_text_v2(response_text: str) -> Optional[str]:
    text = response_text.upper()
    
    # 1. Explicit answer statement (most reliable for CoT)
    m = re.search(r'(?:ANSWER IS|ANSWER:|THE ANSWER|THEREFORE)[^A-Z]*([A-D])\b', text)
    if m:
        return m.group(1)
    
    # 2. Last standalone letter as fallback
    matches = re.findall(r'\b([A-D])\b', text)
    return matches[-1] if matches else None

def score_baseline_response(resp: Dict[str, Any]) -> Tuple[Optional[str], str]:
    """
    Returns (pred, notes). pred is A/B/C/D or None if not extractable.
    """
    #print(resp)
    choices = resp.get("choices", [])
    msg = choices[0].get("message", {}) if choices else {}
    text = (msg.get("content") or "").strip()
    #print(text)

    pred = extract_letter_from_text_v2(text)
    if pred is None:
        return None, f"Regex extraction failed. Raw='{text[:120]}'"
    return pred, ""
