# datasets/scienceqa/prompts.py
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from datasets.scienceqa.data_loader import ScienceQAExample

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _format_options(choices: List[str]) -> str:
    # Paper-style options formatting
    parts = []
    for i, ch in enumerate(choices):
        parts.append(f"({LETTERS[i]}) {ch}")
    return " ".join(parts)


def _format_context(hint: str, caption: Optional[str]) -> str:
    # Paper uses "Context:" field; we combine hint + caption into one context block.
    # Caption is the "visual context".
    ctx_parts = []

    hint_clean = hint.strip() if hint else ""
    if hint_clean:
        ctx_parts.append(hint_clean)

    if caption and ("figure" not in hint_clean.lower()):
        ctx_parts.append(caption.strip())

    return " ".join(ctx_parts).strip()


def format_scienceqa_item(
    *,
    question: str,
    choices: List[str],
    context: str,
    answer_text: str,
    lecture: str,
    explanation: str,
) -> str:
    """
    Matches the pattern you quoted:
    Question: ...
    Options: ...
    Context: ...
    Answer: The answer is <answer>. BECAUSE: <lecture> <explanation>
    """
    opts = _format_options(choices)
    # Keep it compact, paper-like
    return (
        f"Question: {question}\n"
        f"Options: {opts}\n"
        f"Context: {context}\n"
        f"Answer: The answer is {answer_text}. BECAUSE: {lecture} {explanation}\n"
    )


def build_1shot_paper_prompt(
    *,
    test_ex: ScienceQAExample,
    captions: Dict[str, str],
    train_pool_by_group: Dict[Tuple[str, str, str], List[ScienceQAExample]],
    seed: int,
    nshots: int
) -> Tuple[str, str]:
    """
    Returns (prompt, notes).

    1-shot example is sampled seeded-randomly from TRAIN within same (subject, category, topic).
    If no matching train example exists, falls back to no-shot prompt (and notes it).
    """
    group = (test_ex.subject, test_ex.category, test_ex.topic)
    rng = random.Random(f"{seed}::{group}::{test_ex.qid}")

    # Build test context (hint + caption)
    test_caption = captions.get(test_ex.qid, "")
    test_context = _format_context(test_ex.hint, test_caption)

    # Compose the test query (ending with Answer:)
    test_opts = _format_options(test_ex.choices)
    test_block = (
        f"Question: {test_ex.question}\n"
        f"Options: {test_opts}\n"
        f"Context: {test_context}\n"
        f"Answer:"
    )

    # Pick 1-shot from train group
    pool = train_pool_by_group.get(group, [])
    if not pool or nshots == 0:
        # No-shot fallback
        prompt = test_block + "\n"
        return prompt, "no_train_match_for_group -> used_0shot"

    shot_ex = pool[rng.randrange(len(pool))]

    # For the 1-shot example, we need answer text and CoT fields:
    # We'll use the gold choice text as "answer text"
    shot_answer_text = shot_ex.choices[shot_ex.answer_index]
    shot_caption = captions.get(shot_ex.qid, "")  # might be absent if you only captioned test; ok
    shot_context = _format_context(shot_ex.hint, shot_caption)

    shot_block = format_scienceqa_item(
        question=shot_ex.question,
        choices=shot_ex.choices,
        context=shot_context,
        answer_text=shot_answer_text,
        lecture=shot_ex.lecture.strip(),
        explanation=shot_ex.solution.strip(),
    )

    prompt = shot_block + "\n" + test_block + "\n"
    return prompt, ""
