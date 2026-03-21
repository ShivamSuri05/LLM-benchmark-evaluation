def build_zero_shot_paper_style_prompt(item: dict) -> str:
    """Original: Variation 0 - Raw paper style, no added instruction."""
    question_text = item["question"]
    return question_text.strip()


def build_prompt_v1_baseline(item: dict) -> str:
    """
    Variation 1: Direct instruction baseline.
    Mirrors MMLU v1 - explicit subject context + constrained output.
    """
    subject = item.get("subject", "Science")
    question_text = item["question"].strip()
    return f"""You will be given a multiple-choice question from {subject}.
Choose the best answer based only on the information provided.

{question_text}

Return ONLY the letter: A, B, C, D, or E.
"""


def build_prompt_v2_reasoning(item: dict) -> str:
    """
    Variation 2: Step-by-step reasoning before final answer.
    Mirrors MMLU v2 - encourages chain-of-thought.
    """
    subject = item.get("subject", "Science")
    question_text = item["question"].strip()
    return f"""Answer a multiple-choice question from {subject}.

{question_text}

Think through this step-by-step, then provide your final answer as a single letter (A, B, C, D, or E).
"""


def build_prompt_v3_concise(item: dict) -> str:
    """
    Variation 3: Minimal concise version.
    Mirrors MMLU v3 - reduced instruction overhead.
    """
    subject = item.get("subject", "Science")
    question_text = item["question"].strip()
    return f"""Question ({subject}):
{question_text}

Answer: [A/B/C/D/E]
"""


def build_prompt_v4_detailed(item: dict) -> str:
    """
    Variation 4: Emphasizes careful analysis with explicit constraints.
    Mirrors MMLU v4 - strict output formatting instruction.
    """
    subject = item.get("subject", "Science")
    question_text = item["question"].strip()
    return f"""You are answering a {subject} multiple-choice question.

{question_text}

IMPORTANT: Your response must end with ONLY the letter of your answer (A, B, C, D, or E). No other text after the letter.
"""