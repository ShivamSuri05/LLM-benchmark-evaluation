from typing import Any, Dict, List

PAPER_HEADER_TEMPLATE = "The following are multiple choice questions about {subject}."

def subject_pretty(subject: str) -> str:
    return subject.replace("_", " ")

def format_item_paper(item: Dict[str, Any], include_answer: bool) -> str:
    q = item["question"].strip()
    A, B, C, D = [c.strip() for c in item["choices"]]
    s = f"{q}\n(A) {A} (B) {B} (C) {C} (D) {D}\nAnswer: "
    if include_answer:
        s += f"{item['answer']}"
    return s

def build_5shot_paper_prompt(subject: str, dev: List[Dict[str, Any]], test_item: Dict[str, Any], n_shots: int = 5) -> str:
    header = PAPER_HEADER_TEMPLATE.format(subject=subject_pretty(subject))
    examples = [format_item_paper(dev[i], include_answer=True) for i in range(n_shots)]
    query = format_item_paper(test_item, include_answer=False)
    return "\n".join([header] + examples + [query]) + " "
