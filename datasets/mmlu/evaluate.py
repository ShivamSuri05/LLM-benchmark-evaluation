import random
from typing import Dict, List

from core.llm_client import OpenRouterClient
from datasets.mmlu.data_loader import load_subject
from datasets.mmlu.prompts import build_5shot_paper_prompt
from datasets.mmlu.scoring import extract_top_logprobs, choose_from_top_logprobs_strict

def smoke_test_subject(
    client: OpenRouterClient,
    *,
    mmlu_root: str,
    subject: str,
    models: List[str],
    seed: int,
    n_shots: int,
    limit_questions: int,
) -> List[Dict]:
    dev, test = load_subject(mmlu_root, subject)

    rng = random.Random(f"smoke::{seed}::{subject}")
    idxs = list(range(len(test)))
    rng.shuffle(idxs)
    idxs = idxs[:limit_questions]

    rows = []
    for model in models:
        for qid in idxs:
            item = dict(test[qid])
            prompt = build_5shot_paper_prompt(subject, dev, item, n_shots=n_shots)

            resp = client.chat_completion(model=model, prompt=prompt)
            try:
                top = extract_top_logprobs(resp)
                pred, _scores = choose_from_top_logprobs_strict(top)

                if pred is None:
                    model_out = "__INVALID__"
                    notes = "No A/B/C/D token in top_logprobs"
                    correct_flag = "N"
                else:
                    model_out = pred
                    notes = ""
                    correct_flag = "Y" if pred == item["answer"] else "N"
            except Exception as e:
                model_out = "__ERROR__"
                notes = f"Exception: {str(e)}"
                correct_flag = "N"

            rows.append({
                "question_id": qid,
                "subject": subject,
                "model": model,
                "model_output": model_out,
                "correct_answer": item["answer"],
                "correct": correct_flag,
                "notes": notes,
            })

    return rows
