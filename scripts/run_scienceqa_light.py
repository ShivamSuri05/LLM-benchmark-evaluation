# scripts/run_scienceqa_light.py
import os
import sys
import csv
import json
import time
from collections import defaultdict
from dotenv import load_dotenv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.llm_client import OpenRouterClient
from core.logging_utils import ensure_dir

from datasets.scienceqa.data_loader import load_split_examples, ScienceQAExample
from datasets.scienceqa.prompts import build_1shot_paper_prompt
from datasets.scienceqa.scoring import score_prediction


ITEM_FIELDS = [
    "qid",
    "subject",
    "category",
    "topic",
    "model",
    "shots",
    "gold_index",
    "gold_text",
    "pred_index",
    "pred_text",
    "correct",
    "notes",
    "scores_json",
    "raw_output",
    "request_time_ms",
]

SUMMARY_FIELDS = ["model", "shots", "n", "accuracy"]


def load_captions(cache_path: str) -> dict:
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Missing captions cache: {cache_path}")
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_train_group_index(train_examples):
    """
    group key: (subject, category, topic) -> list of train examples
    """
    idx = defaultdict(list)
    for ex in train_examples:
        key = (ex.subject, ex.category, ex.topic)
        idx[key].append(ex)
    return idx


def filter_test_examples(test_examples, captions: dict, max_qid: int):
    """
    Keep only:
    - qid numeric
    - qid <= max_qid
    - caption exists (since you only captioned up to some id)
    """
    kept = []
    for ex in test_examples:
        if not ex.qid.isdigit():
            continue
        if int(ex.qid) > max_qid:
            continue
        kept.append(ex)
    return kept


def print_progress(done, total, start_time, extra=""):
    pct = (done / total) * 100 if total else 100.0
    elapsed = time.time() - start_time
    print(f"[ScienceQA] {done}/{total} ({pct:.1f}%) elapsed={elapsed:.1f}s {extra}".strip())


def main():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing OPENROUTER_API_KEY in env/.env")

    # ---- Config ----
    model = "openai/gpt-3.5-turbo"
    shots = 1
    seed = 123

    max_qid = 8230  # per your constraint
    max_tokens = 512  # paper mentions 512 output tokens for generation stage
    temperature = 0.0

    data_root = os.path.join(REPO_ROOT, "datasets/scienceqa/data") 
    out_dir = os.path.join(REPO_ROOT, "outputs")
    ensure_dir(out_dir)

    captions_path = os.path.join(data_root, "outputs/scienceqa_captions.json")

    items_csv = os.path.join(out_dir, f"scienceqa_light_qid_le_{max_qid}_{model.replace('/', '_')}_{shots}shot.csv")
    summary_csv = os.path.join(out_dir, f"scienceqa_light_summary_qid_le_{max_qid}_{model.replace('/', '_')}_{shots}shot.csv")

    # ---------------

    captions = load_captions(captions_path)

    # Load splits
    train_examples = load_split_examples(data_root, split="train")
    test_examples = load_split_examples(data_root, split="test")

    train_by_group = build_train_group_index(train_examples)

    # Filter to your requested range and only those with captions already generated
    test_subset = filter_test_examples(test_examples, captions, max_qid=max_qid)
    if not test_subset:
        raise RuntimeError(f"No test examples found with qid<= {max_qid} AND present in captions cache.")

    print(f"Loaded train={len(train_examples)} test={len(test_examples)}")
    print(f"Running subset: {len(test_subset)} test questions (qid<= {max_qid} and captioned).")

    client = OpenRouterClient(api_key=api_key)

    correct = 0
    total = 0

    t0 = time.time()

    with open(items_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ITEM_FIELDS)
        w.writeheader()

        for ex in test_subset:
            total += 1

            prompt, pnotes = build_1shot_paper_prompt(
                test_ex=ex,
                captions=captions,
                train_pool_by_group=train_by_group,
                seed=seed,
                nshots=shots
            )

            #print(prompt)
            try:
                resp = client.chat_completion(
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                pred_idx, ok, scores, snotes, raw = score_prediction(resp, ex)
                correct += 1 if ok else 0

                gold_text = ex.choices[ex.answer_index]
                pred_text = ex.choices[pred_idx] if 0 <= pred_idx < len(ex.choices) else ""

                w.writerow({
                    "qid": ex.qid,
                    "subject": ex.subject,
                    "category": ex.category,
                    "topic": ex.topic,
                    "model": model,
                    "shots": shots,
                    "gold_index": ex.answer_index,
                    "gold_text": gold_text,
                    "pred_index": pred_idx,
                    "pred_text": pred_text,
                    "correct": "Y" if ok else "N",
                    "notes": " | ".join([x for x in [pnotes, snotes] if x]),
                    "scores_json": json.dumps(scores, ensure_ascii=False),
                    "raw_output": raw,
                    "request_time_ms": resp.get("_request_time_ms"),
                })

            except Exception as e:
                # Log error row; counts as incorrect
                w.writerow({
                    "qid": ex.qid,
                    "subject": ex.subject,
                    "category": ex.category,
                    "topic": ex.topic,
                    "model": model,
                    "shots": shots,
                    "gold_index": ex.answer_index,
                    "gold_text": ex.choices[ex.answer_index] if ex.choices else "",
                    "pred_index": "",
                    "pred_text": "",
                    "correct": "N",
                    "notes": f"Exception: {type(e).__name__}: {str(e)}",
                    "scores_json": "",
                    "raw_output": "",
                    "request_time_ms": "",
                })

            if total % 10 == 0 or total == len(test_subset):
                acc = correct / total if total else 0.0
                print_progress(total, len(test_subset), t0, extra=f"acc={acc:.3f}")

    accuracy = correct / total if total else 0.0

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerow({"model": model, "shots": shots, "n": total, "accuracy": accuracy})

    print("\nDone.")
    print(f"Wrote items:   {items_csv}")
    print(f"Wrote summary: {summary_csv}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
