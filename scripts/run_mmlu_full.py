# scripts/run_mmlu_full.py
#
# Run MMLU for selected model(s) + selected prompt condition(s),
# and write outputs to uniquely named CSV files.
#
# Examples:
#   # Only GPT-4o, paper prompt, first 2 subjects:
#   python scripts/run_mmlu_full.py --models openai/gpt-4o --prompts paper --max-subjects 2
#
#   # Only GPT-3.5-turbo, baseline prompt, all subjects:
#   python scripts/run_mmlu_full.py --models openai/gpt-3.5-turbo --prompts baseline --max-subjects 0
#
#   # Both models, only paper:
#   python scripts/run_mmlu_full.py --models openai/gpt-3.5-turbo openai/gpt-4o --prompts paper
#
# Outputs:
#   outputs/<auto_name>_items.csv
#   outputs/<auto_name>_summary.csv

import os
import sys
import csv
import time
import json
import argparse
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Ensure repo root is on sys.path so `core` and `datasets` imports work
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.llm_client import OpenRouterClient
from core.logging_utils import ensure_dir

from datasets.mmlu.data_loader import list_subjects, load_subject, sample_test_rows
from datasets.mmlu.prompts import build_5shot_paper_prompt
from datasets.mmlu.scoring import extract_top_logprobs, choose_from_top_logprobs_strict, extract_usage
from datasets.mmlu.baseline import build_baseline_prompt, score_baseline_response, BASELINE_PROMPT_VERSION

PAPER_PROMPT_VERSION = "paper_5shot_strict_logprobs"

ITEM_FIELDS = [
    "prompt_version",
    "question_id",
    "subject",
    "model",
    "model_output",
    "correct_answer",
    "correct",
    "notes",
    "choice_logprobs_json",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cost",
    "request_time_ms",
]

SUMMARY_FIELDS = ["model", "prompt_version", "subject", "n", "accuracy"]


@dataclass(frozen=True)
class FullRunConfig:
    mmlu_root: str
    out_dir: str
    seed: int
    n_shots: int
    sample_frac: float
    models: Tuple[str, ...]
    prompt_versions: Tuple[str, ...]          # subset of {PAPER_PROMPT_VERSION, BASELINE_PROMPT_VERSION}
    max_questions_per_subject: Optional[int]
    max_subjects: Optional[int]               # None => all
    base_url: str
    baseline_max_tokens: int
    baseline_temperature: float


def load_api_key(env_name: str = "OPENROUTER_API_KEY") -> str:
    api_key = os.getenv(env_name)
    if not api_key:
        raise EnvironmentError(f"Missing {env_name}. Set it in .env or export it.")
    return api_key


def sanitize_name(s: str) -> str:
    # file-name safe
    return (
        s.replace("/", "_")
         .replace(":", "_")
         .replace(" ", "_")
         .replace("__", "_")
    )


def make_run_name(cfg: FullRunConfig, subjects: List[str]) -> str:
    models_part = "+".join(sanitize_name(m) for m in cfg.models)
    prompts_part = "+".join(
        "paper" if pv == PAPER_PROMPT_VERSION else "baseline"
        for pv in cfg.prompt_versions
    )
    subj_part = f"subj{len(subjects)}"
    frac_part = f"frac{int(cfg.sample_frac*100)}"
    seed_part = f"seed{cfg.seed}"
    shots_part = f"{cfg.n_shots}shot"
    return f"mmlu_{models_part}_{prompts_part}_{shots_part}_{subj_part}_{frac_part}_{seed_part}"


def get_subjects(mmlu_root: str, max_subjects: Optional[int]) -> List[str]:
    subjects = list_subjects(mmlu_root)
    if max_subjects is not None:
        subjects = subjects[:max_subjects]
    return subjects


def iter_subject_items(cfg: FullRunConfig, subject: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    dev, test = load_subject(cfg.mmlu_root, subject)
    sampled = sample_test_rows(test, cfg.sample_frac, cfg.seed, subject)
    if cfg.max_questions_per_subject is not None:
        sampled = sampled[: cfg.max_questions_per_subject]
    return dev, test, sampled


def evaluate_one_item_paper(
    client: OpenRouterClient,
    *,
    model: str,
    subject: str,
    dev: List[Dict],
    item: Dict,
    n_shots: int,
) -> Dict:
    qid = item["qid"]
    gold = item["answer"]
    prompt = build_5shot_paper_prompt(subject, dev, item, n_shots=n_shots)
    #print(prompt)

    try:
        resp = client.chat_completion(
            model=model,
            prompt=prompt,
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=20,
        )
        top = extract_top_logprobs(resp)
        pred, scores = choose_from_top_logprobs_strict(top)
        usage = extract_usage(resp)

        if pred is None:
            model_out = "__INVALID__"
            notes = "No A/B/C/D token in top_logprobs"
            correct_flag = "N"
        else:
            model_out = pred
            notes = ""
            correct_flag = "Y" if pred == gold else "N"

        return {
            "prompt_version": PAPER_PROMPT_VERSION,
            "question_id": qid,
            "subject": subject,
            "model": model,
            "model_output": model_out,
            "correct_answer": gold,
            "correct": correct_flag,
            "notes": notes,
            "choice_logprobs_json": json.dumps(scores, ensure_ascii=False),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "cost": usage.get("cost"),
            "request_time_ms": resp.get("_request_time_ms"),
        }

    except Exception as e:
        return {
            "prompt_version": PAPER_PROMPT_VERSION,
            "question_id": qid,
            "subject": subject,
            "model": model,
            "model_output": "__ERROR__",
            "correct_answer": gold,
            "correct": "N",
            "notes": f"Exception: {str(e)}",
            "choice_logprobs_json": "",
            "prompt_tokens": "",
            "completion_tokens": "",
            "total_tokens": "",
            "cost": "",
            "request_time_ms": "",
        }


def evaluate_one_item_baseline(
    client: OpenRouterClient,
    *,
    model: str,
    subject: str,
    item: Dict,
    max_tokens: int,
    temperature: float,
) -> Dict:
    qid = item["qid"]
    gold = item["answer"]
    prompt = build_baseline_prompt(item, subject)
    #print(prompt)

    try:
        resp = client.chat_completion(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            logprobs=False,
            top_logprobs=None,
        )

        pred, notes = score_baseline_response(resp)
        usage = extract_usage(resp)

        if pred is None:
            model_out = "__INVALID__"
            correct_flag = "N"
        else:
            model_out = pred
            correct_flag = "Y" if pred == gold else "N"

        return {
            "prompt_version": BASELINE_PROMPT_VERSION,
            "question_id": qid,
            "subject": subject,
            "model": model,
            "model_output": model_out,
            "correct_answer": gold,
            "correct": correct_flag,
            "notes": notes,
            "choice_logprobs_json": "",
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "cost": usage.get("cost"),
            "request_time_ms": resp.get("_request_time_ms"),
        }

    except Exception as e:
        return {
            "prompt_version": BASELINE_PROMPT_VERSION,
            "question_id": qid,
            "subject": subject,
            "model": model,
            "model_output": "__ERROR__",
            "correct_answer": gold,
            "correct": "N",
            "notes": f"Exception: {str(e)}",
            "choice_logprobs_json": "",
            "prompt_tokens": "",
            "completion_tokens": "",
            "total_tokens": "",
            "cost": "",
            "request_time_ms": "",
        }


def precompute_total_rows(cfg: FullRunConfig, subjects: List[str]) -> int:
    total_items = 0
    for subject in subjects:
        _, test = load_subject(cfg.mmlu_root, subject)
        sampled = sample_test_rows(test, cfg.sample_frac, cfg.seed, subject)
        if cfg.max_questions_per_subject is not None:
            sampled = sampled[: cfg.max_questions_per_subject]
        total_items += len(sampled) * len(cfg.models)
    return total_items * len(cfg.prompt_versions)


def write_items_csv(
    cfg: FullRunConfig,
    client: OpenRouterClient,
    subjects: List[str],
    items_csv: str,
) -> Tuple[Dict[Tuple[str, str, str], int], Dict[Tuple[str, str, str], int]]:
    correct_counts = defaultdict(int)  # (model, prompt_version, subject) -> correct
    total_counts = defaultdict(int)    # (model, prompt_version, subject) -> total

    total_rows = precompute_total_rows(cfg, subjects)
    done = 0

    with open(items_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ITEM_FIELDS)
        writer.writeheader()

        for subject_idx, subject in enumerate(subjects, start=1):
            dev, test, sampled = iter_subject_items(cfg, subject)
            print(f"[{subject_idx}/{len(subjects)}] {subject}: {len(sampled)}/{len(test)} questions (frac={cfg.sample_frac})")

            for model in cfg.models:
                for item in sampled:
                    for prompt_version in cfg.prompt_versions:
                        if prompt_version == PAPER_PROMPT_VERSION:
                            row = evaluate_one_item_paper(
                                client,
                                model=model,
                                subject=subject,
                                dev=dev,
                                item=item,
                                n_shots=cfg.n_shots,
                            )
                        elif prompt_version == BASELINE_PROMPT_VERSION:
                            row = evaluate_one_item_baseline(
                                client,
                                model=model,
                                subject=subject,
                                item=item,
                                max_tokens=cfg.baseline_max_tokens,
                                temperature=cfg.baseline_temperature,
                            )
                        else:
                            raise ValueError(f"Unknown prompt_version: {prompt_version}")

                        writer.writerow(row)

                        key = (row["model"], row["prompt_version"], row["subject"])
                        total_counts[key] += 1
                        if row["correct"] == "Y":
                            correct_counts[key] += 1

                        done += 1
                        pct = 100.0 * done / max(1, total_rows)
                        print(f"Progress: {done}/{total_rows} ({pct:.1f}%)", end="\r")

    print()
    return correct_counts, total_counts


def write_summary_csv(
    summary_csv: str,
    subjects: List[str],
    models: Tuple[str, ...],
    prompt_versions: Tuple[str, ...],
    correct_counts: Dict[Tuple[str, str, str], int],
    total_counts: Dict[Tuple[str, str, str], int],
) -> None:
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()

        for model in models:
            for pv in prompt_versions:
                overall_correct = 0
                overall_total = 0

                for subject in subjects:
                    key = (model, pv, subject)
                    c = correct_counts.get(key, 0)
                    n = total_counts.get(key, 0)
                    if n == 0:
                        continue
                    writer.writerow({
                        "model": model,
                        "prompt_version": pv,
                        "subject": subject,
                        "n": n,
                        "accuracy": c / n,
                    })
                    overall_correct += c
                    overall_total += n

                if overall_total > 0:
                    writer.writerow({
                        "model": model,
                        "prompt_version": pv,
                        "subject": "__OVERALL_WEIGHTED__",
                        "n": overall_total,
                        "accuracy": overall_correct / overall_total,
                    })


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models",
        nargs="+",
        default=["openai/gpt-3.5-turbo", "openai/gpt-4o"],
        help="One or more OpenRouter model IDs.",
    )
    p.add_argument(
        "--prompts",
        nargs="+",
        default=["paper", "baseline"],
        choices=["paper", "baseline"],
        help="Which prompt condition(s) to run.",
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--shots", type=int, default=5)
    p.add_argument("--sample-frac", type=float, default=0.4)

    # Use 0 to mean "all subjects" (easier CLI)
    p.add_argument("--max-subjects", type=int, default=2, help="How many subjects to run. Use 0 for all.")
    p.add_argument("--max-questions-per-subject", type=int, default=0, help="Use 0 for no limit.")

    p.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "outputs"))
    return p.parse_args()


def run(cfg: FullRunConfig) -> None:
    load_dotenv()
    api_key = load_api_key("OPENROUTER_API_KEY")
    client = OpenRouterClient(api_key=api_key, base_url=cfg.base_url)

    subjects = get_subjects(cfg.mmlu_root, cfg.max_subjects)
    run_name = make_run_name(cfg, subjects)

    ensure_dir(cfg.out_dir)
    items_csv = os.path.join(cfg.out_dir, f"{run_name}_items.csv")
    summary_csv = os.path.join(cfg.out_dir, f"{run_name}_summary.csv")

    print(f"Items CSV:   {items_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Models: {cfg.models}")
    print(f"Prompts: {cfg.prompt_versions}")

    t0 = time.time()
    correct_counts, total_counts = write_items_csv(cfg, client, subjects, items_csv)
    write_summary_csv(summary_csv, subjects, cfg.models, cfg.prompt_versions, correct_counts, total_counts)
    dt = time.time() - t0

    print("\nDone.")
    print(f"Wrote: {items_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Elapsed: {dt:.1f}s")


def main() -> None:

    max_subjects = None
    max_q = None

    #prompt_versions = [BASELINE_PROMPT_VERSION, PAPER_PROMPT_VERSION]
    #models = ["openai/gpt-4o", "openai/gpt-3.5-turbo"]

    prompt_versions = [PAPER_PROMPT_VERSION]
    models = ["openai/gpt-3.5-turbo"]

    cfg = FullRunConfig(
        mmlu_root=os.path.join(REPO_ROOT, "datasets/mmlu"),  # expects REPO_ROOT/data/dev and REPO_ROOT/data/test
        out_dir=os.path.join(REPO_ROOT, "outputs"),
        seed=123,
        n_shots=4,
        sample_frac=0.4,
        models=tuple(models),
        prompt_versions=tuple(prompt_versions),
        max_questions_per_subject=max_q,
        max_subjects=max_subjects,
        base_url="https://openrouter.ai/api/v1",
        baseline_max_tokens=5,
        baseline_temperature=0.0,
    )
    run(cfg)


if __name__ == "__main__":
    main()
