import os
import sys
import csv
import time
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Callable
from dotenv import load_dotenv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.llm_client import OpenRouterClient
from core.logging_utils import ensure_dir

from datasets.scemqa.data_loader import (
    load_multiple_choice,
    list_subjects,
    sample_items,
)
from datasets.scemqa.prompts import (
    build_zero_shot_paper_style_prompt,
    build_prompt_v1_baseline,
    build_prompt_v2_reasoning,
    build_prompt_v3_concise,
    build_prompt_v4_detailed,
)
from datasets.scemqa.scoring import extract_mc_answer


# ── Prompt version constants ───────────────────────────────────────────────────
PAPER_STYLE_PROMPT_VERSION    = "paper_style"
BASELINE_PROMPT_VERSION_V1    = "v1_baseline"
BASELINE_PROMPT_VERSION_V2    = "v2_reasoning"
BASELINE_PROMPT_VERSION_V3    = "v3_concise"
BASELINE_PROMPT_VERSION_V4    = "v4_detailed"

PROMPT_REGISTRY: Dict[str, Callable[[dict], str]] = {
    PAPER_STYLE_PROMPT_VERSION: build_zero_shot_paper_style_prompt,
    BASELINE_PROMPT_VERSION_V1: build_prompt_v1_baseline,
    BASELINE_PROMPT_VERSION_V2: build_prompt_v2_reasoning,
    BASELINE_PROMPT_VERSION_V3: build_prompt_v3_concise,
    BASELINE_PROMPT_VERSION_V4: build_prompt_v4_detailed,
}

# ── CSV fields ─────────────────────────────────────────────────────────────────
ITEM_FIELDS = [
    "question_id",
    "subject",
    "model",
    "prompt_version",
    "mode",
    "model_output",
    "correct_answer",
    "correct",
    "notes",
    "request_time_ms",
]

SUMMARY_FIELDS = ["model", "prompt_version", "subject", "n", "accuracy"]


@dataclass(frozen=True)
class ScemqaRunConfig:
    scemqa_root: str
    out_dir: str
    seed: int
    sample_frac: float
    models: Tuple[str, ...]
    prompt_versions: Tuple[str, ...]          # ← new
    max_questions_per_subject: Optional[int]
    max_subjects: Optional[int]
    base_url: str
    mode: str  # "multimodal" or "text_only"


def load_api_key():
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise EnvironmentError("Missing OPENROUTER_API_KEY")
    return key


def make_run_name(cfg: ScemqaRunConfig, subjects: List[str]) -> str:
    models_part   = "+".join(m.replace("/", "_") for m in cfg.models)
    prompts_part  = "+".join(cfg.prompt_versions)
    subj_part     = f"subj{len(subjects)}"
    frac_part     = f"frac{int(cfg.sample_frac * 100)}"
    seed_part     = f"seed{cfg.seed}"
    return f"scemqa_mc_{models_part}_{prompts_part}_{subj_part}_{frac_part}_{seed_part}"


def evaluate_one_item(client, model, item, mode, prompt_version: str):
    build_prompt = PROMPT_REGISTRY[prompt_version]
    prompt = build_prompt(item)

    #print(f"[{prompt_version}] Prompt:\n{prompt}")

    if mode == "multimodal":
        resp = client.chat_completion_multimodal(
            model=model,
            text=prompt,
            image_path=item["image_path"],
            max_tokens=750,
            temperature=0.0,
        )
    elif mode == "text_only":
        resp = client.chat_completion(
            model=model,
            prompt=prompt,
            max_tokens=750,
            temperature=0.0,
            logprobs=False,
            top_logprobs=None,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    pred, notes = extract_mc_answer(resp)

    return {
        "question_id":    item["qid"],
        "subject":        item["subject"],
        "model":          model,
        "prompt_version": prompt_version,
        "mode":           mode,
        "model_output":   pred if pred else "__INVALID__",
        "correct_answer": item["answer"],
        "correct":        "Y" if pred == item["answer"] else "N",
        "notes":          notes,
        "usage_raw":      resp.get("usage", {}),
        "request_time_ms": resp.get("_request_time_ms"),
    }


def run(cfg: ScemqaRunConfig):
    # Validate prompt versions up front
    for pv in cfg.prompt_versions:
        if pv not in PROMPT_REGISTRY:
            raise ValueError(f"Unknown prompt version '{pv}'. Valid: {list(PROMPT_REGISTRY)}")

    total_prompt_tokens      = 0
    total_completion_tokens  = 0
    total_tokens             = 0
    total_cost               = 0.0

    load_dotenv()
    api_key = load_api_key()
    client  = OpenRouterClient(api_key=api_key, base_url=cfg.base_url)

    data     = load_multiple_choice(cfg.scemqa_root)
    subjects = list(data.keys())

    if cfg.max_subjects:
        subjects = subjects[:cfg.max_subjects]

    run_name    = make_run_name(cfg, subjects)
    print(f"Subjects : {subjects}")
    print(f"Prompts  : {list(cfg.prompt_versions)}")

    ensure_dir(cfg.out_dir)
    items_csv   = os.path.join(cfg.out_dir, f"{run_name}_items.csv")
    summary_csv = os.path.join(cfg.out_dir, f"{run_name}_summary.csv")

    correct_counts = defaultdict(int)
    total_counts   = defaultdict(int)

    # Pre-count total items for progress display
    total_items = 0
    for subject in subjects:
        items = data[subject]
        items = sample_items(items, cfg.sample_frac, cfg.seed)
        if cfg.max_questions_per_subject:
            items = items[:cfg.max_questions_per_subject]
        total_items += len(items) * len(cfg.models) * len(cfg.prompt_versions)

    done = 0

    with open(items_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ITEM_FIELDS)
        writer.writeheader()

        print(subjects)
        # only keep last 2 entries in subjects for quick testing
        subjects = subjects[-3:]
        subjects = [subjects.pop(0)]
        print(subjects)
        #return
        for subject in subjects:
            items = data[subject]
            items = sample_items(items, cfg.sample_frac, cfg.seed)
            if cfg.max_questions_per_subject:
                items = items[:cfg.max_questions_per_subject]

            print(f"\n{subject}: {len(items)} questions")

            for model in cfg.models:
                for prompt_version in cfg.prompt_versions:      # ← new loop
                    for item in items:
                        row   = evaluate_one_item(client, model, item, cfg.mode, prompt_version)
                        usage = row.pop("usage_raw", None)

                        if usage:
                            total_prompt_tokens     += usage.get("prompt_tokens", 0)
                            total_completion_tokens += usage.get("completion_tokens", 0)
                            total_tokens            += usage.get("total_tokens", 0)
                            total_cost              += usage.get("cost", 0.0)

                        writer.writerow(row)

                        key = (model, prompt_version, subject)
                        total_counts[key]   += 1
                        if row["correct"] == "Y":
                            correct_counts[key] += 1

                        done += 1
                        print(f"Progress: {done}/{total_items}", end="\r")

    print()
    print("\n=== COST SUMMARY ===")
    print(f"Total prompt tokens:      {total_prompt_tokens}")
    print(f"Total completion tokens:  {total_completion_tokens}")
    print(f"Total tokens:             {total_tokens}")
    print(f"Total cost:               ${total_cost:.4f}")
    if total_items > 0:
        print(f"Average cost per question: ${total_cost / total_items:.6f}")

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()

        for model in cfg.models:
            for prompt_version in cfg.prompt_versions:
                overall_correct = 0
                overall_total   = 0

                for subject in subjects:
                    key = (model, prompt_version, subject)
                    c   = correct_counts[key]
                    n   = total_counts[key]

                    if n == 0:
                        continue

                    writer.writerow({
                        "model":          model,
                        "prompt_version": prompt_version,
                        "subject":        subject,
                        "n":              n,
                        "accuracy":       c / n,
                    })

                    overall_correct += c
                    overall_total   += n

                writer.writerow({
                    "model":          model,
                    "prompt_version": prompt_version,
                    "subject":        "__OVERALL_WEIGHTED__",
                    "n":              overall_total,
                    "accuracy":       overall_correct / overall_total if overall_total else 0,
                })

    print("Done.")


def main():
    load_dotenv()

    models = ["openai/gpt-4o"]

    cfg = ScemqaRunConfig(
        scemqa_root=os.path.join(REPO_ROOT, "datasets/scemqa/data/Multiple_Choice"),
        out_dir=os.path.join(REPO_ROOT, "outputs"),
        seed=1723,
        sample_frac=1.0,
        models=tuple(models),
        prompt_versions=(                           # ← pick any subset here
            #PAPER_STYLE_PROMPT_VERSION,
            #BASELINE_PROMPT_VERSION_V1,
            #BASELINE_PROMPT_VERSION_V2,
            BASELINE_PROMPT_VERSION_V3,
            BASELINE_PROMPT_VERSION_V4,
        ),
        max_questions_per_subject=None,
        max_subjects=None,
        base_url="https://openrouter.ai/api/v1",
        mode="multimodal",
    )

    t0 = time.time()
    run(cfg)
    dt = time.time() - t0
    print(f"Time Duration: {dt:.1f}s")


if __name__ == "__main__":
    main()