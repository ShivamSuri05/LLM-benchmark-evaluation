import os
import sys
import csv
import time
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.llm_client import OpenRouterClient
from core.logging_utils import ensure_dir

from datasets.matscibench.data_loader import load_qa, filter_by_mode,sample_items
from datasets.matscibench.prompts import build_cot_prompt
from datasets.matscibench.scoring import is_correct, llm_judge, extract_final_answer
from dotenv import load_dotenv


ITEM_FIELDS = [
    "question_id",
    "subject",
    "model",
    "mode",
    "model_extracted_output",
    "correct_answer",
    "correct",
    "notes",
    "request_time_ms",
]



SUMMARY_FIELDS = [
    "model",
    "group",
    "group_value",
    "n",
    "accuracy",
]

CATEGORY_COLUMNS = [
    "Failure Mechanisms",
    "Fundamental Mechanisms",
    "Materials",
    "Processes",
    "Properties",
    "Structures",
]

def load_api_key(env_name: str = "OPENROUTER_API_KEY") -> str:
    api_key = os.getenv(env_name)
    if not api_key:
        raise EnvironmentError(f"Missing {env_name}. Set it in .env or export it.")
    return api_key


@dataclass(frozen=True)
class RunConfig:
    csv_path: str
    images_root: str
    out_dir: str
    solver_model: str
    judge_model: str
    base_url: str
    mode: str  # text_only or multimodal
    eval_mode: str  # "rule" | "llm" | "hybrid"
    seed: int
    sample_frac: float
    max_items: Optional[int]

def sanitize_name(s: str) -> str:
    return (
        s.replace("/", "_")
         .replace(":", "_")
         .replace(" ", "_")
         .replace("__", "_")
    )


def make_run_name(cfg: RunConfig, n_items: int) -> str:
    model_part = sanitize_name(cfg.solver_model)
    mode_part = cfg.mode
    frac_part = f"frac{int(cfg.sample_frac * 100)}"
    seed_part = f"seed{cfg.seed}"
    n_part = f"n{n_items}"
    mode= f"{cfg.eval_mode}"
    return f"matscibench_{model_part}_{mode_part}_{mode}_{frac_part}_{seed_part}_{n_part}"



def evaluate_one(solver_client, judge_client, cfg, item):
    prompt = build_cot_prompt(item["question"])

    if cfg.mode == "multimodal":
        resp = solver_client.chat_completion_multimodal_multi(
            model=cfg.solver_model,
            text=prompt,
            image_paths=item["image_paths"],
            temperature=0.0,
            max_tokens=8192,
        )
    else:
        resp = solver_client.chat_completion(
            model=cfg.solver_model,
            prompt=prompt,
            temperature=0.0,
            max_tokens=8192,
            logprobs=False,
            top_logprobs=0,
        )

    #print(resp["usage"])
    output = resp["choices"][0]["message"]["content"]

    #print("Output\n\n",output)
    #print("\n\nActual Answer\n\n",item["answer"])
    final_answer = extract_final_answer(output)
    #print("Extracted Output\n\n",final_answer)
    rule_correct = is_correct(final_answer, item["answer"])
    llm_correct = None
    llm_reasoning = None

    # --- Apply evaluation mode ---
    if cfg.eval_mode == "rule":
        final_correct = rule_correct
        notes = "rule_only"

    elif cfg.eval_mode == "llm":
        llm_correct, llm_reasoning = llm_judge(
            judge_client,
            cfg.judge_model,
            item["question"],
            item["answer"],
            output,
        )
        #print("\n###### LLM Output #########\n")
        #print(llm_correct)
        #print(llm_reasoning)
        final_correct = llm_correct
        notes = "llm_only"

    elif cfg.eval_mode == "hybrid":
        llm_correct, llm_reasoning = llm_judge(
            judge_client,
            cfg.judge_model,
            item["question"],
            item["answer"],
            output,
        )
        final_correct = llm_correct  # LLM overrides (faithful to their logic)
        notes = "hybrid_llm_override"

    else:
        raise ValueError("Invalid eval_mode")

    return {
        "question_id": item["qid"],
        "subject": item["primary_category"],
        "model": cfg.solver_model,
        "mode": cfg.mode,
        "model_extracted_output": final_answer.strip(),
        "correct_answer": item["answer"],
        "correct": "Y" if final_correct else "N",
        "notes": notes,
        "request_time_ms": resp.get("_request_time_ms"),
    }



def run(cfg: RunConfig):
    load_dotenv()
    api_key = load_api_key("OPENROUTER_API_KEY")
    ensure_dir(cfg.out_dir)

    solver = OpenRouterClient(api_key=api_key, base_url=cfg.base_url)
    judge = None
    if cfg.eval_mode in ["llm", "hybrid"]:
        judge = OpenRouterClient(api_key=api_key, base_url=cfg.base_url)

    items = load_qa(cfg.csv_path, cfg.images_root)
    items = filter_by_mode(items, cfg.mode)
    items = sample_items(items, cfg.sample_frac, cfg.seed)

    if cfg.max_items:
        items = items[:cfg.max_items]

    run_name = make_run_name(cfg, len(items))

    items_csv = os.path.join(cfg.out_dir, f"{run_name}_items.csv")
    summary_csv = os.path.join(cfg.out_dir, f"{run_name}_summary.csv")

    print(f"Items CSV:   {items_csv}")
    print(f"Summary CSV: {summary_csv}")

    group_counts = defaultdict(int)
    group_correct = defaultdict(int)

    total_rows = len(items)
    done = 0

    with open(items_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ITEM_FIELDS)
        writer.writeheader()

        for item in items:
            try:
                row = evaluate_one(solver, judge, cfg, item)
            except Exception as e:
                print("Exception caught:::: ---Skipping Row:", item["qid"])
                continue

            writer.writerow(row)

            key = ("overall", "all")
            group_counts[key] += 1
            if row["correct"] == "Y":
                group_correct[key] += 1

            # difficulty
            key = ("difficulty", item["difficulty"])
            group_counts[key] += 1
            if row["correct"] == "Y":
                group_correct[key] += 1

            for col in CATEGORY_COLUMNS:
                #print(col)
                value = item.get(col)
                #print(value)
                #print("@@@@@@@")
                #print(item)
                if value and str(value).strip() != "":
                    key = ("category", col)
                    group_counts[key] += 1
                    if row["correct"] == "Y":
                        group_correct[key] += 1
            
            done += 1
            pct = 100.0 * done / max(1, total_rows)
            print(f"Progress: {done}/{total_rows} ({pct:.1f}%)", end="\r")


    print()

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()

        for (group, value), n in group_counts.items():
            acc = group_correct[(group, value)] / n
            writer.writerow({
                "model": cfg.solver_model,
                "group": group,
                "group_value": value,
                "n": n,
                "accuracy": acc,
            })

    print("Done.")


def main():
    cfg = RunConfig(
        csv_path=os.path.join(REPO_ROOT, "datasets/matscibench/data/qa.csv"),
        images_root=os.path.join(REPO_ROOT, "datasets/matscibench/data"),
        out_dir=os.path.join(REPO_ROOT, "outputs"),
        solver_model="openai/o4-mini",
        judge_model="google/gemini-2.0-flash-001",
        base_url="https://openrouter.ai/api/v1",
        mode= "text_only", #"text_only" or "multimodal",
        eval_mode="llm",  # change to "rule" or "llm" or "hybrid"
        seed=12,
        sample_frac=1.0,
        max_items=None,
    )
    
    t0 = time.time()
    run(cfg)
    dt = time.time() - t0
    print(f"Time Duration: {dt:.1f}s")


if __name__ == "__main__":
    main()
