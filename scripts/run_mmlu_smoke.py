# scripts/run_mmlu_smoke.py
#
# Run:
#   1) Put OPENROUTER_API_KEY in your .env (or export it in your shell)
#   2) From repo root:
#        python scripts/run_mmlu_smoke.py

import os
import sys
from dotenv import load_dotenv

# Ensure repo root is on sys.path so `core` and `datasets` imports work
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.llm_client import OpenRouterClient
from core.logging_utils import write_csv, DEFAULT_FIELDS


def main() -> None:
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing OPENROUTER_API_KEY. Set it in .env or export it.")

    # Configure smoke test
    mmlu_root = os.path.join(REPO_ROOT, "datasets/mmlu")              # contains ./data/dev and ./data/test
    out_dir = os.path.join(REPO_ROOT, "outputs")
    out_path = os.path.join(out_dir, "mmlu_smoke_test_log.csv")

    subject = "astronomy"              # change if you want
    limit_questions = 2               # smoke size
    seed = 123
    n_shots = 5

    models = [
        "openai/gpt-3.5-turbo",
        "openai/gpt-4o",
    ]

    client = OpenRouterClient(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    # Import here so sys.path change is already applied
    from datasets.mmlu.evaluate import smoke_test_subject

    rows = smoke_test_subject(
        client,
        mmlu_root=mmlu_root,
        subject=subject,
        models=models,
        seed=seed,
        n_shots=n_shots,
        limit_questions=limit_questions,
    )

    write_csv(out_path, DEFAULT_FIELDS, rows)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
