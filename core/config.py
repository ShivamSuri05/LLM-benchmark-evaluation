from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class RunConfig:
    # dataset selection / sampling
    dataset_root: str      # repo root containing ./data/dev and ./data/test for MMLU
    sample_frac: float     # e.g. 0.4
    seed: int              # reproducible
    n_shots: int           # e.g. 5

    # LLM / API
    openrouter_api_key_env: str  # "OPENROUTER_API_KEY"
    openrouter_base_url: str     # "https://openrouter.ai/api/v1"
    models: List[str]            # ["openai/gpt-3.5-turbo", "openai/gpt-4o"]

    # output
    out_dir: str                 # e.g. "./outputs"
