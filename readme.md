# LLM Benchmark Evaluation

Reproduction study and meta-analysis of LLM benchmarks, with a focus on engineering and materials science domains.

This repository accompanies the research project:
**"Meta-Analysis and Reproduction Study of LLM Benchmarks with Focus on Engineering Domains"**
Hamburg University of Technology (TUHH), Institute of Continuum and Material Mechanics, 2026.

---

## Overview

This project reproduces five LLM benchmarks — MMLU, ScienceQA, SceMQA, MaterialBENCH, and MatSciBench — and systematically analyses how evaluation design choices (prompt wording, pipeline architecture, dataset accessibility) affect reported scores.

---

## Benchmark Survey

`Benchmarking Survey.xlsx` contains a systematic catalogue of **60 LLM benchmarks** across 11 categories.

Each entry records 26 attributes:

| Attribute | Description |
|---|---|
| Name | Benchmark name |
| Category | Task category (e.g. Mathematical Reasoning, Multimodal) |
| Year | Publication year |
| Domain | General / domain-specific |
| Task type | MCQ, free-response, code generation, etc. |
| Input format | Text-only / multimodal |
| Evaluation method | Exact match, similarity, LLM-as-judge, etc. |
| Automation level | Fully automated / semi-automated |
| Open source | Dataset publicly available |
| Citation count | Approximate citations (Google Scholar) |
| Reproducibility rating | High / Medium / Low |
| ...and more | |

**Categories covered:** Commonsense & Knowledge QA (9), Code Generation (7), Multimodal (7), Mathematical Reasoning (7), General Reasoning (6), Instruction Following & LLM-as-Judge (5), Engineering & Materials Science (5), Long-Context (4), Domain-Specific (4), Safety & Robustness (3), Multilingual (3).

---

## Five Benchmarks Studied

| Benchmark | Domain | Reproducibility | Notes |
|---|---|---|---|
| MMLU | General reasoning | High | 57 subjects, text-only |
| ScienceQA | K–12 science | Medium | Multimodal, K–12 level |
| SceMQA | College-entrance science | Medium | 4 subjects, multimodal |
| MaterialBENCH | Materials science | Low | Dataset inaccessible |
| MatSciBench | Materials science | Low | 1,340 problems, agentic pipeline |

---

## Setup

```bash
git clone https://github.com/ShivamSuri05/LLM-benchmark-evaluation
cd LLM-benchmark-evaluation
pip install -r requirements.txt
```

All experiments use the [OpenRouter](https://openrouter.ai) API for model access. Set your API key in .env file:

```bash
OPENROUTER_API_KEY=your_key_here
```

In order to reproduce the benchmark evaluation experiments, run the scripts under the `scripts/` directory. You can change the config of the scripts by changing values in the particular script file.

```bash
python scripts/run_scemqa_mcq.py
```

---

## Models Used

| Model | Used for |
|---|---|
| GPT-3.5-turbo | MMLU experiments |
| GPT-4o | ScienceQA, SceMQA experiments |
| GPT-4o-mini | MatSciBench baseline |
| o4-mini | MatSciBench primary reproduction |
| Gemini-2.0-Flash | MatSciBench judge model |
