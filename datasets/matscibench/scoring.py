import re
import regex
from math import isclose
from typing import Union
from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from latex2sympy2 import latex2sympy
from datasets.matscibench.prompts import build_judge_prompt


# ===============================
# Numeric Utilities
# ===============================

def parse_digits(num):
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except:
        if num.endswith("%"):
            num = num[:-1]
            try:
                return float(num) / 100
            except:
                return None
    return None


def is_digit(num):
    return parse_digits(num) is not None


def numeric_equal(pred: float, ref: float):
    return isclose(ref, pred, rel_tol=5e-2)  # 5% tolerance


# ===============================
# Symbolic Equality
# ===============================

def _parse_expression(s: str):
    for parser in [parse_latex, parse_expr, latex2sympy]:
        try:
            return parser(s.replace("\\\\", "\\"))
        except:
            continue
    return s


def symbolic_equal(a: str, b: str):
    a_parsed = _parse_expression(a)
    b_parsed = _parse_expression(b)

    try:
        if a_parsed == b_parsed:
            return True
    except:
        pass

    try:
        if simplify(a_parsed - b_parsed) == 0:
            return True
    except:
        pass

    try:
        if numeric_equal(float(N(a_parsed)), float(N(b_parsed))):
            return True
    except:
        pass

    return False


# ===============================
# Main math_equal logic
# ===============================

def math_equal(
    prediction: Union[str, float],
    reference: Union[str, float],
):
    if prediction is None or reference is None:
        return False

    pred = str(prediction).strip()
    ref = str(reference).strip()

    # Direct string match
    if pred.lower() == ref.lower():
        return True

    # -------- Numeric comparison --------
    if is_digit(pred) and is_digit(ref):
        p_val = parse_digits(pred)
        r_val = parse_digits(ref)
        if p_val is None or r_val is None:
            return False
        return numeric_equal(p_val, r_val)

    # -------- Tuple / vector handling --------
    if (
        (pred.startswith("(") and pred.endswith(")")) and
        (ref.startswith("(") and ref.endswith(")"))
    ):
        pred_parts = [p.strip() for p in pred[1:-1].split(",")]
        ref_parts = [r.strip() for r in ref[1:-1].split(",")]

        if len(pred_parts) != len(ref_parts):
            return False

        return all(math_equal(p, r) for p, r in zip(pred_parts, ref_parts))

    # -------- Symbolic comparison --------
    return symbolic_equal(pred, ref)


# ===============================
# Public API
# ===============================

def is_correct(model_answer: str, correct_answer: str) -> bool:
    return math_equal(model_answer, correct_answer)


def llm_judge(judge_client, judge_model, question, correct_answer, model_answer):
    system_prompt, user_prompt = build_judge_prompt(
        question, correct_answer, model_answer
    )
    #print(system_prompt + "\n" + user_prompt,)

    resp = judge_client.chat_completion(
        model=judge_model,
        prompt=system_prompt + "\n" + user_prompt,
        temperature=0.0,
        max_tokens=10,
    )

    content = resp["choices"][0]["message"]["content"].strip().lower()
    #print(resp)

    if "incorrect" in content:
        return False, content
    elif "correct" in content:
        return True, content
    else:
        return False, content



import re


# ===============================
# Utilities
# ===============================

def _strip_latex_units(ans: str) -> str:
    """
    Remove LaTeX unit expressions like:
    \, \text{nm}
    \text{Å}
    """
    ans = re.sub(r"\\,?\s*\\text\{[^}]+\}", "", ans)
    ans = ans.replace("\\,", "")
    return ans.strip()


def _clean_percent(ans: str) -> str:
    """
    Convert 4.25% → 4.25
    """
    if ans.endswith("%"):
        return ans[:-1]
    return ans


# ===============================
# Boxed extraction (priority)
# ===============================

def _extract_boxed(response: str) -> str:
    match_index = response.rfind("boxed{")
    if match_index == -1:
        return ""

    start = match_index + len("boxed{")
    brace_count = 1
    i = start

    while brace_count > 0 and i < len(response):
        if response[i] == "{":
            brace_count += 1
        elif response[i] == "}":
            brace_count -= 1
        i += 1

    if brace_count != 0:
        return ""

    return response[start:i-1].strip()


# ===============================
# Safe numeric fallback
# ===============================

def _extract_last_numeric_from_tail(response: str) -> str:
    """
    Extract numeric value from last 500 characters only.
    Prevents mid-derivation grabbing.
    """
    tail = response[-150:]

    numbers = re.findall(
        r"[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?%?",
        tail.lower()
    )

    if not numbers:
        return ""

    return numbers[-1]


# ===============================
# Safe tuple extraction (tail only)
# ===============================

def _extract_last_tuple_from_tail(response: str) -> str:
    tail = response[-150:]

    tuples = re.findall(r"\(([^()]+)\)", tail)
    if not tuples:
        return ""

    # Only accept numeric tuples
    for t in reversed(tuples):
        if re.fullmatch(
            r"\s*[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?\s*(,\s*[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?\s*)+\s*",
            t
        ):
            return f"({t.strip()})"

    return ""


# ===============================
# Public API
# ===============================

def extract_final_answer(response: str) -> str:
    if not response:
        return ""

    # 1️⃣ Boxed first
    boxed = _extract_boxed(response)
    if boxed:
        boxed = _strip_latex_units(boxed)
        boxed = _clean_percent(boxed)
        return boxed

    # 2️⃣ Numeric fallback FIRST
    num = _extract_last_numeric_from_tail(response)
    if num:
        num = _strip_latex_units(num)
        num = _clean_percent(num)
        return num

    # 3️⃣ Tuple fallback (strict numeric tuples only)
    tuple_ans = _extract_last_tuple_from_tail(response)
    if tuple_ans:
        tuple_ans = _strip_latex_units(tuple_ans)
        tuple_ans = _clean_percent(tuple_ans)
        return tuple_ans

    return ""
