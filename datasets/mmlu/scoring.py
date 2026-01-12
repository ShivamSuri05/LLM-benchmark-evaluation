from typing import Any, Dict, List, Optional, Tuple

CHOICES = ("A", "B", "C", "D")

def extract_top_logprobs(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    choices = resp.get("choices", [])
    lp = choices[0].get("logprobs", {}) if choices else {}
    content = lp.get("content", [])
    top = content[0].get("top_logprobs", []) if content else []
    if not top:
        raise ValueError("No top_logprobs found in response.")
    return top

def choose_from_top_logprobs_strict(top_logprobs: List[Dict[str, Any]]) -> Tuple[Optional[str], Dict[str, float]]:
    scores = {c: float("-inf") for c in CHOICES}
    for entry in top_logprobs:
        tok = entry.get("token")
        lp = entry.get("logprob")
        if tok in scores and lp is not None:
            scores[tok] = float(lp)

    if all(v == float("-inf") for v in scores.values()):
        return None, scores

    pred = max(scores.items(), key=lambda kv: kv[1])[0]
    return pred, scores

def extract_usage(resp: Dict[str, Any]) -> Dict[str, Any]:
    return resp.get("usage", {}) or {}
