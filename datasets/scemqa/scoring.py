import re

def extract_mc_answer(resp_json):
    try:
        #print(resp_json)
        text = resp_json["choices"][0]["message"]["content"]
    except:
        return None, "Malformed response"

    text = text.strip()

    # Look for "Answer: X"
    m = re.search(r"Answer\s*[:\-]?\s*\(?([A-E])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1), ""

    # Look for "correct answer is X"
    m = re.search(r"correct answer (is|:)\s*\(?([A-E])\)?", text, re.IGNORECASE)
    if m:
        return m.group(2), ""

    # Look for bold style **D.**
    m = re.search(r"\*\*([A-E])\.", text)
    if m:
        return m.group(1), ""

    # Look for standalone (D)
    m = re.findall(r"\(([A-E])\)", text)
    if m:
        return m[-1], "Paren match"

    # Last standalone letter
    m = re.findall(r"\b([A-E])\b", text)
    if m:
        return m[-1], "Fallback last letter"

    return None, f"Could not parse answer from: {text}"
