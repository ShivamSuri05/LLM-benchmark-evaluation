def build_zero_shot_prompt(question_text: str) -> str:
    return (
        question_text.strip()
        #+ "\n\nAnswer with only the capital letter (A, B, C, D, or E)."
    )
