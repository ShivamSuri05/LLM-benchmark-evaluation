SYSTEM_PROMPT = (
    "You are a renowned materials science engineering professor with extensive knowledge in the field. "
    "Your students have presented you with a challenging question related to materials science. "
    "Please reason step by step, and put the final answer inside a single box using \\boxed{...}. "
    "Include only the final answer inside the box, without the unit."
)


def build_cot_prompt(question: str) -> str:
    return SYSTEM_PROMPT + "\n\nQuestion:\n" + question



def build_judge_prompt(question, correct_answer, model_answer):
    system_prompt = (
        "As an expert judge, evaluate if the following model's answer matches the reference answer. "
        "Focus on the numerical values and key concepts. Small numerical differences are tolerable due to approximation errors. "
        "Don't solve the problem, just judge if the model answer matches the reference answer. "
        "Put the final decision ('correct' (if matching) or 'incorrect' (if not matching)) "
        "inside a single box using \\boxed{...}."
    )

    user_prompt = (
        f"The question is: {question}\n"
        f"Reference answer: {correct_answer}\n"
        f"Model answer: {model_answer}\n"
        "Is the model answer matching the reference answer?"
    )

    return system_prompt, user_prompt

