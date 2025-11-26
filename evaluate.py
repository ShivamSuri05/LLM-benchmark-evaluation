import pandas as pd
from llm_client import LLMClient
import re
import os

def extract_letter(response):
    """Extract A/B/C/D from model output."""
    match = re.search(r"\b([A-D])\b", response.upper())
    return match.group(1) if match else None

def create_prompt(question, A, B, C, D):
    """Neutral, domain-independent MCQ prompt."""
    return f"""
You will be given a multiple-choice question from any subject.
Choose the best answer based only on the information provided.

QUESTION:
{question}

OPTIONS:
A: {A}
B: {B}
C: {C}
D: {D}

Return ONLY the letter: A, B, C, or D.
"""

def evaluate(file_path,agg_correct,agg_wrong,agg_all):
    df = pd.read_csv(file_path, header=None)
    df.columns = ["question", "A", "B", "C", "D", "correct"]

    client = LLMClient(model="openai/gpt-4o")

    predictions = []

    for _, row in df.iterrows():
        prompt = create_prompt(row["question"], row["A"], row["B"], row["C"], row["D"])
        response = client.generate(prompt)
        prediction = extract_letter(response)
        predictions.append(prediction)

    df["prediction"] = predictions
    agg_all += len(predictions)
    agg_correct += (df["prediction"] == df["correct"]).sum()
    agg_wrong += (df["prediction"] != df["correct"]).sum()
    df["correct_bool"] = df["prediction"] == df["correct"].str.strip().str.upper()

    accuracy = df["correct_bool"].mean()
    print("\nModel Accuracy:", accuracy)
    return agg_correct,agg_wrong,agg_all


def main(dir_path):
    num = len(os.listdir(dir_path))
    count = 0
    agg_correct = 0
    agg_wrong = 0
    agg_all = 0
    for filename in os.listdir(dir_path):
        if filename.lower().endswith(".csv"):
            file_path = os.path.join(dir_path, filename)
            print("Reading file:",file_path)
            agg_correct,agg_wrong,agg_all = evaluate(file_path,agg_correct,agg_wrong,agg_all)
            count += 1
            print("Percentage of files evaluated:",(100*count)/num)
            #print(agg_correct,agg_wrong,agg_all)
            print("Overall accuracy till now:",(100*agg_correct)/agg_all)

if __name__ == "__main__":
    main(dir_path = "data/dev")
