from openai import OpenAI
import pandas as pd
import json
import re
import chardet
import random
from typing import Any
from pandas._libs.missing import NAType

### NOTE: Make sure OpenAI_API_KEY is set up in your system environment variables ###
client = OpenAI()


def preprocess_text(text: Any) -> str | NAType:
    if pd.isna(text):
        return pd.NA

    text = str(text).lower()
    # Convert one or more of any kind of space to single space
    text = re.sub(r"\s+", " ", text)
    # Remove special characters
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = text.strip()
    return text


def get_random_sample_from_series(series: pd.Series, sample_size: int) -> pd.Series:
    if sample_size > len(series):
        raise ValueError("Sample size n cannot be greater than the length of the series")
    return series.sample(sample_size, random_state=random.randint(1, 10000))


def export_dataframe_to_csv(file_path: str, export_df: pd.DataFrame, header: bool = False) -> None:
    try:
        if export_df.empty:
            return
        export_df.to_csv(file_path, index=False, header=header)
    except Exception as e:
        print(f"Error while writing to CSV: {e}")


def generate_categories_GPT(
    client: OpenAI,
    question: str,
    responses_sample: list[str],
    number_of_categories: int = 20,
):
    user_prompt = f"""List the {number_of_categories} most relevant thematic categories for this sample of survey responses.
    Return only the category names, in the format: `["name1", "name2", ...]`\n\n
    Question:\n`{question}`\n\n
    Responses:\n```\n{responses_sample}\n```"""

    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": user_prompt}], model="gpt-4-1106-preview"
        )
        categories = json.loads(completion.choices[0].message.content)  # type: ignore

    except Exception as e:
        print(f"An error occurred: {e}")
        categories = "Error"
        raise

    return categories


file_name = "New Year Resolution - A2 open ends.csv"
print("Loading data...")
with open(file_name, "rb") as file:
    encoding = chardet.detect(file.read())["encoding"]  # Detect encoding
df = pd.read_csv(file_name, encoding=encoding)

print("Cleaning responses...")
processed_responses = df[df.columns[1:]].map(preprocess_text)
unique_responses = processed_responses.stack().drop_duplicates().dropna().reset_index(drop=True)
print("\nResponses:\n", unique_responses.head(10))

print("\nFetching sample...")
responses_sample = get_random_sample_from_series(unique_responses, 200).to_list()  # type: ignore

print("Generating categories with GPT-4...")
questionnaire_question = "What is your new year resolution?"
categories = generate_categories_GPT(
    client, questionnaire_question, responses_sample, number_of_categories=20
)
categories.extend(["Other", "Bad response", "Uncategorized", "Missing data"])
categories_df = pd.DataFrame(categories)
print(f"\nCategories:\n{categories}")

file_path = "categories.csv"
print(f"\nSaving to {file_path} ...")
export_dataframe_to_csv(file_path, categories_df)

print("\nDone")
