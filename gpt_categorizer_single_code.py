# TODO: need to strip csv's after loading before sending to gpt

from openai import OpenAI
import asyncio
import pandas as pd
import json
import re
import chardet
from itertools import islice
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


def create_batches(data, batch_size=3):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


async def GPT_categorize_response_batch(
    client: OpenAI, question: str, responses_batch: list[str], categories: list[str], max_retries=3
):
    combined_responses = "\n".join(
        [f"{i+1}: {response}" for i, response in enumerate(responses_batch)]
    )

    user_prompt = f"""Categorize these responses to the following survey question using one of the provided categories.
    Return only the category names, in the format: `["name 1", "name 2", ...]`.\n\n
    Question:\n`{question}`\n\n
    Response:\n`{combined_responses}`\n\n
    Categories:\n```\n{categories}\n```"""

    for attempt in range(max_retries):
        try:
            loop = asyncio.get_event_loop()
            completion = await loop.run_in_executor(
                None,  # Executor
                lambda: client.chat.completions.create(
                    messages=[{"role": "user", "content": user_prompt}], model="gpt-4-1106-preview"
                ),
            )

            output_categories = json.loads(completion.choices[0].message.content)  # type: ignore

            if any(category not in categories for category in output_categories):
                print(
                    f"\nUnexpected category returned in categories: {output_categories}\nRetrying attempt {attempt + 1}/{max_retries}..."
                )
                continue

            return output_categories  # type: ignore

        except Exception as e:
            print(
                f"\nAn error occurred: {e}.\nResponses in batch: {responses_batch}\nRetrying attempt {attempt + 1}/{max_retries}..."
            )

    print(f"Max retries reached for responses: {responses_batch}")
    return ["Error"] * len(responses_batch)


async def process_batches(client, question, categories_list, responses, batch_size, max_retries):
    categorized_dict = {}
    batches = list(create_batches(list(responses), batch_size))
    tasks = []

    for batch in batches:
        task = asyncio.create_task(
            GPT_categorize_response_batch(client, question, batch, categories_list, max_retries)
        )
        tasks.append(task)

    for i, task in enumerate(tasks):
        output_categories = await task
        for response, category in zip(batches[i], output_categories):
            categorized_dict[response] = category

    return categorized_dict


async def GPT_categorize_responses_main(
    client, question, categories_list, unique_responses, batch_size, max_retries
):
    categorized_dict = await process_batches(
        client, question, categories_list, unique_responses, batch_size, max_retries
    )
    return categorized_dict


def export_dict_to_csv(file_path: str, export_dict: dict, header: bool = True) -> None:
    try:
        if not export_dict:
            raise ValueError("Data is empty")

        df = pd.DataFrame(list(export_dict.items()), columns=["key", "value"])
        df.to_csv(file_path, index=False, header=header)

    except Exception as e:
        print(f"\nError while writing to CSV: {e}")


# Load open ends
data_file_path = "New Year Resolution - A2 open ends.csv"
print("Loading data...")
with open(data_file_path, "rb") as file:
    encoding = chardet.detect(file.read())["encoding"]  # Detect encoding
df = pd.read_csv(data_file_path, encoding=encoding)

# Clean open ends
print("Cleaning responses...")
# Assume first column UUIDs, remaining columns are responses
df_preprocessed = df.iloc[:, 1:].map(preprocess_text)  # type: ignore
print(f"\nResponses (first 10):\n{df_preprocessed.head(10)}")

unique_responses = set(df_preprocessed.stack().dropna().reset_index(drop=True))
# we don't want to match empty strings against every row
unique_responses = unique_responses - {""}

# Load categories
categories_file_path = "categories.csv"
print("Loading categories...")
with open(categories_file_path, "rb") as file:
    encoding = chardet.detect(file.read())["encoding"]  # Detect encoding
categories = pd.read_csv(categories_file_path, encoding=encoding, header=None)
print(f"\nCategories:\n{categories}")

categories_list = categories.iloc[:, 0].tolist()
# Missing data and uncategorized are helper categories for later, we don't want ChatGPT to use them.
categories_list.remove("Missing data")
categories_list.remove("Uncategorized")

# Categorize responses using GPT API
question = "What is your new year resolution?"
print("Categorizing data with GPT-4...")
# unique_responses_sample = list(unique_responses)[:20]
categorized_dict = asyncio.run(
    GPT_categorize_responses_main(
        client, question, categories_list, unique_responses, batch_size=3, max_retries=5
    )
)
categorized_dict.pop("", None)  # removing empty string since it matches against every row
print("Codeframe (first 10):\n")
print("\n".join(f"{key}: {value}" for key, value in islice(categorized_dict.items(), 10)))
print("Finished categorizing with GPT-4...")

# Saving codeframe (dictionary of response-category pairs)
result_file_path = "codeframe.csv"
print(f"\nSaving codeframe to {result_file_path} ...")
export_dict_to_csv(result_file_path, categorized_dict)

print("\nFinished")
