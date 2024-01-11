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
    client: OpenAI,
    question: str,
    responses_batch: list[str],
    valid_categories: list[str],
    max_retries=3,
):
    combined_responses = "\n".join(
        [f"{i+1}: {response}" for i, response in enumerate(responses_batch)]
    )

    user_prompt = f"""Categorize these responses to the following survey question using one or multiple of the provided categories.
    Return only a list where each element is a list of category names for each response, in the format: `[["name 1", "name 2", ...], ["name 1", "name 2", ...], ...]`.\n\n
    Question:\n`{question}`\n\n
    Response:\n`{combined_responses}`\n\n
    Categories:\n```\n{valid_categories}\n```"""

    for attempt in range(max_retries):
        try:
            loop = asyncio.get_event_loop()
            completion = await loop.run_in_executor(
                None,  # Executor
                lambda: client.chat.completions.create(
                    messages=[{"role": "user", "content": user_prompt}], model="gpt-4-1106-preview"
                ),
            )

            output_categories_list = json.loads(completion.choices[0].message.content)  # type: ignore

            # Validating output
            if not isinstance(output_categories_list, list) or not all(
                isinstance(category, list) for category in output_categories_list
            ):
                raise ValueError("Output format is not as expected")

            for response_categories in output_categories_list:
                if any(category not in valid_categories for category in response_categories):
                    raise ValueError(
                        f"\nUnexpected category returned in categories: {output_categories_list}"
                    )

            return output_categories_list  # type: ignore

        except Exception as e:
            print(
                f"""\nAn error occurred:\n{e}.\n
                Responses in batch:\n{responses_batch}\n
                Retrying attempt {attempt + 1}/{max_retries}..."""
            )

    print(f"Max retries reached for responses:\n{responses_batch}")
    return [["Error"]] * len(responses_batch)


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
        output_categories_list = await task
        for response, response_categories in zip(batches[i], output_categories_list):
            categorized_dict[response] = response_categories

    return categorized_dict


async def GPT_categorize_responses_main(
    client, question, categories_list, unique_responses, batch_size, max_retries
):
    categorized_dict = await process_batches(
        client, question, categories_list, unique_responses, batch_size, max_retries
    )
    return categorized_dict


def export_dict_of_lists_to_csv(file_path: str, dict_to_export: dict, header: bool = True) -> None:
    try:
        if not dict_to_export:
            raise ValueError("Data is empty")

        formatted_data = [(key, ", ".join(value)) for key, value in dict_to_export.items()]
        df = pd.DataFrame(formatted_data, columns=["key", "value"])
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
# Uncategorized is a helper category for later, we don't want ChatGPT to use it.
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
export_dict_of_lists_to_csv(result_file_path, categorized_dict)

print("\nFinished")
