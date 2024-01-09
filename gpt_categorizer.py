from openai import OpenAI
import asyncio
import pandas as pd
import json
import re
import chardet

### NOTE: Make sure OpenAI_API_KEY is set up in your system environment variables ###
client = OpenAI()


def preprocess_text(text) -> str:
    text = str(text).lower()
    # Convert one or more of any kind of space to single space
    text = re.sub(r"\s+", " ", text)
    # Remove special characters
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = text.strip()
    return text


def create_batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


async def categorize_response_batch_GPT(
    client: OpenAI,
    question: str,
    responses_batch: list[str],
    categories: list[str],
):
    combined_responses = "\n".join(
        [f"{i+1}: {response}" for i, response in enumerate(responses_batch)]
    )

    user_prompt = f"""Categorize these responses to the following survey question using one of the provided categories.
    Return only the category names, in the format: `["name 1", "name 2", ...]`.\n\n
    Question:\n`{question}`\n\n
    Response:\n`{combined_responses}`\n\n
    Categories:\n```\n{categories}\n```"""

    try:
        loop = asyncio.get_event_loop()
        completion = await loop.run_in_executor(
            None,  # Executor
            lambda: client.chat.completions.create(
                messages=[{"role": "user", "content": user_prompt}], model="gpt-4-1106-preview"
            ),
        )
        output_categories = completion.choices[0].message.content
        return json.loads(output_categories)  # type: ignore

    except Exception as e:
        print(f"An error occurred: {e}")
        output_categories = ["Error"] * len(responses_batch)
    return output_categories


async def process_batches(client, question, categories_list, responses, batch_size=3):
    categorized_responses = {}
    batches = list(create_batches(list(responses), batch_size))
    tasks = []

    for batch in batches:
        task = asyncio.create_task(
            categorize_response_batch_GPT(client, question, batch, categories_list)
        )
        tasks.append(task)

    for i, task in enumerate(tasks):
        output_categories = await task
        for response, category in zip(batches[i], output_categories):
            categorized_responses[response] = category

    return categorized_responses


async def categorize_responses_gpt_main(
    client, question, categories_list, unique_responses, batch_size
):
    categorized_responses = await process_batches(
        client, question, categories_list, unique_responses, batch_size
    )
    return categorized_responses


def categorize_response_in_dataframe(
    response: str,
    category: str,
    categorized_data: pd.DataFrame,
    response_columns: list[str],
):
    # Boolean mask for rows in categorized_data containing selected responses
    mask = pd.Series([False] * len(categorized_data))

    for column in categorized_data[response_columns]:
        mask |= categorized_data[column] == response

    categorized_data.loc[mask, "Uncategorized"] = 0
    categorized_data.loc[mask, category] = 1


def categorize_missing_data(categorized_data: pd.DataFrame) -> pd.DataFrame:
    def is_missing(value):
        return pd.isna(value) or value is None or value == "missing data" or value == "nan"

    # Boolean mask where each row is True if all elements are missing
    all_missing_mask = df_preprocessed.map(is_missing).all(axis=1)  # type: ignore
    categorized_data.loc[all_missing_mask, "Missing data"] = 1
    categorized_data.loc[all_missing_mask, "Uncategorized"] = 0
    return categorized_data


def export_dataframe_to_csv(file_path: str, export_df: pd.DataFrame, header: bool = False) -> None:
    try:
        if export_df.empty:
            raise pd.errors.EmptyDataError

        export_df.to_csv(file_path, index=False, header=header)

    except Exception as e:
        print(f"Error while writing to CSV: {e}")


# Load data
data_file_path = "New Year Resolution - A2 open ends.csv"
print("Loading data...")
with open(data_file_path, "rb") as file:
    encoding = chardet.detect(file.read())["encoding"]  # Detect encoding
df = pd.read_csv(data_file_path, encoding=encoding)

print("Cleaning responses...")
df_preprocessed = df.iloc[:, 1:].map(preprocess_text)  # type: ignore
print("\nResponses:\n", df_preprocessed.head(10))

# Load categories
categories_file_path = "categories_output.csv"
print("Loading categories...")
with open(categories_file_path, "rb") as file:
    encoding = chardet.detect(file.read())["encoding"]  # Detect encoding
categories = pd.read_csv(categories_file_path, encoding=encoding)
print("Categories:\n", categories)

# Create data structures
categories_list = categories.iloc[:, 0].tolist()

unique_responses = set(df_preprocessed.stack().dropna().reset_index(drop=True)) - {""}
uuids = df.iloc[:, 0]
response_columns = list(df_preprocessed.columns)
categorized_data = pd.concat([uuids, df_preprocessed], axis=1)
categorized_data["Uncategorized"] = 1  # Everything starts uncategorized
for category in categories_list:
    categorized_data[category] = 0
categorized_data["Missing data"] = 0
categorize_missing_data(categorized_data)


# Get GPT responses
question = "Why were you or your child consuming media at this time?"
print("Categorizing data with GPT-4...")
# unique_responses_sample = list(unique_responses)[:20]
categorized_responses = asyncio.run(
    categorize_responses_gpt_main(client, question, categories_list, unique_responses, batch_size=3)
)
print("Finished categorizing with GPT-4...")

print("Preparing data...")
for response, category in categorized_responses.items():
    if category != "Error":
        categorize_response_in_dataframe(response, category, categorized_data, response_columns)
    else:
        print(f"Response '{response}' was not categorized.")
categorized_data = categorize_missing_data(categorized_data)
print("Categorized results:\n", categorized_data.head(10))

# Save to csv
result_file_path = "categorized_data.csv"
print(f"\nSaving to {result_file_path} ...")
export_dataframe_to_csv(result_file_path, categorized_data, header=True)

print("\nDone")
