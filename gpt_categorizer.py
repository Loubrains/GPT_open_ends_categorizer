from openai import OpenAI
import pandas as pd
import json
import re
import chardet
import random

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


def categorize_missing_data(categorized_data: pd.DataFrame) -> pd.DataFrame:
    def is_missing(value):
        return pd.isna(value) or value is None or value == "missing data" or value == "nan"

    # Boolean mask where each row is True if all elements are missing
    all_missing_mask = df_preprocessed.map(is_missing).all(axis=1)  # type: ignore
    categorized_data.loc[all_missing_mask, "Missing data"] = 1
    categorized_data.loc[all_missing_mask, "Uncategorized"] = 0
    return categorized_data


def categorize_response_in_dataframe(
    response: str,
    category: str,
    categorized_data: pd.DataFrame,
    response_columns: list[str],
) -> pd.DataFrame:
    # Boolean mask for rows in categorized_data containing selected responses
    mask = pd.Series([False] * len(categorized_data))

    for column in categorized_data[response_columns]:
        mask |= categorized_data[column] == response

    categorized_data.loc[mask, "Uncategorized"] = 0
    categorized_data.loc[mask, category] = 1

    return categorized_data


def export_dataframe_to_csv(file_path: str, export_df: pd.DataFrame) -> None:
    try:
        if export_df.empty:
            return
        export_df.to_csv(file_path, index=False, header=False)
    except Exception as e:
        print(f"Error while writing to CSV: {e}")


def categorize_response_GPT(
    client: OpenAI,
    question: str,
    response: str,
    categories: list[str],
):
    system = [
        {
            "role": "system",
            "content": """You are a data analyst specialized in categorizing open-ended responses from questionnaires. 
            Your task is to assign the most appropriate category to each response based on the provided question and response content.""",
        }
    ]

    user = [
        {
            "role": "user",
            "content": f"""Please categorize the following questionnaire response. 
            Use only one of the provided categories for your classification.\n\n
            Question:\n{question}\n\n
            Response:\n{response}\n\n
            Categories:\n{categories}\n\n
            Return only the category that best fits the response.""",
        }
    ]
    try:
        completion = client.chat.completions.create(messages=system + user, model="gpt-4")
        category = json.loads(completion.choices[0].message.content)

    except Exception as e:
        print(f"An error occurred: {e}")
        category = "Error"

    return category


# Load data
data_file_path = "BBC Need States - B3_OPEN open ends.csv"
print("Loading data...")
with open(data_file_path, "rb") as file:
    encoding = chardet.detect(file.read())["encoding"]  # Detect encoding
df = pd.read_csv(data_file_path, encoding=encoding)

print("Cleaning responses...")
processed_df = df[df.columns[1:]].map(preprocess_text)
print("\nResponses:\n", processed_df.head(10))

# Load categories
categories_file_path = ""
print("Loading categories...")
with open(categories_file_path, "rb") as file:
    encoding = chardet.detect(file.read())["encoding"]  # Detect encoding
categories = pd.read_csv(categories_file_path, encoding=encoding)

# Create data structures
categories_list = categories.iloc[:, 0].tolist()
df_preprocessed = pd.DataFrame(
    df.iloc[:, 1:].map(
        preprocess_text  # , na_action="ignore"
    )  # type: ignore
)
unique_responses = set(df_preprocessed.stack().reset_index(drop=True).dropna())
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
for response in unique_responses:
    # Call your categorize_response_GPT function
    gpt_response = categorize_response_GPT(client, question, str(response), categories_list)

    if gpt_response != "Error":
        categorized_data = categorize_response_in_dataframe(
            str(response), gpt_response, categorized_data, response_columns
        )
    else:
        print(f"Response '{response}' was not categorized.")

print("Categorized results:\n", categorized_data.head(20))

# Save to csv
result_file_path = "categories_output.csv"
print(f"\nSaving to {result_file_path} ...")
export_dataframe_to_csv(result_file_path, categorized_data)

print("\nDone")
