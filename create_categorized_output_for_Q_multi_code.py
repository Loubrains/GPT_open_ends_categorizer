# TODO: need to strip csv's after loading before sending to gpt

import pandas as pd
import chardet
import re
from itertools import islice
import sys
from typing import Any
from pandas._libs.missing import NAType


def load_csv_to_dict_of_lists(file_path: str) -> dict:
    try:
        df = pd.read_csv(file_path)
        df["value"] = df["value"].apply(lambda x: x.split(", ") if isinstance(x, str) else [])
        return dict(zip(df["key"], df["value"]))

    except Exception as e:
        print(f"\nError while reading CSV: {e}")
        sys.exit(1)


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


def construct_default_categorized_dataframe(
    categorized_data: pd.DataFrame, response_column_names: list[str], categories_list: list[str]
):
    for response_column in response_column_names:
        for category in categories_list:
            col_name = f"{category}_{response_column}"
            if category == "Uncategorized":
                categorized_data[col_name] = 1
            else:
                categorized_data[col_name] = 0
    return categorized_data


def categorize_missing_data_for_response_column(
    categorized_data: pd.DataFrame,
    response_column: str,
    categories_list: list[str],
) -> pd.DataFrame:
    def _is_missing(value):
        return pd.isna(value)

    # Boolean mask where each row is True if all elements are missing
    missing_data_mask = categorized_data[response_column].map(_is_missing)

    for category in categories_list:
        col_name = f"{category}_{response_column}"
        categorized_data.loc[missing_data_mask, col_name] = pd.NA

    return categorized_data


def categorize_responses_for_response_column(
    response: str,
    categories: list[str],
    response_column: str,
    categorized_data: pd.DataFrame,
):
    # Boolean mask for rows in response_column containing selected response
    mask = categorized_data[response_column] == response

    for category in categories:
        col_name = f"{category}_{response_column}"

        if col_name in categorized_data.columns:
            categorized_data.loc[mask, f"Uncategorized_{response_column}"] = 0
            categorized_data.loc[mask, col_name] = 1
        else:
            print(f"\nUnknown category: {category} for response: {response}")


def export_dataframe_to_csv(file_path: str, export_df: pd.DataFrame, header: bool = True) -> None:
    try:
        if export_df.empty:
            raise pd.errors.EmptyDataError

        export_df.to_csv(file_path, index=False, header=header)

    except Exception as e:
        print(f"\nError while writing to CSV: {e}")


# Load open ends
data_file_path = "New Year Resolution - A2 open ends.csv"
print("Loading data...")
with open(data_file_path, "rb") as file:
    encoding = chardet.detect(file.read())["encoding"]  # Detect encoding
df = pd.read_csv(data_file_path, encoding=encoding)
print(f"Raw data:\n{df.head(20)}")

# Clean open ends
print("Cleaning responses...")
response_columns = df.iloc[:, 1:].map(preprocess_text)  # type: ignore
print(f"\nResponses (first 10):\n{response_columns.head(10)}")

# Load categories
categories_file_path = "categories.csv"
print("Loading categories...")
with open(categories_file_path, "rb") as file:
    encoding = chardet.detect(file.read())["encoding"]  # Detect encoding
categories = pd.read_csv(categories_file_path, encoding=encoding, header=None)
print(f"\nCategories:\n{categories}")

# Load codeframe (dictionary of response-category pairs)
print("Loading codeframe...")
categorized_dict = load_csv_to_dict_of_lists("codeframe.csv")
print("Codeframe (first 10):\n")
print("\n".join(f"{key}: {value}" for key, value in islice(categorized_dict.items(), 10)))

# Create data structures
categories_list = categories.iloc[:, 0].tolist()
uuids = df.iloc[:, 0]
response_column_names = list(response_columns.columns)
categorized_data = pd.concat([uuids, response_columns], axis=1)
# repeat categories columns for each response column
categorized_data = construct_default_categorized_dataframe(
    categorized_data, response_column_names, categories_list
)
for response_column in response_column_names:
    categorized_data = categorize_missing_data_for_response_column(
        categorized_data, response_column, categories_list
    )


# Populate categorized dataframe
print("Preparing output data...")
for response_column in response_column_names:
    for response, categories in categorized_dict.items():
        if "Error" in categories:
            print(f"\nResponse '{response}' was not categorized.")

        categorize_responses_for_response_column(
            response, categories, response_column, categorized_data
        )

print(f"\nCategorized results:\n{categorized_data.head(10)}")

# Save to csv
result_file_path = "categorized_data.csv"
print(f"\nSaving to {result_file_path} ...")
export_dataframe_to_csv(result_file_path, categorized_data)

print("\nFinished")
