import pandas as pd
import chardet
import re
from itertools import islice
import sys


def load_csv_to_dict(file_path: str) -> dict:
    try:
        df = pd.read_csv(file_path)
        return dict(zip(df["key"], df["value"]))

    except Exception as e:
        print(f"\nError while reading CSV: {e}")
        sys.exit(1)


def preprocess_text(text) -> str:
    text = str(text).lower()
    # Convert one or more of any kind of space to single space
    text = re.sub(r"\s+", " ", text)
    # Remove special characters
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = text.strip()
    return text


def categorize_responses_in_entire_dataframe(
    response: str,
    category: str,
    categorized_data: pd.DataFrame,
    response_columns: list[str],
):
    # Boolean mask for rows in categorized_data containing selected responses
    mask = pd.Series([False] * len(categorized_data))

    for column in categorized_data[response_columns]:
        mask |= categorized_data[column] == response

    if category in categorized_data.columns:
        categorized_data.loc[mask, "Uncategorized"] = 0
        categorized_data.loc[mask, category] = 1
    else:
        print(f"\nUnknown category: {category} for response: {response}")


def categorize_missing_data_in_entire_dataframe(categorized_data: pd.DataFrame) -> pd.DataFrame:
    def _is_missing(value):
        return (
            value == "missing data" or value == "nan"
        )  # after processing the data all nan values become lowercase text

    # Boolean mask where each row is True if all elements are missing
    all_missing_mask = df_preprocessed.map(_is_missing).all(axis=1)  # type: ignore
    categorized_data.loc[all_missing_mask, "Missing data"] = 1
    categorized_data.loc[all_missing_mask, "Uncategorized"] = 0
    return categorized_data


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

# Clean open ends
print("Cleaning responses...")
df_preprocessed = df.iloc[:, 1:].map(preprocess_text)  # type: ignore
print(f"\nResponses (first 10):\n{df_preprocessed.head(10)}")

# Load categories
categories_file_path = "categories.csv"
print("Loading categories...")
with open(categories_file_path, "rb") as file:
    encoding = chardet.detect(file.read())["encoding"]  # Detect encoding
categories = pd.read_csv(categories_file_path, encoding=encoding)
print(f"\nCategories:\n{categories}")

# Load codeframe (dictionary of response-category pairs)
print("Loading codeframe...")
categorized_dict = load_csv_to_dict("codeframe.csv")
print("Codeframe (first 10):\n")
print("\n".join(f"{key}: {value}" for key, value in islice(categorized_dict.items(), 10)))

# Create data structures
categories_list = categories.iloc[:, 0].tolist()
uuids = df.iloc[:, 0]
response_columns = list(df_preprocessed.columns)
categorized_data = pd.concat([uuids, df_preprocessed], axis=1)
for category in categories_list:
    categorized_data[category] = 0
categorized_data["Uncategorized"] = 1  # Everything starts uncategorized
# putting this here as insurance in case "Missing data" is not in the list of categories)
categorized_data["Missing data"] = 0  # all start as 0 before calculating missing data rows.
categorize_missing_data_in_entire_dataframe(categorized_data)


# Populate categorized dataframe
print("Preparing output data...")
for response, category in categorized_dict.items():
    if category != "Error":
        categorize_responses_in_entire_dataframe(
            response, category, categorized_data, response_columns
        )
    else:
        print(f"\nResponse '{response}' was not categorized.")

categorized_data = categorize_missing_data_in_entire_dataframe(categorized_data)
print(f"\nCategorized results:\n{categorized_data.head(10)}")

# Save to csv
result_file_path = "categorized_data.csv"
print(f"\nSaving to {result_file_path} ...")
export_dataframe_to_csv(result_file_path, categorized_data)

print("\nFinished")
