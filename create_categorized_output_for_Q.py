import pandas as pd
import chardet
from itertools import islice
import general_utils
import dataframe_utils
from config import *

# Load open ends
print("\nLoading data...")
with open(open_end_data_file_path, "rb") as file:
    encoding = chardet.detect(file.read())["encoding"]  # Detect encoding
df = pd.read_csv(open_end_data_file_path, encoding=encoding)
print(f"\nRaw data:\n{df.head(20)}")

# Clean open ends
print("\nCleaning responses...")
response_columns = df.iloc[:, 1:].map(general_utils.preprocess_text)
print(f"\nResponses (first 10):\n{response_columns.head(10)}")

# Load categories
print("\nLoading categories...")
with open(categories_file_path, "rb") as file:
    encoding = chardet.detect(file.read())["encoding"]  # Detect encoding
categories = pd.read_csv(categories_file_path, encoding=encoding, header=None)
print(f"\nCategories:\n{categories}")

# Load codeframe (dictionary of response-category pairs)
print("\nLoading codeframe...")
if is_multicode:
    categorized_dict = general_utils.load_csv_to_dict_of_lists(codeframe_file_path)
else:
    categorized_dict = general_utils.load_csv_to_dict(codeframe_file_path)
print("\nCodeframe (first 10):\n")
print("\n".join(f"{key}: {value}" for key, value in islice(categorized_dict.items(), 10)))

# Create data structures
categories_list = categories.iloc[:, 0].tolist()
uuids = df.iloc[:, 0]
response_column_names = list(response_columns.columns)
categorized_data = pd.concat([uuids, response_columns], axis=1)
# repeat categories columns for each response column
categorized_data = dataframe_utils.construct_default_categorized_dataframe(
    categorized_data, response_column_names, categories_list
)
for response_column in response_column_names:
    categorized_data = dataframe_utils.categorize_missing_data_for_response_column(
        categorized_data, response_column, categories_list
    )

# Populate categorized dataframe
print("\nPreparing output data...")
for response_column in response_column_names:
    for response, categories in categorized_dict.items():
        if is_multicode and "Error" in categories:
            print(f"\nResponse '{response}' was not categorized.")
        elif categories == "Error":
            print(f"\nResponse '{response}' was not categorized.")

        dataframe_utils.categorize_responses_for_response_column(
            response, categories, response_column, categorized_data, is_multicode
        )

print(f"\nCategorized results:\n{categorized_data.head(10)}")

# Save to csv
print(f"\nSaving to {categorized_data_file_path} ...")
general_utils.export_dataframe_to_csv(categorized_data_file_path, categorized_data)

print("\nFinished")
