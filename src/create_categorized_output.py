"""
This script generates a dataframe of categorized open-ended survey responses based on a pre-defined codeframe.
It exports the DataFrame to a CSV file.

Steps:
1. Load the raw response data from a specified CSV file.
2. Clean the responses using preprocessing functions.
3. Load the categories and codeframe (response-category pairs) from specified CSV files.
4. Prepare a DataFrame to hold the categorized responses.
5. Populate the DataFrame by mapping responses to categories according to the codeframe.
6. Save the categorized data to a CSV file.

Input Files:
- Open-ended response data file (open_end_data_file_path): A CSV file containing the raw survey responses. Expects the first column to be uuids, and the following columns to be response columns.
- Categories file (categories_file_path): A CSV file containing the list of categories. Expects no header.
- Codeframe file (codeframe_file_path): A CSV file containing the predefined response-category pairs. Expects two columns, with headers 'key' and 'value'. Supports single or multiple categories per response based on the is_multicode flag.

Output File:
- Categorized data file (categorized_data_file_path): A CSV file where the processed and categorized data is saved.

The script utilizes utility functions from 'general_utils' and 'dataframe_utils' modules for processing and uses configurations defined in the 'config' module.

Note:
- User-defined variables should be properly set in the config.py file.
- The script supports both single-category and multi-category (is_multicode) response categorization.
- The script prints the progress at each major step and provides a summary of the categorized results.
"""

import pandas as pd
import chardet
from itertools import islice
from utils import general_utils
from utils import dataframe_utils
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
