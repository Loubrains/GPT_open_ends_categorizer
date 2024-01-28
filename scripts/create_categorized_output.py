"""
Script that generates a dataframe of categorized open-ended survey responses based on a pre-defined codeframe.
Exports the DataFrame to a CSV file.

Steps:
1. Load the raw response data from a specified CSV file.
2. Clean the responses using preprocessing functions.
3. Load the categories and codeframe (response-category pairs) from specified CSV files.
4. Prepare a DataFrame to hold the categorized responses.
5. Populate the DataFrame by mapping responses to categories according to the codeframe.
6. Save the categorized data to a CSV file.

Input Files:
- Open-ended response data file (`open_end_data_file_path`): A CSV file containing the raw survey responses. Expects the first column to be uuids, and the following columns to be response columns.
- Categories file (`categories_file_path`): A CSV file containing the list of categories. Expects no header.
- Codeframe file (`codeframe_file_path`): A CSV file containing the predefined response-category pairs. Expects two columns, with headers `key` and `value`. Supports single or multiple categories per response based on the `is_multicode` flag.

Output File:
- Categorized data file (`categorized_data_file_path`): A CSV file where the processed and categorized data is saved.

Notes:
- The script utilizes utility functions from `general_utils` and `dataframe_utils` modules.
- User-defined variables such as file paths should be properly set in the `config.py` file before running this script.
- The script supports both single-category and multi-category (`is_multicode`) response categorization.
- The script terminates if exceptions are raised at any point.
"""

import pandas as pd
import chardet
from itertools import islice
import sys
from gpt_categorizer_utils import general_utils, dataframe_utils
import config as cfg

if __name__ == "__main__":
    try:
        # Load open ends
        print("\nLoading data...")
        with open(cfg.open_end_data_file_path_load, "rb") as file:
            encoding = chardet.detect(file.read())["encoding"]  # Detect encoding
        df = pd.read_csv(cfg.open_end_data_file_path_load, encoding=encoding)
        print(f"\nRaw data:\n{df.head(20)}")

        # Clean open ends
        print("\nCleaning responses...")
        response_columns = df.iloc[:, 1:].map(general_utils.preprocess_text)
        print(f"\nResponses (first 10):\n{response_columns.head(10)}")

        # Load categories
        print("\nLoading categories...")
        with open(cfg.categories_file_path_load, "rb") as file:
            encoding = chardet.detect(file.read())["encoding"]  # Detect encoding
        categories = pd.read_csv(cfg.categories_file_path_load, encoding=encoding, header=None)
        print(f"\nCategories:\n{categories}")

        # Load codeframe (dictionary of response-category pairs)
        print("\nLoading codeframe...")
        if cfg.is_multicode:
            categorized_dict = general_utils.load_csv_to_dict_of_lists(cfg.codeframe_file_path_load)
        else:
            categorized_dict = general_utils.load_csv_to_dict(cfg.codeframe_file_path_load)
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
                if cfg.is_multicode and "Error" in categories:
                    print(f"\nResponse '{response}' was not categorized.")
                elif categories == "Error":
                    print(f"\nResponse '{response}' was not categorized.")

                dataframe_utils.categorize_responses_for_response_column(
                    response, categories, response_column, categorized_data, cfg.is_multicode
                )

        print(f"\nCategorized results:\n{categorized_data.head(10)}")

        # Save to csv
        print(f"\nSaving to {cfg.categorized_data_file_path_save} ...")
        general_utils.export_dataframe_to_csv(cfg.categorized_data_file_path_save, categorized_data)

        print("\nFinished")

    except Exception as e:
        print(e)
        sys.exit(1)
