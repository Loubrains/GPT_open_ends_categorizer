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
    - User-defined variables such as file paths should be properly set in the `config` file before running this script.
    - The script supports both single-category and multi-category (`is_multicode`) response categorization.
    - The script terminates if exceptions are raised at any point.

Author: Louie Atkins-Turkish (louie@tapestryresearch.com)
"""

import pandas as pd
import chardet
from itertools import islice
import sys
from gpt_categorizer_utils import general_utils, dataframe_utils
import config as cfg
from logging_utils import setup_logging


if __name__ == "__main__":
    logger = setup_logging()

    try:
        # Load open ends
        logger.info("Loading data")
        with open(cfg.OPEN_END_DATA_FILE_PATH_LOAD, "rb") as file:
            encoding = chardet.detect(file.read())["encoding"]  # Detect encoding
        df = pd.read_csv(cfg.OPEN_END_DATA_FILE_PATH_LOAD, encoding=encoding)
        logger.debug(f"\nRaw data (first 20):\n{df.head(20)}")

        # Clean open ends
        logger.info("Cleaning data")
        response_columns = df.iloc[:, 1:].map(general_utils.preprocess_text)
        logger.debug(f"\nResponses (first 10):\n{response_columns.head(10)}")

        # Load categories
        logger.info("Loading categories")
        with open(cfg.CATEGORIES_FILE_PATH_LOAD, "rb") as file:
            encoding = chardet.detect(file.read())["encoding"]  # Detect encoding
        categories = pd.read_csv(cfg.CATEGORIES_FILE_PATH_LOAD, encoding=encoding, header=None)
        logger.debug(f"Categories:\n{categories}")

        # Load codeframe (dictionary of response-category pairs)
        logger.info("Loading codeframe...")
        if cfg.IS_MULTICODE:
            categorized_dict = general_utils.load_csv_to_dict_of_lists(cfg.CODEFRAME_FILE_PATH_LOAD)
        else:
            categorized_dict = general_utils.load_csv_to_dict(cfg.CODEFRAME_FILE_PATH_LOAD)
        logger.debug(
            "Codeframe (first 10):\n",
            "\n".join(f"{key}: {value}" for key, value in islice(categorized_dict.items(), 10)),
        )

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
        logger.info("Preparing output data...")
        for response_column in response_column_names:
            for response, categories in categorized_dict.items():
                if cfg.IS_MULTICODE and "Error" in categories:
                    logger.error(f"\nResponse '{response}' was not categorized.")
                elif categories == "Error":
                    logger.error(f"\nResponse '{response}' was not categorized.")

                dataframe_utils.categorize_responses_for_response_column(
                    response, categories, response_column, categorized_data, cfg.IS_MULTICODE
                )

        logger.debug(f"\nCategorized results (first 10):\n{categorized_data.head(10)}")

        # Save to csv
        logger.info(f"Saving to {cfg.CATEGORIZED_DATA_FILE_PATH_SAVE} ...")
        general_utils.export_dataframe_to_csv(cfg.CATEGORIZED_DATA_FILE_PATH_SAVE, categorized_data)

        logger.info("Finished")

    except Exception as e:
        logger.exception(e)
        sys.exit(1)
