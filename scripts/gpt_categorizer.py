"""
Script that generates a thematic codeframe for survey responses using the GPT-4 API.
The codeframe is a dictionary of responses to categories.
Exports the codeframe to a CSV file.

Steps:
    1. Load the raw response data from a specified CSV file.
    2. Clean the responses by preprocessing the text.
    3. Extract a unique set of cleaned responses, excluding any empty strings.
    4. Load the predefined categories from a specified CSV file (excluding the 'Uncategorized' category).
    5. Use the GPT-4 API to categorize each unique response according to the predefined categories.
    6. Export the categorized responses (codeframe) to a CSV file.

Input Files:
    - Open-ended response data file (`open_end_data_file_path`): A CSV file containing the raw survey responses. Expects the first column to be uuids, and the following columns to be response columns.
    - Categories file (`categories_file_path`): A CSV file containing the list of predefined categories. Expects no header.

Output File:
    - Codeframe file (`codeframe_file_path`): A CSV file where the response-category pairs (codeframe) are saved.

Notes:
    - Make sure OPENAI_API_KEY is set up in your system environment variables.
    - The script uses utility functions from the `general_utils` and `gpt_utils` modules.
    - User-defined variables should be properly set in the `config` file before running this script.
    - The script terminates if exceptions are raised at any point.

Author: Louie Atkins-Turkish (louie@tapestryresearch.com)
"""

from openai import AsyncOpenAI
import asyncio
import pandas as pd
import chardet
from itertools import islice
import sys
from gpt_categorizer_utils import general_utils, gpt_utils
import config as cfg
from logging_utils import setup_logging

### NOTE: MAKE SURE TO SET USER DEFINED VARIABLES IN config.py
### NOTE: Make sure OPENAI_API_KEY is set up in your system environment variables ###


if __name__ == "__main__":
    logger = setup_logging()

    try:
        client = AsyncOpenAI()

        # Load open ends
        logger.info("Loading data")
        with open(cfg.OPEN_END_DATA_FILE_PATH_LOAD, "rb") as file:
            encoding = chardet.detect(file.read())["encoding"]  # Detect encoding
        df = pd.read_csv(cfg.OPEN_END_DATA_FILE_PATH_LOAD, encoding=encoding)
        logger.debug(f"\nRaw data (first 20):\n{df.head(20)}")

        # Clean open ends
        logger.info("Cleaning responses")
        # Assume first column UUIDs, remaining columns are responses
        df_preprocessed = df.iloc[:, 1:].map(general_utils.preprocess_text)
        logger.debug(f"Responses (first 10):\n{df_preprocessed.head(10)}")

        unique_responses = set(df_preprocessed.stack().dropna().reset_index(drop=True))
        # we don't want to match empty string against every row
        unique_responses = unique_responses - {""}
        unique_responses = [str(item) for item in unique_responses]  # convert to list[str]

        # Load categories
        logger.info("Loading categories")
        with open(cfg.CATEGORIES_FILE_PATH_LOAD, "rb") as file:
            encoding = chardet.detect(file.read())["encoding"]  # Detect encoding
        categories = pd.read_csv(cfg.CATEGORIES_FILE_PATH_LOAD, encoding=encoding, header=None)
        logger.debug(f"Categories:\n{categories}")

        categories_list = categories.iloc[:, 0].tolist()
        categories_list = [str(x) for x in categories_list if not pd.isna(x)]
        # Uncategorized is a helper category for later, we don't want ChatGPT to use it.
        categories_list.remove("Uncategorized")

        # Categorize responses using the GPT API
        logger.info("Categorizing data with GPT-4")
        # unique_responses_sample = unique_responses[:20]
        categorized_dict = asyncio.run(
            gpt_utils.gpt_categorize_response_batches_main(
                client,
                cfg.QUESTIONNAIRE_QUESTION,
                unique_responses,
                categories_list,
                cfg.IS_MULTICODE,
            )
        )
        categorized_dict.pop("", None)  # removing empty string since it matches against every row

        logger.debug(
            "Codeframe (first 10):\n",
            "".join(f"{key}: {value}" for key, value in islice(categorized_dict.items(), 10)),
        )
        logger.info("Finished categorizing with GPT-4")

        # Saving codeframe (dictionary of response-category pairs)
        logger.info(f"Saving codeframe to {cfg.CODEFRAME_FILE_PATH_SAVE}")
        general_utils.export_dict_to_csv(cfg.CODEFRAME_FILE_PATH_SAVE, categorized_dict)

        logger.info("Finished")

    except Exception as e:
        logger.exception(e)
        sys.exit(1)
