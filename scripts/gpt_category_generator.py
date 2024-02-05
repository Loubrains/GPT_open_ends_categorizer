"""
Script that generates a list of thematic categories based on a sample of open-ended survey responses using the GPT-4 API.
Exports the generated categories to a CSV file.

Steps:
    1. Load raw response data from a specified CSV file.
    2. Clean the responses by preprocessing the text.
    3. Extract a unique set of responses and fetch a random sample from these unique responses.
    4. Use the GPT-4 API to generate a list of categories based on the sample of responses.
    5. Append additional default categories such as "Other", "Bad response", and "Uncategorized".
    6. Export the generated categories to a CSV file.

Input Files:
    - Open-ended response data file (`open_end_data_file_path`): A CSV file containing the raw survey responses. Expects the first column to be uuids, and the following columns to be response columns.

Output Files:
    - Categories file (`categories_file_path`): A CSV file where the generated categories are saved.

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

        # Clean open ends
        logger.info("Cleaning responses")
        processed_responses = df[df.columns[1:]].map(general_utils.preprocess_text)
        unique_responses = (
            processed_responses.stack().drop_duplicates().dropna().reset_index(drop=True)
        )
        logger.debug(f"Responses (first 10):\n{unique_responses.head(10)}")

        # Get sample of responses
        logger.info("Fetching sample of responses")
        responses_sample = general_utils.get_random_sample_from_series(unique_responses, cfg.responses_sample_size).to_list()  # type: ignore
        logger.debug(f"Sample (size {cfg.RESPONSES_SAMPLE_SIZE}):\n{responses_sample}")

        # Generate categories using the GPT API
        logger.info("Generating categories with GPT-4")
        categories = asyncio.run(
            gpt_utils.gpt_generate_categories_list(
                client, cfg.QUESTIONNAIRE_QUESTION, responses_sample, cfg.NUMBER_OF_CATEGORIES
            )
        )
        categories.extend(["Other", "Bad response", "Uncategorized"])
        categories_df = pd.DataFrame(categories)
        logger.debug(f"Categories generated:\n{categories}")

        # Save results
        logger.info(f"Saving categories to {cfg.CATEGORIES_FILE_PATH_SAVE}")
        general_utils.export_dataframe_to_csv(
            cfg.CATEGORIES_FILE_PATH_SAVE, categories_df, header=False
        )

        logger.info("Finished")

    except Exception as e:
        logger.exception(e)
        sys.exit(1)
