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
- User-defined variables should be properly set in the `config.py` file before running this script.
- The script terminates if exceptions are raised at any point.
"""


from openai import AsyncOpenAI
import asyncio
import pandas as pd
from pathlib import Path
import chardet
import sys
from gpt_categorizer_utils import general_utils, gpt_utils
import config as cfg

### NOTE: MAKE SURE TO SET USER DEFINED VARIABLES IN config.py
### NOTE: Make sure OPENAI_API_KEY is set up in your system environment variables ###


if __name__ == "__main__":
    try:
        client = AsyncOpenAI()

        # Load open ends
        print("Loading data...")

        with open(cfg.open_end_data_file_path_load, "rb") as file:
            encoding = chardet.detect(file.read())["encoding"]  # Detect encoding
        df = pd.read_csv(cfg.open_end_data_file_path_load, encoding=encoding)

        # Clean open ends
        print("Cleaning responses...")
        processed_responses = df[df.columns[1:]].map(general_utils.preprocess_text)
        unique_responses = (
            processed_responses.stack().drop_duplicates().dropna().reset_index(drop=True)
        )
        print("\nResponses:\n", unique_responses.head(10))

        # Get sample of responses
        print("\nFetching sample...")
        responses_sample = general_utils.get_random_sample_from_series(unique_responses, responses_sample_size).to_list()  # type: ignore

        # Generate categories using the GPT API
        print("Generating categories with GPT-4...")
        categories = asyncio.run(
            gpt_utils.gpt_generate_categories_list(
                client,
                cfg.questionnaire_question,
                responses_sample,
                cfg.number_of_categories,
                cfg.max_retries,
            )
        )
        categories.extend(["Other", "Bad response", "Uncategorized"])
        categories_df = pd.DataFrame(categories)
        print(f"\nCategories:\n{categories}")

        # Save results
        print(f"\nSaving to {cfg.categories_file_path_save} ...")
        general_utils.export_dataframe_to_csv(
            cfg.categories_file_path_save, categories_df, header=False
        )

        print("\nFinished")

    except Exception as e:
        print(e)
        sys.exit(1)
