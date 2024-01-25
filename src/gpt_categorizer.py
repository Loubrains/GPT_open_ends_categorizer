"""
This script generates a thematic codeframe for survey responses using the GPT-4 API.
The codeframe is a dictionary of responses to categories. It exports the codeframe to a CSV file.

Steps:
1. Load the raw response data from a specified CSV file.
2. Clean the responses by preprocessing the text.
3. Extract a unique set of cleaned responses, excluding any empty strings.
4. Load the predefined categories from a specified CSV file (excluding the 'Uncategorized' category).
5. Use the GPT-4 API to categorize each unique response according to the predefined categories.
6. Export the categorized responses (codeframe) to a CSV file.

Input Files:
- Open-ended response data file (open_end_data_file_path): A CSV file containing the raw survey responses. Expects the first column to be uuids, and the following columns to be response columns.
- Categories file (categories_file_path): A CSV file containing the list of predefined categories. Expects no header.

Output File:
- Codeframe file (codeframe_file_path): A CSV file where the response-category pairs (codeframe) are saved.

The script uses utility functions from the 'general_utils' and 'gpt_utils' modules for processing. Configurations, including file paths and GPT model parameters, are defined in the 'config' module.

Notes:
- Make sure OPENAI_API_KEY is set up in your system environment variables.
- User-defined variables should be properly set in the config.py file.
- The script supports asynchronous operations for calling the GPT-4 API.
- The script provides console output at each major step to track progress and outputs.
"""

from openai import AsyncOpenAI
import asyncio
import pandas as pd
import chardet
from itertools import islice
from utils import general_utils
from utils import gpt_utils
from config import *

### NOTE: MAKE SURE TO SET USER DEFINED VARIABLES IN config.py

### NOTE: Make sure OPENAI_API_KEY is set up in your system environment variables ###
client = AsyncOpenAI()

# Load open ends
print("\nLoading data...")
with open(open_end_data_file_path, "rb") as file:
    encoding = chardet.detect(file.read())["encoding"]  # Detect encoding
df = pd.read_csv(open_end_data_file_path, encoding=encoding)

# Clean open ends
print("Cleaning responses...")
# Assume first column UUIDs, remaining columns are responses
df_preprocessed = df.iloc[:, 1:].map(general_utils.preprocess_text)
print(f"\nResponses (first 10):\n{df_preprocessed.head(10)}")

unique_responses = set(df_preprocessed.stack().dropna().reset_index(drop=True))
# we don't want to match empty string against every row
unique_responses = unique_responses - {""}
unique_responses = [str(item) for item in unique_responses]  # convert to list[str]

# Load categories
print("\nLoading categories...")
with open(categories_file_path, "rb") as file:
    encoding = chardet.detect(file.read())["encoding"]  # Detect encoding
categories = pd.read_csv(categories_file_path, encoding=encoding, header=None)
print(f"\nCategories:\n{categories}")

categories_list = categories.iloc[:, 0].tolist()
# Uncategorized is a helper category for later, we don't want ChatGPT to use it.
categories_list.remove("Uncategorized")

# Categorize responses using the GPT API
print("\nCategorizing data with GPT-4...")
# unique_responses_sample = unique_responses[:20]
categorized_dict = asyncio.run(
    gpt_utils.gpt_categorize_response_batches_main(
        client,
        questionnaire_question,
        unique_responses,
        categories_list,
        batch_size,
        max_retries,
        is_multicode,
    )
)

categorized_dict.pop("", None)  # removing empty string since it matches against every row

print("\nCodeframe (first 10):")
print("\n".join(f"{key}: {value}" for key, value in islice(categorized_dict.items(), 10)))
print("\nFinished categorizing with GPT-4...")

# Saving codeframe (dictionary of response-category pairs)
print(f"\nSaving codeframe to {codeframe_file_path} ...")
general_utils.export_dict_to_csv(codeframe_file_path, categorized_dict)

print("\nFinished")
