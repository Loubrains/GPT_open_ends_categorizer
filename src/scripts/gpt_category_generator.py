from openai import AsyncOpenAI
import asyncio
import pandas as pd
import chardet
from utils import general_utils
from utils import gpt_utils
from config.config import *

### NOTE: MAKE SURE TO SET USER DEFINED VARIABLES IN config.py

### NOTE: Make sure OpenAI_API_KEY is set up in your system environment variables ###
client = AsyncOpenAI()

# Load open ends
print("Loading data...")
with open(open_end_data_file_path, "rb") as file:
    encoding = chardet.detect(file.read())["encoding"]  # Detect encoding
df = pd.read_csv(open_end_data_file_path, encoding=encoding)

# Clean open ends
print("Cleaning responses...")
processed_responses = df[df.columns[1:]].map(general_utils.preprocess_text)
unique_responses = processed_responses.stack().drop_duplicates().dropna().reset_index(drop=True)
print("\nResponses:\n", unique_responses.head(10))

# Get sample of responses
print("\nFetching sample...")
responses_sample = general_utils.get_random_sample_from_series(unique_responses, responses_sample_size).to_list()  # type: ignore

# Generate categories using the GPT API
print("Generating categories with GPT-4...")
categories = asyncio.run(
    gpt_utils.GPT_generate_categories_list(
        client, questionnaire_question, responses_sample, number_of_categories, max_retries
    )
)
categories.extend(["Other", "Bad response", "Uncategorized"])
categories_df = pd.DataFrame(categories)
print(f"\nCategories:\n{categories}")

# Save results
print(f"\nSaving to {categories_file_path} ...")
general_utils.export_dataframe_to_csv(categories_file_path, categories_df, header=False)

print("\nFinished")
