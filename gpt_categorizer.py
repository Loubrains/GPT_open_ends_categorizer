from openai import OpenAI
import asyncio
import pandas as pd
import chardet
from itertools import islice
import general_utils
import gpt_utils
from config import *

### NOTE: MAKE SURE TO SET USER DEFINED VARIABLES IN config.py
### NOTE: IF YOU SEE EVERY BATCH OF RESPONSES IS REACHING 5/5 RETRIES, TERMINATE THE PROGRAM AND DEBUG.

### NOTE: Make sure OpenAI_API_KEY is set up in your system environment variables ###
client = OpenAI()

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
# unique_responses_sample = list(unique_responses)[:20]
categorized_dict = asyncio.run(
    gpt_utils.GPT_categorize_response_batches_main(
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
