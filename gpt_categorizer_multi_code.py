# TODO: need to strip csv's after loading before sending to gpt

from openai import OpenAI
import asyncio
import pandas as pd
import chardet
from itertools import islice
import general_utils
import gpt_utils

### NOTE: Make sure OpenAI_API_KEY is set up in your system environment variables ###
client = OpenAI()

### USER DEFINED VARIABLES
data_file_path = "C3.csv"
categories_file_path = "categories.csv"
batch_size = 3
max_retries = 5
result_codeframe_file_path = "codeframe.csv"
questionnaire_question = (
    "Why do you not like the always-on player feature in this streaming service?"
)


# Load open ends
print("\nLoading data...")
with open(data_file_path, "rb") as file:
    encoding = chardet.detect(file.read())["encoding"]  # Detect encoding
df = pd.read_csv(data_file_path, encoding=encoding)

# Clean open ends
print("Cleaning responses...")
# Assume first column UUIDs, remaining columns are responses
df_preprocessed = df.iloc[:, 1:].map(general_utils.preprocess_text)  # type: ignore
print(f"\nResponses (first 10):\n{df_preprocessed.head(10)}")

unique_responses = set(df_preprocessed.stack().dropna().reset_index(drop=True))
# we don't want to match empty string against every row
unique_responses = unique_responses - {""}

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
    gpt_utils.GPT_categorize_responses_multicode_main(
        client,
        questionnaire_question,
        categories_list,
        unique_responses,
        batch_size,
        max_retries,
    )
)
categorized_dict.pop("", None)  # removing empty string since it matches against every row
print("\nCodeframe (first 10):")
print("\n".join(f"{key}: {value}" for key, value in islice(categorized_dict.items(), 10)))
print("\nFinished categorizing with GPT-4...")

# Saving codeframe (dictionary of response-category pairs)
print(f"\nSaving codeframe to {result_codeframe_file_path} ...")
general_utils.export_dict_of_lists_to_csv(result_codeframe_file_path, categorized_dict)

print("\nFinished")
