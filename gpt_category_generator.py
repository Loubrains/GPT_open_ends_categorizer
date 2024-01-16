from openai import OpenAI
import pandas as pd
import chardet
import general_utils
import gpt_utils

### NOTE: Make sure OpenAI_API_KEY is set up in your system environment variables ###
client = OpenAI()

### CHANGE THESE VALUES TO WHAT YOU NEED
file_name = "C3.csv"
result_categories_file_path = "categories.csv"
questionnaire_question = (
    "Why do you not like the always-on player feature in this streaming service?"
)


print("Loading data...")
with open(file_name, "rb") as file:
    encoding = chardet.detect(file.read())["encoding"]  # Detect encoding
df = pd.read_csv(file_name, encoding=encoding)

print("Cleaning responses...")
processed_responses = df[df.columns[1:]].map(general_utils.preprocess_text)
unique_responses = processed_responses.stack().drop_duplicates().dropna().reset_index(drop=True)
print("\nResponses:\n", unique_responses.head(10))

print("\nFetching sample...")
responses_sample = general_utils.get_random_sample_from_series(unique_responses, 200).to_list()  # type: ignore

print("Generating categories with GPT-4...")
categories = gpt_utils.generate_categories_GPT(
    client, questionnaire_question, responses_sample, number_of_categories=20
)
categories.extend(["Other", "Bad response", "Uncategorized"])
categories_df = pd.DataFrame(categories)
print(f"\nCategories:\n{categories}")

print(f"\nSaving to {result_categories_file_path} ...")
general_utils.export_dataframe_to_csv(result_categories_file_path, categories_df, header=False)

print("\nFinished")
