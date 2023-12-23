from openai import OpenAI
import pandas as pd
import json
import re
import chardet
import random

client = OpenAI()


def preprocess_text(text) -> str:
    # Lowercase, removing special characters and numbers
    text = str(text).lower()
    text = re.sub(
        r"\s+", " ", text
    )  # Convert one or more of any kind of space to single space
    text = re.sub(r"[^a-z0-9\s]", "", text)  # Remove special characters
    text = text.strip()
    return text


def get_random_sample_from_series(series: pd.Series, sample_size: int) -> pd.Series:
    if sample_size > len(series):
        raise ValueError(
            "Sample size n cannot be greater than the length of the series"
        )
    return series.sample(sample_size, random_state=random.randint(1, 10000))


def write_list_to_csv(my_list: list, file_name: str):
    try:
        df = pd.DataFrame(my_list)
        df.to_csv(file_name, index=False, header=False)
    except Exception as e:
        print(f"Error while writing to CSV: {e}")


def generate_categories_GPT(
    client: OpenAI,
    questionnaire_question: str,
    responses_sample: pd.Series,
    number_of_categories: int = 20,
):
    system = [
        {
            "role": "system",
            "content": "You are data analyst, working with open-ended questionnaire responses.",
        }
    ]
    user = [
        {
            "role": "user",
            "content": f"""Generate a list of {number_of_categories} distinct thematic categories
                            for the following responses. Return the category names in the following format,
                            `["name1", "name2", ...]`\n\n
                            Question:\n`{questionnaire_question}`\n\n
                            Responses:\n`{responses_sample}`
                            """,
        }
    ]
    try:
        completion = client.chat.completions.create(
            messages=system + user, model="gpt-4"
        )
        categories = json.loads(completion.choices[0].message.content)

    except Exception as e:
        print(f"An error occurred: {e}")
        categories = "Error"
        raise

    return categories


file_name = "BBC Need States - B3_OPEN open ends.csv"
print("Loading data...")
with open(file_name, "rb") as file:
    encoding = chardet.detect(file.read())["encoding"]  # Detect encoding
df = pd.read_csv(file_name, encoding=encoding)

print("Cleaning responses...")
processed_responses = df["B3_OPEN"].map(preprocess_text).dropna()  # Process data
print("\nResponses:\n", processed_responses.head(10))

print("\nFetching sample...")
responses_sample = get_random_sample_from_series(processed_responses, 100)

print("Generating categories with GPT-4...")
questionnaire_question = "Why were you or your child consuming media at this time?"
categories = generate_categories_GPT(
    client, questionnaire_question, responses_sample, number_of_categories=20
)
print(f"\nCategories:\n{categories}")

output_file_name = "categories_output.csv"
print(f"\nSaving to {output_file_name} ...")
write_list_to_csv(categories, output_file_name)

print("\nDone")
