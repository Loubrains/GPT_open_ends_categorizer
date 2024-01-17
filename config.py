open_end_data_file_path = "C3.csv"
categories_file_path = "categories.csv"  # Save to or load from
codeframe_file_path = "codeframe.csv"  # Save to or load from
categorized_data_file_path = "categorized_data.csv"

# Number of responses sent to GPT to generate initial list of categories
responses_sample_size = 200
# Number of categories to generate
number_of_categories = 20
# Number of responses to send to GPT per request
batch_size = 3
# Number of retry GPT requests upon error
max_retries = 5

# Whether each response can be put into multiple categories
is_multicode = False

questionnaire_question = (
    "Why do you not like the always-on player feature in this streaming service?"
)
