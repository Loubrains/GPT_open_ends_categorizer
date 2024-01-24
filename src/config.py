### User defined settings - change these before running the scripts!

open_end_data_file_path = "data/test_data/New Year Resolution - A2 open ends.csv"
categories_file_path = "data/output_data/categories.csv"  # Save to or load from
codeframe_file_path = "data/output_data/codeframe.csv"  # Save to or load from
categorized_data_file_path = "data/output_data/categorized_data.csv"  # Save to

# Number of responses sent to GPT to generate initial list of categories
responses_sample_size = 200
# Number of categories to generate
number_of_categories = 20
# Number of responses to send to GPT per request
batch_size = 3
# Number of retry GPT requests upon error
max_retries = 5

# Whether each response can be put into multiple categories
is_multicode = True

questionnaire_question = "What is your new year resolution?"
