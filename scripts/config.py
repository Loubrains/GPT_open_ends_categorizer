"""
### MODIFY THIS BEFORE RUNNING ANY SCRIPTS ###

Configuration file for user-defined parameters used in the open-end response categorization process.

It includes file paths for loading and saving data, the questionnaire question, categorization options, sample sizes, and other settings.

Configuration Variables:
    `open_end_data_file_path`: File path to the CSV containing open-ended responses. Expects the first column to contain uuids, and subsequent columns to contain responses. Expects column headers.
    `categories_file_path`: File path for saving or loading the GPT generated categories to CSV.
    `codeframe_file_path`: File path for saving or loading the GPT generated codeframe to CSV.
    `categorized_data_file_path`: File path for saving the final categorized data to CSV.    
    `questionnaire_question`: Text of the questionnaire question associated with the open-ended responses.    
    `is_multicode`: Boolean flag indicating whether each response can belong to multiple categories.
    `max_retries`: Number of retry attempts for GPT requests upon encountering errors.
    `number_of_categories`: Number of categories to generate.
    `responses_sample_size`: Number of responses sent to GPT to generate the initial list of categories.
    `batch_size`: Number of responses to send to GPT per request.

Note:
- File paths expect foreward slashes.
"""

# File paths (expects foreward slashes)
open_end_data_file_path = "C:/Users/LouieAtkins-Turkish/Tapestry Research/BBC - BBC Studios - Need States/Data/AI text coding/C2a open ends.csv"
categories_file_path = "C:/Users/LouieAtkins-Turkish/Tapestry Research/BBC - BBC Studios - Need States/Data/AI text coding/C2a gpt categories.csv"  # Save to or load from
codeframe_file_path = "C:/Users/LouieAtkins-Turkish/Tapestry Research/BBC - BBC Studios - Need States/Data/AI text coding/C2a gpt codeframe.csv"  # Save to or load from
categorized_data_file_path = "C:/Users/LouieAtkins-Turkish/Tapestry Research/BBC - BBC Studios - Need States/Data/AI text coding/C2a gpt categorized_data.csv"  # Save to

# Global settings
questionnaire_question = "What best describes the type of media content it is that you consumed?"
is_multicode = False
max_retries = 5

# Category generation settings
number_of_categories = 20
responses_sample_size = 200

# Codeframe generation settings
batch_size = 3
