"""
### MODIFY THIS BEFORE RUNNING ANY SCRIPTS ###

Configuration file for user-defined parameters used in the open-end response categorization process.

It includes file paths for loading and saving data, the questionnaire question, categorization options, sample sizes, and other settings.

Configuration Variables:
    `open_end_data_file_path_load`: File path to the CSV containing open-ended responses. Expects the first column to contain uuids, and subsequent columns to contain responses. Expects column headers.
    `categories_file_path_save`: File path for saving the GPT generated categories to CSV.
    `categories_file_path_load`: File path for loading a list of categories from CSV.
    `codeframe_file_path_save`: File path for saving the GPT generated codeframe to CSV.
    `codeframe_file_path_load`: File path for loading a codeframe from CSV.
    `categorized_data_file_path_save`: File path for saving the final categorized data to CSV.
    `questionnaire_question`: Text of the questionnaire question associated with the open-ended responses.    
    `is_multicode`: Boolean flag indicating whether each response can belong to multiple categories.
    `max_retries`: Number of retry attempts for GPT requests upon encountering errors.
    `number_of_categories`: Number of categories to generate.
    `responses_sample_size`: Number of responses sent to GPT to generate the initial list of categories.
    `batch_size`: Number of responses to send to GPT per request.

Note:
- File paths expect foreward slashes.
"""

from pathlib import Path

# File paths (expects foreward slashes)
open_end_data_file_path_load = Path("path/to/your/data")
categories_file_path_save = Path("path/to/save/categories")
categories_file_path_load = Path("path/to/load/categories")
codeframe_file_path_save = Path("path/to/save/codeframe")
codeframe_file_path_load = Path("path/to/load/codeframe")
categorized_data_file_path_save = Path("path/to/save/categorized/data")

# Global settings
questionnaire_question = "your_questionnaire_question"
is_multicode = False
max_retries = 5

# Category generation settings
number_of_categories = 20
responses_sample_size = 200

# Codeframe generation settings
batch_size = 3
