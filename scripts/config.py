"""
### MODIFY THIS FILE BEFORE RUNNING ANY SCRIPTS ###

Configuration file containing user-defined parameters used in the open-end response categorization process.

It includes file paths for loading and saving data, the questionnaire question, categorization options, sample sizes, and other settings.

Settings for interacting with the OpenAI API (for example, parameters for handling rate limits) can be found in `gpt_config.py` in the `gpt_categorizer_utils` package.

Configuration Variables:
    `OPEN_END_DATA_FILE_PATH_LOAD`: File path to the CSV containing open-ended responses. Expects the first column to contain uuids, and subsequent columns to contain responses. Expects column headers.
    `CATEGORIES_FILE_PATH_SAVE`: File path for saving the GPT generated categories to CSV.
    `CATEGORIES_FILE_PATH_LOAD`: File path for loading a list of categories from CSV.
    `CODEFRAME_FILE_PATH_SAVE`: File path for saving the GPT generated codeframe to CSV.
    `CODEFRAME_FILE_PATH_LOAD`: File path for loading a codeframe from CSV.
    `CATEGORIZED_DATA_FILE_PATH_SAVE`: File path for saving the final categorized data to CSV.
    `QUESTIONNAIRE_QUESTION`: Text of the questionnaire question associated with the open-ended responses.    
    `IS_MULTICODE`: Boolean flag indicating whether each response can belong to multiple categories.

Note:
- File paths expect foreward slashes.
"""

from pathlib import Path

# File paths (expects /foreward/slashes/ and file extension .csv)
OPEN_END_DATA_FILE_PATH_LOAD = Path(
    "C:/Users/LouieAtkins-Turkish/Tapestry Research/BBC - BBC Studios - Need States/Data/AI text coding/C2a/C2a_open_ends - confidential.csv"
)
CATEGORIES_FILE_PATH_SAVE = Path("")
CATEGORIES_FILE_PATH_LOAD = Path(
    "C:/Users/LouieAtkins-Turkish/Tapestry Research/BBC - BBC Studios - Need States/Data/AI text coding/C2a/C2a_categories_to_use - confidential.csv"
)
CODEFRAME_FILE_PATH_SAVE = Path(
    "C:/Users/LouieAtkins-Turkish/Tapestry Research/BBC - BBC Studios - Need States/Data/AI text coding/C2a/C2a_code_frame_test_2 - confidential.csv"
)
CODEFRAME_FILE_PATH_LOAD = Path(
    "C:/Users/LouieAtkins-Turkish/Tapestry Research/BBC - BBC Studios - Need States/Data/AI text coding/C2a/C2a_code_frame_test_2 - confidential.csv"
)
CATEGORIZED_DATA_FILE_PATH_SAVE = Path(
    "C:/Users/LouieAtkins-Turkish/Tapestry Research/BBC - BBC Studios - Need States/Data/AI text coding/C2a/C2a_categorized_data_test_2 - confidential.csv"
)

# Global settings
QUESTIONNAIRE_QUESTION = "What best describes the type of media content it is that you consumed?"
IS_MULTICODE = False

# Category generation settings
NUMBER_OF_CATEGORIES = 20
RESPONSES_SAMPLE_SIZE = 200
