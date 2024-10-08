"""
### MODIFY THIS FILE BEFORE RUNNING ANY SCRIPTS ###

Configuration file containing user-defined parameters used in the open-end response categorization process.

It includes file paths for loading and saving data, the questionnaire question, categorization options, sample sizes, and other settings.

Settings for interacting with the OpenAI API (e.g. parameters for handling rate limits) can be found in the config file `gpt_categorizer_utils.gpt_config`.

Configuration Variables:
    - `OPEN_END_DATA_FILE_PATH_LOAD`: File path to the CSV containing open-ended responses. Expects the first column to contain uuids, and subsequent columns to contain responses. Expects column headers.
    - `CATEGORIES_FILE_PATH_SAVE`: File path for saving the GPT generated categories to CSV.
    - `CATEGORIES_FILE_PATH_LOAD`: File path for loading a list of categories from CSV.
    - `CODEFRAME_FILE_PATH_SAVE`: File path for saving the GPT generated codeframe to CSV.
    - `CODEFRAME_FILE_PATH_LOAD`: File path for loading a codeframe from CSV.
    - `CATEGORIZED_DATA_FILE_PATH_SAVE`: File path for saving the final categorized data to CSV.
    - `QUESTIONNAIRE_QUESTION`: Text of the questionnaire question associated with the open-ended responses.    
    - `IS_MULTICODE`: Boolean flag indicating whether each response can belong to multiple categories.
    - `NUMBER_OF_CATEGORIES`: Number of categories to generate.
    - `RESPONSES_SAMPLE_SIZE`: Number of responses sent to GPT to generate the initial list of categories.

Note:
    - File paths expect /foreward/slashes/.

Author: Louie Atkins-Turkish (louie@tapestryresearch.com)
"""

from pathlib import Path

# File paths (expects /foreward/slashes/ and file extension .csv)
OPEN_END_DATA_FILE_PATH_LOAD = Path("path/to/your/data.csv")
CATEGORIES_FILE_PATH_SAVE = Path("path/to/save/categories.csv")
CATEGORIES_FILE_PATH_LOAD = Path("path/to/load/categories.csv")
CODEFRAME_FILE_PATH_SAVE = Path("path/to/save/codeframe.csv")
CODEFRAME_FILE_PATH_LOAD = Path("path/to/load/codeframe.csv")
CATEGORIZED_DATA_FILE_PATH_SAVE = Path("path/to/save/categorized/data.csv")

# Global settings
QUESTIONNAIRE_QUESTION = "your_questionnaire_question?"
IS_MULTICODE = False

# Category generation settings
NUMBER_OF_CATEGORIES = 20
RESPONSES_SAMPLE_SIZE = 200
