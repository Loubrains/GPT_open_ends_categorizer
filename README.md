# Intro

This project contains scripts and modules for the automatic categorization of open-ended text responses to survey questions using GPT-4.

It costs less than 1$ and takes about 15 seconds to categorize a dataset of 1500 responses.

# Prerequisites

**Python**: Ensure Python is installed on your system. Download and install the latest version from [Python's official website](https://www.python.org/downloads/). During installation, ensure you select the option to 'Add Python to PATH'.

**Pip**: It's recommended to use the latest version of pip. Update pip using the following command in your command line tool _(such as Windows PowerShell)_:

```powershell
python -m pip install --upgrade pip
```

**OpenAI API key**: Obtain your API key from the [OpenAI API Keys portal](https://platform.openai.com/api-keys). Once you have your key, set it as an environment variable named OPENAI_API_KEY on your system. This will be used for authenticating with OpenAI account and billing to your account.

```powershell
setx OPENAI_API_KEY "your_api_key_here"
```

# Installation

1. Make a copy of the project folder on your local system.

2. In your command line tool, navigate to the copied project root directory:

```powershell
cd 'C:\Users\[user]\path\to\project'
```

3. Run the following command to install the required packages:

```powershell
pip install -r requirements.txt
```

# Usage

### Scripts

1. **config.py**: Contains user defined variables (file paths, questionnaire text, settings) that are used by the scripts.<br>
   **Modify this file before running any scripts.**
2. **gpt_category_generator.py**: Generates a list of categories based a sample of the response data. Exports to CSV.
3. **gpt_categorizer.py**: Generates a codeframe, a map of responses to their corresponding categories, based on a list of predefined list of categories. Exports to CSV.
4. **create_categorized_output.py**: Applies a codeframe to the open-ends data to construct a final categorized DataFrame. Exports to a CSV.

### Running a script

From the project root directory, execute a script using the following command:

```powershell
python scripts/script_name.py
```

**Note on loading the data**:

- Accepts only `.csv` files.
- The first column should contains UUIDs.
- Subsequent columns should contain open-ended text responses.

# Documentation

The HTML documentation for this project is generated using `pdoc3`. You can view it by opening the files located in the `docs` directory with your web browser.

# Tests

To run the test suite, enter the following command in the project root directory:

```powershell
pytest
```

You can run tests in a specific file or directory by providing the path:

```powershell
pytest tests/test_specific_module.py
```

For more verbose output, use the -v flag:

```powershell
pytest -v
```

# Contributing

When contributing, make sure to configure Git to use `obfuscator.py` to clean your sensitive data from `config.py`

Run the following commands in the project root directory to set up the Git filters:

```powershell
git config filter.obfuscate.clean './obfuscator.py'
git config filter.obfuscate.smudge cat
```

Any new developments you would like to be obfuscated, make sure to include that file in `.gitattributes` and update the patterns in `obfuscator.py`.

# Future developments

- **General gpt utility function**: Create a more general gpt utility function that isn't locked into the prompts that the other functions are - but does include cleaning, validation, retries, error handling, etc.
- **Automation script**: Create a script that sequentially runs all the necessary scripts for a full automation (use with caution)
- **User experience**: Create a smoother experience for new users to dive right in without modifying files, running multiple scripts, etc.

# Author

Louie Atkins-Turkish  
Data Analyst at Tapestry Research  
Email: louie@tapestryresearch.com  
Created on: December 26, 2023
