# Intro

This project contains scripts and modules for the automatic categorization of open-ended text responses to survey questions using GPT-4.

## Prerequisites

**Python**: Ensure Python is installed on your system. Download and install the latest version from [Python's official website](https://www.python.org/downloads/).

**Pip**: It's recommended to use the latest version of pip. Update pip using the following command in your command line tool _(such as Windows PowerShell)_:

```sh
python -m pip install --upgrade pip
```

## Installation

1. Make a copy of the project folder on your local system.

2. In your command line tool, navigate to the copied project root directory.<br>
   E.g. `cd 'C:\Users\[user]\path\to\project'`

3. Run the following command to install the required packages:

```sh
pip install -r requirements.txt
```

# Usage

### Scripts

1. **config.py**: Holds user defined variables (file paths, questionnaire text, settings) that are used by the scripts.<br>
   **Modify this file before running any scripts.**
2. **gpt_category_generator.py**: Generates categories based a sample of the response data. Exports to CSV.
3. **gpt_categorizer.py**: Generates a codeframe, a map of responses to their corresponding categories, based on a list of predefined list of categories. Exports to CSV.
4. **create_categorized_output.py**: Applies a codeframe to the open-ends data to construct a final categorized DataFrame. Exports to a CSV.

### Running a script

From the project root directory, execute a script using:

```sh
python src/[script_name].py
```

**Note on loading the data**:

- Accepts only `.csv` files.
- The first column should contains UUIDs.
- Subsequent columns should contain open-ended text responses.

# Future developments

- **Packaging**: Convert this project into a Python package for easier distribution and use.
- **General gpt utility function**: Create a more general gpt utility function that isn't locked into the prompts that the other functions are - but does include cleaning, validation, retries, error handling, etc.
- **Automation script**: Create a script that sequentially runs all the necessary scripts for a full automation (use with caution)
- **User experience**: Create a smoother experience for new users to dive right in without modifying files, running multiple scripts, etc.

# Author

Louie Atkins-Turkish  
Data Analyst at Tapestry Research  
Email: louie@tapestryresearch.com  
Created on: December 26, 2023
