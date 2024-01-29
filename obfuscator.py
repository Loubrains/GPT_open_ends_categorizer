#!/usr/bin/env python

"""
A script for obfuscating known sensitive data within files prior to committing to Git.

This script is intended to be used as a filter in a Git environment to automatically replace 
sensitive information in files like 'config.py' with placeholder data. The script is triggered by
the 'clean' filter of a .gitattributes file, ensuring that sensitive information is never
committed to the repository. The original file remains unchanged in the 
working directory, while the staged version is cleaned of sensitive content.

The script defines a set of patterns for sensitive data that it expects to find in the input 
content and replaces any matches with predefined placeholders.

Usage:
    To use this script as a Git filter, add the following to your .gitattributes file in the project root directory:
    
        ```powershell
        config.py filter=obfuscate
        ```

    Then configure the Git filter with the following commands in the project root directory:

        ```powershell
        git config filter.obfuscate.clean './obfuscator.py'
        git config filter.obfuscate.smudge cat
        ```
        
    This will configure Git to clean config.py with this obfuscator script,
    but not clean it when checking out code (e.g. switching branches)
"""

import sys
import re


def obfuscate_sensitive_data(content: str) -> str:
    """
    Obfuscates sensitive data in a given string by replacing it with default values.

    Intended to be used on specific files (e.g. config.py) containing known specific sequences of sensitive data, as defined within the function.

    Parameters:
    - content (str): The string content that potentially contains sensitive data.

    Returns:
    - str: The obfuscated version of the input content with sensitive data replaced by safe placeholders.
    """
    path_pattern = r'Path\(".*?"\)'

    replacements = {
        rf"open_end_data_file_path_load = {path_pattern}": 'open_end_data_file_path_load = Path("path/to/your/data.csv")',
        rf"categories_file_path_save = {path_pattern}": 'categories_file_path_save = Path("path/to/save/categories.csv")',
        rf"categories_file_path_load = {path_pattern}": 'categories_file_path_load = Path("path/to/load/categories.csv")',
        rf"codeframe_file_path_save = {path_pattern}": 'codeframe_file_path_save = Path("path/to/save/codeframe.csv")',
        rf"codeframe_file_path_load = {path_pattern}": 'codeframe_file_path_load = Path("path/to/load/codeframe.csv")',
        rf"categorized_data_file_path_save = {path_pattern}": 'categorized_data_file_path_save = Path("path/to/save/categorized/data.csv")',
        #
        rf'questionnaire_question = ".*?"': 'questionnaire_question = "your_questionnaire_question?"',
    }

    for pattern, replacement in replacements.items():
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        # re.DOTALL catches multiline strings

    return content


def main():
    content = sys.stdin.read()
    cleaned_content = obfuscate_sensitive_data(content)
    sys.stdout.write(cleaned_content)


if __name__ == "__main__":
    main()
