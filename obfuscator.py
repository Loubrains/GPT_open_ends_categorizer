#!/usr/bin/env python
import sys
import re


def obfuscate_sensitive_data(content):
    replacements = {
        r'open_end_data_file_path_load = Path\(".*?"\)': 'open_end_data_file_path_load = Path("path/to/your/data")',
        r'categories_file_path_save = Path\(".*?"\)': 'categories_file_path_save = Path("path/to/save/categories")',
        r'categories_file_path_load = Path\(".*?"\)': 'categories_file_path_load = Path("path/to/load/categories")',
        r'codeframe_file_path_save = Path\(".*?"\)': 'codeframe_file_path_save = Path("path/to/save/codeframe")',
        r'codeframe_file_path_load = Path\(".*?"\)': 'codeframe_file_path_load = Path("path/to/load/codeframe")',
        r'categorized_data_file_path_save = Path\(".*?"\)': 'categorized_data_file_path_save = Path("path/to/save/categorized/data")',
        r'questionnaire_question = ".*?"': 'questionnaire_question = "your_questionnaire_question"',
    }

    for pattern, replacement in replacements.items():
        content = re.sub(pattern, replacement, content)

    return content


def main():
    content = sys.stdin.read()
    cleaned_content = obfuscate_sensitive_data(content)
    sys.stdout.write(cleaned_content)


if __name__ == "__main__":
    main()
