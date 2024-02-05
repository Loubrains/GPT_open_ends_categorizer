"""
Package of modules for categorizing open-ended survey responses with GPT-4.

This package provides utilities designed for the processing of open-ended survey responses, categorizating them using GPT-4, and formatting the results for data analysis.

Modules:
    - `general_utils`:
        Contains functions for general data handling and processing tasks, such as text preprocessing, random sampling from series, 
        batch creation, loading and exporting data to and from CSV files.
    - `gpt_utils`:
        Provides utilities for interacting with the OpenAI GPT model asyncronously. It includes functions for sending prompts to the GPT model, 
        generating lists of thematic categories, generating codeframes for responses, validating categorized outputs, and batch processing.
        It imports config parameters from `gpt_config`.
    - `dataframe_utils`:
        Provides utilities for manipulating DataFrames, with the goal of preparing the categorized data for further analysis.
        It includes functions for initializing categorized DataFrames, handling missing data, and categorizing responses based on a codeframe.
    - `gpt_config`:
        Configuration file containing user-defined parameters for rate limiting

Author: Louie Atkins-Turkish (louie@tapestryresearch.com)
"""
