"""
This module provides utilities for building a dataframe for the categorization of open-ended survey responses.
It is designed with an output format in mind that is suitable for use in Q Research Software (or data analysis in general).
It expects the input data to have a column for uuids, and then subsequent columns for responses.

Functions:
    construct_default_categorized_dataframe: Initializes a DataFrame with category columns for each specified response column, setting all entries to 0 except for "Uncategorized" which are set to 1.
    categorize_missing_data_for_response_column: Marks category columns as missing (pd.NA) for rows where the corresponding response column has missing data.
    categorize_responses_for_response_column: Categorizes responses in a response column by setting the corresponding category columns to 1 and the 'Uncategorized' column to 0 for matched responses.
"""

import pandas as pd


def construct_default_categorized_dataframe(
    categorized_data: pd.DataFrame, response_column_names: list[str], categories_list: list[str]
) -> pd.DataFrame:
    """
    Modifies a DataFrame by addingm category columns for each specified response column.

    This function appends columns to the input DataFrame for each combination of response columns and categories.
    Newly added columns are named in the format "{category}_{response_column}".

    All entries are initialized to 0, except for those under the "Uncategorized" category, which are initialized to 1.

    Args:
        categorized_data (pd.DataFrame): The DataFrame to modify. Must include the columns specified in response_column_names.
        response_column_names (list[str]): The names of columns in categorized_data containing open-ended text responses.
        categories_list (list[str]): The list of categories for which columns will be added to the DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with added category columns for each response column.
    """

    for response_column in response_column_names:
        for category in categories_list:
            col_name = f"{category}_{response_column}"
            if category == "Uncategorized":
                categorized_data[col_name] = 1
            else:
                categorized_data[col_name] = 0
    return categorized_data


def categorize_missing_data_for_response_column(
    categorized_data: pd.DataFrame,
    response_column: str,
    categories_list: list[str],
) -> pd.DataFrame:
    """
    Modifies a DataFrame by marking category columns as missing (pd.NA) for rows where the corresponding response column has missing data.

    This function checks the specified response column for missing value (pd.NA) rows, and updates the corresponding category
    columns to pd.NA for those rows. This is done for all categories in categories_list.

    Args:
        categorized_data (pd.DataFrame): The DataFrame to modify. Must include the columns specified in response_column.
        response_column (str): The name of the column in categorized_data to check for missing data.
        categories_list (list[str]): The list of categories that have corresponding columns in the categorized_data DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with updated category columns to reflect missing data in the response column.
    """

    def _is_missing(value):
        return pd.isna(value)

    # Boolean mask where each row is True if all elements are missing
    missing_data_mask = categorized_data[response_column].map(_is_missing)

    for category in categories_list:
        col_name = f"{category}_{response_column}"
        categorized_data.loc[missing_data_mask, col_name] = pd.NA

    return categorized_data


def categorize_responses_for_response_column(
    response: str,
    categories: list[str] | str,
    response_column: str,
    categorized_data: pd.DataFrame,
    is_multicode: bool,
) -> None:
    """
    Categorizes responses in a DataFrame column based on the specified response and categories.

    This function identifies rows in the response_column that match the given response. It then sets the corresponding
    category columns to 1 and the 'Uncategorized' column to 0 for those rows.

    If is_multicode is True, multiple category columns can be modified; otherwise, only a single category column is modified.

    Args:
        response (str): The response text to match against in the categorized_data DataFrame's response_column.
        categories (list[str] | str): The category or categories corresponding to the response. Must be a list if is_multicode is True.
        response_column (str): The name of the column in categorized_data that contains the responses to categorize.
        categorized_data (pd.DataFrame): The DataFrame to modify. Must include the response_column.
        is_multicode (bool): If True, allows categorization across multiple category columns. Otherwise, restricts categorization to a single column.

    Note:
        - If a specified category does not exist in the DataFrame, a warning is printed.
        - This function modifies the categorized_data DataFrame in place and does not return a value.
    """

    # Boolean mask for rows in response_column containing selected response
    mask = categorized_data[response_column] == response

    if is_multicode:
        for category in categories:
            col_name = f"{category}_{response_column}"

            if col_name in categorized_data.columns:
                categorized_data.loc[mask, f"Uncategorized_{response_column}"] = 0
                categorized_data.loc[mask, col_name] = 1
            else:
                print(f"\nUnknown category: {category} for response: {response}")
    else:
        col_name = f"{categories}_{response_column}"

        if col_name in categorized_data.columns:
            categorized_data.loc[mask, f"Uncategorized_{response_column}"] = 0
            categorized_data.loc[mask, col_name] = 1
        else:
            print(f"\nUnknown category: {categories} for response: {response}")
