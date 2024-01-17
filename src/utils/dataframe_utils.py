import pandas as pd


def construct_default_categorized_dataframe(
    categorized_data: pd.DataFrame, response_column_names: list[str], categories_list: list[str]
):
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
):
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
