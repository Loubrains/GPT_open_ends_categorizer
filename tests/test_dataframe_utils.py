import pytest
import pandas as pd
from src.utils import dataframe_utils


@pytest.mark.parametrize(
    "input_data, response_column_names, categories_list, expected_output",
    [
        # Standard case with multiple response columns and categories
        (
            pd.DataFrame({"uuid": ["a", "b"], "response1": ["c", "d"], "response2": ["e", "f"]}),
            ["response1", "response2"],
            ["Category1", "Category2", "Uncategorized"],
            pd.DataFrame(
                {
                    "uuid": ["a", "b"],
                    "response1": ["c", "d"],
                    "response2": ["e", "f"],
                    "Category1_response1": pd.array([0, 0], dtype=pd.Int64Dtype()),
                    "Category2_response1": pd.array([0, 0], dtype=pd.Int64Dtype()),
                    "Uncategorized_response1": pd.array([1, 1], dtype=pd.Int64Dtype()),
                    "Category1_response2": pd.array([0, 0], dtype=pd.Int64Dtype()),
                    "Category2_response2": pd.array([0, 0], dtype=pd.Int64Dtype()),
                    "Uncategorized_response2": pd.array([1, 1], dtype=pd.Int64Dtype()),
                }
            ),
        )
    ],
)
def test_construct_default_categorized_dataframe(
    input_data, response_column_names, categories_list, expected_output
):
    result_df = dataframe_utils.construct_default_categorized_dataframe(
        input_data, response_column_names, categories_list
    )

    pd.testing.assert_frame_equal(result_df, expected_output)


@pytest.mark.parametrize(
    "input_data, response_column, categories_list, expected_output",
    [
        # Multiple response columns, missing data in one response column
        (
            pd.DataFrame(
                {
                    "response1": [pd.NA, "a", "b", pd.NA],
                    "response2": ["c", pd.NA, pd.NA, "d"],
                    "Category1_response1": pd.array([0, 1, 0, 0], dtype=pd.Int64Dtype()),
                    "Category2_response1": pd.array([0, 0, 1, 0], dtype=pd.Int64Dtype()),
                    "Uncategorized_response1": pd.array([1, 0, 0, 1], dtype=pd.Int64Dtype()),
                    "Category1_response2": pd.array([1, 0, 0, 0], dtype=pd.Int64Dtype()),
                    "Category2_response2": pd.array([0, 0, 0, 1], dtype=pd.Int64Dtype()),
                    "Uncategorized_response2": pd.array([0, 1, 1, 0], dtype=pd.Int64Dtype()),
                }
            ),
            "response1",
            ["Category1", "Category2", "Uncategorized"],
            pd.DataFrame(
                {
                    "response1": [pd.NA, "a", "b", pd.NA],
                    "response2": ["c", pd.NA, pd.NA, "d"],
                    "Category1_response1": pd.array([pd.NA, 1, 0, pd.NA], dtype=pd.Int64Dtype()),
                    "Category2_response1": pd.array([pd.NA, 0, 1, pd.NA], dtype=pd.Int64Dtype()),
                    "Uncategorized_response1": pd.array(
                        [pd.NA, 0, 0, pd.NA], dtype=pd.Int64Dtype()
                    ),
                    "Category1_response2": pd.array([1, 0, 0, 0], dtype=pd.Int64Dtype()),
                    "Category2_response2": pd.array([0, 0, 0, 1], dtype=pd.Int64Dtype()),
                    "Uncategorized_response2": pd.array([0, 1, 1, 0], dtype=pd.Int64Dtype()),
                }
            ),
        ),
        # No missing data in the response column
        (
            pd.DataFrame(
                {
                    "response1": ["a", "b", "c"],
                    "Category1_response1": pd.array([1, 0, 0], dtype=pd.Int64Dtype()),
                    "Category2_response1": pd.array([0, 0, 1], dtype=pd.Int64Dtype()),
                    "Uncategorized_response1": pd.array([0, 1, 0], dtype=pd.Int64Dtype()),
                }
            ),
            "response1",
            ["Category1", "Category2", "Uncategorized"],
            pd.DataFrame(
                {
                    "response1": ["a", "b", "c"],
                    "Category1_response1": pd.array([1, 0, 0], dtype=pd.Int64Dtype()),
                    "Category2_response1": pd.array([0, 0, 1], dtype=pd.Int64Dtype()),
                    "Uncategorized_response1": pd.array([0, 1, 0], dtype=pd.Int64Dtype()),
                }
            ),
        ),
    ],
)
def test_categorize_missing_data_for_response_column(
    input_data, response_column, categories_list, expected_output
):
    result_df = dataframe_utils.categorize_missing_data_for_response_column(
        input_data, response_column, categories_list
    )

    pd.testing.assert_frame_equal(result_df, expected_output)


@pytest.mark.parametrize(
    "input_data, response, categories, response_column, is_multicode, expected_output",
    [
        # Single category, not multicode
        (
            pd.DataFrame(
                {
                    "response1": ["yes", "no", "yes"],
                    "Category1_response1": [0, 0, 0],
                    "Uncategorized_response1": [1, 1, 1],
                }
            ),
            "yes",
            "Category1",
            "response1",
            False,
            pd.DataFrame(
                {
                    "response1": ["yes", "no", "yes"],
                    "Category1_response1": [1, 0, 1],
                    "Uncategorized_response1": [0, 1, 0],
                }
            ),
        ),
        # Multiple categories, multicode
        (
            pd.DataFrame(
                {
                    "response1": ["yes", "no", "yes"],
                    "Category1_response1": [0, 0, 0],
                    "Category2_response1": [0, 0, 0],
                    "Uncategorized_response1": [1, 1, 1],
                }
            ),
            "yes",
            ["Category1", "Category2"],
            "response1",
            True,
            pd.DataFrame(
                {
                    "response1": ["yes", "no", "yes"],
                    "Category1_response1": [1, 0, 1],
                    "Category2_response1": [1, 0, 1],
                    "Uncategorized_response1": [0, 1, 0],
                }
            ),
        ),
        # Multiple response columns present in dataframe
        (
            pd.DataFrame(
                {
                    "response1": ["yes", "no", "yes"],
                    "response2": ["no", "yes", "no"],
                    "Category1_response1": [0, 0, 0],
                    "Category2_response1": [0, 0, 0],
                    "Uncategorized_response1": [1, 1, 1],
                    "Category1_response2": [0, 0, 0],
                    "Category2_response2": [0, 0, 0],
                    "Uncategorized_response2": [1, 1, 1],
                }
            ),
            "no",
            "Category1",
            "response2",
            False,
            pd.DataFrame(
                {
                    "response1": ["yes", "no", "yes"],
                    "response2": ["no", "yes", "no"],
                    "Category1_response1": [0, 0, 0],
                    "Category2_response1": [0, 0, 0],
                    "Uncategorized_response1": [1, 1, 1],
                    "Category1_response2": [1, 0, 1],
                    "Category2_response2": [0, 0, 0],
                    "Uncategorized_response2": [0, 1, 0],
                }
            ),
        ),
    ],
)
def test_categorize_responses_for_response_column(
    input_data, response, categories, response_column, is_multicode, expected_output
):
    dataframe_utils.categorize_responses_for_response_column(
        response, categories, response_column, input_data, is_multicode
    )

    pd.testing.assert_frame_equal(input_data, expected_output)
