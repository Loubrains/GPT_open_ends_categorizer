import pytest
import pandas as pd
from src.utils import gpt_utils


@pytest.mark.parametrize(
    "output_categories, categories_list, is_multicode, expect_exception",
    [
        # is_multicode is False, correct format, valid categories
        (["sports", "health"], ["sports", "health", "finance"], False, False),
        # is_multicode is True, correct format, valid categories
        ([["sports"], ["health"]], ["sports", "health", "finance"], True, False),
        # is_multicode is False, invalid category
        (["sports", "unknown"], ["sports", "health", "finance"], False, True),
        # is_multicode is True, invalid category
        ([["sports"], ["unknown"]], ["sports", "health", "finance"], True, True),
        # is_multicode is False, incorrect format (not a list of strings)
        (["sports", 123], ["sports", "health", "finance"], False, True),
        # is_multicode is True, incorrect format (not a list of list of strings)
        ([["sports"], "health"], ["sports", "health", "finance"], True, True),
    ],
)
def test_validate_gpt_categorized_output(
    output_categories, categories_list, is_multicode, expect_exception
):
    if expect_exception:
        with pytest.raises(ValueError):
            gpt_utils.validate_gpt_categorized_output(
                output_categories, categories_list, is_multicode
            )
    else:
        gpt_utils.validate_gpt_categorized_output(output_categories, categories_list, is_multicode)
