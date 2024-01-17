def validate_gpt_categorized_output(output_categories, categories_list, is_multicode):
    def _check_categories_are_valid(categories_to_check):
        if any(category not in categories_list for category in categories_to_check):
            raise ValueError(
                f"Unexpected category returned in output_categories:\n{categories_to_check}"
            )

    # Check if output is a list
    if not isinstance(output_categories, list):
        raise ValueError(
            f"Output format is not a as expected (expected list [..., ...]):\n{output_categories}\n"
        )

    if is_multicode:
        # Check if each element is itself a list
        if not all(isinstance(category, list) for category in output_categories):
            raise ValueError(
                f"Output format is not as expected (expected list of lists [[...], [...], ...]):\n{output_categories}"
            )

        for response_categories in output_categories:
            _check_categories_are_valid(response_categories)

    else:
        _check_categories_are_valid(output_categories)


output_categories = [["yo", "yo"]]
categories_list = ["yoo"]
is_multicode = True

validate_gpt_categorized_output(output_categories, categories_list, is_multicode)
