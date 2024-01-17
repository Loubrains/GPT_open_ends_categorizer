from openai import AsyncOpenAI
import json
import asyncio


async def call_gpt(
    client: AsyncOpenAI,
    user_prompt: str,
) -> str | None:
    try:
        completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": user_prompt}], model="gpt-4-1106-preview"
        )
        content = completion.choices[0].message.content

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        content = "Error"
        raise

    return content


async def GPT_generate_categories_list(
    client: AsyncOpenAI,
    question: str,
    responses_sample: list[str],
    number_of_categories: int = 20,
    max_retries: int = 5,
) -> list[str]:
    user_prompt = f"""List the {number_of_categories} most relevant thematic categories for this sample of survey responses.
    Return only a JSON list of category names, in the format: `["name1", "name2", ...]`
    
    Question:
    `{question}`
    
    Responses:
    ```
    {responses_sample}
    ```"""
    for attempt in range(max_retries):
        try:
            output = await call_gpt(client, user_prompt)
            output_cleaned = output.strip().replace("json", "").replace("`", "").replace("\n", "")  # type: ignore
            output_categories_list = json.loads(output_cleaned)

            # Check if loaded json is a list
            if not isinstance(output_categories_list, list):
                raise ValueError(f"Output format is not as expected:\n{output_categories_list}")

            return output_categories_list

        except Exception as e:
            print(
                f"""\nAn error occurred:\n{e}
            Retrying attempt {attempt + 1}/{max_retries}..."""
            )

    print("\nMax retries reached for responses")
    output_categories_list = ["Error"]

    return output_categories_list


def validate_gpt_categorized_output(output_categories, categories_list, is_multicode):
    def _check_elements_of_list_are_strings(list_to_check):
        for element in list_to_check:
            if not isinstance(element, str):
                raise ValueError(
                    f"""Output format is not a as expected (expected string)
                    output_categories:\n{list_to_check}
                    element:\n{element}"""
                )

    def _check_categories_are_valid(categories_to_check):
        for category in categories_to_check:
            if category not in categories_list:
                raise ValueError(
                    f"""Unexpected category returned in output_categories
                    output_categories:\n{categories_to_check}
                    unexpected_category:\n{category}"""
                )

    # Check if overall output is a list
    if not isinstance(output_categories, list):
        raise ValueError(
            f"Output format is not a as expected (expected list [..., ...]):\n{output_categories}\n"
        )

    if is_multicode:
        # Check if all elements themselvers are lists
        if not all(isinstance(element, list) for element in output_categories):
            raise ValueError(
                f"Output format is not as expected (expected list of lists [[...], [...], ...]):\n{output_categories}"
            )

        for response_categories in output_categories:
            _check_elements_of_list_are_strings(response_categories)
            _check_categories_are_valid(response_categories)

    else:
        _check_elements_of_list_are_strings(output_categories)
        _check_categories_are_valid(output_categories)


def create_user_prompt_for_gpt_categorization(question, responses, categories_list, is_multicode):
    combined_responses = "\n".join([f"{i+1}: {response}" for i, response in enumerate(responses)])
    combined_categories_list = "\n".join(categories_list)

    if is_multicode:
        multiple_categories_text = "or multiple "
        output_format_text = "a list of category names "
        output_format = '`[["category 1 for response 1", "category 2 for response 1", ...], ["category 1 for response 2", "category 2 for response 2", ...], ...]`'
    else:
        multiple_categories_text = ""
        output_format_text = "a category name "
        output_format = '`["category for response 1", "category for response 2", ...]`'

    user_prompt = f"""Categorize these responses to the following survey question using one {multiple_categories_text}of the provided categories.
    Return only a JSON list where each element is {output_format_text}for each response, in the format: {output_format}.
    
    Question:
    `{question}`
    
    Responses:
    `{combined_responses}`
    
    categories:
    ```
    {combined_categories_list}
    ```"""

    return user_prompt


async def GPT_categorize_responses(
    client: AsyncOpenAI,
    question: str,
    responses: list[str],
    categories_list: list[str],
    max_retries: int = 3,
    is_multicode: bool = False,
) -> list[str] | list[list[str]]:
    user_prompt = create_user_prompt_for_gpt_categorization(
        question, responses, categories_list, is_multicode
    )

    for attempt in range(max_retries):
        try:
            output = await call_gpt(client, user_prompt)
            output_cleaned = output.strip().replace("json", "").replace("`", "").replace("\n", "")  # type: ignore
            output_categories = json.loads(output_cleaned)

            validate_gpt_categorized_output(output_categories, categories_list, is_multicode)

            return output_categories

        except Exception as e:
            print(
                f"""\nAn error occurred:\n{e}
            Responses:\n{responses}
            Retrying attempt {attempt + 1}/{max_retries}..."""
            )

    print(f"\nMax retries reached for responses:\n{responses}")
    if is_multicode:
        output_categories = [["Error"]] * len(responses)
    else:
        output_categories = ["Error"] * len(responses)

    return output_categories


def create_batches(data: list[str], batch_size: int = 3):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


async def GPT_categorize_response_batches_main(
    client: AsyncOpenAI,
    question: str,
    responses: list[str] | set[str],
    categories_list: list[str],
    batch_size: int = 3,
    max_retries: int = 5,
    is_multicode: bool = False,
) -> dict[str, str] | dict[str, list[str]]:
    categorized_dict = {}
    batches = list(create_batches(list(responses), batch_size))
    tasks = []

    for batch in batches:
        task = GPT_categorize_responses(
            client, question, batch, categories_list, max_retries, is_multicode
        )
        tasks.append(task)

    output_categories = await asyncio.gather(*tasks)
    print("output_categories:", output_categories)

    for i, categories_in_batch in enumerate(output_categories):
        for response, categories in zip(batches[i], categories_in_batch):
            categorized_dict[response] = categories

    return categorized_dict
