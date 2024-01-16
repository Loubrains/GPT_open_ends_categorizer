from openai import OpenAI
import json
import asyncio


def call_gpt(
    client: OpenAI,
    user_prompt: str,
):
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": user_prompt}], model="gpt-4-1106-preview"
        )
        content = completion.choices[0].message.content

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        content = "Error"
        raise

    return content


def GPT_generate_categories_list(
    client: OpenAI,
    question: str,
    responses_sample: list[str],
    number_of_categories: int = 20,
):
    user_prompt = f"""List the {number_of_categories} most relevant thematic categories for this sample of survey responses.
    Return only a JSON list of category names, in the format: `["name1", "name2", ...]`
    
    Question:
    `{question}`
    
    Responses:
    ```
    {responses_sample}
    ```"""

    try:
        output = call_gpt(client, user_prompt)
        output_cleaned = output.strip().replace("json", "").replace("`", "").replace("\n", "")  # type: ignore
        output_categories_list = json.loads(output_cleaned)

        # Check if loaded json is a list
        if not isinstance(output_categories_list, list):
            raise ValueError(f"Output format is not as expected:\n{output_categories_list}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        output_categories_list = "Error"
        raise

    return output_categories_list


def validate_gpt_categorized_output(output_categories, categories_list, is_multicode):
    def _check_categories_are_valid(categories_to_check):
        if any(category not in categories_list for category in categories_to_check):
            raise ValueError(
                f"Unexpected category returned in output_categories:\n{categories_to_check}"
            )

    if is_multicode:
        # Check if output is list of lists
        if not isinstance(output_categories, list) or not all(
            isinstance(category, list) for category in output_categories
        ):
            raise ValueError(f"Output format is not as expected:\n{output_categories}")

        for response_categories in output_categories:
            _check_categories_are_valid(response_categories)

    else:
        _check_categories_are_valid(output_categories)


async def GPT_categorize_response_batch(
    client: OpenAI,
    question: str,
    responses_batch: list[str],
    categories_list: list[str],
    max_retries=3,
    is_multicode=False,
):
    combined_responses = "\n".join(
        [f"{i+1}: {response}" for i, response in enumerate(responses_batch)]
    )
    combined_categories_list = "\n".join(categories_list)

    if is_multicode:
        output_format = '`[["name 1", "name 2", ...], ["name 1", "name 2", ...], ...]`'
        multiple_categories_text = "or multiple "
        output_format_text = "a list of category names "
    else:
        output_format = '`["name 1", "name 2", ...]`'
        multiple_categories_text = ""
        output_format_text = "a category name "

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

    for attempt in range(max_retries):
        try:
            output = call_gpt(client, user_prompt)
            output_cleaned = output.strip().replace("json", "").replace("`", "").replace("\n", "")  # type: ignore
            output_categories = json.loads(output_cleaned)

            validate_gpt_categorized_output(output_categories, categories_list, is_multicode)

            return output_categories

        except Exception as e:
            print(
                f"""\nAn error occurred:\n{e}
            Responses in batch:\n{responses_batch}
            Retrying attempt {attempt + 1}/{max_retries}..."""
            )

    # Error case
    print(f"\nMax retries reached for responses:\n{responses_batch}")
    if is_multicode:
        output_categories = [["Error"]] * len(responses_batch)
    else:
        output_categories = ["Error"] * len(responses_batch)

    return output_categories


def create_batches(data, batch_size=3):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


async def process_batches(
    client,
    question,
    categories_list,
    responses,
    batch_size,
    max_retries,
    is_multicode,
):
    categorized_dict = {}
    batches = list(create_batches(list(responses), batch_size))
    tasks = []

    for batch in batches:
        task = asyncio.create_task(
            GPT_categorize_response_batch(
                client, question, batch, categories_list, max_retries, is_multicode
            )
        )
        tasks.append(task)

    for i, task in enumerate(tasks):
        output_categories = await task
        for response, categories in zip(batches[i], output_categories):
            categorized_dict[response] = categories

    return categorized_dict


async def GPT_categorize_responses_main(
    client, question, unique_responses, categories_list, batch_size, max_retries, is_multicode
):
    categorized_dict = await process_batches(
        client, question, categories_list, unique_responses, batch_size, max_retries, is_multicode
    )
    return categorized_dict
