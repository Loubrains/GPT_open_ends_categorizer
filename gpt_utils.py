from openai import OpenAI
import json
import asyncio


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
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": user_prompt}], model="gpt-4-1106-preview"
        )
        output = completion.choices[0].message.content
        output_cleaned = output.strip().replace("json", "").replace("`", "").replace("\n", "")  # type: ignore
        output_categories_list = json.loads(output_cleaned)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        output_categories_list = "Error"
        raise

    return output_categories_list


async def GPT_categorize_responses_main(
    client, question, unique_responses, categories_list, batch_size, max_retries, is_multicode
):
    categorized_dict = await process_batches(
        client, question, categories_list, unique_responses, batch_size, max_retries, is_multicode
    )
    return categorized_dict


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
        output_categories_list = await task
        for response, categories in zip(batches[i], output_categories_list):
            categorized_dict[response] = categories

    return categorized_dict


def create_batches(data, batch_size=3):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


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
        user_prompt = f"""Categorize these responses to the following survey question using one or multiple of the provided categories.
        Return only a JSON list where each element is a list of category names for each response, in the format: `[["name 1", "name 2", ...], ["name 1", "name 2", ...], ...]`.
        
        Question:
        `{question}`
        
        Responses:
        `{combined_responses}`
        
        categories:
        ```
        {combined_categories_list}
        ```"""
    else:
        user_prompt = f"""Categorize these responses to the following survey question using one or multiple of the provided categories.
        Return only a JSON list where each element is a list of category names for each response, in the format: `[["name 1", "name 2", ...], ["name 1", "name 2", ...], ...]`.
        
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
            loop = asyncio.get_event_loop()
            completion = await loop.run_in_executor(
                None,  # Executor
                lambda: client.chat.completions.create(
                    messages=[{"role": "user", "content": user_prompt}], model="gpt-4-1106-preview"
                ),
            )
            output = completion.choices[0].message.content
            output_cleaned = output.strip().replace("json", "").replace("`", "").replace("\n", "")  # type: ignore
            output_categories_list = json.loads(output_cleaned)

            # Validating output
            if not isinstance(output_categories_list, list) or not all(
                isinstance(category, list) for category in output_categories_list
            ):
                raise ValueError(f"Output format is not as expected:\n{output_categories_list}")

            for response_categories in output_categories_list:
                if any(category not in categories_list for category in response_categories):
                    raise ValueError(
                        f"Unexpected category returned in output_categories_list:\n{output_categories_list}"
                    )

            return output_categories_list  # type: ignore

        except Exception as e:
            print(
                f"""\nAn error occurred:\n{e}
            Responses in batch:\n{responses_batch}
            Retrying attempt {attempt + 1}/{max_retries}..."""
            )

    # Error case
    print(f"\nMax retries reached for responses:\n{responses_batch}")
    if is_multicode:
        output_categories_list = [["Error"]] * len(responses_batch)
    else:
        output_categories_list = ["Error"] * len(responses_batch)

    return output_categories_list
