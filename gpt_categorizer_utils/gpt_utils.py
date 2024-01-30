"""
Utilities for interacting with the OpenAI GPT model to categorize survey responses.

This module allows for the asynchronous sending of prompts to the GPT model, categorization of survey responses into thematic categories, 
validation of categorization results, and batch processing of responses for efficient categorization.
It implements token and request limiting using the `TokenBucket` class, backoff decorator, and uses `tiktoken` for accurate token counting.
https://cookbook.openai.com/examples/how_to_handle_rate_limits
https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
Another potential option for limiting rate of async tasks:
https://stackoverflow.com/questions/48483348/how-to-limit-concurrency-with-python-asyncio

Functions:
    `call_gpt`: Asynchronously sends a user prompt to the GPT model and retrieves the completion. It manages token and request rate limiting using the TokenBucket class
    `gpt_generate_categories_list`: Asynchronously generates a list of thematic categories relevant to a sample of survey responses.
    `validate_gpt_categorized_output`: Validates the GPT output received by the categorizer.
    `create_user_prompt_for_gpt_categorization`: Creates a user prompt to send to the GPT model to categorize survey question responses.
    `gpt_categorize_responses`: Asynchronously categorizes a list of responses using the GPT model.
    `gpt_categorize_response_batches_main`: Asynchronously sends batches of survey responses to be categorized using the GPT model.
    
Note: A potential future update includes calling the GPT model with JSON mode for structured responses.
"""

### NOTE: potential future update to these utils: call gpt with JSON mode
### Put the following in the client.chat.completions.create() arguments:
### `response_format={ "type": "json_object" }`
### Make sure the prompt specifies the JSON structure, and then parse the output

import openai
from openai import AsyncOpenAI
import json
import asyncio
import tiktoken
import backoff
import time
from .general_utils import create_batches
import logging

logger = logging.getLogger(__name__)

### NOTE: SET USAGE LIMITS HERE
REQUESTS_PER_MINUTE = 450  # Actual limit is 500
TOKENS_PER_MINUTE = 140000  # Actual limit is 150000

# For counting tokens
encoding = tiktoken.encoding_for_model("gpt-4")


class TokenBucket:
    """
    TokenBucket is a rate-limiting mechanism using the token bucket algorithm.
    It enforces a maximum capacity of tokens that can be consumed over a specific time period.
    Extended to handle requests limiting too.

    Attributes:
        max_capacity (int): The maximum number of tokens in the bucket.
        refill_rate_per_second (float): The rate at which tokens are added to the bucket per second.
        current_token_count (float): The current number of tokens in the bucket.
        timestamp_of_last_refill (float): The timestamp when the bucket was last refilled.
    """

    def __init__(self, max_capacity: int, refill_rate_per_second: float):
        self.max_capacity = max_capacity
        self.refill_rate_per_second = refill_rate_per_second
        self.current_token_count = max_capacity
        self.timestamp_of_last_refill = time.time()

    async def consume_tokens(self, tokens_required: int):
        """
        Consumes a specified number of tokens from the bucket. If enough tokens are not available, waits asynchronously until they are refilled.

        Args:
            tokens_required (int): The number of tokens to consume from the bucket.

        Raises:
            ValueError: If tokens_required exceeds the maximum capacity of the bucket.
        """
        if tokens_required > self.max_capacity:
            raise ValueError(
                f"Requested tokens ({tokens_required}) exceed the maximum capacity of the bucket ({self.max_capacity})."
            )

        while True:
            self.refill()
            if self.current_token_count >= tokens_required:
                self.current_token_count -= tokens_required
                break
            else:
                logger.debug("Token limit per minute exceeded. Exponential backoff.")
                raise ValueError("Token limit per minute exceeded. Waiting.")

    async def consume_request(self):
        """
        Consumes a token for a request. If no tokens are available for requests, waits asynchronously until they are refilled.
        """
        await self.consume_tokens(1)

    def refill(self):
        """
        Refills the tokens in the bucket based on the refill rate and the time elapsed since the last refill.
        """
        current_time = time.time()
        time_since_last_refill = current_time - self.timestamp_of_last_refill
        tokens_to_add = time_since_last_refill * self.refill_rate_per_second
        # Refill only up to max_capacity
        self.current_token_count = min(self.max_capacity, self.current_token_count + tokens_to_add)
        self.timestamp_of_last_refill = current_time


# Instantiate TokenBuckets for token and request limiting
token_bucket = TokenBucket(TOKENS_PER_MINUTE, TOKENS_PER_MINUTE / 60)
request_bucket = TokenBucket(REQUESTS_PER_MINUTE, REQUESTS_PER_MINUTE / 60)


@backoff.on_exception(backoff.expo, (openai.RateLimitError, ValueError), jitter=backoff.full_jitter)
async def call_gpt(
    client: AsyncOpenAI,
    user_prompt: str,
) -> str | None:
    """
    Asynchronously sends a user prompt to the GPT-4 model and retrieves the completion.
    Tokens usage is managed with the token bucket algrithm.
    Failed requests are exponentially backed off and jittered using the backoff library.

    Args:
        client (AsyncOpenAI): The client instance used to communicate with the GPT model.
        user_prompt (str): The prompt text to send to the model.

    Returns:
        str | None: The content of the model's completion, or None if an error occurs.

    Raises:
        Raises an exception if the API call fails.
    """

    estimated_tokens = len(encoding.encode(user_prompt)) + 10  # add extra for the system message
    # Limit tokens and requests per minute
    await token_bucket.consume_tokens(estimated_tokens)
    await request_bucket.consume_request()

    try:
        completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": user_prompt}], model="gpt-4-turbo-preview"
        )
        content = completion.choices[0].message.content
        if content:
            logger.debug(f"Success: {content}")

        if content:
            response_tokens = len(encoding.encode(content))
            await token_bucket.consume_tokens(response_tokens)
        else:
            raise ValueError("No completion returned")

    except Exception as e:
        logger.error(f"\nAn error occurred: {e}")
        content = "Error"
        raise

    return content


async def gpt_generate_categories_list(
    client: AsyncOpenAI,
    question: str,
    responses_sample: list[str],
    number_of_categories: int = 20,
    max_retries: int = 5,
) -> list[str]:
    """
    Asynchronously generates a list of thematic categories relevant to a sample of survey responses.

    Args:
        client (AsyncOpenAI): The client instance used to communicate with the GPT model.
        question (str): The survey question related to the responses.
        responses_sample (list[str]): A sample of survey responses.
        number_of_categories (int): The number of categories to generate. Defaults to 20.
        max_retries (int): The maximum number of retries for the API call. Defaults to 5.

    Returns:
        list[str]: A list of generated category names.

    Notes:
        - Expects the GPT model to return a JSON list of category names.
        - Retries the API call up to max_retries times if errors occur.
    """

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
            logger.info(
                f"""\nAn error occurred:\n{e}
            Retrying attempt {attempt + 1}/{max_retries}..."""
            )

    # Error case
    logger.info("\nMax retries reached for responses")
    output_categories_list = ["Error"]

    return output_categories_list


def validate_gpt_categorized_output(output_categories, categories_list, is_multicode):
    """
    Validates the GPT output received by the the categorizer.
    Checks the that the format is correct (i.e. a list, whose elements are strings (is_multicode = False) or a list of strings (is_multicode = True)).
    Checks the categories are valid.

    Args:
        output_categories (list[str] | list[list[str]]): The GPT output to validate.
        categories_list (list[str]): The list of valid category names.
        is_multicode (bool): If True, the output should be a list of lists of strings. If False, the output should be a list of strings.

    Raises:
        ValueError: If the output_categories format is incorrect or contains invalid categories.
    """

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
    """
    Creates a user prompt to send to the GPT model to categorize survey question responses.

    Args:
        question (str): The survey question related to the responses.
        responses (list[str]): The list of responses to categorize.
        categories_list (list[str]): The list of valid category names.
        is_multicode (bool): Changes the prompt to explain whether multiple categories can be used, and the expected output format.

    Returns:
        str: The user prompt formatted for the GPT model.
    """

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


async def gpt_categorize_responses(
    client: AsyncOpenAI,
    question: str,
    responses: list[str],
    categories_list: list[str],
    max_retries: int = 5,
    is_multicode: bool = False,
) -> list[str] | list[list[str]]:
    """
    Asynchronously categorizes a list of responses using the GPT model.
    `is_multicode = True` allows multiple categories to be associated with each response.

    Args:
        client (AsyncOpenAI): The client instance used to communicate with the GPT model.
        question (str): The survey question related to the responses.
        responses (list[str]): The list of responses to categorize.
        categories_list (list[str]): The list of valid category names.
        max_retries (int): The maximum number of retries for the API call. Defaults to 5.
        is_multicode (bool): If True, each response can belong to multiple categories. Defaults to False.

    Returns:
        list[str] | list[list[str]]:
            - If 'is_multi' is False, returns a list of categories, where each category is assigned to a single response in the same order they were fed in.
            - If 'is_multi' is True, returns a list of lists of categories, where each inner list contains multiple categories assigned to the corresponding response.

    Notes:
        - Retries the API call up to max_retries times if errors occur.
    """

    user_prompt = create_user_prompt_for_gpt_categorization(
        question, responses, categories_list, is_multicode
    )

    # logger.debug(user_prompt)

    for attempt in range(max_retries):
        try:
            output = await call_gpt(client, user_prompt)
            output_cleaned = output.strip().replace("json", "").replace("`", "").replace("\n", "")  # type: ignore
            output_categories = json.loads(output_cleaned)

            validate_gpt_categorized_output(output_categories, categories_list, is_multicode)

            return output_categories

        except Exception as e:
            logger.info(
                f"""\nAn error occurred:\n{e}
            Responses:\n{responses}
            Retrying attempt {attempt + 1}/{max_retries}..."""
            )

    # Error case
    logger.info(f"\nMax retries reached for responses:\n{responses}")
    if is_multicode:
        output_categories = [["Error"]] * len(responses)
    else:
        output_categories = ["Error"] * len(responses)

    return output_categories


async def gpt_categorize_response_batches_main(
    client: AsyncOpenAI,
    question: str,
    responses: list[str] | set[str],
    categories_list: list[str],
    batch_size: int = 3,
    max_retries: int = 5,
    is_multicode: bool = False,
) -> dict[str, str] | dict[str, list[str]]:
    """
    Asynchronously sends batches of survey responses to be categorized using the GPT model,
    and constructs a codeframe from the output (i.e. dictionary of responses to categories)

    Args:
        client (AsyncOpenAI): The client instance used to communicate with the GPT model.
        question (str): The survey question related to the responses.
        responses (list[str] | set[str]): The responses to categorize.
        categories_list (list[str]): The list of valid category names.
        batch_size (int): The number of responses to process in each batch. Defaults to 3.
        max_retries (int): The maximum number of retries for the API call. Defaults to 5.
        is_multicode (bool): If True, each response can belong to multiple categories. Defaults to False.

    Returns:
        dict[str, str] | dict[str, list[str]]: A dictionary mapping each response to its category (if is_multicode if False) or categories (if is_multicode if True).

    Notes:
        - Processes the responses in batches for efficiency.
        - Retries the API call for a batch up to max_retries times if errors occur.
    """

    categorized_dict = {}
    batches = list(create_batches(list(responses), batch_size))
    tasks = []

    # Create async gpt tasks
    for batch in batches:
        task = gpt_categorize_responses(
            client, question, batch, categories_list, max_retries, is_multicode
        )
        logger.debug(task)
        tasks.append(task)

    output_categories = await asyncio.gather(*tasks)

    # Construct codeframe (dict of responses to categories)
    for i, categories_in_batch in enumerate(output_categories):
        for response, categories in zip(batches[i], categories_in_batch):
            categorized_dict[response] = categories

    return categorized_dict
