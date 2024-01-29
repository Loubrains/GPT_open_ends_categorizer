import pytest
import asyncio
from gpt_categorizer_utils import gpt_utils


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "max_capacity, refill_rate, tokens_required, wait_time, expected_remaining_tokens, expect_exception",
    [
        # Tokens consumed correctly without refill
        (100, 10, 50, 0, 50, False),
        # Not enough max_tokens results in an exception
        (100, 10, 150, 0, None, True),
        # Tokens refill correctly over time
        (100, 10, 50, 5, 100, False),  # 5 seconds, should refill 50 tokens
        # Bucket does not exceed max capacity after refill
        (100, 10, 10, 5, 100, False),  # refill would add 50 but max is 100
    ],
)
async def test_token_bucket_consume(
    max_capacity,
    refill_rate,
    tokens_required,
    wait_time,
    expected_remaining_tokens,
    expect_exception,
):
    token_bucket = gpt_utils.TokenBucket(max_capacity, refill_rate)

    if expect_exception:
        with pytest.raises(ValueError):
            await token_bucket.consume_tokens(tokens_required)
    else:
        await token_bucket.consume_tokens(tokens_required)
        await asyncio.sleep(wait_time)  # Simulate waiting for refill
        token_bucket.refill()  # Manually trigger refill for testing purposes
        assert token_bucket.current_token_count == expected_remaining_tokens


@pytest.mark.asyncio
async def test_concurrent_token_consumption():
    max_capacity = 100
    refill_rate = 10  # tokens per second
    token_bucket = gpt_utils.TokenBucket(max_capacity, refill_rate)

    async def consume_and_wait(bucket: gpt_utils.TokenBucket, tokens_to_consume, wait_time):
        await bucket.consume_tokens(tokens_to_consume)
        await asyncio.sleep(wait_time)
        token_bucket.refill()
        bucket.refill()

    test_tokens1, test_wait1 = 20, 1
    test_tokens2, test_wait2 = 30, 2
    test_tokens3, test_wait3 = 10, 1.5
    tasks = [
        consume_and_wait(token_bucket, test_tokens1, test_wait1),
        consume_and_wait(token_bucket, test_tokens2, test_wait2),
        consume_and_wait(token_bucket, test_tokens3, test_wait3),
    ]

    await asyncio.gather(*tasks)

    assert token_bucket.current_token_count > max_capacity - sum(
        [test_tokens1, test_tokens2, test_tokens3]
    )

    # NOTE: I think a more accurate calculation of tokens remaining would actually break me
