"""
Configuration file containing user-defined parameters for interacting with the OpenAI API, such as rate limiting.

Configuration Variables:
    `BATCH_SIZE`: Number of responses to send to GPT per request.
    `MAX_RETRIES`: Number of retry attempts for GPT requests upon encountering errors.
    `REQUESTS_PER_MINUTE`: The maximum number of requests allowed per minute.
    `TOKENS_PER_MINUTE`: The maximum number of tokens allowed per minute.
    `CONCURRENT_TASKS`: The maximum number of concurrent tasks calling the GPT model.
"""

BATCH_SIZE = 3
MAX_RETRIES = 6
REQUESTS_PER_MINUTE = 450  # Actual limit is 500
TOKENS_PER_MINUTE = 140000  # Actual limit is 150000
CONCURRENT_TASKS = 6
