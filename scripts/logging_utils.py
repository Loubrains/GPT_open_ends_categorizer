"""
This module provides utility functions for configuring and managing logging.

Functions:
    - setup_logging(): Initializes and configures logging for the application.
        It sets up logging to a file and the console, with specified formatting and log levels.
        
Note: The logging level for each module in the gpt_categorizer_utils package is set within that module.
"""

import logging

# NOTE: The logging level for each module in the gpt_categorizer_utils package is set within that module.


def setup_logging():
    """
    Configures logging to output to a file named 'app.log', and less verbose logging to the console.

    File logging:
        - Outputs to 'app.log', overwriting the file each time the application is started.
        - Log level is set to DEBUG by default.
        - Log format includes the timestamp, file name, line number, log level, and message.

    Console logging:
        - Outputs log messages to the console.
        - Log level is set to INFO.
        - Log format is simplified to show only the message.

    Returns:
        logger (Logger): A configured Logger object which can be used to log messages in the application.
    """

    # Set up file logging
    logging.basicConfig(
        filename="app.log",
        filemode="w",
        level=logging.DEBUG,  # suppress all loggers from dependencies
        format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Set up console logging
    console_handler = logging.StreamHandler()
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # Suppress noisey libraries
    logging.getLogger("chardet").setLevel(logging.WARNING)

    logger.info("Logging initialized")

    return logger
