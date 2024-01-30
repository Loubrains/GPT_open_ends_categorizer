import logging


def setup_logging():
    # Set up file logging
    logging.basicConfig(
        filename="app.log",
        filemode="w",
        level=logging.WARNING,  # suppress all loggers from dependencies
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
