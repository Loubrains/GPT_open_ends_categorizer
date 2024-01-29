import logging


def setup_logging():
    # Set up file logging
    logging.basicConfig(
        filename="app.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger()

    # Set up console logging
    console_handler = logging.StreamHandler()
    console_format = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_format)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # Reduce noise
    logging.getLogger("chardet").setLevel(logging.WARNING)

    logger.info("Logging initialized")

    return logger
