import logging


def setup_logging():
    # Set up file logging
    file_handler = logging.FileHandler("app.log", mode="w")
    file_format = logging.Formatter(
        "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_format)
    file_handler.setLevel(logging.DEBUG)

    # Set up console logging
    console_handler = logging.StreamHandler()
    console_format = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_format)
    console_handler.setLevel(logging.INFO)

    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.getLogger("chardet").setLevel(logging.WARNING)  # Noisy library

    logger.info("Logging initialized")
