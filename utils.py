import logging

MASK_TOKEN="[MASK]"


def get_logger():
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)
    return logger