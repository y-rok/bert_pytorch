import logging

MASK_TOKEN="[MASK]"


def get_logger():
    logging.basicConfig()
    logger = logging.getLogger("logger")
    return logger