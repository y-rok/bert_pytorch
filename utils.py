import logging

MASK_TOKEN="[MASK]"
VOCAB_FILE_NAME="vocab"
VOCAB_TXT_FILE_NAME="vocab.txt"
MODEL_FILE_NAME="model.pt"


def get_logger():
    logging.basicConfig()
    logger = logging.getLogger("logger")
    return logger