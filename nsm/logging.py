import logging
import stanza


def configure_logging(level=logging.INFO):
    logging.getLogger("stanza").setLevel(logging.WARNING)

    logging.basicConfig(level=logging.INFO)
