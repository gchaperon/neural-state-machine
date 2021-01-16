import logging
import stanza


def configure_logging(level=logging.INFO):
    logging.getLogger("stanza").setLevel(logging.WARNING)
    fmt_str = "🕓{asctime} 📍{name: <18} 📨{msg}"
    datefmt = "%d/%m-%H:%M:%S"
    logging.basicConfig(format=fmt_str, datefmt=datefmt, level=logging.INFO, style="{")
