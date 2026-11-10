"""Application logging configuration."""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure application logging at the requested level, defaulting to INFO.

    Module loggers inherit this configuration through normal propagation. The
    logging exercise runs entry operations at INFO and DEBUG to compare output.
    Never log journal text, settings objects, or credentials.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # basicConfig() is a no-op when handlers already exist (uvicorn, pytest),
    # so set the root level explicitly as well.
    logging.getLogger().setLevel(level)
