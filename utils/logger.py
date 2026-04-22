import logging
import os
import sys
import colorlog

_LEVEL_MAP = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}

def _parse_level(level: str | int) -> int:
    """
    Convert a logging level from string or integer to the corresponding logging constant.

    Parameters
    ----------
    level : str or int
        Logging level as a string ('debug', 'info', ...) or numeric constant.

    Returns
    -------
    int
        Corresponding logging level constant.

    Raises
    ------
    ValueError
        If the string level is not recognized.
    """
    if isinstance(level, str):
        level = level.lower()
        if level not in _LEVEL_MAP:
            raise ValueError(f"Unknown logging level: {level}")
        return _LEVEL_MAP[level]
    return level

def configure_logging(level: str | int = "info", *, stream=sys.stdout) -> None:
    """
    Configure global colored logging (call ONCE in main entry point).

    Parameters
    ----------
    level : str or int, optional
        Logging level as a string ('debug', 'info', 'warning', 'error', 'critical') or numeric constant (e.g., logging.DEBUG). Default is 'info'.
    stream : file-like object, optional
        Stream to which logs will be written. Default is sys.stdout.

    Features
    --------
    - Colored console output
    - Environment override via LOG_LEVEL
    - Clean root logger configuration
    - Safe handler replacement
    """
    # Environment override
    env_level = os.getenv("LOG_LEVEL")
    if env_level:
        level = env_level
    level = _parse_level(level)
    root = logging.getLogger()
    root.setLevel(level)
    # Prevent duplicate handlers safely
    for h in root.handlers[:]:
        root.removeHandler(h)
    handler = colorlog.StreamHandler(stream)
    # Let root logger control filtering → handler stays flexible
    handler.setLevel(logging.DEBUG)
    formatter = colorlog.ColoredFormatter(
        fmt="%(asctime)s | %(log_color)s%(levelname)s%(reset)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)

def get_logger(name: str) -> logging.Logger:
    """
    Retrieve a logger instance.

    This function does not configure logging. It assumes that
    `configure_logging` has already been called.

    Parameters
    ----------
    name : str
        Name of the logger, typically `__name__`.

    Returns
    -------
    logging.Logger
        Logger instance associated with the given name.

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Running solver")
    >>> logger.debug("Internal values")
    """
    return logging.getLogger(name)