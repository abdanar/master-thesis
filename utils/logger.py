import logging
import sys
from typing import Optional, Union

def setup_logger(name: Optional[str] = None, level: Union[str, int] = 'info') -> logging.Logger:
    """
    Set up a reusable logger with standardized formatting.

    Parameters
    ----------
    name : str, optional
        Name of the logger, typically `__name__`. Default is None.
    level : str or int, optional
        Logging level as a string ('debug', 'info', 'warning', 'error', 'critical')
        or numeric logging level (e.g., logging.INFO). Default is 'info'.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    
    Notes
    -----
    - If the logger already has handlers, no new handlers are added to prevent duplicate messages.
    - Standard log format includes timestamp, logger name, level, and message.
    """
    # Convert string levels to logging constants
    if isinstance(level, str):
        level = level.lower()
        level_map = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL
        }
        if level not in level_map:
            raise ValueError(f"Unknown logging level: {level}")
        level = level_map[level]

    # Create or get logger
    logger = logging.getLogger(name)

    # Only add handler if none exist to avoid duplicates
    if not logger.hasHandlers():
        logger.setLevel(level)
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

def log_progress(logger: logging.Logger, step: int, nsteps: int, t: float) -> None:
    """
    Log a colorful progress bar for time-stepping loops.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance returned by `setup_logger`
    step : int
        Current step index (0-based)
    nsteps : int
        Total number of steps
    t : float
        Current time
    """

    use_color = sys.stdout.isatty()

    step = min(step, nsteps - 1)

    width = 28
    progress = (step + 1) / nsteps
    filled = int(width * progress)

    if use_color:
        bar = (
            "\033[92m" + "█" * filled +
            "\033[90m" + "─" * (width - filled) +
            "\033[0m"
        )
        msg = (
            "%s  "
            "\033[96mStep %4d/%d\033[0m | "
            "\033[93mt = %.3e\033[0m | "
            "\033[95m%5.1f%%\033[0m"
        )
    else:
        bar = "[" + "=" * filled + "-" * (width - filled) + "]"
        msg = "%s  Step %4d/%d | t = %.3e | %5.1f%%"

    logger.info(
        msg,
        bar,
        step + 1,
        nsteps,
        t,
        100 * progress
    )