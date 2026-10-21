"""
File: logging.py
Author: Patrick Hannawald (patrick.hannawald@dlr.de)
Date: 2026-01-13
Description: Utility functions for logging
"""

import numpy as np
import logging
import sys
import traceback

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    return logger

def set_basic_logging_config(loglevel: str, log_file: str|None, show_only_warnings_for = ['matplotlib', 'rasterio']):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(format='%(asctime)s, %(levelname)s, %(name)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=getattr(logging, loglevel),
                        handlers=handlers)

    for x in show_only_warnings_for:
        logging.getLogger(x).setLevel(logging.WARNING)


def handle_exception(exc_type, exc_value, exc_traceback):
# Do not print exception when user cancels the program
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.critical("An uncaught exception occurred:")
    logging.critical("Type: %s", exc_type)
    logging.critical("Value: %s", exc_value)

    if exc_traceback:
        format_exception = traceback.format_tb(exc_traceback)
        for line in format_exception:
            logging.critical(repr(line))

def image_logstr(img: np.ndarray) -> str:
    """
    Generate a log string for an image.

    Parameters:
    img (np.ndarray): The image to log.

    Returns:
    str: A log string containing information about the image.
    """
    # Get the type and bit depth of the image
    dtype = img.dtype
    # bit_depth = np.dtype(dtype).itemsize * 8

    # Calculate the minimum, maximum, and mean values of the image
    min_val = np.min(img)
    max_val = np.max(img)
    mean_val = np.mean(img)

    # Check for NaN values
    nan_count = np.isnan(img).sum()

    # Generate the log string
    log_str = (
        f"({dtype}), "
        f"min={min_val:.2f}, max={max_val:.2f}, mean={mean_val:.2f}"
    )

    log_str += f", NaN={nan_count}" if nan_count > 0 else ""

    # Add additional metrics if the image is 2D
    if len(img.shape) == 2:
        # Calculate the standard deviation and variance of the image
        std_dev = np.std(img)

        # Add the metrics to the log string
        log_str += (
            f", sd={std_dev:.2f}"
        )

    return log_str
