"""
File: utility.py
Author: Patrick Hannawald (patrick.hannawald@dlr.de)
Date: 2026-01-13
Description: Utility functions
"""


import numpy as np
import random
import os, re
from scipy.signal import fftconvolve
from PIL import Image, PngImagePlugin
import torch

# for retry
import functools
import time
import logging
from typing import Callable, Type, Tuple, Optional


def convert_azimuth(az: float):
    """Convert azimuth between mathematical orientation (0 = E, 90 = N) and geographic orientation (0 = N, 90 = E)"""
    return (90 - az) % 360

def normal_dist_limited(min_val, max_val):
    """Create normal distributed value and clip between min and max."""
    mu = (max_val + min_val) / 2
    sigma = (mu - min_val) / 3
    r = random.normalvariate(mu, sigma)
    if r > max_val:
        r = max_val
    if r < min_val:
        r = min_val
    return r
    # plt.hist([log_normal_dist_limited(1, 20) for i in range(10000)], bins = 50)


def log_normal_dist_limited(min_val, max_val, mu = 0.00001, sigma = 1.3):
    """Create log normal distributed value and clip between min and max."""
    r = random.lognormvariate(mu, sigma)
    if r > max_val:
        r = max_val
    if r < min_val:
        r = min_val
    return r
    # plt.hist([log_normal_dist_limited(1, 20, 0.00001, 1.3) for i in range(10000)], bins = 50) # <- difficult to find useful values.


def map_range(img, from_range, to_range):
    """
    Differentiable mapping of a tensor from one range to another.
    Clamps values outside the from_range.
    """
    if not torch.is_tensor(img):
        img = torch.as_tensor(img, dtype=torch.float32)

    min_in, max_in = from_range
    min_out, max_out = to_range

    # 1. Clamp the input to the expected bounds
    img_clamped = torch.clamp(img, min=min_in, max=max_in)

    # 2. Normalize to [0.0, 1.0]
    # Add a small epsilon to prevent division by zero if max_in == min_in
    normalized = (img_clamped - min_in) / ((max_in - min_in) + 1e-8)

    # 3. Scale to the target range
    mapped = normalized * (max_out - min_out) + min_out

    return mapped

def adjust_int_parameter(value: int, 
                     min_value: int = 0, 
                     force_odd: bool = False):
    """Adjust parameter for min-range (inclusive) and also 
    allow for odd numbers (adding 1 if number is even).
    
    Example usage could be to make sure that a correct kernel size is used 
    (e.g. for GaussianBlur which should be at least 3 and odd)
    """
    if value < min_value:
        value = min_value
    if force_odd and value % 2 == 0:
        value += 1
    return value
    

def blur(img: np.ndarray, kernelsize: int = 5):
    """Blur an image with the given kernel of 'ones' with size 'kernelsize'."""
    kernel = np.ones((kernelsize, kernelsize), dtype = "float")
    return fftconvolve(img, kernel, mode = "same")


def convert_value_to_range(x: int|float, max_val:int|float = 255, min_val:int|float = 0, need_odd: bool = False):
    """Convert a value into the specified value range and optionally force it to be odd. min_val is inclusive, max_val is exclusive."""
    if min_val > max_val:
        min_val, max_val = max_val, min_val
    if min_val == max_val:
        raise ValueError(f"max_val ({max_val}) has to be greater than min_val ({min_val})!")
    if x < min_val:
        x = min_val
    if x >= max_val:
        x = max_val - 1
    if need_odd:
        if x % 2 == 0:
            if x + 1 > max_val - 1:
                x = x - 1
            else:
                x = x + 1
    return x

def gaussian_noise(img_shape, sd: float, rng: np.random.Generator, _type: type = np.float32):
    """Generate an array with gaussian noise."""
    noise = rng.normal(0, sd, img_shape).astype(_type)
    return noise

def poisson_noise(image: np.ndarray, rng: np.random.Generator, _type: type = np.float32, subtract_image: bool = True):
    """Generate an array with Poisson noise (only the noise, not the noise added to the image)."""
    noise = rng.poisson(image, image.shape).astype(_type) - (image if subtract_image else 0)
    return noise

def list_files(directory: str|list[str], 
               full_names: bool = True, 
               pattern: str|None = None, 
               recursive: bool = False) -> list[str]:
    """Lists files and directories in a specified directory or list of directories.

    Parameters:
    - directory (str | list[str]): The directory path(s) to search within.
    - full_names (bool): If True, returns absolute paths; if False, returns only basenames.
    - pattern (str | None): A regex pattern to match file and directory names. 
            Use e.g. '\\.txt$' or '\\.(csv|xls)$' or '^log' or '\\d{4}-\\d{2}-\\d{2}' for YYYY-MM-DD dates. 
            Or use negative lookahead to avoid certain extensions: '^(?!.*\\.(tmp|bak|swp)$).+'
    - recursive (bool): If True, lists files recursively in subdirectories.

    Returns:
    - List[str]: A list of files and directories matching the specified criteria.
    """
    # Ensure directory is a list for unified processing
    if isinstance(directory, str):
        directory = [directory]
    
    matched_files = []
    
    for dir_path in directory:
        if recursive:
            # Walk through directory structure
            for root, _, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Use basename if full_names is False
                    name_to_check = file_path if full_names else os.path.basename(file_path)
                    # Check if the file matches the pattern
                    if pattern is None or re.search(pattern, name_to_check):
                        matched_files.append(name_to_check)
        else:
            # Only list the files in the top level of the directory
            try:
                for item in os.listdir(dir_path):
                    item_path = os.path.join(dir_path, item)
                    if os.path.isfile(item_path) or os.path.isdir(item_path):
                        # Use basename if full_names is False
                        name_to_check = item_path if full_names else os.path.basename(item_path)
                        # Check if the item matches the pattern
                        if pattern is None or re.search(pattern, name_to_check):
                            matched_files.append(name_to_check)
            except FileNotFoundError:
                # Handle the case where the directory does not exist
                print(f"Directory not found: {dir_path}")
    
    return matched_files

def write_to_png_with_metadata(img: np.ndarray, filename: str, metadata: dict) -> None:
    """
    Save a NumPy image as PNG and embed key/value metadata.

    Supported formats (Anything else raises ValueError):
        * uint8  (H,W)          → "L"
        * uint8  (H,W,3)        → "RGB"
        * uint8  (H,W,4)        → "RGBA"
        * uint16 (H,W)          → "I;16"
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a NumPy ndarray")
    if img.ndim not in (2, 3):
        raise ValueError(f"img must be 2-D or 3-D, got shape {img.shape}")

    # choose Pillow mode
    if img.dtype == np.uint8:
        if img.ndim == 2:
            mode = "L"
        elif img.ndim == 3 and img.shape[2] == 3:
            mode = "RGB"
        elif img.ndim == 3 and img.shape[2] == 4:
            mode = "RGBA"
        else:
            raise ValueError(f"Unsupported uint8 shape {img.shape}")
    elif img.dtype == np.uint16 and img.ndim == 2:
        mode = "I;16"
    else:
        raise ValueError(f"Unsupported dtype/shape: dtype={img.dtype}, shape={img.shape}")

    # create image with metadata
    pil_img = Image.fromarray(img, mode=mode)
    png_info = PngImagePlugin.PngInfo()
    for k, v in metadata.items():
        png_info.add_text(str(k), str(v))

    try:
        pil_img.save(filename, format="PNG", pnginfo=png_info)
    except Exception as exc:
        raise PNGWriteError(f"Failed to write PNG file '{filename}': {exc}") from exc
