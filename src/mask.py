"""
File: mask.py
Author: Patrick Hannawald (patrick.hannawald@dlr.de)
Date: 2026-01-13
Description: Class BaseMask and classes PerlinNoise. BaseMask acts as a 
prototype class for different mask types. PerlinNoise is a type of quasi-
fractal noise with structures looking like cumulus clouds and is used for 
creating a cloud layers (see clouds.py) and masking wave occurrences to
introduce imperfections ("M-Function" read like "Mask-Function", see waves.py).
"""

"""
@author: Perlin noise by tgirod (see https://stackoverflow.com/questions/42147776/producing-2d-perlin-noise-with-numpy)
other things by Patrick Hannawald
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from pydantic import BaseModel, Field
from typing import Literal, Union, Annotated
from abc import abstractmethod
import numpy as np
from math import cos, sin, radians
import cv2

from .utils.utility import map_range, adjust_int_parameter
from .utils.logging import get_logger

logger = get_logger(__name__)

class BaseMask(BaseModel):
    type: str = Field(..., description="discriminator")
    
    @abstractmethod
    def generate(self, image_shape: tuple[int, int], timestep: int):
        raise NotImplementedError("overwrite me")


class PerlinNoise(BaseMask):
    """Class to hold info about a PerlinNoise structure.
    
    A seed of 0 means that the noise is different each time.
    """
    type: Literal['PerlinNoise'] = 'PerlinNoise'
    seed: int = 0
    angle: float = 20
    velocity: float = 0.05
    clip: tuple[float, float] = (0, 1)
    octaves: int = 4
    gauss_ksize: int = 0
    gauss_sigma: float = 20.0
    blur_ksize: int = 0

    def __post_init__(self):
        logger.debug(f"Created {self.type} object!")

    def generate(self, image_shape: tuple[int, int] = (512, 512),
                 timestep: int = 0):
        """Generate fractal noise out of multiple (octave) perlin noise instances to be multiplied e.g. with waves used as a mask or used directly as cloud layer.
        
        Octaves gives the number of frequencies starting with 2, 4, 8, ...
        move_x and move_y translate the noise pattern which is useful when creating consecutive images (then use the same seed)
        clip_range will clip the values leaving either black or white areas for "no wave at all" or "no variation of the wave", respectively.
        clip_range 0 and 1 will not clip anything. Clip (0.7, 1) will make areas more black, clip (0, 0.5) will make more areas white.
        The perlin noise result can be blurred by Gaussian blur or uniform blur (or even both) depending on the respective _ksize values (0 means no blurring).
        Note: the function will not work properly for high values of velocity and timestep. Artefacts will be introduced. 
        """
        # Use the same logic but with torch operations
        # Create torch tensor to work with
        p = torch.zeros(image_shape)
        mx, my = self.move(timestep, offset = 1024)
        
        for i in range(self.octaves):
            freq = 2**i
            lin_x = torch.linspace(mx, freq + mx, image_shape[1], dtype=torch.float32)
            lin_y = torch.linspace(my, freq + my, image_shape[0], dtype=torch.float32)
            x, y = torch.meshgrid(lin_x, lin_y, indexing='xy')
            # print(f"{mx=}, {my=}, {freq+mx=}, {freq+my=}, {np.min(x)=}, {np.max(x)=}, {np.min(y)=}, {np.max(y)=}")
            p = self.perlin(x, y, seed = self.seed) / freq + p
            
        # map to intensity_range without losing information
        p_min, p_max = p.min(), p.max()
        if p_max != p_min:
            p = (p - p_min) / (p_max - p_min) 
        else:
            p = p.zeros_like(p)
            
        # clip values of defined 'intensity range' based on 'clip_range'
        if self.clip != (0, 1):
            # Convert clip values to tensors
            clip_min, clip_max = torch.tensor(self.clip[0]), torch.tensor(self.clip[1])
            p = (p - clip_min) / (clip_max - clip_min)
            p = torch.clamp(p, 0, 1)
        
        if self.gauss_ksize > 0: 
            ksize = adjust_int_parameter(self.gauss_ksize, min_value=3, force_odd=True)
            # Use torchvision's Gaussian blur implementation for better compatibility
            # Reshape to fit expected input (batch, channels, height, width)
            p = p.unsqueeze(0).unsqueeze(0)  
            
            # Use torchvision's gaussian_blur
            p = TF.gaussian_blur(p, kernel_size=ksize, sigma=self.gauss_sigma)
            
            p = p.squeeze(0).squeeze(0)
            
        if self.blur_ksize > 0:
            ksize = adjust_int_parameter(self.blur_ksize, min_value=3, force_odd=True)
            # Use PyTorch's average pooling implementation
            p = p.unsqueeze(0).unsqueeze(0)
            # Use average pooling as blur
            p = F.avg_pool2d(p, kernel_size=ksize, stride=1, padding=ksize//2)
            p = p.squeeze(0).squeeze(0)
            
        logger.debug(f"Generated {self.type} mask: octaves={self.octaves}, clip={self.clip}, velocity={self.velocity}, direction={self.angle}, gauss_ksize={self.gauss_ksize}, blur_ksize={self.blur_ksize}!")
        return p

    # perlin noise functions from tgirod, https://stackoverflow.com/questions/42147776/producing-2d-perlin-noise-with-numpy:
    def perlin(self, x, y, seed: int = 0):
        # permutation table using torch
        if seed != 0:
            torch.manual_seed(seed)
        p = torch.arange(2048, dtype=torch.long)
        p = p[torch.randperm(2048)]
        p = torch.stack([p, p]).flatten()
        
        # coordinates of the top-left (convert to int)
        xi = x.long()
        yi = y.long()
        
        # internal coordinates
        xf, yf = x - xi, y - yi
        
        # fade factors
        u, v = self.fade(xf), self.fade(yf)
        
        # noise components
        n00 = self.gradient(p[p[xi] + yi], xf, yf)
        n01 = self.gradient(p[p[xi] + yi + 1], xf, yf - 1)
        n11 = self.gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
        n10 = self.gradient(p[p[xi + 1] + yi], xf - 1, yf)
        
        # combine noises
        x1 = self.lerp(n00, n10, u)
        x2 = self.lerp(n01, n11, u)  # FIX1: I was using n10 instead of n01
        return self.lerp(x1, x2, v)  # FIX2: I also had to reverse x1 and x2 here

    def lerp(self, a, b, x):
        "linear interpolation"
        return a + x * (b - a)

    def fade(self, t):
        "6t^5 - 15t^4 + 10t^3"
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    def gradient(self, h, x, y):
        "grad converts h to the right gradient vector and return the dot product with (x,y)"
        # Using torch instead of numpy array
        vectors = torch.tensor([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=torch.float32)
        g = vectors[h % 4] 
        return g[:, :, 0] * x + g[:, :, 1] * y


    def move(self, timestep: int = 0, offset: float = 1000):
        """Calculate movement of noise pattern based on angle and velocity for a given timestep."""
        return (cos(radians(self.angle)) * self.velocity * timestep + offset, 
                sin(radians(self.angle)) * self.velocity * timestep + offset)

class BrownianNoise(BaseMask):
    """
    Holds information about and generates a brownian noise field of the size of the image based on 
    the selection of randomly chosen frequencies modified by the Hurst-exponent.
    
    Brownian noise, also red noise is an alternative name to random walk noise.

    Code modified based on Jakob Strutz' code.

    Never tested this code, because I found out that fractional Brownian Noise is actually rather similar to Perlin Noise.
    In PerlinNoise() I have already implemented the possibility to move the field which is not done here. So this function/class
    is not meant to be used, yet, as it is mostly redundant to PerlinNoise() and not so far developed!

    """
    type: Literal['BrownianNoise'] = 'BrownianNoise'
    hurst_exp: float = 1.5
    seed: int = 0
    
    def __post_init__(self):
        logger.debug(f"Created {self.type} object!")
    
    def generate(self, imagesize: tuple[int, int], timestep: int):
        self.width = imagesize[0]
        self.height = imagesize[1]
        fbm_field = self.fbm2d(self.hurst_exp, width = self.width, height = self.height)
        fbm_field = (fbm_field - fbm_field.min()) / (fbm_field.max() - fbm_field.min())
        logger.debug(f"Generated {self.type} mask!")
        return fbm_field

    def fbm2d(self, hurst_exp, width, height):
        # Spectrum
        kx = np.fft.fftfreq(width)
        ky = np.fft.fftfreq(height)
        kx[0], ky[0] = 1, 1  # no division by zero!
        kx, ky = np.meshgrid(kx, ky)
        spectrum = (kx**2 + ky**2)**(-hurst_exp-0.5)
        spectrum[0, 0] = 0
        
        # Zufällige Phasen
        if self.seed != 0:
            np.random.seed(self.seed)
        noise = np.random.normal(size=(height, width)) + 1j * np.random.normal(size=(height, width))
        
        # Anwendung der inversen Fourier-Transformation
        fbm = np.fft.ifft2(noise * np.sqrt(spectrum)).real
        
        return fbm

Mask = Annotated[
    Union[PerlinNoise, BrownianNoise], Field(discriminator="type")
]

BaseMask.model_rebuild()