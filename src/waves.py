"""
File: waves.py
Author: Patrick Hannawald (patrick.hannawald@dlr.de)
Date: 2026-01-13
Description: Classes BaseWave and PlaneWave2D. BaseWave acts as a prototype 
class. The PlaneWave2D creates wave fields using a plane wave equation. 
Additionally, imperfections are added to make the wave fields realistic and
simular to real-world observations. First, multiple similar wave fields are 
created and superimposed to introduce "curviness" and constructive/destrictive 
interferences. Second, a Mask is applied to the wave field to break stationarity
of the wave.
"""

import random
from abc import abstractmethod
from math import sin, cos, pi
from typing import Union, Literal, Annotated

import torch
from pydantic import BaseModel, Field, PrivateAttr, computed_field, ConfigDict

from .mask import Mask
from .utils.logging import get_logger
from .utils.utility import map_range, convert_azimuth

logger = get_logger(__name__)

class BaseWave(BaseModel):
    # Allow PyTorch tensors to pass through Pydantic validation untouched
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: str = Field(..., description="discriminator")

    mask: None|Mask = None # additional imperfection: Wave not present equally over whole grid. White in the mask means 100% of the amplitude, black means 0% of the amplitude.
    _wave_image: None|torch.Tensor = PrivateAttr(default=None) # will be set upon generation
    _mask_image: None|torch.Tensor = PrivateAttr(default=None) # will be set upon generation
    
    @property
    def wave_image(self):
        return self._wave_image
    
    @property
    def mask_image(self):
        return self._mask_image
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.mask:
            logger.debug(f"Mask of type {self.mask.type} is set for {self.type} object.")
    
    @abstractmethod
    def generate(self, img_shape: tuple[int, int], timestep: int = 0, resolution_px_per_km: float = 1, resolution_seconds_per_time_step: float = 1):
        raise NotImplementedError("overwrite me")

    def get_mask(self, lower_threshold: float = 0, upper_threshold: float = 1):
        """Get the noise mask used for the wave."""
        if self._mask_image:
            try:
                return map_range(self._mask_image, [lower_threshold, upper_threshold], [0, 1])
            except:
                raise Exception("Wave not generated, yet! No noise mask available and img_shape unknown to create one.")
        else:
            logger.warning("No mask set, maybe the wave is not 'generate'd, yet?")
            
    def get_binary_mask(self, threshold = 0.5):
        """Get a binary noise mask."""
        self.get_mask(threshold, threshold)


class PlaneWave2D(BaseWave):
    """Class to hold info about a 2D plane wave.
    
    The class is meant to simulate a 2D plane wave moving.
    """
    type: Literal['PlaneWave2D'] = 'PlaneWave2D'

    # Update type hints to accept Tensors so Pydantic doesn't cast them to floats
    wavelength: Union[float, torch.Tensor] = 50.0 # in km
    angle: Union[float, torch.Tensor] = 30.0 # geog. orientation (N = 0°, E = 90°)
    amplitude: Union[float, torch.Tensor] = 1.0 # (arbitrary) amplitude of the wave
    phase: Union[float, torch.Tensor] = 0.0 # initial phase for t = 0
    period: Union[float, torch.Tensor] = 350.0 # period of the wave in seconds
    temperature_amplitude: Union[float, torch.Tensor] = 10.0 # in K
    seed: int = 0 # random seed for wave imperfections, 0: different value each call, all other values will generate the same result each time
    wl_tol: float = 0.2 # wavelength tolerance imperfection (i.e. 0.2 = 20% variation on wavelength parameter for n_waves waves)
    angle_noise: float = 5 # angle tolerance imperfection
    phase_noise: float = pi/12 # phase tolerance imperfection
    n_waves: int = 10 # number of waves to use for imperfections
    individual_mask: None|Mask = None
    _individual_mask_images: None|list[torch.Tensor] = PrivateAttr(default=None)

    def __init__(self, **data) -> None:
      super().__init__(**data)
    #   self.random_wavelengths
    #   self.random_angles
    #   self.random_phases
      logger.debug(f"Created {self.type} object with random wavelengths {[f'{w:.2f}' for w in self.random_wavelengths]} and angles {[f'{a:.2f}' for a in self.random_angles]}!")

    @property
    def angle_geog(self):
        return self.angle

    @computed_field
    @property
    def random_wavelengths(self) -> list[float]:
        return self._compute_random_values_tolerance(self.wl_tol, self.wavelength, self.n_waves - 1, self.seed) # -1 because we already have the original value (e.g. 10 waves = 9 random values needed)

    @computed_field
    @property
    def random_angles(self) -> list[float]:
        return self._compute_random_values_circular_offset(self.angle_noise, self.angle_math, self.n_waves - 1, self.seed)

    @computed_field
    @property
    def random_phases(self) -> list[float]:
        return self._compute_random_values_circular_offset(self.phase_noise, self.phase, self.n_waves - 1, self.seed)
    
    @computed_field
    @property
    def angle_math(self) -> float:
        return convert_azimuth(self.angle)
    
    def create_individual_masks(self, img_shape, timestep):
        start_seed = self.individual_mask.seed
        self._individual_mask_images = []
        for _ in range(self.n_waves):
            self.individual_mask.seed += 1
            self._individual_mask_images.append(self.individual_mask.generate(img_shape, timestep))
        self.individual_mask.seed = start_seed # reset seed if function is called multiple times (also important for varying timesteps)

    def generate(self, amplitude,
                 img_shape: tuple[int, int],
                 timestep: int = 0,
                 resolution_px_per_km: float = 1,
                 resolution_seconds_per_time_step: float = 1):
        """Generate an array with a plane wave (either using the intensity or temperature amplitude) that is distorted by a superposition of multiple wave structures and by perlin noise."""
        if self.individual_mask is not None:
            self.create_individual_masks(img_shape, timestep)
            
        wave = self.generate_plane_wave(img_shape, resolution_px_per_km, resolution_seconds_per_time_step, timestep, 
                                        self.angle_math, self.wavelength, amplitude, self.phase, self.period)

        # Move individual mask to wave.device
        if self.individual_mask is not None:
            mask_img = torch.as_tensor(self._individual_mask_images[0], dtype=torch.float32, device=wave.device)
            wave *= mask_img

        if self.n_waves > 0:
            for i in range(len(self.random_wavelengths)):
                wave_component = self.generate_plane_wave(img_shape, resolution_px_per_km, resolution_seconds_per_time_step, timestep,
                                                 self.random_angles[i], self.random_wavelengths[i], amplitude, self.random_phases[i], self.period)

                # Move component mask to wave.device
                if self.individual_mask is not None:
                    mask_img_i = torch.as_tensor(self._individual_mask_images[i + 1], dtype=torch.float32,
                                                 device=wave.device)
                    wave_component *= mask_img_i
                wave += wave_component
            wave /= self.n_waves

        # Move global mask to wave.device
        if self.mask is not None:
            mask_out = self.mask.generate(img_shape, timestep)
            self._mask_image = torch.as_tensor(mask_out, dtype=torch.float32, device=wave.device)
            wave *= self._mask_image

        self._wave_image = wave
        logger.debug(f"Generated {self.type} image based on wl={self.wavelength}, angle={self.angle}, I={amplitude}, etc. for time step {timestep} and shape {img_shape}!")
    
    def assumptions(self):
        s = f"{self.type}:\n"
        s+= "  2D plane sin-wave.\n"
        s+= "  Create multiple similar such waves based on random wavelength, angle, phase, then superimpose and average by number of waves\n"
        s+= "  Limit wave visibility to (random) mask"
        return s

    # def wavevector(self, Angle_math: float, Wavelength: float):
    #     """Calculate wave vector with trigonometry."""
    #     Angle_math = Angle_math - 90
    #     k2 = 2 * pi / Wavelength * tan(pi/180 * Angle_math) / sqrt(tan(pi/180 * Angle_math)**2 + 1)
    #     k1 = sqrt(4 * pi**2 / Wavelength**2 - k2**2)
    #     return (k1, k2)

    def wavevector(self, Angle_math, Wavelength):
        """Calculate wave vector based on polar coordinates (Differentiable)."""
        Angle_math = Angle_math - 90
        pre = 2 * pi / Wavelength

        # Check if we are optimizing (tensor) or just generating normally (float)
        if isinstance(Angle_math, torch.Tensor):
            return (pre * torch.cos(pi / 180 * Angle_math), pre * torch.sin(pi / 180 * Angle_math))
        else:
            return (pre * cos(pi / 180 * Angle_math), pre * sin(pi / 180 * Angle_math))
    
    def get_current_phase(self, Phase: float, Period: float, timestep: float = 0):
        """Calculate current phase based on given timestep."""
        return Phase + (timestep / Period * 2 * pi)
    
    def plane_wave_function(self, x, y, k, Amplitude, Phase, resolution_px_per_km):
        """Get value of a 2D plane wave for given x and y coordinate for a given wave vector."""
        return Amplitude * (torch.cos(k[0] * x / resolution_px_per_km + k[1] * y / resolution_px_per_km - Phase) - 
                            torch.sin(k[0] * x / resolution_px_per_km + k[1] * y / resolution_px_per_km - Phase))

    def generate_plane_wave(self, img_shape: tuple[int, int], resolution_px_per_km: float,
                            resolution_seconds_per_time_step: float,
                            timestep: int, angle, wavelength, amplitude, phase, period):
        """Generate an array with a plane/plane wave with a given wavelength in km and angle in math. orientation."""

        # Detect the target device from the amplitude tensor (fallback to cpu if float)
        device = amplitude.device if isinstance(amplitude, torch.Tensor) else 'cpu'

        # Add device=device to the linspace creations
        x = torch.linspace(0, img_shape[1] - 1, img_shape[1], dtype=torch.float32, device=device)
        y = torch.linspace(0, img_shape[0] - 1, img_shape[0], dtype=torch.float32, device=device)
        x_grid, y_grid = torch.meshgrid(x, y, indexing='xy')
        
        # Get wave vector
        k = self.wavevector(angle, wavelength)
        
        # Calculate current phase
        current_phase = self.get_current_phase(phase, period, float(timestep) * resolution_seconds_per_time_step)
        
        # Create the wave pattern using tensor operations
        wave = amplitude * (torch.cos(k[0] * x_grid / resolution_px_per_km + k[1] * y_grid / resolution_px_per_km - current_phase) - 
                            torch.sin(k[0] * x_grid / resolution_px_per_km + k[1] * y_grid / resolution_px_per_km - current_phase))
        
        return wave

    def _compute_random_values_tolerance(self, variation: float, parameter, n: int, seed: int = 0):
        """Compute n random values. REMOVED round() to preserve gradients."""
        if seed != 0:
            random.seed(seed)
        return [(1 + random.uniform(-variation, variation)) * parameter for _ in range(n)]

    def _compute_random_values_circular_offset(self, offset: float, parameter, n: int, seed: int = 0):
        """Compute n random values. REMOVED round() to preserve gradients."""
        if seed != 0:
            random.seed(seed)
        return [(random.uniform(-offset, offset) + parameter) % 360 for _ in range(n)]


class RadialWave2D(BaseWave):
    """Class to hold info about a radial wave.
    
    The class is meant to simulate a radial wave moving.
    """
    type: Literal['RadialWave2D'] = 'RadialWave2D'
    wavelength: int = 50 # in pixels
    center: tuple[float, float] = (0, 0) # epicenter of the wave
    phase: float = 0 # initial phase
    amplitude: float = 10
    period: float = 60
    
    # parameters for imperfect waves
    seed: int = 0
    n_waves: int = 10 # generate this much waves with rather similar wavelength and angle and phase
    w_tol: float = 0.1
    c_tol: float = 0.1
    p_tol: float = 0.05
    
    def __post_init__(self):
        """Initialse the random values of varying wavelength, angle, and phase after the usual init of dataclasses."""
        if self.n_waves > 0:
            self.wl = [(1 + random.uniform(-self.w_tol, self.w_tol)) * self.wavelength for i in range(self.n_waves - 1)]
            self.ce = [((1 + random.uniform(-self.c_tol, self.c_tol)) * self.center[0],
                        (1 + random.uniform(-self.c_tol, self.c_tol)) * self.center[1]) for i in range(self.n_waves - 1)]
            self.ph = [(1 + random.uniform(-self.p_tol, self.p_tol)) * self.phase for i in range(self.n_waves - 1)]
        logger.debug(f"Created {self.type} object!")

    @property
    def angle_geog(self):
        return self.angle

    def generate(self, img_shape: tuple[int, int], timestep: int = 0, resolution_px_per_km: float = 1, resolution_seconds_per_time_step: float = 1):
        """Generate an array with a radial wave that is distorted by a superposition of multiple wave structures and by perlin noise."""
        Wave = self.generate_radial_wave(img_shape, resolution_px_per_km, resolution_seconds_per_time_step, timestep, self.wavelength, self.center, self.amplitude, self.phase, self.period)
        if self.n_waves > 0:
            Wave /= self.n_waves
            for i in range(len(self.wl)):
                Wave += self.generate_radial_wave(img_shape, resolution_px_per_km, resolution_seconds_per_time_step, timestep, self.wl[i], self.ce[i], self.amplitude, self.ph[i], self.period) / self.n_waves

        # Move global mask to Wave.device
        if self.mask is not None:
            mask_out = self.mask.generate(img_shape, timestep)
            self._mask_image = torch.as_tensor(mask_out, dtype=torch.float32, device=Wave.device)
            Wave *= self._mask_image

        self._wave_image = Wave
        logger.debug(f"Generated {self.type} image based on λ={self.wavelength}, center={self.center}, I={self.amplitude}, etc. for time step {timestep} and shape {img_shape}!")
    
    def get_current_phase(self, Phase: float, Period: float, timestep: float = 0):
        """Calculate current phase based on given timestep."""
        return Phase + (timestep / Period * 2 * pi)
    
    def radial_wave_function(self, x, y, Center, Wavelength, Amplitude, Phase):
        """Get value of a 2D plane wave for given x and y coordinate for a given wave vector."""
        x0, y0 = Center
        r = torch.sqrt((x - x0)**2 + (y - y0)**2)
        k = 2 * pi / Wavelength
        return Amplitude * (torch.cos(k * r - Phase) - torch.sin(k * r - Phase))

    def generate_radial_wave(self, img_shape: tuple[int, int], resolution_px_per_km: float,
                             resolution_seconds_per_time_step: float,
                             timestep: int, Wavelength: float, Center: tuple[float, float],
                             Amplitude, Phase, period: float):
        """Generate an array with a plane/plane wave with a given wavelength in km and angle in math. orientation."""

        device = Amplitude.device if isinstance(Amplitude, torch.Tensor) else 'cpu'

        x = torch.linspace(0, img_shape[1] - 1, img_shape[1], dtype=torch.float32, device=device)
        y = torch.linspace(0, img_shape[0] - 1, img_shape[0], dtype=torch.float32, device=device)
        x_grid, y_grid = torch.meshgrid(x, y, indexing='xy')
        
        # Calculate current phase
        current_phase = self.get_current_phase(Phase, period, float(timestep) * resolution_seconds_per_time_step)
        
        # Create the radial wave pattern using tensor operations
        x0, y0 = Center
        r = torch.sqrt((x_grid - x0)**2 + (y_grid - y0)**2)
        k = 2 * pi / Wavelength * resolution_px_per_km
        wave = Amplitude * (torch.cos(k * r - current_phase) - torch.sin(k * r - current_phase))
        
        return wave


BaseWave.model_rebuild()

Wave = Annotated[Union[PlaneWave2D, RadialWave2D], Field(discriminator="type")]