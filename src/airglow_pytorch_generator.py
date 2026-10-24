"""
PyTorch Module Wrapper for Airglow Wave Sequence Generator

This module provides a PyTorch nn.Module implementation for generating 
superimposed wave sequences. It's designed to be differentiable and can be 
used as part of neural networks for end-to-end training.

The module accepts configurations similar to the airglow_sequence_generator 
but operates with PyTorch tensors and maintains full differentiability.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Optional

from src.mask import PerlinNoise, BrownianNoise
from src.waves import PlaneWave2D, RadialWave2D


class AirglowPyTorchGenerator(nn.Module):
    """
    PyTorch Module for generating superimposed wave sequences.
    
    This module provides a differentiable wrapper around the wave generation 
    functionality, allowing it to be integrated into neural networks for 
    end-to-end training.
    """
    
    def __init__(self, wave_configs: List[Dict[str, Any]], global_params: Dict[str, Any]):
        """
        Initialize the PyTorch wave generator module.
        
        Args:
            wave_configs: List of dictionaries containing hyperparameters for each wave component.
                         Each dictionary should contain parameters for PlaneWave2D class.
            global_params: Dictionary containing global parameters for the sequence generation:
                - timesteps: array of time steps for the sequence
                - image_shape: tuple of (height, width) for generated images
                - resolution_px_per_km: spatial resolution
                - resolution_seconds_per_time_step: temporal resolution
        """
        super(AirglowPyTorchGenerator, self).__init__()
        
        self.wave_configs = wave_configs
        self.global_params = global_params
        
        # Validate parameters
        self.timesteps = global_params.get('timesteps', torch.arange(0, 10))
        self.image_shape = global_params.get('image_shape', (256, 320))
        self.resolution_px_per_km = global_params.get('resolution_px_per_km', 1)
        self.resolution_seconds_per_time_step = global_params.get('resolution_seconds_per_time_step', 1)

        sensor_noise_config = global_params.get('sensor_noise', {})
        # Default to 4 to match your original numpy implementation
        self.sensor_noise_sd = sensor_noise_config.get('sd', None)
        self.sensor_noise_seed = sensor_noise_config.get('seed', None)

        # Create wave objects from configurations
        self.waves = []
        for config in wave_configs:
            # Use a shallow copy so PyTorch tensors keep their computation graph
            wave_kwargs = config.copy()
            # Manually shallow-copy the nested mask dict to prevent mutating it
            mask = wave_kwargs.get('mask', None)
            if isinstance(mask, dict):
                mask = mask.copy()
            if mask is not None:
                wave_kwargs['mask'] = mask
                mask_type = mask.get('type', None)
                if mask_type == 'PerlinNoise':
                    mask_obj = PerlinNoise(**mask)
                elif mask_type == 'BrownianNoise':
                    mask_obj = BrownianNoise(**mask)
                else:
                    mask_obj = mask
                wave_kwargs['mask'] = mask_obj
            wave_type = wave_kwargs.get('type', 'PlaneWave2D')
            if wave_type == 'PlaneWave2D':
                wave = PlaneWave2D(**wave_kwargs)
            elif wave_type == 'RadialWave2D':
                wave = RadialWave2D(**wave_kwargs)
            self.waves.append(wave)
    
    def forward(self, 
                timesteps: Optional[torch.Tensor] = None,
                image_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Generate wave sequence for given timesteps.
        
        Args:
            timesteps: Optional tensor of time steps to generate. If None, uses self.timesteps.
            image_shape: Optional tuple of (height, width) for images. If None, uses self.image_shape.
            
        Returns:
            Tensor of shape (num_timesteps, height, width) containing generated wave images.
        """
        # Use provided values or fallback to defaults
        timesteps = timesteps if timesteps is not None else self.timesteps
        image_shape = image_shape if image_shape is not None else self.image_shape
        
        # Convert to tensor if needed
        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.tensor(timesteps, dtype=torch.int64)
        if not isinstance(image_shape, tuple):
            image_shape = tuple(image_shape)
        
        # Generate the sequence with superposition
        wave_sequence = []
        for t in timesteps:
            wave_image_superposition = None  # Initialize as None
            for wave in self.waves:
                wave.generate(
                    amplitude=wave.amplitude,
                    img_shape=image_shape,
                    timestep=int(t),
                    resolution_px_per_km=self.resolution_px_per_km,
                    resolution_seconds_per_time_step=self.resolution_seconds_per_time_step
                )
                if hasattr(wave._wave_image, 'detach'):
                    tensor_image = wave._wave_image
                else:
                    tensor_image = torch.as_tensor(wave._wave_image, dtype=torch.float32)

                # Dynamically create the canvas on the correct device
                if wave_image_superposition is None:
                    wave_image_superposition = torch.zeros_like(tensor_image)

                wave_image_superposition += tensor_image

            if self.sensor_noise_sd:
                # 1. Create a generator on the same device as our images
                noise_generator = torch.Generator(device=wave_image_superposition.device)

                # 2. Seed it safely
                if self.sensor_noise_seed is not None:
                    # Add 't' to the base seed so every frame gets unique noise,
                    # but the overall sequence remains completely reproducible.
                    noise_generator.manual_seed(self.sensor_noise_seed + int(t))
                else:
                    noise_generator.seed()

                # 3. Generate normal (Gaussian) noise and scale by standard deviation
                noise_tensor = torch.randn(
                    wave_image_superposition.shape,
                    generator=noise_generator,
                    dtype=wave_image_superposition.dtype,
                    device=wave_image_superposition.device
                ) * self.sensor_noise_sd

                # 4. Add the noise to the canvas
                wave_image_superposition += noise_tensor

            wave_sequence.append(wave_image_superposition)
        
        return torch.stack(wave_sequence, dim=0)
    
    def generate_sequence(self, 
                         timesteps: Optional[torch.Tensor] = None,
                         image_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Alternative method to generate wave sequence.
        
        Args:
            timesteps: Optional tensor of time steps to generate. 
            image_shape: Optional tuple of (height, width) for images.
            
        Returns:
            Tensor of shape (num_timesteps, height, width) containing generated wave images.
        """
        return self.forward(timesteps, image_shape)


# Convenience function for creating generator (equivalent to airglow_sequence_generator)
def create_airglow_pytorch_generator(wave_configs: List[Dict[str, Any]], 
                                    global_params: Dict[str, Any]) -> AirglowPyTorchGenerator:
    """
    Create a PyTorch generator with the same interface as the original wrapper.
    
    Args:
        wave_configs: List of dictionaries containing hyperparameters for each wave component.
        global_params: Dictionary containing global parameters for the sequence generation.
        
    Returns:
        AirglowPyTorchGenerator instance.
    """
    return AirglowPyTorchGenerator(wave_configs, global_params)